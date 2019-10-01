import argparse
import better_exceptions
from pathlib import Path
from collections import defaultdict
from itertools import chain
import json
import pandas as pd
from tqdm import tqdm
import cv2

from util import get_hierarchy, find_contour


def get_args():
    parser = argparse.ArgumentParser(description="This script creates coco format dataset for maskrcnn-benchmark",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--layer", "-l", type=int, default=0,
                        help="target layer; 0 or 1")
    parser.add_argument("--mode", "-m", type=str, default="train",
                        help="target dataset; train or validation")
    parser.add_argument("--img_num", type=int, default=1500,
                        help="max image num for each class")
    args = parser.parse_args()
    return args


def calc_overlap_rate(rect1, rect2):
    x_left = max(rect1.x1, rect2.x1)
    y_top = max(rect1.y1, rect2.y1)
    x_right = min(rect1.x2, rect2.x2)
    y_bottom = min(rect1.y2, rect2.y2)
    intersection = max(0, x_right - x_left) * max(0, y_bottom - y_top)
    iou = intersection / rect1.area

    return iou


class Rect:
    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.cx = (x1 + x2) / 2
        self.cy = (y1 + y2) / 2
        self.area = (self.x2 - self.x1) * (self.y2 - self.y1)

    def is_inside(self, x, y):
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2


def main():
    args = get_args()
    layer = args.layer
    mode = args.mode
    images = []
    annotations = []
    layer0_class_strings, layer1_class_strings, class_string_to_parent = get_hierarchy()
    target_class_strings = get_hierarchy()[layer]
    target_class_string_to_class_id = {class_string: i + 1 for i, class_string in
                                       enumerate(sorted(target_class_strings))}
    parent_class_strings = list(set(class_string_to_parent.values()))
    layer0_independent_class_strings = [class_string for class_string in layer0_class_strings if
                                        class_string not in parent_class_strings]
    data_dir = Path(__file__).parent.joinpath("datasets")
    img_dir = data_dir.joinpath(f"{mode}")
    mask_dir = data_dir.joinpath(f"{mode}_masks")
    mask_csv_path = data_dir.joinpath(f"challenge-2019-{mode}-segmentation-masks.csv")
    df = pd.read_csv(str(mask_csv_path))

    output_dir = data_dir.joinpath("coco")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_annotation_dir = output_dir.joinpath("annotations")
    output_annotation_dir.mkdir(parents=True, exist_ok=True)
    output_img_dir = output_dir.joinpath("train2017")
    output_img_dir.mkdir(parents=True, exist_ok=True)

    class_string_to_img_ids = defaultdict(list)
    img_id_to_meta = defaultdict(list)

    print("=> parsing {}".format(mask_csv_path.name))

    for i, row in tqdm(df.iterrows(), total=len(df)):
        mask_path, img_id, label_name, _, xp1, xp2, yp1, yp2, _, _ = row.values
        class_string_to_img_ids[label_name].append(img_id)
        img_id_to_meta[img_id].append({"mask_path": mask_path, "class_string": label_name,
                                       "bbox": [xp1, xp2, yp1, yp2]})

    # use only args.img_num images for each class
    target_img_ids = list(
        set(chain.from_iterable(
            [class_string_to_img_ids[class_string][:args.img_num] for class_string in target_class_strings])))
    print("=> use {} images for training".format(len(target_img_ids)))
    bbox_id = 0

    for i, img_id in enumerate(tqdm(target_img_ids)):
        added = False
        img_path = img_dir.joinpath(img_id + ".jpg")
        img = cv2.imread(str(img_path), 1)
        h, w, _ = img.shape
        target_rects = []

        # collect target bboxes
        for m in img_id_to_meta[img_id]:
            class_string = m["class_string"]

            # non target
            if class_string not in target_class_strings:
                continue

            xp1, xp2, yp1, yp2 = m["bbox"]
            target_rects.append(Rect(xp1, yp1, xp2, yp2))

        for m in img_id_to_meta[img_id]:
            class_string = m["class_string"]
            xp1, xp2, yp1, yp2 = m["bbox"]
            x1, x2, y1, y2 = xp1 * w, xp2 * w, yp1 * h, yp2 * h

            # for layer1: remove layer0 classes with no child class
            if layer == 1 and class_string in layer0_independent_class_strings:
                continue

            # for both layer0 and layer1: non-target object is removed if it occludes target bbox over 25%
            if class_string not in target_class_strings:
                curr_rect = Rect(xp1, yp1, xp2, yp2)
                overlap_rate = max([calc_overlap_rate(r, curr_rect) for r in target_rects])

                if overlap_rate > 0.25:
                    continue

            # layer0: convert layer1 and layer2 classes to their parent layer0 classes
            if layer == 0:
                if class_string in class_string_to_parent.keys():
                    class_string = class_string_to_parent[class_string]

                if class_string in class_string_to_parent.keys():  # needed for layer2 classes
                    class_string = class_string_to_parent[class_string]

            if class_string in target_class_strings:
                mask_path = mask_dir.joinpath(m["mask_path"])
                mask_img = cv2.imread(str(mask_path), 0)
                mask_img = cv2.resize(mask_img, (w, h), cv2.INTER_NEAREST)
                contour = find_contour(mask_img)
                contour = [p for p in contour if len(p) > 4]

                if not contour:
                    continue

                class_id = target_class_string_to_class_id[class_string]
                gt_bbox = [x1, y1, x2 - x1, y2 - y1]
                box_w, box_h = x2 - x1, y2 - y1

                if box_w < 10 or box_h < 10:
                    print(box_w, box_h)

                annotations.append({# "area": gt_h * gt_h,
                                    "segmentation": contour,
                                    "iscrowd": 0,
                                    "image_id": i,
                                    "bbox": gt_bbox,
                                    "category_id": class_id,
                                    "id": bbox_id})
                bbox_id += 1
                added = True

            # for layer1: fill non-target bbox with gray; this class has its child class in layer1
            else:
                x1, x2, y1, y2 = int(x1), int(x2), int(y1), int(y2)
                img[y1:y2, x1:x2] = 128

        if not added:
            continue

        images.append({"file_name": img_path.name,
                       "height": h,
                       "width": w,
                       "id": i})

        output_img_path = output_img_dir.joinpath(img_path.name)
        cv2.imwrite(str(output_img_path), img)

    categories = [{"supercategory": "object", "id": class_id, "name": class_string} for class_string, class_id in
                  target_class_string_to_class_id.items()]

    with output_annotation_dir.joinpath("instances_train2017.json").open("w") as f:
        json.dump({"images": images, "annotations": annotations, "categories": categories}, f)


if __name__ == '__main__':
    main()
