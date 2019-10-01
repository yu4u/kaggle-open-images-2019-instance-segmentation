import argparse
import better_exceptions
from pathlib import Path
import json
from tqdm import tqdm
import numpy as np
import pandas as pd
import cv2
from joblib import Parallel, delayed
from util import encode_binary_mask, get_hierarchy


def get_args():
    parser = argparse.ArgumentParser(description="This script creates submission file from maskrcnn results.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--th", type=int, default=128,
                        help="threshold for mask")
    parser.add_argument("--layer", "-l", type=int, default=0,
                        help="target layer; 0 or 1")
    args = parser.parse_args()
    return args


def create_result_per_img(json_path, results_dir, th, class_strings):
    with json_path.open("r") as f:
        d = json.load(f)

    new_w, new_h = d["new_img_size"]
    boxes = d["boxes"]
    scores = d["scores"]
    labels = d["labels"]
    img_id = json_path.stem
    predicted_strings = []

    for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
        mask_path = results_dir.joinpath("{}_{}.png".format(json_path.stem, i))
        predicted_mask_img = cv2.imread(str(mask_path), 0)
        mask_img = (predicted_mask_img > th)
        mask_string = encode_binary_mask(mask_img)
        class_string = class_strings[label - 1]
        predicted_string = "{} {} {}".format(class_string, score, mask_string.decode())
        predicted_strings.append(predicted_string)

    predicted_strings = " ".join(predicted_strings)

    return img_id, new_w, new_h, predicted_strings


def main():
    args = get_args()
    th = args.th
    layer = args.layer
    data_dir = Path(__file__).parent.joinpath("datasets")
    results_dir = data_dir.joinpath(f"test_{layer}_results")
    layer0, layer1, class_string_to_parent = get_hierarchy()
    class_strings = layer0 if layer == 0 else layer1

    r = Parallel(n_jobs=-1, verbose=10)(
        [delayed(create_result_per_img)(json_path, results_dir, th, class_strings) for json_path in
         results_dir.glob("*.json")])

    df = pd.DataFrame(data=r, columns=["ImageID", "ImageWidth", "ImageHeight", "PredictionString"])
    df.dropna(inplace=True)
    df["ImageWidth"] = df["ImageWidth"].astype(np.int)
    df["ImageHeight"] = df["ImageHeight"].astype(np.int)
    df.to_csv("output_{}.csv".format(layer), index=False)


if __name__ == '__main__':
    main()
