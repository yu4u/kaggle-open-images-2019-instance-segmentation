import argparse
import better_exceptions
from pathlib import Path
import json
from collections import defaultdict
import pandas as pd
import cv2
import base64
import numpy as np
from pycocotools import _mask as coco_mask
import typing as t
import zlib


def get_class_mapping():
    csv_path = Path(__file__).parent.joinpath("datasets", "classes-segmentation.txt")
    df = pd.read_csv(str(csv_path), header=None, names=["class_string"])
    class_string_to_class_id = dict(zip(df.class_string, df.index))
    class_id_to_class_string = dict(zip(df.index, df.class_string))
    return class_string_to_class_id, class_id_to_class_string


def get_string_to_name():
    csv_path = Path(__file__).parent.joinpath("datasets", "challenge-2019-classes-description-segmentable.csv")
    df = pd.read_csv(str(csv_path), header=None, names=["class_string", "class_name"])
    class_string_to_class_name = dict(zip(df.class_string, df.class_name))
    return class_string_to_class_name


def get_layer_names():
    layer0, layer1, class_string_to_parent = get_hierarchy()
    class_string_to_class_name = get_string_to_name()
    layer0_names = [class_string_to_class_name[s] for s in sorted(layer0)]
    layer1_names = [class_string_to_class_name[s] for s in sorted(layer1)]
    return layer0_names, layer1_names


def get_hierarchy():
    json_path = Path(__file__).parent.joinpath("datasets", "challenge-2019-label300-segmentable-hierarchy.json")

    with json_path.open("r") as f:
        d = json.load(f)

    level_to_class_strings = defaultdict(list)
    class_string_to_parent = {}

    def register(c, level):
        class_string = c["LabelName"]
        level_to_class_strings[level].append(class_string)

        if "Subcategory" in c.keys():
            for sub_c in c["Subcategory"]:
                class_string_to_parent[sub_c["LabelName"]] = class_string
                register(sub_c, level + 1)

    for c in d["Subcategory"]:
        register(c, 0)

    class_string_to_parent["/m/0kmg4"] = "/m/0138tl"  # teddy bear is toy not bear...
    layer0 = level_to_class_strings[0]
    layer1 = list(set(level_to_class_strings[1] + level_to_class_strings[2]))
    return sorted(layer0), sorted(layer1), class_string_to_parent


def find_contour(mask):
    mask = cv2.UMat(mask)
    contour, hierarchy = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1
    )

    reshaped_contour = []
    for entity in contour:
        assert len(entity.shape) == 3
        assert entity.shape[1] == 1, \
            'Hierarchical contours are not allowed'
        reshaped_contour.append(entity.reshape(-1).tolist())
    return reshaped_contour


def encode_binary_mask(mask: np.ndarray) -> t.Text:
    """Converts a binary mask into OID challenge encoding ascii text."""

    # check input mask --
    if mask.dtype != np.bool:
        raise ValueError(
            "encode_binary_mask expects a binary mask, received dtype == %s" %
            mask.dtype)

    mask = np.squeeze(mask)
    if len(mask.shape) != 2:
        raise ValueError(
            "encode_binary_mask expects a 2d mask, received shape == %s" %
            mask.shape)

    # convert input mask to expected COCO API input --
    mask_to_encode = mask.reshape(mask.shape[0], mask.shape[1], 1)
    mask_to_encode = mask_to_encode.astype(np.uint8)
    mask_to_encode = np.asfortranarray(mask_to_encode)

    # RLE encode mask --
    encoded_mask = coco_mask.encode(mask_to_encode)[0]["counts"]

    # compress and base64 encoding --
    binary_str = zlib.compress(encoded_mask, zlib.Z_BEST_COMPRESSION)
    base64_str = base64.b64encode(binary_str)
    return base64_str


def main():
    get_hierarchy()
    print(get_layer_names()[1])


if __name__ == '__main__':
    main()
