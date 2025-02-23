import json
from typing import Optional

from torch import tensor
from pathlib import Path
import os


def parse_annotation_to_fasterrcnn_format(
    annotation_data: dict, labels_list: dict = {'curl': 0, 'healthy': 1, 'slug': 2, 'spot': 3}
) -> Optional[dict]:
    try:
        value = annotation_data['result'][0]['value']
        rectangle_labels = value['rectanglelabels']
        x_min = value['x']
        y_min = value['y']
        x_max = value['width']
        y_max = value['height']
        label = convert_ls_to_boxes([x_min, y_min, x_max, y_max], 224, 224)
        if label[0] > label[2] or label[1] > label[3]:
            return None
        label.append(labels_list[rectangle_labels[0]])
    except (KeyError, IndexError):
        return None
    return {'label': label}


def parse_annotation_to_fasterrcnn_format_from_list(annotation_data_list: list[dict]) -> list[dict]:
    data = []
    for annotation_data in annotation_data_list:
        if annotation_data['annotations'][0]['result'] != []:
            parsed = parse_annotation_to_fasterrcnn_format(annotation_data['annotations'][0])
        if parsed is not None:
            image_path = annotation_data['data']['image']
            if "plant-disease" in image_path:
                clean_path = image_path.split("plant-disease", 1)[1]
                parsed['image'] = Path(os.getcwd() + clean_path)
            else:
                parsed['image'] = Path(image_path)
            data.append(parsed)
    return data


def convert_boxes_to_ls_format(box: list, shape: tensor) -> list:
    image_width, image_height = shape[1:]
    x_min, y_min, x_max, y_max = box
    x = (x_min / image_width) * 100
    y = (y_min / image_height) * 100
    width = (abs(x_max - x_min) / image_width) * 100
    height = (abs(y_max - y_min) / image_height) * 100
    return [x, y, width, height]


def convert_ls_to_boxes(box: list, image_width: int, image_height: int) -> list:
    x, y, width, height = box
    x_min = (x / 100) * image_width
    y_min = (y / 100) * image_height
    x_max = (abs(x + width) / 100) * image_width
    y_max = (abs(y + height) / 100) * image_height
    return [x_min, y_min, x_max, y_max]


def serialize_data(data: dict) -> Optional[str]:
    try:
        return json.dumps(data)
    except Exception as e:
        print(f'Error serializing data: {e}')
        return None


def deserialize_data(data: str) -> list:
    try:
        return json.loads(data)
    except Exception as e:
        print(f'Error deserializing data: {e}')
        return []


def verify_annotation(annotation_data: dict) -> bool:
    if parse_annotation_to_fasterrcnn_format(annotation_data) is not None:
        return True
    return False
