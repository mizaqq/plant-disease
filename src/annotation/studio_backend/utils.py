import json
from torch import tensor
from typing import Optional


def parse_annotation_to_fasterrcnn_format(annotation_data: dict) -> dict:
    labels_list = {'curl': 0, 'healthy': 1, 'slug': 2, 'spot': 3}
    result = annotation_data.get('result', [])

    labels = []

    for item in result:
        value = item.get('value', {})
        rectangle_labels = value.get('rectanglelabels', [])
        if rectangle_labels:
            x = value['x']
            y = value['y']
            width = value['width']
            height = value['height']

            label = convert_ls_to_boxes([x, y, width, height], 224, 224)
            label.append(labels_list[rectangle_labels[0]])
            labels.append(label)

    return {
        'labels': labels,
    }


def convert_boxes_to_ls_format(box: list, shape: tensor) -> list:
    image_width, image_height = shape[1:]
    x_min, y_min, x_max, y_max = box
    x = (x_min / image_width) * 100
    y = (y_min / image_height) * 100
    width = ((x_max - x_min) / image_width) * 100
    height = ((y_max - y_min) / image_height) * 100
    return [x, y, width, height]


def convert_ls_to_boxes(box: list, image_width: int, image_height: int) -> list:
    x, y, width, height = box
    x_min = (x / 100) * image_width
    y_min = (y / 100) * image_height
    x_max = ((x + width) / 100) * image_width
    y_max = ((y + height) / 100) * image_height
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
