# I simply chose them as roughly the smallest median spacing
DISTANCE_THRESHOLDS = {
    # dataset_id: [thresholds] (one for each class)
    "500": [1.0],
    "503": [1.0, 1.0, 1.0],
    "510": [1.5, 1.5, 1.5],
    "511": [1.25, 1.25, 1.25],
    "514": [1.0, 1.0],
    "515": [1.0330772532390826, 1.1328796488598762, 1.1498198361434828],
    # from https://github.com/neheller/kits23/blob/main/kits23/configuration/labels.py
    "520": [1.0],
    "521": [1.0],
    "531": [2.0, 2.0],
    "540": [0.05, 0.01, 0.02],
    "541": [0.05, 0.01, 0.02],
    "542": [0.05, 0.01, 0.02],
    "560": [0.02, 0.02, 0.02, 0.02, 0.02],
}


def get_distance_thresholds(dataset_id: int | str):
    if isinstance(dataset_id, int):
        dataset_id = f"{dataset_id:03d}"
    return DISTANCE_THRESHOLDS[dataset_id]
