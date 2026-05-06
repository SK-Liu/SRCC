# class_mapping.py
from typing import List, Dict

# utils/class_mapping.py

def normalize_name(s):
    # 允许 None；统一成小写+空格
    if s is None:
        return None
    return str(s).lower().replace("_", " ").strip()

def build_test2train_index_map_from_labels(train_labels_dict, test_classes, alias=None):
    """
    返回: 测试索引 -> 训练索引；若无法映射，置为 -1
    alias: 测试类名 -> 训练类名（若值为 None/"" 表示显式跳过该测试类）
    """
    alias = alias or {}
    # 训练端查找表：用 normalize 后的键
    train_lookup = {normalize_name(k): v for k, v in train_labels_dict.items()}
    mapping = {}

    for j, cls in enumerate(test_classes):
        # 先做别名映射
        mapped = alias.get(cls, cls)
        # 显式跳过（None或空串）
        if mapped is None or str(mapped).strip() == "":
            mapping[j] = -1
            continue

        key = normalize_name(mapped)
        if key is None:
            mapping[j] = -1
            continue

        idx = train_lookup.get(key, -1)
        mapping[j] = idx

    return mapping
