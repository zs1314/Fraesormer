import os
import shutil
import random
from pathlib import Path


def split_and_copy_data(source_dir, dest_dir, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2):
    """
    将数据集按照指定比例划分为训练集、验证集和测试集，并复制文件到目标目录。
    同时统计并打印每个子集的图片总数。
    """
    assert train_ratio + val_ratio + test_ratio == 1, "划分比例之和必须等于1。"

    # 统计各个子集的图片数量
    counts = {'train': 0, 'val': 0, 'test': 0}

    # 创建目标文件夹
    for split in ['train', 'val', 'test']:
        split_path = os.path.join(dest_dir, split)
        for class_dir in os.listdir(source_dir):
            if not class_dir.startswith('.'):  # 跳过隐藏文件夹
                os.makedirs(os.path.join(split_path, class_dir), exist_ok=True)

    # 遍历每个类别
    for class_dir in os.listdir(source_dir):
        class_path = os.path.join(source_dir, class_dir)
        if os.path.isdir(class_path) and not class_dir.startswith('.'):  # 跳过隐藏文件夹
            # 获取该类别下的所有有效文件
            files = [f for f in os.listdir(class_path) if
                     f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'))]
            if not files:  # 如果没有有效文件，则跳过
                continue
            random.shuffle(files)  # 随机打乱文件顺序

            # 计算每个子集的文件数
            total_files = len(files)
            train_count = int(train_ratio * total_files)
            val_count = int(val_ratio * total_files)

            # 分配文件到各个子集
            train_files = files[:train_count]
            val_files = files[train_count:train_count + val_count]
            test_files = files[train_count + val_count:]

            # 复制文件到相应的文件夹
            for file in train_files:
                shutil.copy2(os.path.join(class_path, file), os.path.join(dest_dir, 'train', class_dir, file))
                counts['train'] += 1
            for file in val_files:
                shutil.copy2(os.path.join(class_path, file), os.path.join(dest_dir, 'val', class_dir, file))
                counts['val'] += 1
            for file in test_files:
                shutil.copy2(os.path.join(class_path, file), os.path.join(dest_dir, 'test', class_dir, file))
                counts['test'] += 1

            # 打印调试信息
            print(f"类别: {class_dir}")
            print(f"总文件数: {total_files}")
            print(f"训练集文件数: {len(train_files)}")
            print(f"验证集文件数: {len(val_files)}")
            print(f"测试集文件数: {len(test_files)}")

    # 打印每个子集的图片总数
    print(f"训练集图片总数: {counts['train']}")
    print(f"验证集图片总数: {counts['val']}")
    print(f"测试集图片总数: {counts['test']}")


# 调用函数
source_directory = r"E:\Food Data\UECFOOD256"
destination_directory = r'E:\Food Data\UECFOOD256_huafen'
split_and_copy_data(source_directory, destination_directory)
