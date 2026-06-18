import os

import numpy as np


def is_kitti_bin(bin_path):
    if not os.path.isfile(bin_path):
        return False
    size = os.path.getsize(bin_path)
    if size == 0 or size % 16 != 0:
        return False

    data = np.fromfile(bin_path, dtype=np.float32)
    if data.size == 0 or data.size % 4 != 0:
        return False

    try:
        points = data.reshape(-1, 4)
    except ValueError:
        return False

    if not np.isfinite(points).all():
        return False

    if points.shape[0] == 0:
        return False

    return True


def validate_kitti_bin(bin_path):
    if not os.path.exists(bin_path):
        return False

    if not is_kitti_bin(bin_path):
        return False

    points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)

    if not np.isfinite(points).all():
        return False

    xyz = points[:, :3]
    if xyz.size == 0:
        return False

    # 内容检查：点坐标不是全部相同，也不是全部零
    if np.allclose(xyz, xyz[0], atol=1e-6):
        return False
    if np.allclose(xyz, 0.0, atol=1e-6):
        return False

    # intensity 存在且为 finite
    intensity = points[:, 3]
    if not np.isfinite(intensity).all():
        return False

    return True


def batch_validate(input_dir):
    if not os.path.isdir(input_dir):
        raise ValueError("Input directory does not exist: " + input_dir)

    good = []
    bad = []
    for name in sorted(os.listdir(input_dir)):
        if not name.lower().endswith(".bin"):
            continue
        path = os.path.join(input_dir, name)
        if validate_kitti_bin(path):
            good.append(name)
        else:
            bad.append(name)

    print(f"Valid KITTI bins: {len(good)}")
    print(f"Invalid KITTI bins: {len(bad)}")
    for name in bad:
        print("Invalid:", name)
    return len(bad) == 0


if __name__ == "__main__":
    # 单个文件检查
    # print(validate_kitti_bin('path/to/scan.bin'))

    # 批量检查目录
    batch_validate("./datasets/test_data/velodyne")
