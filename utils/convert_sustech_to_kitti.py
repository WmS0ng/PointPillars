import glob
import json
import math
import os


def convert_sustech_to_kitti(json_path, output_path, calib_file=None):
    """
    将 SUSTechPOINTS JSON 标注转换为 KITTI 格式 .txt 文件

    Args:
        json_path: 输入 JSON 文件路径
        output_path: 输出 KITTI 标签文件路径
        calib_file: 标定文件路径（可选，若无则位置直接填入 LiDAR 坐标）
    """

    with open(json_path, "r") as f:
        data = json.load(f)

    lines = []

    for obj in data:
        obj_id = obj["obj_id"]
        obj_type = obj["obj_type"]

        # 无图像信息时默认值
        truncated = 0.00
        occluded = 0
        alpha = 0.0
        bbox = [0.0, 0.0, 50.0, 50.0]

        # 雷达信息
        height = obj["psr"]["scale"]["z"]
        width = obj["psr"]["scale"]["y"]
        length = obj["psr"]["scale"]["x"]

        loc_x = obj["psr"]["position"]["x"]
        loc_y = obj["psr"]["position"]["y"]
        loc_z = obj["psr"]["position"]["z"]

        # 旋转角转换
        # rotation_y_kitti = - (rotation_z_sustech + pi/2)
        rotation_y = -(obj["psr"]["rotation"]["z"] + math.pi / 2)

        # 组装 kitti 行
        line = (
            f"{obj_type} {truncated:.2f} {occluded} {alpha:.2f} "
            f"{bbox[0]:.2f} {bbox[1]:.2f} {bbox[2]:.2f} {bbox[3]:.2f} "
            f"{height:.2f} {width:.2f} {length:.2f} "
            f"{loc_x:.2f} {loc_y:.2f} {loc_z:.2f} "
            f"{rotation_y:.2f}\n"
        )
        lines.append(line)

    with open(output_path, "w") as f:
        f.writelines(lines)

    print(f"Converted: {json_path} -> {output_path}")


def batch_convert(input_dir, output_dir):
    """批量转换目录下所有 JSON 文件"""
    os.makedirs(output_dir, exist_ok=True)
    json_files = glob.glob(os.path.join(input_dir, "*.json"))

    for json_file in sorted(json_files):
        basename = os.path.basename(json_file).replace(".json", ".txt")
        output_path = os.path.join(output_dir, basename)
        convert_sustech_to_kitti(json_file, output_path)


if __name__ == "__main__":
    # 单个文件转换
    # convert_sustech_to_kitti("0000.json", "0000.txt")

    # 批量转换
    batch_convert("./datasets/test_data/label", "./datasets/test_data/label_kitti")
