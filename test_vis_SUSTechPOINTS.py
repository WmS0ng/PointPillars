import argparse
import json
import os

import numpy as np
import torch

from pointpillars.model import PointPillars
from pointpillars.utils import setup_seed
from pointpillars.utils.vis_o3d import vis_pc


def load_pcd(pcd_path):
    with open(pcd_path, "rb") as f:
        header = {}
        while True:
            line = f.readline().decode("utf-8", errors="ignore").strip()
            if not line:
                continue
            if line.startswith("DATA"):
                header["DATA"] = line.split()[1].lower()
                break
            key, *values = line.split()
            header[key] = values

        if header["DATA"] == "ascii":
            data = np.loadtxt(f, dtype=np.float32)
        else:
            fields = header.get("FIELDS", [])
            sizes = list(map(int, header.get("SIZE", ["4"] * len(fields))))
            types = header.get("TYPE", ["F"] * len(fields))
            counts = list(map(int, header.get("COUNT", ["1"] * len(fields))))
            points = int(header.get("POINTS", [0])[0]) if "POINTS" in header else 0

            def map_type(typ, size):
                if typ == "F":
                    if size == 4:
                        return "<f4"
                    if size == 8:
                        return "<f8"
                    return "<f2"
                if typ == "I":
                    if size == 1:
                        return "<i1"
                    if size == 2:
                        return "<i2"
                    return "<i4"
                if typ == "U":
                    if size == 1:
                        return "<u1"
                    if size == 2:
                        return "<u2"
                    return "<u4"
                raise ValueError(f"Unsupported PCD type: {typ}")

            dtype_list = []
            pad_count = 0
            for name, size, typ, cnt in zip(fields, sizes, types, counts):
                if cnt == 0:
                    continue
                if name == "_":
                    name = f"_pad{pad_count}"
                    pad_count += 1
                np_dtype = np.dtype(map_type(typ, size))
                dtype_list.append(
                    (name, np_dtype) if cnt == 1 else (name, np_dtype, cnt)
                )

            dtype = np.dtype(dtype_list)
            raw = f.read()
            if points <= 0:
                points = len(raw) // dtype.itemsize
            data = np.frombuffer(raw, dtype=dtype, count=points)

    if hasattr(data, "dtype") and data.dtype.names is not None:
        x = data["x"].astype(np.float32)
        y = data["y"].astype(np.float32)
        z = data["z"].astype(np.float32)
        if "intensity" in data.dtype.names:
            intensity = data["intensity"].astype(np.float32)
        elif "i" in data.dtype.names:
            intensity = data["i"].astype(np.float32)
        else:
            intensity = np.zeros_like(x, dtype=np.float32)
        points = np.stack([x, y, z, intensity], axis=-1)
    else:
        points = data.astype(np.float32)
        if points.ndim == 1:
            raise ValueError("PCD points must have at least x,y,z columns")
        if points.shape[1] == 3:
            zeros = np.zeros((points.shape[0], 1), dtype=np.float32)
            points = np.concatenate([points, zeros], axis=1)
        elif points.shape[1] > 4:
            points = points[:, :4]
    return points


def point_range_filter(pts, point_range=[0, -39.68, -3, 69.12, 39.68, 1]):
    flag_x_low = pts[:, 0] > point_range[0]
    flag_y_low = pts[:, 1] > point_range[1]
    flag_z_low = pts[:, 2] > point_range[2]
    flag_x_high = pts[:, 0] < point_range[3]
    flag_y_high = pts[:, 1] < point_range[4]
    flag_z_high = pts[:, 2] < point_range[5]
    keep_mask = (
        flag_x_low & flag_y_low & flag_z_low & flag_x_high & flag_y_high & flag_z_high
    )
    return pts[keep_mask]


def load_json_labels(json_path):
    with open(json_path, "r") as f:
        objs = json.load(f)

    gt_bboxes = []
    gt_labels = []
    for obj in objs:
        obj_type = obj.get("obj_type", "Unknown")
        if obj_type not in ["Pedestrian", "Cyclist", "Car"]:
            continue
        pos = obj["psr"]["position"]
        rot = obj["psr"]["rotation"]
        scale = obj["psr"]["scale"]
        x = float(pos["x"])
        y = float(pos["y"])
        # z = float(pos["z"])
        z = float(pos["z"]) - float(scale["z"]) / 2
        l = float(scale["x"])
        w = float(scale["y"])
        h = float(scale["z"])
        # rotation_z = float(rot.get("z", 0.0))
        # theta = -(rotation_z + np.pi / 2)
        theta = float(rot["z"])
        gt_bboxes.append([x, y, z, l, w, h, theta])
        gt_labels.append(-1)
    if len(gt_bboxes) == 0:
        return np.zeros((0, 7), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    return np.array(gt_bboxes, dtype=np.float32), np.array(gt_labels, dtype=np.int64)


def main(args):
    setup_seed()
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    )
    model = PointPillars(nclasses=args.nclasses).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt)
    model.eval()

    if not os.path.exists(args.pc_path):
        raise FileNotFoundError(f"Point cloud file not found: {args.pc_path}")

    pts = load_pcd(args.pc_path)
    pts = point_range_filter(pts)
    pts_torch = torch.from_numpy(pts).to(device)

    result = None
    with torch.no_grad():
        result = model(batched_pts=[pts_torch], mode="test")[0]

    pred_bboxes = np.array(result["lidar_bboxes"], dtype=np.float32)
    # 修改模型中心点的z，以适配vis_pc函数，因vis_pc函数需要的z是底边坐标
    pred_bboxes[:, 2] = pred_bboxes[:, 2] - pred_bboxes[:, 5] / 2
    pred_labels = np.array(result["labels"], dtype=np.int64)

    if args.gt_path and os.path.exists(args.gt_path):
        gt_bboxes, gt_labels = load_json_labels(args.gt_path)
    else:
        gt_bboxes = np.zeros((0, 7), dtype=np.float32)
        gt_labels = np.zeros((0,), dtype=np.int64)

    vis_bboxes = pred_bboxes
    vis_labels = pred_labels
    if gt_bboxes.shape[0] > 0:
        vis_bboxes = (
            np.concatenate([pred_bboxes, gt_bboxes], axis=0)
            if pred_bboxes.shape[0] > 0
            else gt_bboxes
        )
        vis_labels = (
            np.concatenate([pred_labels, gt_labels], axis=0)
            if pred_labels.shape[0] > 0
            else gt_labels
        )

    vis_pc(pts, bboxes=vis_bboxes, labels=vis_labels)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize model prediction on test_data point cloud"
    )
    parser.add_argument(
        "--ckpt",
        default="my_pillar_logs/checkpoints/epoch_100.pth",
        help="checkpoint path",
    )
    parser.add_argument(
        "--pc_path",
        default="datasets/test_data/lidar/000970.pcd",
        help="input point cloud .pcd path",
    )
    parser.add_argument(
        "--gt_path",
        default="datasets/test_data/label/000970.json",
        help="optional ground truth json path",
    )
    parser.add_argument("--nclasses", type=int, default=3)
    parser.add_argument("--no_cuda", action="store_true", help="force cpu")
    args = parser.parse_args()
    main(args)
