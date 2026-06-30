import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from pointpillars.dataset import get_dataloader
from pointpillars.loss import Loss
from pointpillars.model import PointPillars
from pointpillars.utils import setup_seed


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
                    return "<f4" if size == 4 else "<f8" if size == 8 else "<f2"
                if typ == "I":
                    return "<i1" if size == 1 else "<i2" if size == 2 else "<i4"
                if typ == "U":
                    return "<u1" if size == 1 else "<u2" if size == 2 else "<u4"
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
            arr = np.frombuffer(raw, dtype=dtype, count=points)
            data = arr

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
            if data.ndim == 1:
                raise ValueError("PCD points must have at least x,y,z columns")
            points = data.astype(np.float32)
            if points.shape[1] == 3:
                zeros = np.zeros((points.shape[0], 1), dtype=np.float32)
                points = np.concatenate([points, zeros], axis=1)
            elif points.shape[1] > 4:
                points = points[:, :4]
        return points


class SUSTechPOINTSDataset(Dataset):
    CLASSES = {"TruckBed": 0, "ExcavatorBucket": 1}

    def __init__(self, data_root, split="train", train_ratio=0.8):
        assert split in ["train", "val"]
        self.data_root = data_root
        self.split = split
        self.point_dir = os.path.join(data_root, "lidar")
        self.label_dir = os.path.join(data_root, "label")

        ids = [
            os.path.splitext(x)[0]
            for x in sorted(os.listdir(self.point_dir))
            if x.lower().endswith(".pcd")
        ]
        ids = [
            i for i in ids if os.path.exists(os.path.join(self.label_dir, f"{i}.json"))
        ]
        ids.sort()
        split_idx = int(len(ids) * train_ratio)
        if split == "train":
            self.ids = ids[:split_idx] if split_idx > 0 else ids
        else:
            self.ids = ids[split_idx:] if split_idx < len(ids) else []

        if len(self.ids) == 0:
            raise ValueError(f"No sample ids found for split={split} in {data_root}")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        sample_id = self.ids[index]
        pcd_path = os.path.join(self.point_dir, f"{sample_id}.pcd")
        label_path = os.path.join(self.label_dir, f"{sample_id}.json")

        pts = load_pcd(pcd_path)
        gt_bboxes = []
        gt_labels = []
        difficulty = []

        with open(label_path, "r") as f:
            objs = json.load(f)

        for obj in objs:
            obj_type = obj.get("obj_type", "Unknown")
            if obj_type not in self.CLASSES:
                continue
            pos = obj["psr"]["position"]
            rot = obj["psr"]["rotation"]
            scale = obj["psr"]["scale"]

            x = float(pos["x"])
            y = float(pos["y"])
            z = float(pos["z"])
            l = float(scale["x"])
            w = float(scale["y"])
            h = float(scale["z"])
            # rotation_z = float(rot.get("z", 0.0))
            # theta = -(rotation_z + np.pi / 2)
            theta = float(rot.get("z", 0.0))
            gt_bboxes.append([x, y, z, l, w, h, theta])
            gt_labels.append(self.CLASSES[obj_type])
            difficulty.append(0)

        if len(gt_bboxes) == 0:
            gt_bboxes = np.zeros((0, 7), dtype=np.float32)
            gt_labels = np.zeros((0,), dtype=np.int64)
            difficulty = np.zeros((0,), dtype=np.int64)
        else:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
            difficulty = np.array(difficulty, dtype=np.int64)

        # 动态反向映射
        inv_classes = {v: k for k, v in self.CLASSES.items()}

        data_dict = {
            "pts": pts,
            "gt_bboxes_3d": gt_bboxes,
            "gt_labels": gt_labels,
            "gt_names": [inv_classes[label] for label in gt_labels],
            "difficulty": difficulty,
            "image_info": {},
            "calib_info": {},
        }
        return data_dict


def save_summary(writer, loss_dict, global_step, tag, lr=None, momentum=None):
    for k, v in loss_dict.items():
        writer.add_scalar(f"{tag}/{k}", v, global_step)
    if lr is not None:
        writer.add_scalar("lr", lr, global_step)
    if momentum is not None:
        writer.add_scalar("momentum", momentum, global_step)


def main(args):
    setup_seed()
    train_dataset = SUSTechPOINTSDataset(
        data_root=args.data_root, split="train", train_ratio=args.train_ratio
    )
    val_dataset = SUSTechPOINTSDataset(
        data_root=args.data_root, split="val", train_ratio=args.train_ratio
    )

    train_dataloader = get_dataloader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
    )
    val_dataloader = get_dataloader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )

    if not args.no_cuda:
        pointpillars = PointPillars(nclasses=args.nclasses).cuda()
    else:
        pointpillars = PointPillars(nclasses=args.nclasses)
    loss_func = Loss()

    max_iters = len(train_dataloader) * args.max_epoch
    optimizer = torch.optim.AdamW(
        params=pointpillars.parameters(),
        lr=args.init_lr,
        betas=(0.95, 0.99),
        weight_decay=0.01,
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.init_lr * 10,
        total_steps=max_iters,
        pct_start=0.4,
        anneal_strategy="cos",
        cycle_momentum=True,
        base_momentum=0.95 * 0.895,
        max_momentum=0.95,
        div_factor=10,
    )
    saved_logs_path = os.path.join(args.saved_path, "summary")
    os.makedirs(saved_logs_path, exist_ok=True)
    writer = SummaryWriter(saved_logs_path)
    saved_ckpt_path = os.path.join(args.saved_path, "checkpoints")
    os.makedirs(saved_ckpt_path, exist_ok=True)

    for epoch in range(args.max_epoch):
        print("=" * 20, epoch, "=" * 20)
        train_step, val_step = 0, 0
        pointpillars.train()
        for i, data_dict in enumerate(tqdm(train_dataloader)):
            if not args.no_cuda:
                for key in data_dict:
                    for j, item in enumerate(data_dict[key]):
                        if torch.is_tensor(item):
                            data_dict[key][j] = data_dict[key][j].cuda()

            optimizer.zero_grad()
            batched_pts = data_dict["batched_pts"]
            batched_gt_bboxes = data_dict["batched_gt_bboxes"]
            batched_labels = data_dict["batched_labels"]
            bbox_cls_pred, bbox_pred, bbox_dir_cls_pred, anchor_target_dict = (
                pointpillars(
                    batched_pts=batched_pts,
                    mode="train",
                    batched_gt_bboxes=batched_gt_bboxes,
                    batched_gt_labels=batched_labels,
                )
            )

            bbox_cls_pred = bbox_cls_pred.permute(0, 2, 3, 1).reshape(-1, args.nclasses)
            bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 7)
            bbox_dir_cls_pred = bbox_dir_cls_pred.permute(0, 2, 3, 1).reshape(-1, 2)

            batched_bbox_labels = anchor_target_dict["batched_labels"].reshape(-1)
            batched_label_weights = anchor_target_dict["batched_label_weights"].reshape(
                -1
            )
            batched_bbox_reg = anchor_target_dict["batched_bbox_reg"].reshape(-1, 7)
            batched_dir_labels = anchor_target_dict["batched_dir_labels"].reshape(-1)

            pos_idx = (batched_bbox_labels >= 0) & (batched_bbox_labels < args.nclasses)
            bbox_pred = bbox_pred[pos_idx]
            batched_bbox_reg = batched_bbox_reg[pos_idx]

            pred_angle = bbox_pred[:, -1].clone()
            gt_angle = batched_bbox_reg[:, -1].clone()
            bbox_pred[:, -1] = torch.sin(pred_angle) * torch.cos(gt_angle)
            batched_bbox_reg[:, -1] = torch.cos(pred_angle) * torch.sin(gt_angle)

            bbox_dir_cls_pred = bbox_dir_cls_pred[pos_idx]
            batched_dir_labels = batched_dir_labels[pos_idx]

            num_cls_pos = (batched_bbox_labels < args.nclasses).sum()
            bbox_cls_pred = bbox_cls_pred[batched_label_weights > 0]
            batched_bbox_labels[batched_bbox_labels < 0] = args.nclasses
            batched_bbox_labels = batched_bbox_labels[batched_label_weights > 0]

            loss_dict = loss_func(
                bbox_cls_pred=bbox_cls_pred,
                bbox_pred=bbox_pred,
                bbox_dir_cls_pred=bbox_dir_cls_pred,
                batched_labels=batched_bbox_labels,
                num_cls_pos=num_cls_pos,
                batched_bbox_reg=batched_bbox_reg,
                batched_dir_labels=batched_dir_labels,
            )

            loss = loss_dict["total_loss"]
            loss.backward()
            optimizer.step()
            scheduler.step()

            global_step = epoch * len(train_dataloader) + train_step + 1
            if global_step % args.log_freq == 0:
                save_summary(
                    writer,
                    loss_dict,
                    global_step,
                    "train",
                    lr=optimizer.param_groups[0]["lr"],
                    momentum=optimizer.param_groups[0]["betas"][0],
                )
            train_step += 1

        if (epoch + 1) % args.ckpt_freq_epoch == 0:
            torch.save(
                pointpillars.state_dict(),
                os.path.join(saved_ckpt_path, f"epoch_{epoch+1}.pth"),
            )

        if epoch % 2 == 0:
            continue
        pointpillars.eval()
        with torch.no_grad():
            for i, data_dict in enumerate(tqdm(val_dataloader)):
                if not args.no_cuda:
                    for key in data_dict:
                        for j, item in enumerate(data_dict[key]):
                            if torch.is_tensor(item):
                                data_dict[key][j] = data_dict[key][j].cuda()

                batched_pts = data_dict["batched_pts"]
                batched_gt_bboxes = data_dict["batched_gt_bboxes"]
                batched_labels = data_dict["batched_labels"]
                bbox_cls_pred, bbox_pred, bbox_dir_cls_pred, anchor_target_dict = (
                    pointpillars(
                        batched_pts=batched_pts,
                        mode="train",
                        batched_gt_bboxes=batched_gt_bboxes,
                        batched_gt_labels=batched_labels,
                    )
                )

                bbox_cls_pred = bbox_cls_pred.permute(0, 2, 3, 1).reshape(
                    -1, args.nclasses
                )
                bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 7)
                bbox_dir_cls_pred = bbox_dir_cls_pred.permute(0, 2, 3, 1).reshape(-1, 2)

                batched_bbox_labels = anchor_target_dict["batched_labels"].reshape(-1)
                batched_label_weights = anchor_target_dict[
                    "batched_label_weights"
                ].reshape(-1)
                batched_bbox_reg = anchor_target_dict["batched_bbox_reg"].reshape(-1, 7)
                batched_dir_labels = anchor_target_dict["batched_dir_labels"].reshape(
                    -1
                )

                pos_idx = (batched_bbox_labels >= 0) & (
                    batched_bbox_labels < args.nclasses
                )
                bbox_pred = bbox_pred[pos_idx]
                batched_bbox_reg = batched_bbox_reg[pos_idx]

                pred_angle = bbox_pred[:, -1].clone()
                gt_angle = batched_bbox_reg[:, -1].clone()
                bbox_pred[:, -1] = torch.sin(pred_angle) * torch.cos(gt_angle)
                batched_bbox_reg[:, -1] = torch.cos(pred_angle) * torch.sin(gt_angle)

                bbox_dir_cls_pred = bbox_dir_cls_pred[pos_idx]
                batched_dir_labels = batched_dir_labels[pos_idx]

                num_cls_pos = (batched_bbox_labels < args.nclasses).sum()
                bbox_cls_pred = bbox_cls_pred[batched_label_weights > 0]
                batched_bbox_labels[batched_bbox_labels < 0] = args.nclasses
                batched_bbox_labels = batched_bbox_labels[batched_label_weights > 0]

                loss_dict = loss_func(
                    bbox_cls_pred=bbox_cls_pred,
                    bbox_pred=bbox_pred,
                    bbox_dir_cls_pred=bbox_dir_cls_pred,
                    batched_labels=batched_bbox_labels,
                    num_cls_pos=num_cls_pos,
                    batched_bbox_reg=batched_bbox_reg,
                    batched_dir_labels=batched_dir_labels,
                )

                global_step = epoch * len(val_dataloader) + val_step + 1
                if global_step % args.log_freq == 0:
                    save_summary(writer, loss_dict, global_step, "val")
                val_step += 1
        pointpillars.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train on test_data dataset")
    parser.add_argument(
        "--data_root", default="./datasets/test_truck_bed", help="your test_data root"
    )
    parser.add_argument("--saved_path", default="my_pillar_logs")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=2)
    # 不要修改，暂时没有别的数量的适配
    parser.add_argument("--nclasses", type=int, default=3)
    parser.add_argument("--init_lr", type=float, default=0.00025)
    parser.add_argument("--max_epoch", type=int, default=160)
    parser.add_argument("--log_freq", type=int, default=8)
    parser.add_argument("--ckpt_freq_epoch", type=int, default=10)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--no_cuda", action="store_true", help="whether to use cuda")
    args = parser.parse_args()

    main(args)
