import os

import numpy as np


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

        fields = header.get("FIELDS", [])
        sizes = list(map(int, header.get("SIZE", ["4"] * len(fields))))
        types = header.get("TYPE", ["F"] * len(fields))
        counts = list(map(int, header.get("COUNT", ["1"] * len(fields))))
        points = int(header.get("POINTS", [0])[0]) if "POINTS" in header else 0

        if header["DATA"] == "ascii":
            data = np.loadtxt(f, dtype=np.float32)
            return data

        # build structured dtype with explicit little-endian types
        dtype_list = []
        pad_count = 0

        def map_type(typ, size):
            # return numpy dtype string with explicit little-endian
            if typ == "F":
                if size == 4:
                    return "<f4"
                if size == 8:
                    return "<f8"
                if size == 2:
                    return "<f2"
                return "<f4"
            if typ == "I":
                if size == 1:
                    return "<i1"
                if size == 2:
                    return "<i2"
                if size == 4:
                    return "<i4"
                if size == 8:
                    return "<i8"
                return "<i4"
            if typ == "U":
                if size == 1:
                    return "<u1"
                if size == 2:
                    return "<u2"
                if size == 4:
                    return "<u4"
                if size == 8:
                    return "<u8"
                return "<u4"
            raise ValueError(f"Unsupported PCD type: {typ}")

        for name, size, typ, cnt in zip(fields, sizes, types, counts):
            if cnt == 0:
                continue
            if name == "_":
                name = f"_pad{pad_count}"
                pad_count += 1
            np_dtype = np.dtype(map_type(typ, size))
            if cnt == 1:
                dtype_list.append((name, np_dtype))
            else:
                dtype_list.append((name, np_dtype, cnt))

        dtype = np.dtype(dtype_list)

        # read the remaining bytes and parse
        if points <= 0:
            # try width * height
            width = int(header.get("WIDTH", [1])[0])
            height = int(header.get("HEIGHT", [1])[0])
            points = width * height

        # read raw bytes from file (current position is after header)
        raw = f.read()
        expected_bytes = points * dtype.itemsize
        if len(raw) < expected_bytes:
            # maybe header POINTS wrong; try to infer from data length
            if dtype.itemsize > 0:
                points = len(raw) // dtype.itemsize
        arr = np.frombuffer(raw, dtype=dtype, count=points)
        return arr


def pcd2kitti(pcd_path, bin_path):
    points = load_pcd(pcd_path)
    if points.size == 0:
        raise ValueError("No points loaded from " + pcd_path)

    if points.dtype.names is not None:
        x = points["x"].astype(np.float32)
        y = points["y"].astype(np.float32)
        z = points["z"].astype(np.float32)
        if "intensity" in points.dtype.names:
            intensity = points["intensity"].astype(np.float32)
        elif "i" in points.dtype.names:
            intensity = points["i"].astype(np.float32)
        else:
            intensity = np.zeros_like(x, dtype=np.float32)
    else:
        if points.ndim == 1:
            raise ValueError("PCD points must have at least x, y, z columns")
        x = points[:, 0].astype(np.float32)
        y = points[:, 1].astype(np.float32)
        z = points[:, 2].astype(np.float32)
        intensity = (
            points[:, 3].astype(np.float32)
            if points.shape[1] > 3
            else np.zeros_like(x, dtype=np.float32)
        )

    lidar = np.stack([x, y, z, intensity], axis=-1).astype(np.float32)
    os.makedirs(os.path.dirname(bin_path) or ".", exist_ok=True)
    lidar.tofile(bin_path)
    print(f"Converted: {pcd_path} -> {bin_path} ({lidar.shape[0]} points)")


def batch_convert(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for name in sorted(os.listdir(input_dir)):
        if name.lower().endswith(".pcd"):
            src = os.path.join(input_dir, name)
            dst = os.path.join(output_dir, os.path.splitext(name)[0] + ".bin")
            pcd2kitti(src, dst)


if __name__ == "__main__":
    # 单个文件转换
    # pcd2kitti('path/to/scan.pcd', 'path/to/scan.bin')

    # 批量转换
    batch_convert("./datasets/test_data/lidar", "./datasets/test_data/velodyne")
