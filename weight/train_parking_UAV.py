# -*- coding: utf-8 -*-
"""
无人机停车位巡检 - YOLO26s-Seg 训练优化版

目标：
1. 保留 YOLO26s-seg 作为可靠基线；
2. 增加无人机图传常见退化：运动模糊、失焦、雨/雾/雪、压缩、降采样、噪声、阴影；
3. 增加遮挡增强，同时保持 segmentation mask 不被擦除；
4. 使用较温和的航拍几何增强，避免破坏停车位空间结构；
5. 训练前检查“同一视频/航次序列同时出现在 train 和 val”的数据泄漏风险；
6. 保留清晰的参数区，便于后续做消融实验。

依赖建议：
    pip install -U ultralytics albumentations>=1.4.22

注意：
- “运动模糊增强”属于训练期鲁棒性增强；
- 真正的无人机全局运动补偿（Homography / ECC / Optical Flow）应放在实时视频推理管线，
  不应塞进本训练脚本。
"""

from __future__ import annotations

import inspect
import random
from pathlib import Path

import numpy as np
import yaml
import albumentations as A
from ultralytics import YOLO


# =========================================================
# 1. 基础配置：通常只需要修改这里
# =========================================================
DATA_YAML = r"D:\Pycharm\PythonProject\car_park_detect\xh-new0911_dataset\xh-new0911_dataset\dataset.yaml"
MODEL_PATH = "yolo26s-seg.pt"

DEVICE = "cuda"
EPOCHS = 200
IMG_SIZE = 1280
BATCH = 4
WORKERS = 0          # Windows/PyCharm 若出现多进程卡住，可改回 0

PROJECT = "runs_uav_parking"
RUN_NAME = "yolo26s_seg_uav_aug_v1"

SEED = 42
USE_UAV_AUG = True
CHECK_SEQUENCE_LEAKAGE = True
FAIL_ON_SEQUENCE_LEAKAGE = False  # 正式论文对比时建议改为 True，并先重新划分数据
RUN_FINAL_VAL = True


# =========================================================
# 2. UAV 专用 Albumentations
#    原则：增强要贴近真实图传，强度宁可温和，不要把停车线细节直接抹掉。
# =========================================================
def _supports(transform_cls, parameter: str) -> bool:
    """兼容 Albumentations 1.4.x 与 2.x 的参数命名差异。"""
    return parameter in inspect.signature(transform_cls).parameters


def _image_compression():
    if _supports(A.ImageCompression, "quality_range"):
        return A.ImageCompression(quality_range=(55, 92), compression_type="jpeg", p=0.65)
    return A.ImageCompression(quality_lower=55, quality_upper=92, p=0.65)


def _downscale():
    if _supports(A.Downscale, "scale_range"):
        return A.Downscale(scale_range=(0.55, 0.85), p=0.35)
    return A.Downscale(scale_min=0.55, scale_max=0.85, p=0.35)


def _random_fog():
    if _supports(A.RandomFog, "fog_coef_range"):
        return A.RandomFog(fog_coef_range=(0.10, 0.35), alpha_coef=0.05, p=0.40)
    return A.RandomFog(fog_coef_lower=0.10, fog_coef_upper=0.35, alpha_coef=0.05, p=0.40)


def _random_snow():
    if _supports(A.RandomSnow, "snow_point_range"):
        return A.RandomSnow(
            brightness_coeff=1.4, snow_point_range=(0.05, 0.12), method="bleach", p=0.10
        )
    return A.RandomSnow(
        brightness_coeff=1.4, snow_point_lower=0.05, snow_point_upper=0.12, p=0.10
    )


def _gauss_noise():
    if _supports(A.GaussNoise, "std_range"):
        return A.GaussNoise(std_range=(0.005, 0.020), p=0.75)
    # 旧版 var_limit 是像素尺度方差；这里对应轻度噪声，避免抹掉停车线。
    return A.GaussNoise(var_limit=(2.0, 25.0), mean=0, p=0.75)


class UAVImageOnlyOcclusion(A.ImageOnlyTransform):
    """
    只修改图像像素，不参与 bbox / segment 的空间变换。
    目的：模拟车辆、树枝、杆件、图传局部遮挡，同时避免 Ultralytics-Seg
    中 Albumentations 对 bbox 过滤后与 polygon 数量不同步的问题。
    """

    def __init__(
        self,
        num_holes_range=(1, 4),
        hole_side_range=(0.03, 0.10),
        p=0.12,
    ):
        super().__init__(p=p)
        self.num_holes_range = num_holes_range
        self.hole_side_range = hole_side_range

    def apply(self, img, **params):
        h, w = img.shape[:2]
        out = img.copy()
        n = random.randint(self.num_holes_range[0], self.num_holes_range[1])

        for _ in range(n):
            hole_h = max(2, int(h * random.uniform(*self.hole_side_range)))
            hole_w = max(2, int(w * random.uniform(*self.hole_side_range)))

            y1 = random.randint(0, max(0, h - hole_h))
            x1 = random.randint(0, max(0, w - hole_w))
            y2 = min(h, y1 + hole_h)
            x2 = min(w, x1 + hole_w)

            # 使用局部均值/随机轻微扰动填充，避免纯黑块过于人工
            patch = out[y1:y2, x1:x2]
            if patch.size == 0:
                continue

            mean_color = patch.reshape(-1, patch.shape[-1]).mean(axis=0)
            noise = np.random.normal(0, 8, patch.shape)
            fill = np.clip(mean_color + noise, 0, 255).astype(out.dtype)
            out[y1:y2, x1:x2] = fill

        return out


def _image_only_occlusion():
    return UAVImageOnlyOcclusion(
        num_holes_range=(1, 4),
        hole_side_range=(0.03, 0.10),
        p=0.12,
    )

def build_uav_augmentations() -> list:
    return [
        # -------------------------------------------------
        # A. 无人机/云台运动模糊 + 轻度失焦
        # 总触发概率约 18%，避免正常清晰图像占比过低
        # -------------------------------------------------
        A.OneOf(
            [
                A.MotionBlur(blur_limit=(3, 7), p=0.60),
                A.GaussianBlur(blur_limit=(3, 5), p=0.25),
                A.Defocus(radius=(2, 4), alias_blur=(0.10, 0.30), p=0.15),
            ],
            p=0.18,
        ),

        # -------------------------------------------------
        # B. 无人机图传压缩 / 码率下降 / 远距离低分辨率
        # -------------------------------------------------
        A.OneOf([_image_compression(), _downscale()], p=0.18),

        # -------------------------------------------------
        # C. 雨 / 雾 / 雪；雪权重最低
        # -------------------------------------------------
        A.OneOf(
            [
                A.RandomRain(
                    rain_type="drizzle",
                    drop_width=1,
                    blur_value=3,
                    brightness_coefficient=0.85,
                    p=0.50,
                ),
                _random_fog(),
                _random_snow(),
            ],
            p=0.10,
        ),

        # -------------------------------------------------
        # D. 图传/传感器噪声
        # -------------------------------------------------
        A.OneOf(
            [
                _gauss_noise(),
                A.ISONoise(
                    color_shift=(0.01, 0.03),
                    intensity=(0.08, 0.25),
                    p=0.25,
                ),
            ],
            p=0.10,
        ),

        # -------------------------------------------------
        # E. 光照/阴影变化
        # -------------------------------------------------
        A.RandomBrightnessContrast(
            brightness_limit=0.20,
            contrast_limit=0.22,
            p=0.20,
        ),
        A.CLAHE(clip_limit=3.0, tile_grid_size=(8, 8), p=0.08),
        A.RandomShadow(p=0.08),

        # -------------------------------------------------
        # F. 局部遮挡：严格 ImageOnly，不参与 bbox/segment 变换
        # -------------------------------------------------
        _image_only_occlusion(),
    ]


# =========================================================
# 3. 数据集视频序列泄漏检查
#    文件命名示例：8-12_frame_00000600_time_0000005.00s.jpg
#    会把 _frame_ 前面的内容作为“视频/航次 ID”。
# =========================================================
def _source_id(path: Path) -> str:
    stem = path.stem
    return stem.split("_frame_", 1)[0] if "_frame_" in stem else stem


def _list_images(path: Path) -> list[Path]:
    suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    if path.is_dir():
        return [p for p in path.rglob("*") if p.suffix.lower() in suffixes]
    if path.is_file() and path.suffix.lower() == ".txt":
        images = []
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            s = line.strip()
            if s:
                p = Path(s)
                if not p.is_absolute():
                    p = path.parent / p
                images.append(p)
        return images
    return []


def _resolve_split_path(cfg: dict, yaml_path: Path, key: str) -> Path | None:
    value = cfg.get(key)
    if not isinstance(value, str):
        return None

    value_path = Path(value)
    if value_path.is_absolute():
        return value_path

    root = cfg.get("path")
    if isinstance(root, str) and root:
        root_path = Path(root)
        if not root_path.is_absolute():
            root_path = yaml_path.parent / root_path
        return root_path / value_path

    return yaml_path.parent / value_path


def check_sequence_leakage(data_yaml: str) -> set[str]:
    yaml_path = Path(data_yaml)
    if not yaml_path.exists():
        print(f"[数据检查] dataset.yaml 暂未找到：{yaml_path}")
        print("[数据检查] 跳过序列泄漏检查；请确认 DATA_YAML 路径后再训练。")
        return set()

    cfg = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    train_path = _resolve_split_path(cfg, yaml_path, "train")
    val_path = _resolve_split_path(cfg, yaml_path, "val")

    if train_path is None or val_path is None:
        print("[数据检查] dataset.yaml 中未解析到 train/val 路径，跳过检查。")
        return set()

    train_images = _list_images(train_path)
    val_images = _list_images(val_path)

    train_ids = {_source_id(p) for p in train_images}
    val_ids = {_source_id(p) for p in val_images}
    overlap = train_ids & val_ids

    print("\n" + "=" * 68)
    print("数据集序列检查")
    print(f"Train images : {len(train_images)}")
    print(f"Val images   : {len(val_images)}")
    print(f"Train sources: {len(train_ids)}")
    print(f"Val sources  : {len(val_ids)}")

    if overlap:
        print(f"[警告] train/val 发现 {len(overlap)} 个重复视频/航次来源：")
        print("       " + ", ".join(sorted(overlap)))
        print("[建议] 按整段视频/整次航飞划分 train 与 val，不要随机拆相邻帧。")
        print("       否则 mAP/IoU 可能虚高，新停车场泛化能力被高估。")
    else:
        print("[通过] 未发现基于文件名的视频/航次交叉。")
    print("=" * 68 + "\n")

    return overlap


# =========================================================
# 4. 训练
# =========================================================
def train():
    if CHECK_SEQUENCE_LEAKAGE:
        overlap = check_sequence_leakage(DATA_YAML)
        if overlap and FAIL_ON_SEQUENCE_LEAKAGE:
            raise RuntimeError(
                "检测到 train/val 视频序列泄漏。请先按整段视频重新划分数据集，再进行正式训练。"
            )

    custom_aug = build_uav_augmentations() if USE_UAV_AUG else None

    print(f"Model : {MODEL_PATH}")
    print(f"Data  : {DATA_YAML}")
    print(f"UAV augmentation: {'ON' if USE_UAV_AUG else 'OFF'}")

    model = YOLO(MODEL_PATH)

    results = model.train(
        data=DATA_YAML,
        device=DEVICE,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH,
        workers=WORKERS,

        # ------------------------------
        # 训练稳定性 / 可复现性
        # ------------------------------
        seed=SEED,
        deterministic=True,
        amp=True,
        optimizer="auto",
        cos_lr=True,
        warmup_epochs=5.0,
        patience=45,

        # ------------------------------
        # UAV 航拍几何变化
        # 保持“温和”，避免把停车位几何结构扭得不真实
        # ------------------------------
        degrees=12.0,
        translate=0.08,
        scale=0.30,
        shear=2.0,
        perspective=0.0005,
        fliplr=0.50,
        flipud=0.10,

        # ------------------------------
        # 颜色/曝光变化
        # ------------------------------
        hsv_h=0.012,
        hsv_s=0.45,
        hsv_v=0.30,

        # ------------------------------
        # 多图增强
        # 停车位依赖空间上下文；Seg 任务为稳定标签对应关系，关闭 CutMix/MixUp/Copy-Paste
        # ------------------------------
        mosaic=0.35,
        mixup=0.00,
        cutmix=0.02,
        copy_paste=0.00,
        close_mosaic=20,

        # ------------------------------
        # 自定义 UAV Albumentations
        # ------------------------------
        augmentations=custom_aug,

        # ------------------------------
        # 分割训练与输出
        # ------------------------------
        overlap_mask=True,
        mask_ratio=4,
        project=PROJECT,
        name=RUN_NAME,
        exist_ok=False,
        save=True,
        save_period=25,
        plots=True,
        verbose=True,
    )

    # 训练结束后，用 best.pt 再做一次明确的 val
    if RUN_FINAL_VAL:
        # 使用 trainer 记录的真实 best 路径，兼容同名实验自动递增目录。
        trainer_best = getattr(getattr(model, "trainer", None), "best", None)
        best_pt = Path(str(trainer_best)) if trainer_best else Path(PROJECT) / RUN_NAME / "weights" / "best.pt"
        if best_pt.exists():
            print(f"\n使用 best.pt 进行最终验证：{best_pt}")
            best_model = YOLO(str(best_pt))
            best_model.val(
                data=DATA_YAML,
                imgsz=IMG_SIZE,
                batch=BATCH,
                device=DEVICE,
                split="val",
                plots=True,
            )
        else:
            print(f"\n未找到 {best_pt}，跳过额外验证。")

    return results


if __name__ == "__main__":
    # Windows 下必须保留 main 入口，尤其 WORKERS > 0 时。
    train()
