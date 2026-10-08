# -*- coding: utf-8 -*-
"""
UAV roadside-parking real-time patrol recognizer (v6.8.3 tcw0914 custom-checkpoint compatibility + prominent real-slot-number display).

"""

from __future__ import annotations

import json
import math
import os
import random
import sys
import threading
import time
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Callable, Deque, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

# ============================================================================
# v6.8.3 compatibility for parking checkpoints trained with custom UAV augmentation
# ============================================================================
try:
    import albumentations as A
except Exception:
    A = None

if A is not None:
    class UAVImageOnlyOcclusion(A.ImageOnlyTransform):
        """Training-checkpoint compatibility class; not applied during inference."""

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
                patch = out[y1:y2, x1:x2]
                if patch.size == 0:
                    continue
                mean_color = patch.reshape(-1, patch.shape[-1]).mean(axis=0)
                noise = np.random.normal(0, 8, patch.shape)
                fill = np.clip(mean_color + noise, 0, 255).astype(out.dtype)
                out[y1:y2, x1:x2] = fill
            return out
else:
    class UAVImageOnlyOcclusion:
        """Minimal pickle shim used only if albumentations is unavailable at inference."""

        def __init__(self, *args, **kwargs):
            self.num_holes_range = kwargs.get("num_holes_range", (1, 4))
            self.hole_side_range = kwargs.get("hole_side_range", (0.03, 0.10))
            self.p = kwargs.get("p", 0.12)

        def __setstate__(self, state):
            if isinstance(state, dict):
                self.__dict__.update(state)

# The checkpoint records the class path as ``__main__.UAVImageOnlyOcclusion``.
# Explicit registration also covers production platforms that IMPORT this file instead of running it directly.
_main_module = sys.modules.get("__main__")
if _main_module is not None and not hasattr(_main_module, "UAVImageOnlyOcclusion"):
    setattr(_main_module, "UAVImageOnlyOcclusion", UAVImageOnlyOcclusion)

from ultralytics import YOLO

from plate_recognition.double_plate_split_merge import get_split_merge
from plate_recognition.plate_rec import get_plate_result, init_model


# ============================================================================
#                         配置区
# ============================================================================
# ---------- 1）视频 / 无人机实时图传地址 ----------
# 本地视频示例： VIDEO_SOURCE = r"D:\UAV\parking_test.mp4"
# USB摄像头示例：VIDEO_SOURCE = 0
# RTSP示例：     VIDEO_SOURCE = "rtsp://192.168.1.10:8554/live"
VIDEO_SOURCE = os.environ.get("UAV_VIDEO_SOURCE", r"C:\Users\Administrator\Videos\9-17.mp4")  # 平台/容器可用环境变量覆盖；默认摄像头0

# ---------- 2）模型地址 ----------
# 车辆检测模型（COCO车辆类别 car/motorcycle/bus/truck）
VEHICLE_MODEL_PATH = os.environ.get("UAV_VEHICLE_MODEL", r"weights\yolo26n.pt")

# 停车位分割 TensorRT engine：tcw0813.engine
PARKING_MODEL_PATH = os.environ.get("UAV_PARKING_MODEL", r"weights\tcw0914.pt")

# 车牌四关键点检测 TensorRT engine
PLATE_DET_MODEL_PATH = os.environ.get("UAV_PLATE_DET_MODEL", r"weights\yolo26s-plate-detect.pt")

# 车牌字符 + 颜色识别模型
PLATE_REC_MODEL_PATH = os.environ.get("UAV_PLATE_REC_MODEL", r"weights\plate_rec_color.pth")

# 中文字体
FONT_PATH = os.environ.get("UAV_FONT_PATH", r"weights\platech.ttf")

# ---------- 3）GPU ----------
# 单卡通常写 "0"；如只想调试车辆 PT 模型可用 "cpu"，但 TensorRT engine 仍要求 NVIDIA GPU。
DEVICE = "0"

# ---------- 4）实时检测参数 ----------
VEHICLE_IMGSZ = 640
VEHICLE_CONF = 0.28
PARKING_CONF = 0.30
PLATE_CONF = 0.35
IOU_THRESHOLD = 0.55
VEHICLE_CLASSES = (2, 3, 5, 7)  # COCO: car=2, motorcycle=3, bus=5, truck=7

# 停车位模型较重：每 N 个处理帧重新分割一次，其余帧由背景运动补偿传播车位多边形。
PARKING_EVERY_N_FRAMES = 3

# 每个处理帧最多对多少辆车运行车牌检测+识别；无人机实时图传建议 1~2。
MAX_PLATE_ROIS_PER_FRAME = 2
PLATE_RETRY_SECONDS = 0.28
PLATE_MIN_HITS = 2

# ---------- 5）静止 / 正常停车 / 违停判定 ----------
# 注意：静止判定已经扣除了无人机自身导致的背景运动。
# 6.5.4：基础静止确认时间由 1.2 s 调整为 0.90 s；只有连续残差“非常小”时，
# 才允许使用 FAST_SECONDS 提前确认，避免无人机快速扫过时已停车辆来不及进入累计。
STATIONARY_SECONDS = 0.90
STATIONARY_FAST_SECONDS = 0.65
STATIONARY_STRONG_RESIDUAL_FACTOR = 0.55
STATIONARY_RESIDUAL_RATIO = 0.075

# 正常停车：车辆中心在车位内，且车辆面积至少此比例位于同一停车位。
NORMAL_MIN_VEHICLE_OVERLAP = 0.55

# 静止车辆与任意车位的最大重叠不超过此比例 -> 违停。
VIOLATION_MAX_VEHICLE_OVERLAP = 0.32

# 状态连续确认帧数，降低瞬时抖动误判。
STATUS_CONFIRM_FRAMES = 3

# ---------- 6）显示 / 保存 ----------
DISPLAY_WINDOW = os.environ.get("UAV_DISPLAY_WINDOW", "1") == "1"  # 部署默认关闭，本地调试设为1
WINDOW_NAME = "UAV Parking Realtime Patrol"

# 实时显示时自动把“整幅画面”等比例缩放到屏幕可见区域。
# 只影响显示，不影响模型检测分辨率，也不影响保存视频的原始尺寸。
AUTO_FIT_DISPLAY_TO_SCREEN = True
DISPLAY_SCREEN_RATIO = 0.90   # 占屏幕宽/高的最大比例，0.90 可避开任务栏和窗口边框
DISPLAY_FALLBACK_WIDTH = 1600
DISPLAY_FALLBACK_HEIGHT = 900

# True：保存标注后的视频；False：只实时显示，不写视频，可减少磁盘开销。
SAVE_RESULT_VIDEO = os.environ.get("UAV_SAVE_RESULT_VIDEO", "1") == "1"  # 部署默认关闭，减少CPU/磁盘开销
RESULT_VIDEO_PATH = r"output\uav_parking_result2.mp4"

# 结果录像优化（6.4）：录像彻底从识别主线程移到后台线程，避免“为了补帧而卡死窗口”。
# 建议 15 FPS：对检测结果录像已经足够流畅，同时显著降低 CPU/磁盘编码压力。
# 如电脑性能充足可改 20/25/30；0 = 自动取 min(源FPS, 15)。
RESULT_VIDEO_FPS = 15.0

# 保存视频最大宽度。原始无人机视频若为 4K，直接 4K+30FPS 用 mp4v CPU 编码很容易拖死主程序。
# 1920 = 最长边宽度限制为 1920，并保持宽高比；0 = 不缩放，保存原始标注分辨率。
# 注意：这里只影响“保存的视频”，完全不影响模型检测输入与本地显示。
RESULT_VIDEO_MAX_WIDTH = 1920

# True：按真实源时间轴写录像。后台录像线程会按源时间持续复用最新标注帧，
# 不再在识别线程里一次性 write() 几十/上百张重复帧。
RESULT_VIDEO_KEEP_REALTIME = True

# 违停 / 正常停泊 + 车牌绑定事件日志。设为 "" 可关闭。
EVENT_LOG_PATH = r"output\uav_parking_events6.8.3.1.jsonl"

# 控制台统计输出间隔（秒）
STATS_INTERVAL_SECONDS = 1.0

# ---------- 7）无人机航线级车位累计 ----------
# 纯检测部署版不包含任何平台网络上传、远程控制或设备令牌配置。
# 目标：累计值只增不减；同一个停车位确认一次后，本次巡检中状态冻结。
# 占用车位通常更容易确认，因此所需连续帧数可略少。
CUMULATIVE_OCCUPIED_CONFIRM_FRAMES = 3

# 空闲车位更容易因车辆漏检而误判，因此要求更长的连续空闲确认。
CUMULATIVE_EMPTY_CONFIRM_FRAMES = 6

# 候选车位中断超过该时间后，连续确认重新计数。用于避免车位短暂消失后直接接着累计。
CUMULATIVE_CONFIRM_MAX_GAP_SECONDS = 0.8

# 车位必须进入画面内部后才允许参加累计。0.03 表示四周留 3% 画面边距。
# 可避免无人机刚扫到半个车位、车辆尚未完全进入画面时误记为空闲。
CUMULATIVE_SLOT_EDGE_MARGIN_RATIO = 0.03

# 6.5.4：车位短时丢失后保留旧 ID 一段时间，并继续随相机运动传播，
# 再次检测到同一车位时优先恢复旧 ID，减少因 segmentation 抖动导致的重复累计。
SLOT_MAX_MISSED_DETECTS = 8
SLOT_REID_RETENTION_FRAMES = 75

# 6.8.3：停车位唯一ID去重。目标不是阻止模型逐帧重新检测，而是保证同一物理车位
# 在当前画面/本次巡检内只保留一个“规范 local slot_id”，从源头避免 P17/P23 同时指向同一车位。
PARKING_DEDUP_IOU = 0.55                 # 同帧重复 mask 的多边形 IoU 阈值
PARKING_DEDUP_MIN_COVER = 0.78           # 较小 mask 被较大 mask 覆盖比例
PARKING_DEDUP_CENTER_RATIO = 0.22        # 中心距离 / 较大对角线
PARKING_DUPLICATE_AREA_SIMILARITY = 0.50 # 面积相似度下限，避免吞并相邻车位
PARKING_ACTIVE_MERGE_IOU = 0.58          # 已产生两个 local ID 后的在线合并阈值
PARKING_ACTIVE_MERGE_MIN_COVER = 0.82
PARKING_ACTIVE_MERGE_CENTER_RATIO = 0.20

# ---------- 7.1）路侧固定航线永久车位编号映射（无GPS，多航段版） ----------
# 适用场景：无人机沿道路按固定方向连续巡检；平台只有现实车位编号，没有每个车位GPS。
# 核心：停车位模型只负责发现/短时跟踪 P-ID；本模块根据“路线真实编号顺序 + 视觉运动方向 + 局部间距”
#       把 P17/P18/... 稳定映射为 191289/191288/...。
# 注意：JSON 可配置多个 segments；每个 segment 的 slots 必须按该航段“无人机实际巡检方向”排序。
ROADSIDE_SLOT_MAP_ENABLED = os.environ.get("UAV_ROADSIDE_SLOT_MAP", "1") == "1"
ROADSIDE_SLOT_MAP_PATH = os.environ.get("UAV_ROADSIDE_SLOT_MAP_PATH", r"roadside_slot_route_星湖南路东1_双航段.json")

# auto：通过同一停车位在连续帧中的整体漂移，自动判断道路编号在画面中的前进方向。
# 若现场相机/云台导致 auto 不稳定，可改为：left_to_right / right_to_left / top_to_bottom / bottom_to_top。
ROADSIDE_SCREEN_ORDER = os.environ.get("UAV_ROADSIDE_SCREEN_ORDER", "auto")
# 无GPS情况下必须有一个起始锚点。通常写无人机从该端起飞时遇到的第一个真实车位号。
# JSON 内 start_real_id 优先；这里作为兜底。
ROADSIDE_START_REAL_ID = os.environ.get("UAV_ROADSIDE_START_REAL_ID", "191356")

# 同一个临时 P-ID 连续获得多少次相同真实编号候选后才正式绑定。
ROADSIDE_SLOT_MIN_HITS = 2
# auto方向学习：至少收集多少个可靠的停车位整体位移样本。
ROADSIDE_DIRECTION_MIN_SAMPLES = 3
ROADSIDE_MIN_MOTION_PIXELS = 1.5
# 漏检容错：相邻视觉车位中心间距明显大于正常间距时，允许推断中间漏掉 1~N 个真实车位。
# 该规则很保守，避免普通透视变化被误判为漏检。
ROADSIDE_GAP_SKIP_RATIO = 1.75
ROADSIDE_MAX_INFERRED_SKIP = 2
# 车位刚从画面边缘露出时不立即赋永久编号，防止半个车位导致顺序错位。
ROADSIDE_ASSIGN_EDGE_MARGIN_RATIO = 0.010
# 真实车位号显示。True 时保留临时视觉 P-ID 作为调试小字；False 时只显示真实编号。
ROADSIDE_SHOW_LOCAL_ID = True
# v6.8.1：真实车位编号单独大字号显示，避免原先 191280[P17] 字号过小、不醒目。
ROADSIDE_REAL_ID_FONT_SIZE = 48
ROADSIDE_LOCAL_ID_FONT_SIZE = 13
ROADSIDE_PENDING_ID_FONT_SIZE = 14

# 6.5.3：累计占用必须由“静止车辆”提供证据。
# 行驶车辆只是临时经过车位时，实时画面仍可显示为临时占用，但不会写入航线累计。
# 该车辆离开后，车位需要重新连续满足空闲确认帧数，才会累计为空闲。
CUMULATIVE_OCCUPIED_REQUIRE_STATIONARY = True

# ---------- 8）无人机实时异步流水线（建议保持开启） ----------
# 核心原则：车辆检测/BoT-SORT 是最高优先级主链；停车位和车牌绝不能阻塞主链。
ASYNC_PARKING_ENABLED = True
ASYNC_PLATE_ENABLED = True

# 停车位 segmentation 改为“按时间”异步调度，而不是依赖主线程每 N 帧同步执行。
# v6.5.7：从 0.35 s 缩短为 0.22 s，启动阶段更快拿到第一个有效停车位结果。
PARKING_ASYNC_INTERVAL_SECONDS = 0.22
# 异步停车位结果太旧时直接丢弃，防止无人机已经移动后仍套用旧位置。
PARKING_RESULT_MAX_AGE_SECONDS = 1.20

# v6.5.7 关键修复：无人机启动阶段背景光流可能暂时不可靠。旧版只要 source->current
# 任意一帧没有有效仿射矩阵，就把停车位结果整体丢弃，可能导致模型已经识别却 5~10 s 不显示。
# 第一个“非空且仍新鲜”的停车位结果现在直接接受；后续很新的结果在运动补偿失败时也允许短暂
# 使用原坐标，下一次 segmentation 会自动校正。这样优先保证“先显示”，而不是一直等光流稳定。
PARKING_ACCEPT_FIRST_FRESH_UNALIGNED = True
PARKING_FRESH_UNALIGNED_FALLBACK = True
PARKING_FRESH_UNALIGNED_MAX_AGE_SECONDS = 0.45
PARKING_FRESH_UNALIGNED_MAX_FRAME_GAP = 4

# 第一张真实图传帧一进入 process_frame 就立即异步提交停车位，不等待车辆主链结束。
# 只提交、不等待，因此不会恢复旧版 0.65 s 的首帧阻塞。
PARKING_EARLY_FIRST_FRAME_SUBMIT = True
# 送入停车位 worker 前可缩小超高清源画面；模型仍会按其 960 输入推理。
# 返回多边形会自动映射回原始画面坐标。0 = 不预缩放。
PARKING_WORKER_MAX_WIDTH = 1600

# 6.5.6 部署默认：首帧停车位不再同步等待。
# 车辆主链先立即出结果，停车位 worker 在后台完成后自动补上，避免首帧被 0.65 s 人为阻塞。
# 如本地演示强制要求“第一张标注图就必须有停车位”，可临时改为 True。
PARKING_FIRST_FRAME_BOOTSTRAP = False
PARKING_FIRST_FRAME_TIMEOUT_SECONDS = 0.12

# 6.5.6 冷启动优化：服务构造阶段完成车辆 TensorRT 与 BoT-SORT 首次初始化。
# 这些耗时发生在平台 READY 之前，而不是压到第一张真实无人机图传帧上。
AUTO_WARMUP_ON_INIT = True
VEHICLE_WARMUP_PASSES = 2
WARMUP_TRACKER_ON_INIT = True
WARMUP_DUMMY_HEIGHT = 720
WARMUP_DUMMY_WIDTH = 1280

# 车牌任务队列只保留少量最新/高优先级 ROI，绝不形成几十帧积压。
PLATE_ASYNC_QUEUE_SIZE = 3
PLATE_RESULT_MAX_AGE_SECONDS = 1.50

# 停车位 worker 与车牌 worker 共用一个“辅助 GPU 锁”，避免两个重模型同时抢 GPU。
# 注意：车辆主链不使用这个锁，所以仍具有最高调度优先级。
SERIALIZE_AUX_GPU_INFERENCE = True

# 车辆 PT 模型在 CUDA 下使用 FP16，可显著降低推理开销；TensorRT engine 会自动使用自身精度。
VEHICLE_USE_FP16 = True

# 背景光流只用于无人机运动补偿，不需要在超高清分辨率上计算。
MOTION_PROCESS_WIDTH = 720
MOTION_MAX_CORNERS = 350

# 自适应降载：当主链 FPS 很低时，辅助模型自动降低提交频率，优先救车辆跟踪。
ADAPTIVE_AUX_LOAD_SHEDDING = True
AUX_LOW_FPS_THRESHOLD = 4.0
AUX_LOW_FPS_PARKING_INTERVAL = 0.80



SCRIPT_DIR = Path(__file__).resolve().parent


def _resolve_local_path(path_value: str) -> str:
    """把代码配置区中的相对路径稳定地解析为相对于当前脚本目录的绝对路径。"""
    text = str(path_value).strip()
    if not text:
        return ""
    p = Path(text)
    if p.is_absolute():
        return str(p)
    # 同时兼容用户在 Windows 上填写反斜杠相对路径。
    normalized = Path(text.replace("\\", os.sep).replace("/", os.sep))
    return str((SCRIPT_DIR / normalized).resolve())


def _resolve_video_source(source: str | int) -> str | int:
    """摄像头编号保持整数；RTSP/HTTP保持URL；本地视频相对路径按脚本目录解析。"""
    if isinstance(source, int):
        return source
    text = str(source).strip()
    if text.isdigit():
        return int(text)
    low = text.lower()
    if low.startswith(("rtsp://", "rtsps://", "http://", "https://", "udp://", "tcp://")):
        return text
    return _resolve_local_path(text)


def _get_screen_size() -> Tuple[int, int]:
    """获取主屏幕分辨率；Windows 下自动读取，其他环境使用备用值。"""
    try:
        if os.name == "nt":
            import ctypes

            user32 = ctypes.windll.user32
            # 让高 DPI 屏幕下读取到尽量真实的像素尺寸。
            try:
                ctypes.windll.shcore.SetProcessDpiAwareness(1)
            except Exception:
                try:
                    user32.SetProcessDPIAware()
                except Exception:
                    pass
            sw = int(user32.GetSystemMetrics(0))
            sh = int(user32.GetSystemMetrics(1))
            if sw > 0 and sh > 0:
                return sw, sh
    except Exception:
        pass
    return int(DISPLAY_FALLBACK_WIDTH), int(DISPLAY_FALLBACK_HEIGHT)


def _fit_frame_for_display(frame: np.ndarray) -> np.ndarray:
    """仅为屏幕显示缩放图像，确保整幅画面可见且不改变宽高比。"""
    if not AUTO_FIT_DISPLAY_TO_SCREEN or frame is None or frame.size == 0:
        return frame

    h, w = frame.shape[:2]
    screen_w, screen_h = _get_screen_size()
    ratio = float(np.clip(DISPLAY_SCREEN_RATIO, 0.20, 1.00))
    max_w = max(320, int(screen_w * ratio))
    max_h = max(240, int(screen_h * ratio))

    scale = min(max_w / max(1, w), max_h / max(1, h), 1.0)
    if scale >= 0.999:
        return frame

    out_w = max(1, int(round(w * scale)))
    out_h = max(1, int(round(h * scale)))
    return cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)


# ----------------------------- basic geometry -----------------------------

Box = Tuple[float, float, float, float]
Point = Tuple[float, float]


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def box_center(box: Box) -> np.ndarray:
    x1, y1, x2, y2 = box
    return np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=np.float32)


def box_wh(box: Box) -> Tuple[float, float]:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1), max(0.0, y2 - y1)


def box_diag(box: Box) -> float:
    w, h = box_wh(box)
    return max(1.0, math.hypot(w, h))


def box_area(box: Box) -> float:
    w, h = box_wh(box)
    return w * h


def box_to_polygon(box: Box) -> np.ndarray:
    x1, y1, x2, y2 = box
    return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)


def polygon_bbox(poly: np.ndarray) -> Box:
    p = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
    return float(p[:, 0].min()), float(p[:, 1].min()), float(p[:, 0].max()), float(p[:, 1].max())


def polygon_area(poly: np.ndarray) -> float:
    p = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
    if len(p) < 3:
        return 0.0
    return abs(float(cv2.contourArea(p)))


def bbox_iou(a: Box, b: Box) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = box_area(a) + box_area(b) - inter
    return inter / union if union > 1e-6 else 0.0


def convex_intersection_area(poly_a: np.ndarray, poly_b: np.ndarray) -> float:
    """Fast intersection area; parking-space masks are expected to be nearly convex quadrilaterals."""
    a = cv2.convexHull(np.asarray(poly_a, dtype=np.float32).reshape(-1, 1, 2))
    b = cv2.convexHull(np.asarray(poly_b, dtype=np.float32).reshape(-1, 1, 2))
    if len(a) < 3 or len(b) < 3:
        return 0.0
    try:
        area, _ = cv2.intersectConvexConvex(a, b)
        return max(0.0, float(area))
    except cv2.error:
        return 0.0


def transform_point(M: Optional[np.ndarray], p: np.ndarray) -> np.ndarray:
    if M is None:
        return p.copy()
    q = np.array([p[0], p[1], 1.0], dtype=np.float32)
    return (M @ q).astype(np.float32)


def transform_polygon(M: Optional[np.ndarray], poly: np.ndarray) -> np.ndarray:
    if M is None:
        return np.asarray(poly, dtype=np.float32).copy()
    p = np.asarray(poly, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.transform(p, M).reshape(-1, 2)


def transform_box(M: Optional[np.ndarray], box: Box) -> Box:
    poly = transform_polygon(M, box_to_polygon(box))
    return polygon_bbox(poly)


def point_in_polygon(p: np.ndarray, poly: np.ndarray) -> bool:
    contour = np.asarray(poly, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.pointPolygonTest(contour, (float(p[0]), float(p[1])), False) >= 0


def clip_box(box: Box, w: int, h: int) -> Box:
    x1, y1, x2, y2 = box
    x1 = clamp(x1, 0, max(0, w - 1))
    y1 = clamp(y1, 0, max(0, h - 1))
    x2 = clamp(x2, 0, w)
    y2 = clamp(y2, 0, h)
    return float(x1), float(y1), float(x2), float(y2)


def expand_box(box: Box, scale: float, w: int, h: int) -> Box:
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    bw, bh = max(2.0, x2 - x1), max(2.0, y2 - y1)
    nw, nh = bw * scale, bh * scale
    return clip_box((cx - nw / 2, cy - nh / 2, cx + nw / 2, cy + nh / 2), w, h)


# ------------------------ moving-camera compensation ----------------------

class BackgroundMotionEstimator:
    """Estimate previous-frame -> current-frame background affine motion.

    This is intentionally independent from BoT-SORT GMC. BoT-SORT uses GMC to keep IDs stable;
    this estimator is used to decide whether a tracked vehicle itself is moving after removing
    drone/camera motion.
    """

    def __init__(
        self,
        process_width: int = 960,
        max_corners: int = 600,
        min_inliers: int = 18,
        ransac_thresh: float = 2.5,
    ) -> None:
        self.process_width = process_width
        self.max_corners = max_corners
        self.min_inliers = min_inliers
        self.ransac_thresh = ransac_thresh
        self.prev_gray: Optional[np.ndarray] = None
        self.prev_scale: float = 1.0

    def reset(self) -> None:
        self.prev_gray = None
        self.prev_scale = 1.0

    def _resize_gray(self, frame: np.ndarray) -> Tuple[np.ndarray, float]:
        h, w = frame.shape[:2]
        scale = min(1.0, self.process_width / float(max(1, w)))
        if scale < 0.999:
            small = cv2.resize(frame, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)
        else:
            small = frame
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        return gray, scale

    def update(self, frame: np.ndarray, prev_exclude_boxes: Sequence[Box] = ()) -> Optional[np.ndarray]:
        gray, scale = self._resize_gray(frame)
        if self.prev_gray is None or self.prev_gray.shape != gray.shape:
            self.prev_gray = gray
            self.prev_scale = scale
            return None

        mask = np.full_like(self.prev_gray, 255, dtype=np.uint8)
        for box in prev_exclude_boxes:
            x1, y1, x2, y2 = box
            s = self.prev_scale
            x1i, y1i = int(max(0, x1 * s)), int(max(0, y1 * s))
            x2i, y2i = int(min(mask.shape[1], x2 * s)), int(min(mask.shape[0], y2 * s))
            if x2i > x1i and y2i > y1i:
                cv2.rectangle(mask, (x1i, y1i), (x2i, y2i), 0, -1)

        pts0 = cv2.goodFeaturesToTrack(
            self.prev_gray,
            maxCorners=self.max_corners,
            qualityLevel=0.01,
            minDistance=7,
            blockSize=7,
            mask=mask,
        )
        if pts0 is None or len(pts0) < self.min_inliers:
            self.prev_gray = gray
            self.prev_scale = scale
            return None

        pts1, st, err = cv2.calcOpticalFlowPyrLK(
            self.prev_gray,
            gray,
            pts0,
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        if pts1 is None or st is None:
            self.prev_gray = gray
            self.prev_scale = scale
            return None

        ok = st.reshape(-1).astype(bool)
        p0 = pts0.reshape(-1, 2)[ok]
        p1 = pts1.reshape(-1, 2)[ok]
        if len(p0) < self.min_inliers:
            self.prev_gray = gray
            self.prev_scale = scale
            return None

        M_small, inliers = cv2.estimateAffinePartial2D(
            p0,
            p1,
            method=cv2.RANSAC,
            ransacReprojThreshold=self.ransac_thresh,
            maxIters=2000,
            confidence=0.995,
            refineIters=10,
        )

        self.prev_gray = gray
        self.prev_scale = scale

        if M_small is None or inliers is None or int(inliers.sum()) < self.min_inliers:
            return None

        M_small = M_small.astype(np.float32)
        a, b = float(M_small[0, 0]), float(M_small[0, 1])
        scale_est = math.sqrt(a * a + b * b)
        rot_deg = abs(math.degrees(math.atan2(b, a)))
        tx, ty = float(M_small[0, 2]), float(M_small[1, 2])
        diag = math.hypot(gray.shape[1], gray.shape[0])

        # Reject broken optical-flow solutions. UAV frame-to-frame motion should not jump this much.
        if not (0.85 <= scale_est <= 1.15) or rot_deg > 18.0 or math.hypot(tx, ty) > 0.45 * diag:
            return None

        # Convert translation from downscaled coordinates back to original frame coordinates.
        M = M_small.copy()
        if scale > 1e-6:
            M[0, 2] /= scale
            M[1, 2] /= scale
        return M


# --------------------------- parking slot tracker -------------------------

@dataclass
class ParkingSlot:
    # 本次视频/巡检内部的临时视觉ID。它可以短时ReID，但不是现实停车位编号。
    slot_id: int
    polygon: np.ndarray
    conf: float = 1.0
    missed_detects: int = 0
    last_detect_frame: int = 0

    # 6.7.0：现实停车位永久身份。由固定航线路侧顺序映射 + 多次确认后写入 real_id。
    real_id: str = ""
    real_id_conf: float = 0.0
    real_id_hits: Counter = field(default_factory=Counter)
    real_id_scores: Dict[str, float] = field(default_factory=lambda: defaultdict(float))
    zone_id: str = ""

    @property
    def bbox(self) -> Box:
        return polygon_bbox(self.polygon)

    @property
    def display_id(self) -> str:
        return self.real_id if self.real_id else f"P{self.slot_id}"


class RoadsideRouteSlotMapper:
    """Fixed-route permanent slot mapper for roadside parking without per-slot GPS.

    v6.8.0 supports multiple patrol legs. Each leg owns an independent ordered real-slot sequence and
    an absolute start anchor. This matters for the current mission because leg-1 and leg-2 fly in
    opposite physical directions, so image-space direction must be relearned after the leg switch.

    The mapper combines:
      1) ParkingSlotTracker's short-lived local P-ID continuity;
      2) the known real-slot order within the active patrol leg;
      3) image-space motion of roadside slots to infer which side of the frame is "forward";
      4) one-dimensional ordering along the roadside axis;
      5) conservative visual-gap inference so one missed detection does not shift every later ID.

    Important limitation: without GPS/landmarks, the exact instant at which the drone has transferred
    from one physical leg to another is not observable from parking-space detections alone. Therefore
    leg switching is explicit: call ``switch_to_next_segment()`` / ``switch_segment()`` from the mission
    controller. In the standalone preview window, press N to switch to the next configured leg.
    """

    VALID_SCREEN_ORDERS = {
        "auto", "left_to_right", "right_to_left", "top_to_bottom", "bottom_to_top"
    }

    def __init__(
        self,
        enabled: bool,
        map_path: str,
        min_hits: int = 2,
        screen_order: str = "auto",
        start_real_id: str = "",
        direction_min_samples: int = 3,
        min_motion_pixels: float = 1.5,
        gap_skip_ratio: float = 1.75,
        max_inferred_skip: int = 2,
        edge_margin_ratio: float = 0.015,
    ) -> None:
        self.enabled = bool(enabled)
        self.map_path = str(map_path or "")
        self.min_hits = max(1, int(min_hits))
        self.default_screen_order = str(screen_order or "auto").strip().lower()
        if self.default_screen_order not in self.VALID_SCREEN_ORDERS:
            self.default_screen_order = "auto"
        self.screen_order = self.default_screen_order
        self.default_start_real_id = str(start_real_id or "").strip()
        self.start_real_id = self.default_start_real_id
        self.direction_min_samples = max(2, int(direction_min_samples))
        self.min_motion_pixels = max(0.2, float(min_motion_pixels))
        self.gap_skip_ratio = float(np.clip(gap_skip_ratio, 1.25, 4.0))
        self.max_inferred_skip = max(0, int(max_inferred_skip))
        self.edge_margin_ratio = float(np.clip(edge_margin_ratio, 0.0, 0.12))

        self.route_id = ""
        self.area_name = ""
        # All real IDs in actual mission order, across every leg. ``real_to_index`` therefore stays
        # globally unique, while candidate generation is clamped to the active leg's index range.
        self.route_slot_ids: List[str] = []
        self.real_to_index: Dict[str, int] = {}
        self.segments: List[dict] = []
        self.active_segment_index = 0
        self.start_index = 0
        self.end_index = -1

        # session state
        self.prev_centers: Dict[int, np.ndarray] = {}
        self.motion_vectors: Deque[np.ndarray] = deque(maxlen=16)
        self.sequence_axis: Optional[np.ndarray] = None
        self.direction_confidence = 0.0
        self.direction_ready = False
        self.furthest_index = -1
        # global across all legs in one patrol, so cumulative uniqueness survives a leg switch
        self.confirmed_owner: Dict[str, int] = {}
        self.completed_segment_ids: set[str] = set()

        # diagnostics / backward-compatible names used by the renderer
        self.last_zone_id = ""
        self.last_registration_ok = False
        self.last_assignments = 0
        self.last_error = ""
        self.last_gap_skips = 0
        self.total_inferred_skips = 0
        self.last_typical_gap = 0.0
        self.last_axis = (0.0, 0.0)

        if self.enabled:
            self._load_map()

    @property
    def active(self) -> bool:
        return self.enabled and bool(self.route_slot_ids) and bool(self.segments)

    @property
    def current_segment(self) -> dict:
        if 0 <= self.active_segment_index < len(self.segments):
            return self.segments[self.active_segment_index]
        return {}

    @property
    def current_segment_id(self) -> str:
        return str(self.current_segment.get("segment_id", ""))

    @property
    def segment_count(self) -> int:
        return len(self.segments)

    @property
    def segment_complete(self) -> bool:
        return self.end_index >= self.start_index and self.furthest_index >= self.end_index

    @property
    def furthest_real_id(self) -> str:
        if 0 <= self.furthest_index < len(self.route_slot_ids):
            return self.route_slot_ids[self.furthest_index]
        return ""

    @staticmethod
    def _parse_slot_sequence(raw_slots, flight_direction: str = "listed_order") -> List[str]:
        parsed: List[Tuple[int, int, str]] = []
        for pos, item in enumerate(raw_slots or []):
            if isinstance(item, (str, int)):
                rid = str(item).strip()
                order = pos + 1
            elif isinstance(item, dict):
                rid = str(item.get("real_id", item.get("slot_id", ""))).strip()
                try:
                    order = int(item.get("order", pos + 1))
                except Exception:
                    order = pos + 1
                if item.get("enabled", True) is False:
                    continue
            else:
                continue
            if rid:
                parsed.append((order, pos, rid))
        parsed.sort(key=lambda x: (x[0], x[1]))
        out, seen = [], set()
        for _order, _pos, rid in parsed:
            if rid not in seen:
                out.append(rid)
                seen.add(rid)
        if str(flight_direction or "listed_order").strip().lower() in {"reverse", "reverse_order", "backward"}:
            out.reverse()
        return out

    def _reset_direction_state(self) -> None:
        self.prev_centers.clear()
        self.motion_vectors.clear()
        self.sequence_axis = None
        self.direction_confidence = 0.0
        self.direction_ready = self.screen_order != "auto"
        self.furthest_index = self.start_index - 1
        self.last_zone_id = f"{self.route_id}/{self.current_segment_id}" if self.current_segment_id else self.route_id
        self.last_registration_ok = self.direction_ready
        self.last_assignments = 0
        self.last_error = ""
        self.last_gap_skips = 0
        self.last_typical_gap = 0.0
        self.last_axis = (0.0, 0.0)

    def _activate_segment(self, index: int) -> bool:
        if not (0 <= int(index) < len(self.segments)):
            return False
        self.active_segment_index = int(index)
        seg = self.segments[self.active_segment_index]
        self.start_index = int(seg["start_index"])
        self.end_index = int(seg["end_index"])
        self.start_real_id = str(seg["start_real_id"])
        self.screen_order = str(seg.get("screen_order", self.default_screen_order)).strip().lower()
        if self.screen_order not in self.VALID_SCREEN_ORDERS:
            self.screen_order = self.default_screen_order
        self._reset_direction_state()
        print(
            f"[RoadsideSlotMapper] 激活航段 {self.active_segment_index + 1}/{len(self.segments)} "
            f"{self.current_segment_id} | {self.route_slot_ids[self.start_index]} -> "
            f"{self.route_slot_ids[self.end_index]} | screen_order={self.screen_order}"
        )
        return True

    def switch_segment(self, segment: int | str) -> bool:
        """Switch active patrol leg while preserving already-confirmed real IDs from previous legs."""
        target = None
        if isinstance(segment, int):
            target = int(segment)
        else:
            key = str(segment).strip()
            for i, seg in enumerate(self.segments):
                if key in {str(seg.get("segment_id", "")), str(i), str(i + 1)}:
                    target = i
                    break
        if target is None or not (0 <= target < len(self.segments)):
            self.last_error = f"无效航段: {segment}"
            return False
        if self.current_segment_id:
            self.completed_segment_ids.add(self.current_segment_id)
        return self._activate_segment(target)

    def switch_to_next_segment(self) -> bool:
        nxt = self.active_segment_index + 1
        if nxt >= len(self.segments):
            self.last_error = "已经是最后一个航段"
            return False
        return self.switch_segment(nxt)

    def reset_session(self) -> None:
        self.confirmed_owner.clear()
        self.completed_segment_ids.clear()
        self.total_inferred_skips = 0
        if self.segments:
            self._activate_segment(0)
        else:
            self.prev_centers.clear()
            self.motion_vectors.clear()
            self.sequence_axis = None
            self.direction_confidence = 0.0
            self.direction_ready = False
            self.furthest_index = -1

    def _load_map(self) -> None:
        path = Path(self.map_path)
        if not path.exists():
            self.enabled = False
            self.last_error = f"路侧车位顺序表不存在: {path}"
            print(f"[RoadsideSlotMapper][WARN] {self.last_error}，自动回退临时P编号。")
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            self.enabled = False
            self.last_error = f"路侧车位顺序JSON读取失败: {e}"
            print(f"[RoadsideSlotMapper][WARN] {self.last_error}，自动回退临时P编号。")
            return

        self.route_id = str(data.get("route_id", data.get("parking_lot_id", ""))).strip() or path.stem
        self.area_name = str(data.get("area", data.get("area_name", ""))).strip()
        global_screen_order = str(data.get("screen_order", self.default_screen_order)).strip().lower()
        if global_screen_order not in self.VALID_SCREEN_ORDERS:
            global_screen_order = self.default_screen_order

        raw_segments = data.get("segments", None)
        if not isinstance(raw_segments, list) or not raw_segments:
            # Backward compatibility with v6.7 single-leg JSON.
            raw_segments = [{
                "segment_id": str(data.get("segment_id", "LEG-1")),
                "screen_order": global_screen_order,
                "start_real_id": str(data.get("start_real_id", self.default_start_real_id)).strip(),
                "flight_direction": str(data.get("flight_direction", "listed_order")),
                "slots": data.get("slots", []) or [],
            }]

        all_ids: List[str] = []
        seen_global: set[str] = set()
        segments: List[dict] = []
        for seg_pos, raw in enumerate(raw_segments):
            if not isinstance(raw, dict) or raw.get("enabled", True) is False:
                continue
            seg_id = str(raw.get("segment_id", raw.get("id", f"LEG-{seg_pos + 1}"))).strip() or f"LEG-{seg_pos + 1}"
            seg_order = str(raw.get("screen_order", global_screen_order)).strip().lower()
            if seg_order not in self.VALID_SCREEN_ORDERS:
                seg_order = global_screen_order
            ids = self._parse_slot_sequence(raw.get("slots", []) or [], raw.get("flight_direction", "listed_order"))
            # A permanent real ID must occur in exactly one leg for unambiguous cumulative counting.
            ids = [rid for rid in ids if rid not in seen_global]
            if not ids:
                continue
            start_idx = len(all_ids)
            all_ids.extend(ids)
            seen_global.update(ids)
            end_idx = len(all_ids) - 1
            requested_start = str(raw.get("start_real_id", ids[0])).strip()
            start_real = requested_start if requested_start in ids else ids[0]
            local_start_offset = ids.index(start_real)
            # For a fixed patrol leg the configured anchor should normally be its first item. If not, preserve it.
            segments.append({
                "segment_id": seg_id,
                "screen_order": seg_order,
                "start_index": start_idx + local_start_offset,
                "segment_first_index": start_idx,
                "end_index": end_idx,
                "start_real_id": start_real,
            })

        if not all_ids or not segments:
            self.enabled = False
            self.last_error = "路侧车位顺序表中没有有效航段/车位编号"
            print(f"[RoadsideSlotMapper][WARN] {self.last_error}，自动回退临时P编号。")
            return

        self.route_slot_ids = all_ids
        self.real_to_index = {rid: i for i, rid in enumerate(self.route_slot_ids)}
        self.segments = segments
        self.reset_session()
        seg_desc = ", ".join(
            f"{s['segment_id']}:{self.route_slot_ids[s['start_index']]}->{self.route_slot_ids[s['end_index']]}"
            for s in self.segments
        )
        print(
            f"[RoadsideSlotMapper] 已加载路线 {self.route_id} | 总车位={len(self.route_slot_ids)} | "
            f"航段={len(self.segments)} [{seg_desc}]"
        )

    @staticmethod
    def _unit(v: np.ndarray) -> Optional[np.ndarray]:
        v = np.asarray(v, dtype=np.float32).reshape(2)
        n = float(np.linalg.norm(v))
        if n <= 1e-6:
            return None
        return (v / n).astype(np.float32)

    def _explicit_axis(self) -> Optional[np.ndarray]:
        return {
            "left_to_right": np.array([1.0, 0.0], dtype=np.float32),
            "right_to_left": np.array([-1.0, 0.0], dtype=np.float32),
            "top_to_bottom": np.array([0.0, 1.0], dtype=np.float32),
            "bottom_to_top": np.array([0.0, -1.0], dtype=np.float32),
        }.get(self.screen_order)

    def _update_motion_hint(self, slots: Sequence[ParkingSlot], frame_shape: Tuple[int, int]) -> Optional[np.ndarray]:
        current = {s.slot_id: box_center(s.bbox) for s in slots}
        h, w = frame_shape
        diag = max(1.0, math.hypot(w, h))
        deltas = []
        for sid, c in current.items():
            old = self.prev_centers.get(sid)
            if old is None:
                continue
            d = np.asarray(c - old, dtype=np.float32)
            mag = float(np.linalg.norm(d))
            # reject tiny jitter and implausible jumps/ID switches
            if self.min_motion_pixels <= mag <= 0.18 * diag:
                deltas.append(d)
        self.prev_centers = {sid: c.copy() for sid, c in current.items()}

        if deltas:
            med = np.median(np.stack(deltas, axis=0), axis=0).astype(np.float32)
            if float(np.linalg.norm(med)) >= self.min_motion_pixels:
                self.motion_vectors.append(med)

        if len(self.motion_vectors) < self.direction_min_samples:
            return None
        vecs = np.stack(list(self.motion_vectors), axis=0).astype(np.float32)
        med = np.median(vecs, axis=0)
        drift = self._unit(med)
        if drift is None:
            return None
        units = []
        for v in vecs:
            u = self._unit(v)
            if u is not None:
                units.append(u)
        if not units:
            return None
        consistency = float(np.mean([max(0.0, float(np.dot(u, drift))) for u in units]))
        self.direction_confidence = float(np.clip(
            min(1.0, len(units) / float(self.direction_min_samples + 2)) * consistency, 0.0, 1.0
        ))
        if consistency < 0.62:
            return None
        # fixed-route world sequence advances opposite to the apparent image drift of stationary road slots
        return (-drift).astype(np.float32)

    def _axis_from_mapped_anchors(self, slots: Sequence[ParkingSlot]) -> Optional[np.ndarray]:
        anchors = []
        for s in slots:
            if s.real_id in self.real_to_index:
                anchors.append((self.real_to_index[s.real_id], box_center(s.bbox)))
        if len(anchors) < 2:
            return None
        anchors.sort(key=lambda x: x[0])
        lo_i, lo_c = anchors[0]
        hi_i, hi_c = anchors[-1]
        if hi_i <= lo_i:
            return None
        return self._unit(hi_c - lo_c)

    def _refine_axis_with_slot_layout(self, axis_hint: np.ndarray, slots: Sequence[ParkingSlot]) -> np.ndarray:
        if len(slots) < 2:
            return axis_hint
        centers = np.stack([box_center(s.bbox) for s in slots], axis=0).astype(np.float32)
        centered = centers - centers.mean(axis=0, keepdims=True)
        try:
            cov = centered.T @ centered
            vals, vecs = np.linalg.eigh(cov)
            pca = self._unit(vecs[:, int(np.argmax(vals))])
        except Exception:
            pca = None
        if pca is None:
            return axis_hint
        if float(np.dot(pca, axis_hint)) < 0:
            pca = -pca
        # Only use PCA when it agrees reasonably with the learned/explicit forward direction.
        if abs(float(np.dot(pca, axis_hint))) >= 0.35:
            return pca.astype(np.float32)
        return axis_hint

    def _resolve_sequence_axis(self, slots: Sequence[ParkingSlot], frame_shape: Tuple[int, int]) -> Optional[np.ndarray]:
        explicit = self._explicit_axis()
        # Always update motion history even in explicit mode, useful for diagnostics/future switching.
        motion_hint = self._update_motion_hint(slots, frame_shape)
        if explicit is not None:
            axis = explicit
            self.direction_confidence = 1.0
            self.direction_ready = True
        else:
            anchor_axis = self._axis_from_mapped_anchors(slots)
            if anchor_axis is not None:
                axis = anchor_axis
                self.direction_confidence = max(self.direction_confidence, 0.95)
                self.direction_ready = True
            elif motion_hint is not None:
                axis = motion_hint
                self.direction_ready = True
            elif self.sequence_axis is not None:
                axis = self.sequence_axis
            else:
                self.direction_ready = False
                return None
        axis = self._refine_axis_with_slot_layout(axis, slots)
        axis = self._unit(axis)
        if axis is None:
            self.direction_ready = False
            return None
        self.sequence_axis = axis
        self.last_axis = (round(float(axis[0]), 4), round(float(axis[1]), 4))
        return axis

    def _slot_inside_mapping_margin(self, slot: ParkingSlot, w: int, h: int) -> bool:
        if self.edge_margin_ratio <= 0:
            return True
        x1, y1, x2, y2 = slot.bbox
        mx = w * self.edge_margin_ratio
        my = h * self.edge_margin_ratio
        return x1 >= mx and y1 >= my and x2 <= w - mx and y2 <= h - my

    @staticmethod
    def _axis_extent(slot: ParkingSlot, axis: np.ndarray) -> float:
        p = np.asarray(slot.polygon, dtype=np.float32).reshape(-1, 2)
        proj = p @ axis.reshape(2, 1)
        return max(1.0, float(proj.max() - proj.min()))

    def _ordered_entries(self, slots: Sequence[ParkingSlot], axis: np.ndarray, w: int, h: int):
        entries = []
        for s in slots:
            if not self._slot_inside_mapping_margin(s, w, h):
                continue
            c = box_center(s.bbox)
            p = float(np.dot(c, axis))
            extent = self._axis_extent(s, axis)
            entries.append({"slot": s, "proj": p, "extent": extent, "gap_norm": 0.0})
        entries.sort(key=lambda e: e["proj"])
        return entries

    def _gap_steps(self, entries) -> List[int]:
        n = len(entries)
        if n <= 1:
            self.last_typical_gap = 0.0
            self.last_gap_skips = 0
            return []
        norms = []
        for i in range(n - 1):
            gap = max(0.0, float(entries[i + 1]["proj"] - entries[i]["proj"]))
            scale = max(1.0, 0.5 * (entries[i]["extent"] + entries[i + 1]["extent"]))
            ng = gap / scale
            entries[i]["gap_norm"] = ng
            norms.append(ng)

        # Need enough neighbours before inferring a missing slot. With 2~3 visible spaces, use strict adjacency.
        if len(norms) < 3:
            self.last_typical_gap = float(np.median(norms)) if norms else 0.0
            self.last_gap_skips = 0
            return [1] * (n - 1)

        arr = np.asarray(norms, dtype=np.float32)
        # Large gaps may themselves be caused by a missed detection, so estimate normal spacing from the lower 70%.
        cutoff = float(np.percentile(arr, 70))
        base_vals = arr[arr <= cutoff + 1e-6]
        typical = float(np.median(base_vals)) if base_vals.size else float(np.median(arr))
        typical = max(0.15, typical)
        self.last_typical_gap = typical

        steps = []
        inferred = 0
        for ng in norms:
            ratio = ng / typical
            step = 1
            if self.max_inferred_skip > 0 and ratio >= self.gap_skip_ratio:
                estimated = max(2, int(round(ratio)))
                step = min(1 + self.max_inferred_skip, estimated)
            steps.append(step)
            inferred += max(0, step - 1)
        self.last_gap_skips = inferred
        self.total_inferred_skips += inferred
        return steps

    def _build_candidate_indices(self, entries, steps: Sequence[int]) -> Dict[int, int]:
        n = len(entries)
        if n == 0:
            return {}
        anchors = []
        for pos, e in enumerate(entries):
            rid = e["slot"].real_id
            idx = self.real_to_index.get(rid)
            if idx is not None:
                anchors.append((pos, idx))

        out: Dict[int, int] = {}
        if not anchors:
            base = self.start_index if self.furthest_index < self.start_index else self.furthest_index + 1
            out[0] = base
            for p in range(n - 1):
                out[p + 1] = out[p] + int(steps[p] if p < len(steps) else 1)
            return out

        anchors.sort()
        for p, idx in anchors:
            out[p] = idx

        # before the first anchor
        p0, i0 = anchors[0]
        cur = i0
        for p in range(p0 - 1, -1, -1):
            cur -= int(steps[p] if p < len(steps) else 1)
            out[p] = cur

        # after the last anchor
        p1, i1 = anchors[-1]
        cur = i1
        for p in range(p1, n - 1):
            cur += int(steps[p] if p < len(steps) else 1)
            out[p + 1] = cur

        # Fill between anchors while forcing consistency with the two trusted real IDs.
        for (left_p, left_i), (right_p, right_i) in zip(anchors[:-1], anchors[1:]):
            if right_p <= left_p + 1:
                continue
            gaps_n = right_p - left_p
            delta_idx = right_i - left_i
            if delta_idx < gaps_n:
                # contradictory order; leave interior uncertain rather than shifting IDs.
                continue
            increments = [1] * gaps_n
            extra = delta_idx - gaps_n
            if extra > 0:
                ranked = sorted(
                    range(gaps_n),
                    key=lambda j: entries[left_p + j].get("gap_norm", 0.0),
                    reverse=True,
                )
                for j in ranked:
                    if extra <= 0:
                        break
                    add = min(self.max_inferred_skip, extra)
                    increments[j] += add
                    extra -= add
            cur = left_i
            for j, p in enumerate(range(left_p, right_p)):
                cur += increments[j]
                if p + 1 < right_p:
                    out[p + 1] = cur
        return out

    def _vote_real_id(self, slot: ParkingSlot, real_id: str, score: float = 1.0) -> bool:
        if real_id not in self.real_to_index:
            return False
        owner = self.confirmed_owner.get(real_id)
        if owner is not None and owner != slot.slot_id:
            return False

        slot.real_id_hits[real_id] += 1
        slot.real_id_scores[real_id] += max(0.01, float(score))
        best = max(slot.real_id_hits.keys(), key=lambda rid: (slot.real_id_hits[rid], slot.real_id_scores[rid]))
        hits = int(slot.real_id_hits[best])
        avg = float(slot.real_id_scores[best]) / max(1, hits)
        if hits < self.min_hits:
            return False

        old = slot.real_id
        old_hits = int(slot.real_id_hits.get(old, 0)) if old else 0
        if old and old != best and hits < old_hits + 2:
            return False
        best_owner = self.confirmed_owner.get(best)
        if best_owner is not None and best_owner != slot.slot_id:
            return False

        if old and old != best and self.confirmed_owner.get(old) == slot.slot_id:
            self.confirmed_owner.pop(old, None)
        slot.real_id = best
        slot.real_id_conf = float(np.clip(avg, 0.0, 1.0))
        slot.zone_id = self.last_zone_id or self.route_id
        self.confirmed_owner[best] = slot.slot_id
        self.furthest_index = max(self.furthest_index, self.real_to_index[best])
        return True

    def update(self, frame: np.ndarray, slots: Sequence[ParkingSlot], frame_idx: int) -> dict:
        self.last_assignments = 0
        self.last_gap_skips = 0
        self.last_error = ""
        if not self.active:
            return self.diagnostics(slots)
        if not slots:
            self.prev_centers.clear()
            self.last_registration_ok = False if self.screen_order == "auto" else True
            return self.diagnostics(slots)

        h, w = frame.shape[:2]
        axis = self._resolve_sequence_axis(slots, (h, w))
        if axis is None:
            self.last_error = "正在学习无人机巡检方向；若长期无法锁定，请在JSON中设置screen_order"
            self.last_registration_ok = False
            return self.diagnostics(slots)

        self.last_registration_ok = True
        self.last_zone_id = f"{self.route_id}/{self.current_segment_id}" if self.current_segment_id else self.route_id
        entries = self._ordered_entries(slots, axis, w, h)
        if not entries:
            self.last_error = "当前车位均位于画面边缘，暂缓永久编号"
            return self.diagnostics(slots)

        steps = self._gap_steps(entries)
        candidates = self._build_candidate_indices(entries, steps)
        used_indices = set()
        for pos, e in enumerate(entries):
            slot = e["slot"]
            idx = candidates.get(pos)
            if idx is None or not (0 <= idx < len(self.route_slot_ids)):
                continue
            # Never cross a configured leg boundary implicitly. The second leg must be activated explicitly.
            if idx < self.start_index or idx > self.end_index:
                continue
            if idx in used_indices:
                continue
            rid = self.route_slot_ids[idx]

            # Existing confirmed slots are anchors. Never force them to another ID from a weak local ordering change.
            if slot.real_id:
                used_indices.add(self.real_to_index.get(slot.real_id, idx))
                continue

            # Do not reuse a real number already confirmed for a different local slot during this one-way patrol.
            owner = self.confirmed_owner.get(rid)
            if owner is not None and owner != slot.slot_id:
                continue

            # Conservative confidence: adjacency=1.0; IDs inferred across a visual gap get a small penalty.
            score = 1.0
            if pos > 0 and pos - 1 < len(steps) and steps[pos - 1] > 1:
                score = max(0.60, 1.0 - 0.12 * (steps[pos - 1] - 1))
            before = bool(slot.real_id)
            confirmed = self._vote_real_id(slot, rid, score)
            after = bool(slot.real_id)
            if confirmed or (not before and after):
                self.last_assignments += 1
            used_indices.add(idx)

        # 最终保险：一个真实编号在当前画面只允许一个 local slot 持有。
        self._enforce_unique_real_ids(slots)
        return self.diagnostics(slots)

    def rebind_tracker_aliases(self, aliases: Dict[int, int], slots: Sequence[ParkingSlot] = ()) -> None:
        """When ParkingSlotTracker merges duplicate local IDs, keep permanent-ID ownership coherent.

        ``aliases`` maps dropped_local_id -> canonical_local_id. Only transfer an ownership entry
        when the canonical slot still carries that same real_id; otherwise remove the stale ownership
        so a wrong/obsolete duplicate cannot poison the rest of the one-way route.
        """
        if not aliases:
            return

        def resolve(sid: int) -> int:
            seen = set()
            cur = int(sid)
            while cur in aliases and cur not in seen:
                seen.add(cur)
                cur = int(aliases[cur])
            return cur

        by_sid = {int(s.slot_id): s for s in slots}
        for rid, owner in list(self.confirmed_owner.items()):
            new_owner = resolve(int(owner))
            if new_owner == int(owner):
                continue
            keep = by_sid.get(new_owner)
            if keep is not None and keep.real_id == rid:
                self.confirmed_owner[rid] = new_owner
            else:
                self.confirmed_owner.pop(rid, None)

        # Direction-learning cache also uses local IDs; collapse aliases there too.
        for old_id, new_id in aliases.items():
            old_id, new_id = int(old_id), resolve(int(new_id))
            if old_id in self.prev_centers:
                if new_id not in self.prev_centers:
                    self.prev_centers[new_id] = self.prev_centers[old_id]
                self.prev_centers.pop(old_id, None)

    def _enforce_unique_real_ids(self, slots: Sequence[ParkingSlot]) -> None:
        """Hard invariant for rendering/business logic: one real slot number -> one current local ID."""
        groups: Dict[str, List[ParkingSlot]] = defaultdict(list)
        for s in slots:
            if s.real_id:
                groups[str(s.real_id)].append(s)
        for rid, group in groups.items():
            if len(group) <= 1:
                if group:
                    self.confirmed_owner[rid] = int(group[0].slot_id)
                continue
            # Prefer stronger historical evidence, then confidence, then the older/lower local ID.
            def strength(s: ParkingSlot):
                hits = int(s.real_id_hits.get(rid, 0))
                score = float(s.real_id_scores.get(rid, 0.0))
                return (hits, score, float(s.real_id_conf), float(s.conf), -int(s.slot_id))
            winner = max(group, key=strength)
            self.confirmed_owner[rid] = int(winner.slot_id)
            for s in group:
                if s is winner:
                    continue
                s.real_id = ""
                s.real_id_conf = 0.0
                # Remove only this conflicting candidate; other votes may still be useful.
                s.real_id_hits.pop(rid, None)
                s.real_id_scores.pop(rid, None)

    def diagnostics(self, slots: Sequence[ParkingSlot] = ()) -> dict:
        # Run the real-ID uniqueness guard even on early-return frames (e.g. direction still learning).
        self._enforce_unique_real_ids(slots)
        mapped = sum(1 for s in slots if s.real_id)
        return {
            # Backward-compatible fields used by existing platform/UI code.
            "permanent_slot_mapping_enabled": bool(self.active),
            "permanent_slot_zone": self.route_id,
            "permanent_slot_registration_ok": bool(self.last_registration_ok),
            "permanent_slot_good_matches": 0,
            "permanent_slot_inliers": 0,
            "permanent_slot_inlier_ratio": 0.0,
            "permanent_slot_mapped_current": int(mapped),
            "permanent_slot_unmapped_current": int(max(0, len(slots) - mapped)),
            "permanent_slot_assignments_last": int(self.last_assignments),
            "permanent_slot_map_error": self.last_error,

            # v6.8.0 roadside multi-leg diagnostics.
            "roadside_slot_mapping_enabled": bool(self.active),
            "roadside_route_id": self.route_id,
            "roadside_area": self.area_name,
            "roadside_segment_id": self.current_segment_id,
            "roadside_segment_index": int(self.active_segment_index + 1),
            "roadside_segment_count": int(len(self.segments)),
            "roadside_segment_start_real_id": self.route_slot_ids[self.start_index] if 0 <= self.start_index < len(self.route_slot_ids) else "",
            "roadside_segment_end_real_id": self.route_slot_ids[self.end_index] if 0 <= self.end_index < len(self.route_slot_ids) else "",
            "roadside_segment_complete": bool(self.segment_complete),
            "roadside_screen_order": self.screen_order,
            "roadside_direction_ready": bool(self.direction_ready),
            "roadside_direction_confidence": round(float(self.direction_confidence), 3),
            "roadside_sequence_axis": [float(self.last_axis[0]), float(self.last_axis[1])],
            "roadside_furthest_real_id": self.furthest_real_id,
            "roadside_confirmed_unique_slots": int(len(self.confirmed_owner)),
            "roadside_gap_skips_last": int(self.last_gap_skips),
            "roadside_gap_skips_total": int(self.total_inferred_skips),
            "roadside_typical_gap_normalized": round(float(self.last_typical_gap), 3),
        }


class ParkingSlotTracker:
    """Track parking-space polygons while enforcing one canonical local ID per physical slot.

    v6.8.3 adds three defensive layers:
      1) same-frame polygon de-duplication before any ID association;
      2) global one-to-one active/retired association (not confidence-order greedy stealing);
      3) active-ID merge, so historical P17/P23 duplicates collapse to one canonical ID.

    The detector is still allowed to detect the same physical slot on every segmentation cycle. The
    invariant is at the tracking/business layer: one physical slot is represented by one current local ID.
    """

    def __init__(
        self,
        max_missed_detects: int = 8,
        match_iou: float = 0.16,
        reid_retention_frames: int = 75,
        dedup_iou: float = PARKING_DEDUP_IOU,
        dedup_min_cover: float = PARKING_DEDUP_MIN_COVER,
        dedup_center_ratio: float = PARKING_DEDUP_CENTER_RATIO,
        duplicate_area_similarity: float = PARKING_DUPLICATE_AREA_SIMILARITY,
        active_merge_iou: float = PARKING_ACTIVE_MERGE_IOU,
        active_merge_min_cover: float = PARKING_ACTIVE_MERGE_MIN_COVER,
        active_merge_center_ratio: float = PARKING_ACTIVE_MERGE_CENTER_RATIO,
    ) -> None:
        self.slots: Dict[int, ParkingSlot] = {}
        self.retired_slots: Dict[int, Tuple[ParkingSlot, int]] = {}
        self.next_id = 1
        self.max_missed_detects = max(1, int(max_missed_detects))
        self.match_iou = float(match_iou)
        self.reid_retention_frames = max(1, int(reid_retention_frames))
        self.dedup_iou = float(dedup_iou)
        self.dedup_min_cover = float(dedup_min_cover)
        self.dedup_center_ratio = float(dedup_center_ratio)
        self.duplicate_area_similarity = float(duplicate_area_similarity)
        self.active_merge_iou = float(active_merge_iou)
        self.active_merge_min_cover = float(active_merge_min_cover)
        self.active_merge_center_ratio = float(active_merge_center_ratio)
        self.total_local_ids_created = 0
        self.total_reidentified = 0
        self.total_detection_duplicates_suppressed = 0
        self.total_duplicate_ids_merged = 0
        self.last_detection_duplicates_suppressed = 0
        self.last_duplicate_ids_merged = 0
        self._merge_aliases: Dict[int, int] = {}

    def warp(self, M: Optional[np.ndarray]) -> None:
        if M is None:
            return
        for slot in self.slots.values():
            slot.polygon = transform_polygon(M, slot.polygon)
        for slot, _retired_frame in self.retired_slots.values():
            slot.polygon = transform_polygon(M, slot.polygon)

    @staticmethod
    def _geometry_metrics(poly_a: np.ndarray, poly_b: np.ndarray) -> Tuple[float, float, float, float, float]:
        a = np.asarray(poly_a, dtype=np.float32).reshape(-1, 2)
        b = np.asarray(poly_b, dtype=np.float32).reshape(-1, 2)
        aa = max(1.0, polygon_area(a))
        ab = max(1.0, polygon_area(b))
        inter = convex_intersection_area(a, b)
        union = max(1.0, aa + ab - inter)
        poly_iou = float(inter / union)
        min_cover = float(inter / max(1.0, min(aa, ab)))
        area_similarity = float(min(aa, ab) / max(aa, ab))
        bba, bbb = polygon_bbox(a), polygon_bbox(b)
        ca, cb = box_center(bba), box_center(bbb)
        diag = max(1.0, box_diag(bba), box_diag(bbb))
        center_ratio = float(np.linalg.norm(ca - cb)) / diag
        bb_iou = bbox_iou(bba, bbb)
        return poly_iou, min_cover, center_ratio, area_similarity, bb_iou

    def _is_same_frame_duplicate(self, a: np.ndarray, b: np.ndarray) -> bool:
        iou, cover, dist, area_sim, bb_iou = self._geometry_metrics(a, b)
        if area_sim < self.duplicate_area_similarity:
            return False
        return bool(
            (iou >= self.dedup_iou and dist <= 0.32)
            or (cover >= self.dedup_min_cover and dist <= self.dedup_center_ratio)
            or (bb_iou >= 0.80 and dist <= 0.16 and area_sim >= 0.68)
        )

    def _is_active_duplicate(self, a: ParkingSlot, b: ParkingSlot) -> bool:
        iou, cover, dist, area_sim, bb_iou = self._geometry_metrics(a.polygon, b.polygon)
        if area_sim < self.duplicate_area_similarity:
            return False
        return bool(
            (iou >= self.active_merge_iou and dist <= 0.28)
            or (cover >= self.active_merge_min_cover and dist <= self.active_merge_center_ratio)
            or (bb_iou >= 0.84 and dist <= 0.14 and area_sim >= 0.70)
        )

    def _deduplicate_detections(self, detections: Sequence[Tuple[np.ndarray, float]]) -> List[Tuple[np.ndarray, float]]:
        kept: List[Tuple[np.ndarray, float]] = []
        suppressed = 0
        for poly, conf in sorted(detections, key=lambda x: float(x[1]), reverse=True):
            p = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
            if len(p) < 3 or polygon_area(p) < 20.0:
                continue
            duplicate = False
            for kp, _kc in kept:
                if self._is_same_frame_duplicate(p, kp):
                    duplicate = True
                    suppressed += 1
                    break
            if not duplicate:
                kept.append((p, float(conf)))
        self.last_detection_duplicates_suppressed = suppressed
        self.total_detection_duplicates_suppressed += suppressed
        return kept

    def _match_metrics(self, poly: np.ndarray, slot: ParkingSlot) -> Tuple[bool, float]:
        iou, cover, dist_ratio, area_similarity, bb_iou = self._geometry_metrics(poly, slot.polygon)
        # Compared with the previous bbox-only matcher, polygon overlap makes the gate more stable after
        # segmentation-shape jitter while the center/area conditions protect adjacent roadside spaces.
        eligible = (
            iou >= self.match_iou
            or (cover >= 0.42 and dist_ratio <= 0.34 and area_similarity >= 0.48)
            or (bb_iou >= 0.08 and dist_ratio <= 0.30 and area_similarity >= 0.52)
            or (dist_ratio <= 0.20 and area_similarity >= 0.66)
        )
        score = (
            2.6 * iou
            + 0.80 * cover
            + 0.70 * max(0.0, 1.0 - dist_ratio)
            + 0.25 * area_similarity
            + 0.15 * bb_iou
        )
        return bool(eligible), float(score)

    @staticmethod
    def _slot_strength(slot: ParkingSlot) -> Tuple[int, int, float, float, int, int]:
        real_hits = max(slot.real_id_hits.values()) if slot.real_id_hits else 0
        return (
            1 if slot.real_id else 0,
            int(real_hits),
            float(slot.real_id_conf),
            float(slot.conf),
            -int(slot.missed_detects),
            -int(slot.slot_id),  # older/lower ID wins final tie
        )

    @staticmethod
    def _copy_or_merge_identity(keep: ParkingSlot, drop: ParkingSlot) -> None:
        # Preserve all voting history first.
        keep.real_id_hits.update(drop.real_id_hits)
        for rid, score in drop.real_id_scores.items():
            keep.real_id_scores[rid] += float(score)

        if not keep.real_id and drop.real_id:
            keep.real_id = drop.real_id
            keep.real_id_conf = drop.real_id_conf
            keep.zone_id = drop.zone_id
        elif keep.real_id and drop.real_id and keep.real_id != drop.real_id:
            # Same geometry cannot legitimately own two real numbers. Keep the stronger evidence only.
            def ev(slot: ParkingSlot, rid: str):
                return (
                    int(slot.real_id_hits.get(rid, 0)),
                    float(slot.real_id_scores.get(rid, 0.0)),
                    float(slot.real_id_conf),
                )
            if ev(drop, drop.real_id) > ev(keep, keep.real_id):
                keep.real_id = drop.real_id
                keep.real_id_conf = drop.real_id_conf
                keep.zone_id = drop.zone_id

    def _record_alias(self, dropped_id: int, kept_id: int) -> None:
        dropped_id, kept_id = int(dropped_id), int(kept_id)
        if dropped_id == kept_id:
            return
        # Collapse alias chains so downstream permanent-ID ownership can be fixed once.
        for old, target in list(self._merge_aliases.items()):
            if int(target) == dropped_id:
                self._merge_aliases[old] = kept_id
        self._merge_aliases[dropped_id] = kept_id

    def consume_merge_aliases(self) -> Dict[int, int]:
        out = dict(self._merge_aliases)
        self._merge_aliases.clear()
        return out

    def _merge_active_duplicates(self, frame_idx: int) -> None:
        self.last_duplicate_ids_merged = 0
        changed = True
        while changed:
            changed = False
            ids = sorted(self.slots.keys())
            for i in range(len(ids)):
                a_id = ids[i]
                if a_id not in self.slots:
                    continue
                for j in range(i + 1, len(ids)):
                    b_id = ids[j]
                    if b_id not in self.slots:
                        continue
                    a, b = self.slots[a_id], self.slots[b_id]
                    if not self._is_active_duplicate(a, b):
                        continue
                    if self._slot_strength(b) > self._slot_strength(a):
                        keep_id, drop_id = b_id, a_id
                    else:
                        keep_id, drop_id = a_id, b_id
                    keep, drop = self.slots[keep_id], self.slots[drop_id]
                    self._copy_or_merge_identity(keep, drop)
                    # Use the most recently observed/high-confidence geometry, not an average that can blur boundaries.
                    if (drop.last_detect_frame, drop.conf) > (keep.last_detect_frame, keep.conf):
                        keep.polygon = np.asarray(drop.polygon, dtype=np.float32).copy()
                    keep.conf = max(float(keep.conf), float(drop.conf))
                    keep.missed_detects = min(int(keep.missed_detects), int(drop.missed_detects))
                    keep.last_detect_frame = max(int(keep.last_detect_frame), int(drop.last_detect_frame), int(frame_idx))
                    self.slots.pop(drop_id, None)
                    self.retired_slots.pop(drop_id, None)
                    self._record_alias(drop_id, keep_id)
                    self.last_duplicate_ids_merged += 1
                    self.total_duplicate_ids_merged += 1
                    changed = True
                    break
                if changed:
                    break

    def _global_assign_active(
        self,
        detections: Sequence[Tuple[np.ndarray, float]],
    ) -> Tuple[Dict[int, int], set[int], set[int]]:
        """Return det_index->slot_id using global descending pair scores, one-to-one on both sides."""
        pairs = []
        for di, (poly, _conf) in enumerate(detections):
            for sid, slot in self.slots.items():
                eligible, score = self._match_metrics(poly, slot)
                if eligible:
                    pairs.append((float(score), int(di), int(sid)))
        pairs.sort(key=lambda x: x[0], reverse=True)
        det_to_sid: Dict[int, int] = {}
        used_det, used_sid = set(), set()
        for _score, di, sid in pairs:
            if di in used_det or sid in used_sid:
                continue
            det_to_sid[di] = sid
            used_det.add(di)
            used_sid.add(sid)
        return det_to_sid, used_det, used_sid

    def _global_assign_retired(
        self,
        detections: Sequence[Tuple[np.ndarray, float]],
        candidate_det_ids: Sequence[int],
    ) -> Dict[int, int]:
        pairs = []
        for di in candidate_det_ids:
            poly = detections[di][0]
            for sid, (slot, _retired_frame) in self.retired_slots.items():
                eligible, score = self._match_metrics(poly, slot)
                if eligible:
                    # Slight penalty so an active ID always wins an otherwise equal association.
                    pairs.append((float(score) - 0.08, int(di), int(sid)))
        pairs.sort(key=lambda x: x[0], reverse=True)
        det_to_sid: Dict[int, int] = {}
        used_det, used_sid = set(), set()
        for _score, di, sid in pairs:
            if di in used_det or sid in used_sid:
                continue
            det_to_sid[di] = sid
            used_det.add(di)
            used_sid.add(sid)
        return det_to_sid

    def update_detections(self, detections: Sequence[Tuple[np.ndarray, float]], frame_idx: int) -> None:
        detections = self._deduplicate_detections(detections)

        expired = [
            sid for sid, (_slot, retired_frame) in self.retired_slots.items()
            if frame_idx - retired_frame > self.reid_retention_frames
        ]
        for sid in expired:
            self.retired_slots.pop(sid, None)

        old_ids = set(self.slots.keys())
        det_to_sid, used_det, used_active = self._global_assign_active(detections)

        # Re-ID only detections that failed active matching.
        remaining_det = [i for i in range(len(detections)) if i not in used_det]
        retired_match = self._global_assign_retired(detections, remaining_det)
        for di, sid in retired_match.items():
            old_slot, _ = self.retired_slots.pop(sid)
            self.slots[sid] = old_slot
            det_to_sid[di] = sid
            used_det.add(di)
            self.total_reidentified += 1

        # Update all matched IDs.
        matched_sids = set()
        for di, sid in det_to_sid.items():
            poly, conf = detections[di]
            slot = self.slots.get(sid)
            if slot is None:
                continue
            slot.polygon = np.asarray(poly, dtype=np.float32)
            slot.conf = float(conf)
            slot.missed_detects = 0
            slot.last_detect_frame = int(frame_idx)
            matched_sids.add(int(sid))

        # Truly new geometry only after de-dup + active global match + retired global match all fail.
        for di, (poly, conf) in enumerate(detections):
            if di in used_det:
                continue
            sid = self.next_id
            self.next_id += 1
            self.total_local_ids_created += 1
            self.slots[sid] = ParkingSlot(sid, np.asarray(poly, dtype=np.float32), float(conf), 0, int(frame_idx))
            matched_sids.add(sid)

        # Only pre-existing unmatched active IDs accrue misses. New IDs are not penalized in their birth frame.
        for sid in old_ids:
            if sid in self.slots and sid not in matched_sids:
                self.slots[sid].missed_detects += 1

        # Collapse any duplicate IDs that already existed from previous frames/older versions.
        self._merge_active_duplicates(frame_idx)

        dead = [sid for sid, s in self.slots.items() if s.missed_detects > self.max_missed_detects]
        for sid in dead:
            slot = self.slots.pop(sid, None)
            if slot is not None:
                self.retired_slots[sid] = (slot, int(frame_idx))

    def visible_slots(self, w: int, h: int) -> List[ParkingSlot]:
        # A final visibility pass cannot create IDs; active-duplicate merging has already enforced uniqueness.
        out = []
        for s in self.slots.values():
            x1, y1, x2, y2 = s.bbox
            if x2 >= 0 and y2 >= 0 and x1 < w and y1 < h and polygon_area(s.polygon) > 20.0:
                out.append(s)
        return out


# ----------------------------- vehicle state ------------------------------

STATUS_MONITORING = "监测中"
STATUS_MOVING = "行驶/未静止"
STATUS_NORMAL = "正常停泊"
STATUS_VIOLATION = "违停"
STATUS_BORDERLINE = "疑似压线"


@dataclass
class VehicleTrackState:
    track_id: int
    cls_id: int
    cls_name: str
    first_seen_t: float
    last_seen_t: float
    last_frame_idx: int
    box: Box
    prev_center: np.ndarray
    conf: float = 0.0

    residual_ratios: Deque[float] = field(default_factory=lambda: deque(maxlen=25))
    stationary_accum_s: float = 0.0
    is_stationary: bool = False
    stationary_effective_seconds: float = 0.0

    best_slot_id: Optional[int] = None
    best_real_slot_id: str = ""
    best_vehicle_overlap: float = 0.0  # intersection / vehicle box area
    best_slot_overlap: float = 0.0     # intersection / slot area
    center_in_slot: bool = False
    overlapping_slot_count: int = 0
    violation_type: str = ""  # OUTSIDE / MULTI_SPACE / PARTIAL_SPACE

    raw_status: str = STATUS_MONITORING
    confirmed_status: str = STATUS_MONITORING
    status_candidate: str = STATUS_MONITORING
    status_streak: int = 0

    plate_scores: Dict[str, float] = field(default_factory=lambda: defaultdict(float))
    plate_hits: Counter = field(default_factory=Counter)
    plate_colors: Counter = field(default_factory=Counter)
    stable_plate: str = ""
    stable_plate_color: str = ""
    last_plate_try_t: float = 0.0
    last_plate_seen_t: float = 0.0
    plate_rel_box: Optional[Tuple[float, float, float, float]] = None
    last_plate_detect_conf: float = 0.0

    def update_motion(
        self,
        new_box: Box,
        now: float,
        frame_idx: int,
        M_prev_to_curr: Optional[np.ndarray],
        stationary_ratio_threshold: float,
        stationary_seconds: float,
        stationary_fast_seconds: float = 0.65,
        strong_residual_factor: float = 0.55,
    ) -> None:
        new_center = box_center(new_box)
        dt = max(1e-3, now - self.last_seen_t)

        # Only compare consecutive processed frames; otherwise a single affine transform is insufficient.
        if M_prev_to_curr is not None and frame_idx == self.last_frame_idx + 1:
            predicted_if_static = transform_point(M_prev_to_curr, self.prev_center)
            residual = float(np.linalg.norm(new_center - predicted_if_static))
            ratio = residual / max(1.0, box_diag(new_box))
            self.residual_ratios.append(ratio)

            if ratio <= stationary_ratio_threshold:
                self.stationary_accum_s += dt
            else:
                # Hysteresis: one noisy optical-flow frame should not instantly erase a parked state.
                self.stationary_accum_s = max(0.0, self.stationary_accum_s - 1.7 * dt)

        # Adaptive confirmation: only very-low residual tracks with enough history may use the fast gate.
        # Borderline residuals still require the normal gate, preserving robustness against slow moving traffic.
        effective = max(0.2, float(stationary_seconds))
        if len(self.residual_ratios) >= 4:
            vals = np.asarray(self.residual_ratios, dtype=np.float32)
            recent = vals[-min(8, len(vals)):]
            med = float(np.median(recent))
            p80 = float(np.percentile(recent, 80))
            strong_gate = stationary_ratio_threshold * max(0.25, min(0.90, strong_residual_factor))
            if med <= strong_gate and p80 <= stationary_ratio_threshold * 0.80:
                effective = min(effective, max(0.2, float(stationary_fast_seconds)))

        self.stationary_effective_seconds = effective
        self.is_stationary = self.stationary_accum_s >= effective
        self.prev_center = new_center
        self.box = new_box
        self.last_seen_t = now
        self.last_frame_idx = frame_idx

    @property
    def motion_score(self) -> Optional[float]:
        if not self.residual_ratios:
            return None
        vals = np.asarray(self.residual_ratios, dtype=np.float32)
        return float(np.median(vals[-min(12, len(vals)):]))

    def update_status(self, raw_status: str, confirm_frames: int) -> None:
        self.raw_status = raw_status
        if raw_status != self.status_candidate:
            self.status_candidate = raw_status
            self.status_streak = 1
        else:
            self.status_streak += 1

        if raw_status in (STATUS_NORMAL, STATUS_VIOLATION, STATUS_BORDERLINE):
            if self.status_streak >= confirm_frames:
                self.confirmed_status = raw_status
        else:
            # Moving/monitoring should update immediately to avoid showing an old parked result on a moving car.
            self.confirmed_status = raw_status

    def update_plate(
        self,
        plate_no: str,
        plate_color: str,
        score: float,
        detect_conf: float,
        global_plate_box: Optional[Box],
        min_hits: int,
        current_vehicle_box: Box,
        now: float,
    ) -> None:
        self.last_plate_try_t = now
        if not plate_no:
            return

        self.last_plate_seen_t = now
        self.plate_scores[plate_no] += max(0.01, float(score))
        self.plate_hits[plate_no] += 1
        if plate_color:
            self.plate_colors[(plate_no, plate_color)] += 1
        self.last_plate_detect_conf = float(detect_conf)

        if global_plate_box is not None:
            vx1, vy1, vx2, vy2 = current_vehicle_box
            vw, vh = max(1.0, vx2 - vx1), max(1.0, vy2 - vy1)
            px1, py1, px2, py2 = global_plate_box
            self.plate_rel_box = (
                (px1 - vx1) / vw,
                (py1 - vy1) / vh,
                (px2 - vx1) / vw,
                (py2 - vy1) / vh,
            )

        # Weighted voting first; hit count prevents one-frame OCR spikes from becoming a permanent plate identity.
        best = max(self.plate_scores.keys(), key=lambda p: (self.plate_hits[p], self.plate_scores[p]))
        if self.plate_hits[best] >= min_hits:
            self.stable_plate = best
            colors = [(cnt, color) for (p, color), cnt in self.plate_colors.items() if p == best]
            self.stable_plate_color = max(colors)[1] if colors else ""

    def current_plate_box(self) -> Optional[Box]:
        if self.plate_rel_box is None:
            return None
        vx1, vy1, vx2, vy2 = self.box
        vw, vh = max(1.0, vx2 - vx1), max(1.0, vy2 - vy1)
        rx1, ry1, rx2, ry2 = self.plate_rel_box
        return vx1 + rx1 * vw, vy1 + ry1 * vh, vx1 + rx2 * vw, vy1 + ry2 * vh


class VehicleStateManager:
    def __init__(self, max_lost_seconds: float = 2.0) -> None:
        self.states: Dict[int, VehicleTrackState] = {}
        self.max_lost_seconds = max_lost_seconds

    def obtain_state(
        self,
        track_id: int,
        cls_id: int,
        cls_name: str,
        box: Box,
        conf: float,
        now: float,
        frame_idx: int,
        M_prev_to_curr: Optional[np.ndarray],
        current_tracker_ids: set[int],
    ) -> Tuple[VehicleTrackState, bool, Optional[int]]:
        """Get tracker state, with a fast one-frame ID-switch repair layer.

        BoT-SORT+GMC is the primary tracker. This extra association is only used when BoT-SORT
        creates a new ID while an old same-class ID vanished on the immediately previous frame.
        It transfers OCR/stationary history instead of starting from zero.

        Returns: (state, is_new_identity, reassociated_from_track_id).
        """
        st = self.states.get(track_id)
        if st is not None:
            return st, False, None

        current_center = box_center(box)
        best_old_id = None
        best_score = -1.0

        for old_id, old in self.states.items():
            if old_id in current_tracker_ids:
                continue  # this old ID still exists in the current frame, so it cannot be an ID switch candidate
            if old.cls_id != cls_id or old.last_frame_idx != frame_idx - 1:
                continue

            pred_box = transform_box(M_prev_to_curr, old.box) if M_prev_to_curr is not None else old.box
            pred_center = box_center(pred_box)
            dist_ratio = float(np.linalg.norm(current_center - pred_center)) / max(1.0, box_diag(box))
            iou = bbox_iou(pred_box, box)

            # Either meaningful overlap or a close motion-compensated center is enough.
            if iou < 0.12 and dist_ratio > 0.40:
                continue
            score = 1.8 * iou + max(0.0, 1.0 - dist_ratio)
            if score > best_score:
                best_score = score
                best_old_id = old_id

        if best_old_id is not None:
            st = self.states.pop(best_old_id)
            st.track_id = track_id
            st.cls_id = cls_id
            st.cls_name = cls_name
            st.conf = conf
            self.states[track_id] = st
            return st, False, best_old_id

        c = box_center(box)
        st = VehicleTrackState(
            track_id=track_id,
            cls_id=cls_id,
            cls_name=cls_name,
            first_seen_t=now,
            last_seen_t=now,
            last_frame_idx=frame_idx,
            box=box,
            prev_center=c,
            conf=conf,
        )
        self.states[track_id] = st
        return st, True, None

    def purge(self, now: float) -> None:
        dead = [tid for tid, st in self.states.items() if now - st.last_seen_t > self.max_lost_seconds]
        for tid in dead:
            self.states.pop(tid, None)


# ------------------------------ plate OCR ---------------------------------


def four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    rect = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    tl, tr, br, bl = rect
    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_width = max(2, int(round(max(width_a, width_b))))
    max_height = max(2, int(round(max(height_a, height_b))))
    dst = np.array(
        [[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]],
        dtype=np.float32,
    )
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (max_width, max_height))


def valid_plate_text(text: str) -> bool:
    if not text:
        return False
    text = text.strip()
    if "#" in text or len(text) < 6 or len(text) > 10:
        return False
    return True


@dataclass
class PlateObservation:
    plate_no: str = ""
    plate_color: str = ""
    score: float = 0.0
    detect_conf: float = 0.0
    global_box: Optional[Box] = None


# ------------------------------- event log --------------------------------

class EventLogger:
    """Log only plate-bound final parking events. One plate/status pair is emitted once per run."""

    def __init__(self, path: Optional[str]) -> None:
        self.path = Path(path) if path else None
        self.seen: set[Tuple[str, str]] = set()
        self.normal_plates: set[str] = set()
        self.violation_plates: set[str] = set()
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def reset(self) -> None:
        """Clear per-session deduplication/cumulative plate sets without deleting the log file."""
        self.seen.clear()
        self.normal_plates.clear()
        self.violation_plates.clear()

    def maybe_log(self, st: VehicleTrackState, now: float) -> Optional[dict]:
        if st.confirmed_status not in (STATUS_NORMAL, STATUS_VIOLATION, STATUS_BORDERLINE):
            return None
        if not st.stable_plate:
            return None

        event_status = STATUS_VIOLATION if st.confirmed_status == STATUS_BORDERLINE else st.confirmed_status
        key = (st.stable_plate, event_status)
        if key in self.seen:
            return None
        self.seen.add(key)

        if event_status == STATUS_NORMAL:
            self.normal_plates.add(st.stable_plate)
        else:
            self.violation_plates.add(st.stable_plate)

        event = {
            "ts": now,
            "time_local": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now)),
            "status": event_status,
            "track_id": st.track_id,
            "vehicle_class": st.cls_name,
            "plate": st.stable_plate,
            "plate_color": st.stable_plate_color,
            # parking_slot_id keeps a single convenient public field: real ID first, local ID fallback.
            "parking_slot_id": st.best_real_slot_id if st.best_real_slot_id else st.best_slot_id,
            "parking_slot_real_id": st.best_real_slot_id,
            "parking_slot_local_id": st.best_slot_id,
            "vehicle_in_slot_ratio": round(st.best_vehicle_overlap, 4),
            "slot_covered_ratio": round(st.best_slot_overlap, 4),
            "camera_compensated_motion_score": None if st.motion_score is None else round(st.motion_score, 5),
            "violation_type": st.violation_type if event_status == STATUS_VIOLATION else "",
        }
        if self.path:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False) + "\n")
        return event


# ------------------------------ live capture ------------------------------

class LatestFrameCapture:
    """Capture thread that always exposes the newest frame and discards stale frames.

    For local video-file testing it throttles reads using source FPS so it does not instantly consume the file.
    For RTSP/camera feeds there is intentionally no queue backlog.
    """

    def __init__(self, source: str | int) -> None:
        self.source = source
        self._lock = threading.Lock()
        self._frame: Optional[np.ndarray] = None
        self._seq = 0
        self._capture_perf = 0.0  # 最新帧真正被采集到的单调时钟时间戳
        self._stopped = threading.Event()
        self._pause_file = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.cap: Optional[cv2.VideoCapture] = None
        self.source_fps = 0.0
        self.source_total_frames = 0
        self.is_file = isinstance(source, str) and Path(source).is_file()

    def start(self) -> "LatestFrameCapture":
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            raise RuntimeError(f"无法打开视频源: {self.source}")
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        self.source_fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)
        self.source_total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) if self.is_file else 0
        self._thread = threading.Thread(target=self._run, name="latest-frame-capture", daemon=True)
        self._thread.start()
        return self

    def _run(self) -> None:
        assert self.cap is not None
        frame_period = 1.0 / self.source_fps if self.is_file and self.source_fps > 1.0 else 0.0
        next_t = time.perf_counter()
        while not self._stopped.is_set():
            if self.is_file and self._pause_file.is_set():
                next_t = time.perf_counter()
                time.sleep(0.02)
                continue
            ok, frame = self.cap.read()
            if not ok:
                self._stopped.set()
                break
            capture_perf = time.perf_counter()
            with self._lock:
                self._frame = frame
                self._seq += 1
                self._capture_perf = capture_perf
            if frame_period > 0:
                next_t += frame_period
                sleep_s = next_t - time.perf_counter()
                if sleep_s > 0:
                    time.sleep(sleep_s)
                else:
                    next_t = time.perf_counter()

    def read_latest(self, last_seq: int) -> Tuple[Optional[np.ndarray], int, float]:
        """返回最新帧、源帧序号、采集时刻。"""
        with self._lock:
            if self._frame is None or self._seq == last_seq:
                return None, last_seq, 0.0
            return self._frame.copy(), self._seq, float(self._capture_perf)

    def get_clock(self) -> Tuple[int, float]:
        """供后台录像线程读取当前源时间轴；只返回轻量标量，不复制视频帧。"""
        with self._lock:
            return int(self._seq), float(self._capture_perf)

    @property
    def stopped(self) -> bool:
        return self._stopped.is_set()

    def set_file_paused(self, paused: bool) -> None:
        """Pause only local-file reading. RTSP/camera capture must keep draining newest frames."""
        if not self.is_file:
            return
        if paused:
            self._pause_file.set()
        else:
            self._pause_file.clear()

    @property
    def progress_percent(self) -> float:
        if not self.is_file or self.source_total_frames <= 0:
            return 0.0
        with self._lock:
            seq = self._seq
        return float(np.clip(seq * 100.0 / max(1, self.source_total_frames), 0.0, 100.0))

    def release(self) -> None:
        self._stopped.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=0.5)
        if self.cap is not None:
            self.cap.release()



# ----------------------- real-time result video writer ---------------------

class AsyncRealTimeResultVideoWriter:
    """后台真实时间录像器。

    设计目标：
    1. 主识别线程调用 ``submit`` 只更新“最新标注帧”，几乎立即返回；
    2. 后台线程按照视频源时间轴和输出 FPS 匀速写盘；
    3. 模型处理慢时自动重复最近一张标注帧，但绝不在主线程里批量补几十帧；
    4. 对 4K 输入可只把“保存录像”缩小到 1080p 左右，检测本身仍使用原始图像。
    """

    def __init__(
        self,
        path: str,
        source_frame_size: Tuple[int, int],
        output_fps: float,
        source_fps: float,
        is_file: bool,
        clock_provider,
        keep_realtime: bool = True,
        max_width: int = 0,
    ) -> None:
        self.path = str(path)
        self.source_frame_size = (int(source_frame_size[0]), int(source_frame_size[1]))
        self.output_fps = float(max(1.0, output_fps))
        self.source_fps = float(source_fps)
        self.is_file = bool(is_file)
        self.clock_provider = clock_provider
        self.keep_realtime = bool(keep_realtime)
        self.max_width = max(0, int(max_width))

        sw, sh = self.source_frame_size
        if self.max_width > 0 and sw > self.max_width:
            scale = self.max_width / float(sw)
            ow = int(round(sw * scale))
            oh = int(round(sh * scale))
            # 编码器更喜欢偶数尺寸
            ow += ow % 2
            oh += oh % 2
            self.frame_size = (max(2, ow), max(2, oh))
        else:
            self.frame_size = (sw, sh)

        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.writer = cv2.VideoWriter(
            self.path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            self.output_fps,
            self.frame_size,
        )
        if not self.writer.isOpened():
            raise RuntimeError(f"无法创建输出视频：{self.path}")

        self._lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._first_source_seq: Optional[int] = None
        self._first_capture_perf: Optional[float] = None
        self._stop = threading.Event()
        self._has_frame = threading.Event()
        self._thread = threading.Thread(target=self._run, name="async-result-video-writer", daemon=True)

        self.frames_written = 0
        self.dropped_submit_frames = 0
        self._last_submit_seq = -1
        self._thread.start()

    def _fit_size(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        tw, th = self.frame_size
        if w == tw and h == th:
            return frame
        return cv2.resize(frame, (tw, th), interpolation=cv2.INTER_AREA)

    def submit(self, frame: np.ndarray, source_seq: int, capture_perf: float) -> None:
        """非阻塞提交。旧的待保存标注帧会被最新结果覆盖，保证低延迟。"""
        fitted = self._fit_size(frame)
        # copy 在主线程中不可完全避免，但远比同步 VideoWriter.write/批量补帧便宜。
        fitted = fitted.copy()
        with self._lock:
            if self._last_submit_seq >= 0 and int(source_seq) > self._last_submit_seq + 1:
                self.dropped_submit_frames += int(source_seq) - self._last_submit_seq - 1
            self._last_submit_seq = int(source_seq)
            self._latest_frame = fitted
            if self._first_source_seq is None:
                self._first_source_seq = int(source_seq)
            if self._first_capture_perf is None and capture_perf > 0:
                self._first_capture_perf = float(capture_perf)
        self._has_frame.set()

    def _source_elapsed(self) -> float:
        with self._lock:
            first_seq = self._first_source_seq
            first_perf = self._first_capture_perf
        if first_seq is None:
            return 0.0

        try:
            current_seq, current_perf = self.clock_provider()
        except Exception:
            current_seq, current_perf = first_seq, first_perf or 0.0

        if not self.keep_realtime:
            # 非真实时间模式下，由后台线程自己的写入节拍推进。
            return self.frames_written / self.output_fps

        if self.is_file and self.source_fps > 1.0:
            return max(0.0, (int(current_seq) - int(first_seq)) / self.source_fps)
        if first_perf is not None and current_perf > 0:
            return max(0.0, float(current_perf) - float(first_perf))
        return self.frames_written / self.output_fps

    def _get_latest_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            return None if self._latest_frame is None else self._latest_frame.copy()

    def _run(self) -> None:
        # 等第一张标注结果；没有结果时不创建空白帧。
        while not self._stop.is_set() and not self._has_frame.wait(timeout=0.05):
            pass
        if self._stop.is_set():
            return

        # 后台线程最多按 output_fps 的节奏写一帧，禁止“瞬间追赶式批量写盘”。
        period = 1.0 / self.output_fps
        next_tick = time.perf_counter()

        while not self._stop.is_set():
            now = time.perf_counter()
            if now < next_tick:
                time.sleep(min(0.01, next_tick - now))
                continue
            next_tick += period
            if next_tick < now - 2.0 * period:
                # 系统曾经阻塞较久时，直接重置节拍，不在后台疯狂追赶。
                next_tick = now + period

            frame = self._get_latest_frame()
            if frame is None:
                continue

            if self.keep_realtime:
                elapsed = self._source_elapsed()
                target_total = 1 + int(elapsed * self.output_fps)
                if self.frames_written >= target_total:
                    continue

            self.writer.write(frame)
            self.frames_written += 1

    @property
    def duration_seconds(self) -> float:
        return self.frames_written / self.output_fps if self.output_fps > 0 else 0.0

    def release(self) -> None:
        self._stop.set()
        self._has_frame.set()
        if self._thread.is_alive():
            self._thread.join(timeout=3.0)
        self.writer.release()


# ----------------------- route cumulative statistics ----------------------

@dataclass
class SlotAccumulationCandidate:
    """Unconfirmed slot state used only for route-level cumulative counting."""

    last_state: Optional[bool] = None
    streak: int = 0
    total_observations: int = 0
    occupied_observations: int = 0
    free_observations: int = 0
    first_seen: float = 0.0
    last_seen: float = 0.0

    def update(self, occupied: bool, now: float, max_gap_seconds: float) -> None:
        occupied = bool(occupied)
        if self.first_seen <= 0.0:
            self.first_seen = now

        # A visibility break invalidates the previous "continuous" confirmation streak.
        if self.last_seen > 0.0 and now - self.last_seen > max_gap_seconds:
            self.last_state = None
            self.streak = 0

        if self.last_state is occupied:
            self.streak += 1
        else:
            self.last_state = occupied
            self.streak = 1

        self.total_observations += 1
        if occupied:
            self.occupied_observations += 1
        else:
            self.free_observations += 1
        self.last_seen = now


class PatrolAccumulator:
    """Session-scoped route-level cumulative metrics.

    Important semantics:
    - ``visible_*`` = current frame only and may rise/fall normally.
    - ``cumulative_*`` = parking spaces confirmed during this patrol session.
    - A confirmed slot is frozen as occupied/free for the remainder of the session.
      Therefore cumulative occupied/free/total never decrease until RESET.
    """

    def __init__(
        self,
        occupied_confirm_frames: int = 3,
        empty_confirm_frames: int = 6,
        max_gap_seconds: float = 0.8,
    ) -> None:
        self.occupied_confirm_frames = max(1, int(occupied_confirm_frames))
        self.empty_confirm_frames = max(1, int(empty_confirm_frames))
        self.max_gap_seconds = max(0.1, float(max_gap_seconds))
        self.reset()

    def reset(self) -> None:
        # Only CONFIRMED slots live here. Once inserted, the bool never changes in this session.
        # 6.6.0: key is the REAL parking-space ID when permanent mapping is enabled (e.g. A037).
        # Legacy mode uses P1/P2/... strings. This prevents the same real slot from being counted twice
        # when it receives a different temporary visual ID later in the route.
        self.slot_states: Dict[str, bool] = {}  # True=occupied, False=free
        self.slot_confirmed_at: Dict[str, float] = {}
        self.slot_last_seen: Dict[str, float] = {}
        self.slot_candidates: Dict[str, SlotAccumulationCandidate] = {}
        self.slot_conflict_observations = 0

        self.all_plates: Dict[str, str] = {}
        self.normal_plates: Dict[str, str] = {}
        self.violation_plates: Dict[str, str] = {}
        self.outside_violation_plates: set[str] = set()
        self.multi_space_violation_plates: set[str] = set()
        self.partial_space_violation_plates: set[str] = set()
        self.last_event_message = "暂无事件"
        self.last_stats: dict = {}

    def _update_slot_ledger(self, stats: dict, now: float) -> None:
        # Tri-state cumulative evidence:
        #   True  = stationary vehicle is stably occupying the slot -> occupied evidence
        #   False = no vehicle overlaps the slot -> free evidence
        #   None  = a MOVING vehicle overlaps the slot -> transient/uncertain, do not accumulate
        # Fall back to the legacy binary map only for backward compatibility.
        current_states = stats.get("_slot_cumulative_state", None)
        if current_states is None:
            current_states = stats.get("_slot_status_current", {}) or {}
        eligible_map = stats.get("_slot_cumulative_eligible", {}) or {}

        current_ids = set()
        for sid, observed_state in current_states.items():
            sid_key = str(sid)
            current_ids.add(sid_key)
            self.slot_last_seen[sid_key] = now

            # Partial parking spaces near frame borders are deliberately excluded from route accumulation.
            if not bool(eligible_map.get(sid_key, eligible_map.get(sid, True))):
                self.slot_candidates.pop(sid_key, None)
                continue

            # A moving vehicle passing across the slot is NOT evidence of occupied or free.
            # It also breaks any previous continuous confirmation streak so separated observations
            # cannot be concatenated across a passing vehicle.
            if observed_state is None:
                self.slot_candidates.pop(sid_key, None)
                continue

            occupied = bool(observed_state)

            # Frozen ledger: later detector flicker can never change the cumulative classification.
            if sid_key in self.slot_states:
                if self.slot_states[sid_key] != occupied:
                    self.slot_conflict_observations += 1
                continue

            cand = self.slot_candidates.get(sid_key)
            if cand is None:
                cand = SlotAccumulationCandidate()
                self.slot_candidates[sid_key] = cand
            cand.update(occupied, now, self.max_gap_seconds)

            need = self.occupied_confirm_frames if occupied else self.empty_confirm_frames
            if cand.streak >= need:
                self.slot_states[sid_key] = occupied
                self.slot_confirmed_at[sid_key] = now
                self.slot_candidates.pop(sid_key, None)

        # A candidate that is no longer visible must not retain a stale continuous streak.
        stale_candidates = []
        for sid, cand in self.slot_candidates.items():
            if sid not in current_ids and cand.last_seen > 0 and now - cand.last_seen > self.max_gap_seconds:
                stale_candidates.append(sid)
        for sid in stale_candidates:
            self.slot_candidates.pop(sid, None)

    def update(self, stats: dict, events: Sequence[dict], now: float) -> None:
        self.last_stats = stats
        self._update_slot_ledger(stats, now)

        for item in stats.get("_stable_plates_current", []):
            plate = str(item.get("plate", "")).strip()
            color = str(item.get("color", "")).strip()
            if plate:
                self.all_plates[plate] = color

        for event in events:
            plate = str(event.get("plate", "")).strip()
            if not plate:
                continue
            color = str(event.get("plate_color", "")).strip()
            self.all_plates[plate] = color
            status = str(event.get("status", ""))
            if status == STATUS_NORMAL:
                self.normal_plates[plate] = color
                self.last_event_message = f"正常停泊：{plate}" + (f"（{color}）" if color else "")
            elif status == STATUS_VIOLATION:
                self.violation_plates[plate] = color
                vtype = str(event.get("violation_type", "OUTSIDE") or "OUTSIDE")
                if vtype == "MULTI_SPACE":
                    self.multi_space_violation_plates.add(plate)
                    label = "一车占多位"
                elif vtype == "PARTIAL_SPACE":
                    self.partial_space_violation_plates.add(plate)
                    label = "压线/偏停"
                else:
                    self.outside_violation_plates.add(plate)
                    label = "车位外停车"
                self.last_event_message = f"{label}：{plate}" + (f"（{color}）" if color else "")

    @staticmethod
    def _plate_text(items: Dict[str, str], max_items: int = 60) -> str:
        if not items:
            return "暂无"
        rows = []
        for plate in sorted(items.keys())[-max_items:]:
            color = items.get(plate, "")
            rows.append(f"{plate}  {color}" if color else plate)
        return "\n".join(rows)

    def _slot_summary(self, max_items: int = 80) -> str:
        if not self.slot_states:
            return "暂无已确认车位"
        ids = sorted(self.slot_states.keys())
        if len(ids) > max_items:
            ids = ids[-max_items:]
        return "\n".join(
            f"{sid}: {'占用' if self.slot_states[sid] else '空闲'}"
            for sid in ids
        )

    def snapshot(self) -> dict:
        # Route-level confirmed counts: all are monotonic until RESET.
        total = len(self.slot_states)
        occupied = sum(1 for v in self.slot_states.values() if v)
        empty = sum(1 for v in self.slot_states.values() if not v)
        rate = 100.0 * occupied / total if total else 0.0
        stats = self.last_stats

        visible_total = int(stats.get("parking_slots_current", 0))
        visible_occupied = int(stats.get("parking_slots_occupied_current", 0))
        visible_empty = int(stats.get("parking_slots_free_current", 0))
        visible_rate = 100.0 * visible_occupied / visible_total if visible_total else 0.0

        return {
            "cumulative_total_spaces": int(total),
            "cumulative_occupied_spaces": int(occupied),
            "cumulative_empty_spaces": int(empty),
            "cumulative_occupancy_rate": round(rate, 1),
            "cumulative_pending_spaces": int(len(self.slot_candidates)),
            "cumulative_slot_conflict_observations": int(self.slot_conflict_observations),

            # Explicit current-frame metrics. These are intentionally NOT cumulative.
            "visible_total_spaces": visible_total,
            "visible_occupied_spaces": visible_occupied,
            "visible_empty_spaces": visible_empty,
            "visible_occupancy_rate": round(visible_rate, 1),
            # 6.5.4 diagnostics: distinguish stationary parked occupancy from moving pass-through occupancy.
            "visible_stationary_occupied_spaces": int(stats.get("parking_slots_stationary_occupied_current", 0)),
            "visible_transient_occupied_spaces": int(stats.get("parking_slots_transient_current", 0)),
            "parking_slot_local_ids_seen": int(stats.get("parking_slot_local_ids_seen", 0)),
            "cumulative_transient_blocked_spaces": int(stats.get("parking_slots_transient_current", 0)),

            "cumulative_recognized_plate_count": len(self.all_plates),
            "cumulative_violation_count": len(self.violation_plates),
            "normal_parked_plate_count": len(self.normal_plates),
            "normal_parked_plates_text": self._plate_text(self.normal_plates),
            "violation_plate_count": len(self.violation_plates),
            "violation_plates_text": self._plate_text(self.violation_plates),
            "cumulative_outside_violation_count": len(self.outside_violation_plates),
            "cumulative_multi_space_violation_count": len(self.multi_space_violation_plates),
            "cumulative_partial_space_violation_count": len(self.partial_space_violation_plates),
            "active_violation_count": int(stats.get("violation_current", 0)),
            "event_message": self.last_event_message,
            "cumulative_slot_summary": self._slot_summary(),
            "vehicle_count": int(stats.get("vehicles_current", 0)),
        }


# ------------------------------- rendering --------------------------------

@lru_cache(maxsize=16)
def _cached_font(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size, encoding="utf-8")


def draw_text_batch(
    frame: np.ndarray,
    items: Sequence[Tuple[str, int, int, Tuple[int, int, int], int]],
    font_path: str,
) -> np.ndarray:
    if not items:
        return frame
    if not Path(font_path).exists():
        # Fallback: ASCII-safe display only.
        for text, x, y, color_bgr, size in items:
            safe = text.encode("ascii", "ignore").decode("ascii")
            if safe:
                cv2.putText(frame, safe, (x, y + size), cv2.FONT_HERSHEY_SIMPLEX, size / 28.0, color_bgr, 1, cv2.LINE_AA)
        return frame

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil)
    for text, x, y, color_bgr, size in items:
        font = _cached_font(font_path, int(size))
        color_rgb = (int(color_bgr[2]), int(color_bgr[1]), int(color_bgr[0]))
        # v6.8.1：给文字增加黑色描边，航拍复杂背景下真实车位号更清楚。
        stroke_w = 2 if int(size) >= 20 else 1
        draw.text(
            (int(x), int(y)), text, fill=color_rgb, font=font,
            stroke_width=stroke_w, stroke_fill=(0, 0, 0),
        )
    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


def render_cumulative_overlay(frame: np.ndarray, snapshot: dict, font_path: str) -> np.ndarray:
    """Small top-right panel to visually verify route-level cumulative counting."""
    if frame is None or frame.size == 0:
        return frame
    out = frame.copy()
    h, w = out.shape[:2]
    panel_w = min(430, max(330, int(w * 0.27)))
    x1 = max(8, w - panel_w - 8)
    y1, y2 = 8, 124
    overlay_img = out.copy()
    cv2.rectangle(overlay_img, (x1, y1), (w - 8, y2), (10, 10, 10), -1)
    cv2.addWeighted(overlay_img, 0.68, out, 0.32, 0, out)

    total = int(snapshot.get("cumulative_total_spaces", 0))
    occ = int(snapshot.get("cumulative_occupied_spaces", 0))
    free = int(snapshot.get("cumulative_empty_spaces", 0))
    pending = int(snapshot.get("cumulative_pending_spaces", 0))
    rate = float(snapshot.get("cumulative_occupancy_rate", 0.0))
    items = [
        ("航线累计（已确认，状态冻结）", x1 + 12, 14, (245, 245, 245), 17),
        (f"总车位 {total}   占用 {occ}   空闲 {free}", x1 + 12, 43, (80, 220, 255), 18),
        (f"累计占用率 {rate:.1f}%   待确认 {pending}", x1 + 12, 75, (210, 210, 210), 17),
    ]
    return draw_text_batch(out, items, font_path)


# -------------------- asynchronous auxiliary inference ---------------------


def _affine_to_h(M: Optional[np.ndarray]) -> np.ndarray:
    H = np.eye(3, dtype=np.float32)
    if M is not None:
        H[:2, :] = np.asarray(M, dtype=np.float32).reshape(2, 3)
    return H


def _compose_affines(mats: Sequence[np.ndarray]) -> Optional[np.ndarray]:
    """Compose affine transforms in chronological order: p_n = M_n ... M_2 M_1 p_0."""
    if not mats:
        return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    H = np.eye(3, dtype=np.float32)
    for M in mats:
        if M is None:
            return None
        H = _affine_to_h(M) @ H
    return H[:2, :].astype(np.float32)


@dataclass
class ParkingAsyncTask:
    frame: np.ndarray
    frame_idx: int
    submit_perf: float


@dataclass
class ParkingAsyncResult:
    polygons: List[Tuple[np.ndarray, float]]
    frame_idx: int
    submit_perf: float
    done_perf: float
    infer_ms: float
    error: str = ""


class AsyncParkingWorker:
    """Latest-only parking segmentation worker.

    A pending task is replaced by a newer task instead of building backlog. The model is loaded
    inside the worker thread so its TensorRT execution context stays with the inference thread.
    """

    def __init__(
        self,
        model_path: str,
        imgsz: int,
        conf: float,
        iou: float,
        device,
        max_width: int = 0,
        aux_gpu_lock: Optional[threading.Lock] = None,
    ) -> None:
        self.model_path = model_path
        self.imgsz = int(imgsz)
        self.conf = float(conf)
        self.iou = float(iou)
        self.device = device
        self.max_width = max(0, int(max_width))
        self.aux_gpu_lock = aux_gpu_lock

        self._cv = threading.Condition()
        self._pending: Optional[ParkingAsyncTask] = None
        self._latest_result: Optional[ParkingAsyncResult] = None
        self._stop = False
        self._thread: Optional[threading.Thread] = None
        self._ready = threading.Event()
        self.init_error = ""
        self.busy = False
        self.dropped_tasks = 0
        self.completed_tasks = 0
        self.last_infer_ms = 0.0

    def start(self, timeout: float = 60.0) -> "AsyncParkingWorker":
        self._thread = threading.Thread(target=self._run, name="parking-seg-worker", daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout=timeout):
            raise RuntimeError("停车位异步 worker 启动超时。")
        if self.init_error:
            raise RuntimeError(f"停车位异步 worker 初始化失败：{self.init_error}")
        return self

    def submit(self, frame: np.ndarray, frame_idx: int) -> None:
        task = ParkingAsyncTask(frame.copy(), int(frame_idx), time.perf_counter())
        with self._cv:
            if self._pending is not None:
                self.dropped_tasks += 1
            self._pending = task
            self._cv.notify()

    def pop_latest_result(self) -> Optional[ParkingAsyncResult]:
        with self._cv:
            r = self._latest_result
            self._latest_result = None
            return r

    def _prepare(self, frame: np.ndarray) -> Tuple[np.ndarray, float]:
        h, w = frame.shape[:2]
        if self.max_width > 0 and w > self.max_width:
            scale = self.max_width / float(w)
            nh = max(1, int(round(h * scale)))
            resized = cv2.resize(frame, (self.max_width, nh), interpolation=cv2.INTER_AREA)
            return resized, scale
        return frame, 1.0

    def _infer(self, model: YOLO, task: ParkingAsyncTask) -> ParkingAsyncResult:
        t0 = time.perf_counter()
        work_frame, scale = self._prepare(task.frame)
        lock = self.aux_gpu_lock
        if lock is None:
            results = model.predict(
                work_frame, imgsz=self.imgsz, conf=self.conf, iou=self.iou,
                device=self.device, verbose=False,
            )
        else:
            with lock:
                results = model.predict(
                    work_frame, imgsz=self.imgsz, conf=self.conf, iou=self.iou,
                    device=self.device, verbose=False,
                )

        out: List[Tuple[np.ndarray, float]] = []
        if results:
            r = results[0]
            confs: List[float] = []
            if r.boxes is not None and len(r.boxes):
                confs = r.boxes.conf.detach().cpu().numpy().tolist()
            if r.masks is not None and getattr(r.masks, "xy", None) is not None:
                for i, p in enumerate(r.masks.xy):
                    poly = np.asarray(p, dtype=np.float32).reshape(-1, 2)
                    if scale > 1e-9 and scale != 1.0:
                        poly = poly / float(scale)
                    if len(poly) >= 3 and polygon_area(poly) >= 80.0:
                        conf = float(confs[i]) if i < len(confs) else 1.0
                        out.append((poly.astype(np.float32), conf))
            elif r.boxes is not None:
                boxes = r.boxes.xyxy.detach().cpu().numpy()
                for i, b in enumerate(boxes):
                    bb = np.asarray(b[:4], dtype=np.float32)
                    if scale > 1e-9 and scale != 1.0:
                        bb = bb / float(scale)
                    conf = float(confs[i]) if i < len(confs) else 1.0
                    out.append((box_to_polygon(tuple(map(float, bb))), conf))
        done = time.perf_counter()
        return ParkingAsyncResult(out, task.frame_idx, task.submit_perf, done, (done - t0) * 1000.0)

    def _run(self) -> None:
        try:
            # v6.8.3: UAVImageOnlyOcclusion is defined at module top-level so torch can
            # deserialize tcw0914.pt checkpoints produced by the custom UAV augmentation trainer.
            model = YOLO(self.model_path, task="segment")
            dummy = np.zeros((720, 1280, 3), dtype=np.uint8)
            try:
                lock = self.aux_gpu_lock
                if lock is None:
                    model.predict(dummy, imgsz=self.imgsz, conf=self.conf, device=self.device, verbose=False)
                else:
                    with lock:
                        model.predict(dummy, imgsz=self.imgsz, conf=self.conf, device=self.device, verbose=False)
            except Exception as e:
                print(f"[ParkingWorker] 预热警告：{e}")
        except Exception as e:
            self.init_error = str(e)
            self._ready.set()
            return
        self._ready.set()

        while True:
            with self._cv:
                while self._pending is None and not self._stop:
                    self._cv.wait(timeout=0.2)
                if self._stop:
                    break
                task = self._pending
                self._pending = None
                self.busy = True
            assert task is not None
            try:
                result = self._infer(model, task)
            except Exception as e:
                done = time.perf_counter()
                result = ParkingAsyncResult([], task.frame_idx, task.submit_perf, done, 0.0, str(e))
            with self._cv:
                self.busy = False
                self.completed_tasks += 1
                self.last_infer_ms = float(result.infer_ms)
                self._latest_result = result

    def close(self) -> None:
        with self._cv:
            self._stop = True
            self._cv.notify_all()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)


@dataclass
class PlateAsyncTask:
    crop: np.ndarray
    crop_origin: Tuple[int, int]
    vehicle_box_snapshot: Box
    identity_key: int
    track_id_snapshot: int
    submit_perf: float
    submit_wall: float


@dataclass
class PlateAsyncResult:
    task: PlateAsyncTask
    observation: PlateObservation
    done_perf: float
    infer_ms: float
    error: str = ""


class AsyncPlateWorker:
    """Small bounded plate ROI worker; old queued ROIs are discarded before they can create latency."""

    def __init__(
        self,
        det_model_path: str,
        rec_model_path: str,
        imgsz: int,
        conf: float,
        iou: float,
        device_yolo,
        torch_device: torch.device,
        queue_size: int = 3,
        aux_gpu_lock: Optional[threading.Lock] = None,
    ) -> None:
        self.det_model_path = det_model_path
        self.rec_model_path = rec_model_path
        self.imgsz = int(imgsz)
        self.conf = float(conf)
        self.iou = float(iou)
        self.device_yolo = device_yolo
        self.torch_device = torch_device
        self.queue_size = max(1, int(queue_size))
        self.aux_gpu_lock = aux_gpu_lock

        self._cv = threading.Condition()
        self._pending: Deque[PlateAsyncTask] = deque()
        self._results: Deque[PlateAsyncResult] = deque(maxlen=12)
        self._stop = False
        self._thread: Optional[threading.Thread] = None
        self._ready = threading.Event()
        self.init_error = ""
        self.busy = False
        self.dropped_tasks = 0
        self.completed_tasks = 0
        self.last_infer_ms = 0.0

    def start(self, timeout: float = 60.0) -> "AsyncPlateWorker":
        self._thread = threading.Thread(target=self._run, name="plate-ocr-worker", daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout=timeout):
            raise RuntimeError("车牌异步 worker 启动超时。")
        if self.init_error:
            raise RuntimeError(f"车牌异步 worker 初始化失败：{self.init_error}")
        return self

    def submit(self, task: PlateAsyncTask) -> None:
        with self._cv:
            # Do not queue duplicate tasks for the same live identity.
            self._pending = deque((x for x in self._pending if x.identity_key != task.identity_key), maxlen=None)
            while len(self._pending) >= self.queue_size:
                self._pending.popleft()
                self.dropped_tasks += 1
            self._pending.append(task)
            self._cv.notify()

    def pop_results(self) -> List[PlateAsyncResult]:
        with self._cv:
            out = list(self._results)
            self._results.clear()
            return out

    @property
    def pending_count(self) -> int:
        with self._cv:
            return len(self._pending)

    def _infer(self, det_model: YOLO, rec_model, task: PlateAsyncTask) -> PlateAsyncResult:
        t0 = time.perf_counter()
        crop = task.crop
        obs = PlateObservation()
        lock = self.aux_gpu_lock
        if lock is None:
            results = det_model.predict(
                crop, imgsz=self.imgsz, conf=self.conf, iou=self.iou,
                device=self.device_yolo, verbose=False,
            )
        else:
            with lock:
                results = det_model.predict(
                    crop, imgsz=self.imgsz, conf=self.conf, iou=self.iou,
                    device=self.device_yolo, verbose=False,
                )

        if results:
            r = results[0]
            if r.boxes is not None and len(r.boxes) > 0 and r.keypoints is not None:
                boxes = r.boxes.xyxy.detach().cpu().numpy()
                confs = r.boxes.conf.detach().cpu().numpy()
                clss = r.boxes.cls.detach().cpu().numpy().astype(int)
                kpts = r.keypoints.xy.detach().cpu().numpy()
                n = min(len(boxes), len(kpts))
                order = np.argsort(-confs[:n]) if n else []
                for idx in order:
                    landmarks = np.asarray(kpts[idx], dtype=np.float32).reshape(-1, 2)
                    if len(landmarks) < 4:
                        continue
                    roi = four_point_transform(crop, landmarks[:4])
                    if roi.size == 0:
                        continue
                    if int(clss[idx]) == 1:
                        roi = get_split_merge(roi)
                    # OCR runs in the same worker. Keep auxiliary GPU jobs serialized if configured.
                    if lock is None:
                        with torch.inference_mode():
                            plate_no, char_probs, plate_color, color_conf = get_plate_result(
                                roi, self.torch_device, rec_model, is_color=True
                            )
                    else:
                        with lock:
                            with torch.inference_mode():
                                plate_no, char_probs, plate_color, color_conf = get_plate_result(
                                    roi, self.torch_device, rec_model, is_color=True
                                )
                    plate_no = str(plate_no).strip()
                    if not valid_plate_text(plate_no):
                        continue
                    probs = np.asarray(char_probs, dtype=np.float32).reshape(-1)
                    char_score = float(probs.mean()) if probs.size else 0.0
                    det_conf = float(confs[idx])
                    score = det_conf * max(0.05, char_score) * max(0.05, float(color_conf))
                    b = boxes[idx]
                    ox, oy = task.crop_origin
                    gbox: Box = (ox + float(b[0]), oy + float(b[1]), ox + float(b[2]), oy + float(b[3]))
                    obs = PlateObservation(plate_no, str(plate_color), score, det_conf, gbox)
                    break
        done = time.perf_counter()
        return PlateAsyncResult(task, obs, done, (done - t0) * 1000.0)

    def _run(self) -> None:
        try:
            det_model = YOLO(self.det_model_path, task="pose")
            rec_model = init_model(self.torch_device, self.rec_model_path, is_color=True)
            dummy = np.zeros((256, 512, 3), dtype=np.uint8)
            try:
                lock = self.aux_gpu_lock
                if lock is None:
                    det_model.predict(dummy, imgsz=self.imgsz, conf=self.conf, device=self.device_yolo, verbose=False)
                else:
                    with lock:
                        det_model.predict(dummy, imgsz=self.imgsz, conf=self.conf, device=self.device_yolo, verbose=False)
            except Exception as e:
                print(f"[PlateWorker] 预热警告：{e}")
        except Exception as e:
            self.init_error = str(e)
            self._ready.set()
            return
        self._ready.set()

        while True:
            with self._cv:
                while not self._pending and not self._stop:
                    self._cv.wait(timeout=0.2)
                if self._stop:
                    break
                task = self._pending.popleft()
                self.busy = True
            try:
                result = self._infer(det_model, rec_model, task)
            except Exception as e:
                done = time.perf_counter()
                result = PlateAsyncResult(task, PlateObservation(), done, 0.0, str(e))
            with self._cv:
                self.busy = False
                self.completed_tasks += 1
                self.last_infer_ms = float(result.infer_ms)
                self._results.append(result)

    def close(self) -> None:
        with self._cv:
            self._stop = True
            self._cv.notify_all()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)


# ------------------------------ main engine -------------------------------

@dataclass
class RealtimeConfig:
    vehicle_model: str
    parking_model: str
    plate_det_model: str
    plate_rec_model: str
    device: str = "0"

    vehicle_imgsz: int = 640
    parking_imgsz: int = 960
    plate_imgsz: int = 640
    vehicle_conf: float = 0.28
    parking_conf: float = 0.30
    plate_conf: float = 0.35
    iou: float = 0.55
    vehicle_classes: Tuple[int, ...] = (2, 3, 5, 7)

    parking_every_n: int = 3  # sync fallback only
    max_plate_rois_per_frame: int = 2
    plate_retry_seconds: float = 0.28
    stable_plate_refresh_seconds: float = 3.0
    plate_min_hits: int = 2
    min_vehicle_box_for_plate: int = 55

    stationary_residual_ratio: float = 0.075
    stationary_seconds: float = 0.90
    stationary_fast_seconds: float = 0.65
    stationary_strong_residual_factor: float = 0.55
    status_confirm_frames: int = 3

    normal_min_vehicle_overlap: float = 0.55
    violation_max_vehicle_overlap: float = 0.32
    occupied_min_slot_overlap: float = 0.15

    state_lost_seconds: float = 2.0
    slot_max_missed_detects: int = 5
    slot_reid_retention_frames: int = 45

    async_parking: bool = True
    async_plate: bool = True
    parking_async_interval_s: float = 0.22
    parking_result_max_age_s: float = 1.20
    parking_accept_first_fresh_unaligned: bool = True
    parking_fresh_unaligned_fallback: bool = True
    parking_fresh_unaligned_max_age_s: float = 0.45
    parking_fresh_unaligned_max_frame_gap: int = 4
    parking_early_first_frame_submit: bool = True
    parking_worker_max_width: int = 1600
    parking_first_frame_bootstrap: bool = False
    parking_first_frame_timeout_s: float = 0.12
    auto_warmup_on_init: bool = True
    vehicle_warmup_passes: int = 2
    warmup_tracker_on_init: bool = True
    warmup_dummy_height: int = 720
    warmup_dummy_width: int = 1280
    plate_async_queue_size: int = 3
    plate_result_max_age_s: float = 1.50
    serialize_aux_gpu: bool = True
    vehicle_half: bool = True
    motion_process_width: int = 720
    motion_max_corners: int = 350
    adaptive_aux_load_shedding: bool = True
    aux_low_fps_threshold: float = 4.0
    aux_low_fps_parking_interval: float = 0.80

    roadside_slot_map_enabled: bool = False
    roadside_slot_map_path: str = ""
    roadside_slot_min_hits: int = 2
    roadside_screen_order: str = "auto"
    roadside_start_real_id: str = ""
    roadside_direction_min_samples: int = 3
    roadside_min_motion_pixels: float = 1.5
    roadside_gap_skip_ratio: float = 1.75
    roadside_max_inferred_skip: int = 2
    roadside_assign_edge_margin_ratio: float = 0.015
    roadside_show_local_id: bool = True
    roadside_real_id_font_size: int = 24
    roadside_local_id_font_size: int = 13
    roadside_pending_id_font_size: int = 14

    font_path: str = "fonts/platech.ttf"
    event_log: Optional[str] = "uav_parking_events.jsonl"


class UAVParkingRealtime:
    """Moving-UAV processor with a high-priority vehicle main chain and non-blocking auxiliary workers."""

    def __init__(self, cfg: RealtimeConfig) -> None:
        self.cfg = cfg
        self.frame_idx = 0
        self.prev_vehicle_boxes: List[Box] = []
        self.motion = BackgroundMotionEstimator(
            process_width=cfg.motion_process_width,
            max_corners=cfg.motion_max_corners,
        )
        self.slots = ParkingSlotTracker(
            max_missed_detects=cfg.slot_max_missed_detects,
            reid_retention_frames=cfg.slot_reid_retention_frames,
        )
        # Keep the attribute name ``permanent_slots`` for compatibility with the existing cumulative/output path.
        # Its implementation is now the no-GPS roadside fixed-route sequence mapper.
        self.permanent_slots = RoadsideRouteSlotMapper(
            enabled=cfg.roadside_slot_map_enabled,
            map_path=cfg.roadside_slot_map_path,
            min_hits=cfg.roadside_slot_min_hits,
            screen_order=cfg.roadside_screen_order,
            start_real_id=cfg.roadside_start_real_id,
            direction_min_samples=cfg.roadside_direction_min_samples,
            min_motion_pixels=cfg.roadside_min_motion_pixels,
            gap_skip_ratio=cfg.roadside_gap_skip_ratio,
            max_inferred_skip=cfg.roadside_max_inferred_skip,
            edge_margin_ratio=cfg.roadside_assign_edge_margin_ratio,
        )
        self.vehicles = VehicleStateManager(max_lost_seconds=cfg.state_lost_seconds)
        self.events = EventLogger(cfg.event_log)
        self.last_events: List[dict] = []

        self.device_yolo = self._normalize_yolo_device(cfg.device)
        self.torch_device = self._normalize_torch_device(cfg.device)
        self._vehicle_is_engine = str(cfg.vehicle_model).lower().endswith('.engine')

        print("加载车辆检测/跟踪主模型 ...")
        self.vehicle_model = YOLO(cfg.vehicle_model, task="detect")
        self.vehicle_names = getattr(self.vehicle_model, "names", {}) or {}

        self._aux_gpu_lock: Optional[threading.Lock] = threading.Lock() if cfg.serialize_aux_gpu else None
        self.parking_worker: Optional[AsyncParkingWorker] = None
        self.plate_worker: Optional[AsyncPlateWorker] = None
        self.parking_model = None
        self.plate_det_model = None
        self.plate_rec_model = None

        # Load auxiliary models in their own worker threads. Startup is sequential to avoid CUDA init spikes.
        if cfg.async_parking:
            print("启动停车位异步 TensorRT worker ...")
            self.parking_worker = AsyncParkingWorker(
                cfg.parking_model, cfg.parking_imgsz, cfg.parking_conf, cfg.iou,
                self.device_yolo, cfg.parking_worker_max_width, self._aux_gpu_lock,
            ).start()
        else:
            print("加载停车位分割模型（同步回退模式） ...")
            self.parking_model = YOLO(cfg.parking_model, task="segment")

        if cfg.async_plate:
            print("启动车牌检测/OCR异步 worker ...")
            self.plate_worker = AsyncPlateWorker(
                cfg.plate_det_model, cfg.plate_rec_model, cfg.plate_imgsz, cfg.plate_conf, cfg.iou,
                self.device_yolo, self.torch_device, cfg.plate_async_queue_size, self._aux_gpu_lock,
            ).start()
        else:
            print("加载车牌模型（同步回退模式） ...")
            self.plate_det_model = YOLO(cfg.plate_det_model, task="pose")
            self.plate_rec_model = init_model(self.torch_device, cfg.plate_rec_model, is_color=True)

        self._warmup_done = False
        self._last_parking_submit_perf = -1e9
        self._last_parking_result_age_ms = 0.0
        self._parking_result_dropped_stale = 0
        self._parking_result_dropped_motion = 0
        self._parking_results_applied = 0
        self._parking_results_applied_unaligned = 0
        self._parking_first_nonempty_result_seen = False
        self._first_real_frame_perf = 0.0
        self._first_parking_visible_ms = 0.0
        self._plate_result_dropped_stale = 0
        self._plate_result_dropped_identity = 0
        self._plate_results_applied = 0
        self._motion_history: Dict[int, Optional[np.ndarray]] = {}
        self._main_fps_ema = 0.0
        self._last_stage_ms: Dict[str, float] = {}
        self._parking_bootstrap_done = False
        self._parking_bootstrap_success = False
        self._parking_bootstrap_ms = 0.0
        self._startup_warmup_ms = 0.0
        self._startup_tracker_warmup_ms = 0.0
        self._startup_tracker_warmup_ok = False

        # 6.5.6: charge one-time TensorRT / Ultralytics / BoT-SORT initialization to service startup,
        # not to the first real UAV frame. Auxiliary workers are already ready and prewarmed here.
        if self.cfg.auto_warmup_on_init:
            self.warmup((self.cfg.warmup_dummy_height, self.cfg.warmup_dummy_width), warmup_tracker=self.cfg.warmup_tracker_on_init)

    @staticmethod
    def _normalize_yolo_device(device: str):
        d = str(device).strip()
        if d.lower() == "cpu":
            return "cpu"
        if "," in d:
            d = d.split(",")[0].strip()
        try:
            return int(d)
        except ValueError:
            return d

    @staticmethod
    def _normalize_torch_device(device: str) -> torch.device:
        d = str(device).strip().lower()
        if d == "cpu":
            return torch.device("cpu")
        idx = d.split(",")[0].strip()
        if not torch.cuda.is_available():
            raise RuntimeError("当前未检测到 CUDA，但 .engine TensorRT 模型需要 NVIDIA CUDA/TensorRT 环境。")
        try:
            return torch.device(f"cuda:{int(idx)}")
        except ValueError:
            return torch.device("cuda:0")

    def _reset_vehicle_tracker_only(self) -> None:
        """Reset BoT-SORT state without touching recognition/cumulative state."""
        try:
            predictor = getattr(self.vehicle_model, "predictor", None)
            trackers = getattr(predictor, "trackers", None)
            if trackers:
                for tracker in trackers:
                    reset_fn = getattr(tracker, "reset", None)
                    if callable(reset_fn):
                        reset_fn()
        except Exception:
            pass

    def warmup(
        self,
        shape: Tuple[int, int] = (720, 1280),
        warmup_tracker: Optional[bool] = None,
        force: bool = False,
    ) -> None:
        """Prewarm vehicle TensorRT/predictor and optionally BoT-SORT before live frames arrive.

        TensorRT engine deserialization/context creation and Ultralytics predictor/tracker initialization
        are one-time costs. Deployment should pay them before reporting service READY.
        """
        if self._warmup_done and not force:
            return
        h, w = max(64, int(shape[0])), max(64, int(shape[1]))
        dummy = np.zeros((h, w, 3), dtype=np.uint8)
        tracker_requested = self.cfg.warmup_tracker_on_init if warmup_tracker is None else bool(warmup_tracker)
        t_all = time.perf_counter()
        try:
            kwargs = dict(
                imgsz=self.cfg.vehicle_imgsz,
                conf=self.cfg.vehicle_conf,
                classes=list(self.cfg.vehicle_classes),
                device=self.device_yolo,
                verbose=False,
            )
            if self.cfg.vehicle_half and not self._vehicle_is_engine and self.device_yolo != "cpu":
                kwargs["half"] = True

            # Multiple passes stabilize lazy CUDA/TensorRT allocations.
            for _ in range(max(1, int(self.cfg.vehicle_warmup_passes))):
                self.vehicle_model.predict(dummy, **kwargs)

            if torch.cuda.is_available() and self.device_yolo != "cpu":
                try:
                    torch.cuda.synchronize(self.torch_device)
                except Exception:
                    pass

            if tracker_requested:
                t_tracker = time.perf_counter()
                track_kwargs = dict(kwargs)
                track_kwargs.update(persist=True, tracker="ultralytics/cfg/trackers/botsort.yaml")
                self.vehicle_model.track(dummy, **track_kwargs)
                if torch.cuda.is_available() and self.device_yolo != "cpu":
                    try:
                        torch.cuda.synchronize(self.torch_device)
                    except Exception:
                        pass
                self._startup_tracker_warmup_ms = (time.perf_counter() - t_tracker) * 1000.0
                self._startup_tracker_warmup_ok = True
                # Dummy frame must never leak into the real patrol tracker state.
                self._reset_vehicle_tracker_only()

            self._warmup_done = True
            self._startup_warmup_ms = (time.perf_counter() - t_all) * 1000.0
            print(
                f"[READY] 车辆主模型预热完成 | total={self._startup_warmup_ms:.1f} ms | "
                f"BoT-SORT={'OK' if self._startup_tracker_warmup_ok else 'SKIP'} "
                f"({self._startup_tracker_warmup_ms:.1f} ms)"
            )
        except Exception as e:
            self._startup_warmup_ms = (time.perf_counter() - t_all) * 1000.0
            print(f"[警告] 车辆模型/跟踪器预热失败，将在真实帧继续运行: {e}")

    def _bootstrap_parking_first_frame(self, frame: np.ndarray) -> None:
        """One-shot parking bootstrap used only after startup/reset.

        Compatibility-only blocking bootstrap. v6.5.6 deployment default disables this path.
        With parking_first_frame_bootstrap=False, process_frame proceeds directly to vehicle tracking and
        schedules parking segmentation asynchronously after the high-priority vehicle inference.
        """
        if self._parking_bootstrap_done:
            return
        self._parking_bootstrap_done = True
        if not self.cfg.parking_first_frame_bootstrap:
            return

        t0 = time.perf_counter()
        target_frame_idx = max(1, self.frame_idx + 1)
        try:
            if self.parking_worker is not None:
                self.parking_worker.submit(frame, target_frame_idx)
                deadline = t0 + max(0.05, float(self.cfg.parking_first_frame_timeout_s))
                while time.perf_counter() < deadline:
                    result = self.parking_worker.pop_latest_result()
                    if result is not None:
                        if not result.error:
                            self.slots.update_detections(result.polygons, target_frame_idx)
                            self._parking_bootstrap_success = True
                            self._last_parking_result_age_ms = max(0.0, (time.perf_counter() - result.done_perf) * 1000.0)
                        else:
                            print(f"[ParkingBootstrap] 首帧停车位推理失败：{result.error}")
                        break
                    time.sleep(0.002)
            else:
                det_slots = self._extract_parking_polygons_sync(frame)
                self.slots.update_detections(det_slots, target_frame_idx)
                self._parking_bootstrap_success = True
        except Exception as e:
            print(f"[ParkingBootstrap] 首帧快速车位初始化失败，继续使用异步流程：{e}")
        finally:
            self._parking_bootstrap_ms = (time.perf_counter() - t0) * 1000.0
            # Prevent immediate duplicate submission when bootstrap succeeded or timed out just now.
            self._last_parking_submit_perf = time.perf_counter()
            state = "成功" if self._parking_bootstrap_success else "超时/回退"
            print(f"[ParkingBootstrap] {state} | {self._parking_bootstrap_ms:.1f} ms")

    def close(self) -> None:
        if self.parking_worker is not None:
            self.parking_worker.close()
        if self.plate_worker is not None:
            self.plate_worker.close()

    def reset_runtime_tracking(self, reset_events: bool = False, reset_slot_ids: bool = False) -> None:
        self.prev_vehicle_boxes = []
        self.motion.reset()
        self.vehicles.states.clear()
        self.slots.slots.clear()
        self.slots.retired_slots.clear()
        self.permanent_slots.reset_session()
        self._motion_history.clear()
        self._last_parking_submit_perf = -1e9
        self._parking_bootstrap_done = False
        self._parking_bootstrap_success = False
        self._parking_bootstrap_ms = 0.0
        if reset_slot_ids:
            self.slots.next_id = 1
            self.slots.total_local_ids_created = 0
            self.slots.total_reidentified = 0
        if reset_events:
            self.events.reset()
        self.last_events = []
        self._reset_vehicle_tracker_only()

    def switch_roadside_segment(self, segment: int | str | None = None) -> bool:
        """Switch roadside patrol leg without clearing cumulative patrol/event results.

        Local camera/vehicle/slot trackers are reset because the drone physically transfers to another
        road end and the apparent image direction reverses. The PatrolAccumulator lives outside this
        engine, so already-confirmed slots from leg-1 remain counted.
        """
        ok = (self.permanent_slots.switch_to_next_segment() if segment is None
              else self.permanent_slots.switch_segment(segment))
        if not ok:
            print(f"[CONTROL][WARN] 航段切换失败: {self.permanent_slots.last_error}")
            return False
        self.prev_vehicle_boxes = []
        self.motion.reset()
        self.vehicles.states.clear()
        self.slots.slots.clear()
        self.slots.retired_slots.clear()
        self._motion_history.clear()
        self._last_parking_submit_perf = -1e9
        self._parking_bootstrap_done = False
        self._parking_bootstrap_success = False
        self._parking_bootstrap_ms = 0.0
        self._reset_vehicle_tracker_only()
        print(
            f"[CONTROL] 已切换到航段 {self.permanent_slots.active_segment_index + 1}/"
            f"{self.permanent_slots.segment_count}: {self.permanent_slots.current_segment_id} | "
            f"起点 {self.permanent_slots.start_real_id}"
        )
        return True

    def _record_motion(self, current_idx: int, M: Optional[np.ndarray]) -> None:
        self._motion_history[int(current_idx)] = None if M is None else np.asarray(M, np.float32).copy()
        min_keep = max(1, int(current_idx) - 80)
        for k in list(self._motion_history.keys()):
            if k < min_keep:
                self._motion_history.pop(k, None)

    def _transform_between_processed_frames(self, source_idx: int, current_idx: int) -> Optional[np.ndarray]:
        if source_idx == current_idx:
            return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        if source_idx > current_idx:
            return None
        mats = []
        for idx in range(int(source_idx) + 1, int(current_idx) + 1):
            if idx not in self._motion_history or self._motion_history[idx] is None:
                return None
            mats.append(self._motion_history[idx])
        return _compose_affines(mats)

    def _extract_parking_polygons_sync(self, frame: np.ndarray) -> List[Tuple[np.ndarray, float]]:
        assert self.parking_model is not None
        results = self.parking_model.predict(
            frame, imgsz=self.cfg.parking_imgsz, conf=self.cfg.parking_conf,
            iou=self.cfg.iou, device=self.device_yolo, verbose=False,
        )
        if not results:
            return []
        r = results[0]
        out: List[Tuple[np.ndarray, float]] = []
        confs = r.boxes.conf.detach().cpu().numpy().tolist() if r.boxes is not None and len(r.boxes) else []
        if r.masks is not None and getattr(r.masks, "xy", None) is not None:
            for i, p in enumerate(r.masks.xy):
                poly = np.asarray(p, dtype=np.float32).reshape(-1, 2)
                if len(poly) >= 3 and polygon_area(poly) >= 80.0:
                    out.append((poly, float(confs[i]) if i < len(confs) else 1.0))
        elif r.boxes is not None:
            for i, b in enumerate(r.boxes.xyxy.detach().cpu().numpy()):
                out.append((box_to_polygon(tuple(map(float, b[:4]))), float(confs[i]) if i < len(confs) else 1.0))
        return out

    def _track_vehicles(self, frame: np.ndarray) -> List[Tuple[int, int, str, float, Box]]:
        kwargs = dict(
            persist=True,
            tracker="ultralytics/cfg/trackers/botsort.yaml",
            imgsz=self.cfg.vehicle_imgsz,
            conf=self.cfg.vehicle_conf,
            iou=self.cfg.iou,
            classes=list(self.cfg.vehicle_classes),
            device=self.device_yolo,
            verbose=False,
        )
        if self.cfg.vehicle_half and not self._vehicle_is_engine and self.device_yolo != "cpu":
            kwargs["half"] = True
        results = self.vehicle_model.track(frame, **kwargs)
        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            return []
        boxes_obj = results[0].boxes
        if boxes_obj.id is None:
            return []
        xyxy = boxes_obj.xyxy.detach().cpu().numpy()
        ids = boxes_obj.id.detach().cpu().numpy().astype(int)
        clss = boxes_obj.cls.detach().cpu().numpy().astype(int)
        confs = boxes_obj.conf.detach().cpu().numpy()
        names = results[0].names if getattr(results[0], "names", None) is not None else self.vehicle_names
        out = []
        for box, tid, cid, conf in zip(xyxy, ids, clss, confs):
            name = names.get(int(cid), str(cid)) if isinstance(names, dict) else str(cid)
            out.append((int(tid), int(cid), str(name), float(conf), tuple(map(float, box[:4]))))
        return out

    def _recognize_plate_in_vehicle_sync(self, frame: np.ndarray, vehicle_box: Box) -> PlateObservation:
        assert self.plate_det_model is not None and self.plate_rec_model is not None
        h, w = frame.shape[:2]
        crop_box = expand_box(vehicle_box, 1.08, w, h)
        x1, y1, x2, y2 = map(int, map(round, crop_box))
        if x2 - x1 < 20 or y2 - y1 < 20:
            return PlateObservation()
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return PlateObservation()
        results = self.plate_det_model.predict(
            crop, imgsz=self.cfg.plate_imgsz, conf=self.cfg.plate_conf,
            iou=self.cfg.iou, device=self.device_yolo, verbose=False,
        )
        if not results:
            return PlateObservation()
        r = results[0]
        if r.boxes is None or len(r.boxes) == 0 or r.keypoints is None:
            return PlateObservation()
        boxes = r.boxes.xyxy.detach().cpu().numpy()
        confs = r.boxes.conf.detach().cpu().numpy()
        clss = r.boxes.cls.detach().cpu().numpy().astype(int)
        kpts = r.keypoints.xy.detach().cpu().numpy()
        n = min(len(boxes), len(kpts))
        for idx in np.argsort(-confs[:n]):
            landmarks = np.asarray(kpts[idx], dtype=np.float32).reshape(-1, 2)
            if len(landmarks) < 4:
                continue
            roi = four_point_transform(crop, landmarks[:4])
            if roi.size == 0:
                continue
            if int(clss[idx]) == 1:
                roi = get_split_merge(roi)
            with torch.inference_mode():
                plate_no, char_probs, plate_color, color_conf = get_plate_result(
                    roi, self.torch_device, self.plate_rec_model, is_color=True
                )
            plate_no = str(plate_no).strip()
            if not valid_plate_text(plate_no):
                continue
            probs = np.asarray(char_probs, dtype=np.float32).reshape(-1)
            char_score = float(probs.mean()) if probs.size else 0.0
            det_conf = float(confs[idx])
            score = det_conf * max(0.05, char_score) * max(0.05, float(color_conf))
            b = boxes[idx]
            return PlateObservation(
                plate_no, str(plate_color), score, det_conf,
                (x1 + float(b[0]), y1 + float(b[1]), x1 + float(b[2]), y1 + float(b[3]))
            )
        return PlateObservation()

    def _consume_parking_result(self, current_perf: float) -> None:
        """Consume the newest parking result with a startup-safe alignment fallback.

        v6.5.6 discarded a result whenever *any* motion matrix between the submitted frame and
        current frame was unavailable. That is too strict during UAV startup: optical flow can be
        unreliable for several frames even though the segmentation itself is already correct.

        v6.5.7 therefore:
        1) always accepts the first fresh NON-EMPTY parking result when no slot has been established;
        2) for later results, allows a very short fresh/raw-coordinate fallback when alignment fails;
        3) still rejects stale results, so old polygons cannot linger far behind a moving drone.
        """
        if self.parking_worker is None:
            return
        result = self.parking_worker.pop_latest_result()
        if result is None:
            return
        if result.error:
            print(f"[ParkingWorker] 推理错误：{result.error}")
            return

        age = max(0.0, current_perf - result.submit_perf)
        self._last_parking_result_age_ms = age * 1000.0
        if age > self.cfg.parking_result_max_age_s:
            self._parking_result_dropped_stale += 1
            return

        frame_gap = max(0, int(self.frame_idx) - int(result.frame_idx))
        M_source_to_now = self._transform_between_processed_frames(result.frame_idx, self.frame_idx)

        # Normal path: accurate motion-aligned polygons.
        if M_source_to_now is not None:
            aligned = [(transform_polygon(M_source_to_now, p), conf) for p, conf in result.polygons]
            self.slots.update_detections(aligned, self.frame_idx)
            self._parking_results_applied += 1
            if result.polygons and not self._parking_first_nonempty_result_seen:
                self._parking_first_nonempty_result_seen = True
                if self._first_real_frame_perf > 0:
                    self._first_parking_visible_ms = max(0.0, (current_perf - self._first_real_frame_perf) * 1000.0)
                print(f"[ParkingStartup] 首个停车位结果已应用(已对齐) | age={age*1000:.1f} ms | gap={frame_gap}")
            return

        # Startup-critical fallback: no established slot yet, but segmentation already produced
        # a fresh non-empty result. Show it immediately rather than throwing it away for seconds.
        first_nonempty_fallback = (
            bool(self.cfg.parking_accept_first_fresh_unaligned)
            and not self._parking_first_nonempty_result_seen
            and len(result.polygons) > 0
            and age <= self.cfg.parking_result_max_age_s
        )

        # Short-lived fallback for later refreshes. Because the result is both young and only a few
        # processed frames old, raw polygons are preferable to showing no parking spaces at all.
        fresh_unaligned_fallback = (
            bool(self.cfg.parking_fresh_unaligned_fallback)
            and len(result.polygons) > 0
            and age <= self.cfg.parking_fresh_unaligned_max_age_s
            and frame_gap <= self.cfg.parking_fresh_unaligned_max_frame_gap
        )

        if first_nonempty_fallback or fresh_unaligned_fallback:
            self.slots.update_detections(result.polygons, self.frame_idx)
            self._parking_results_applied += 1
            self._parking_results_applied_unaligned += 1
            if not self._parking_first_nonempty_result_seen:
                self._parking_first_nonempty_result_seen = True
                if self._first_real_frame_perf > 0:
                    self._first_parking_visible_ms = max(0.0, (current_perf - self._first_real_frame_perf) * 1000.0)
                print(
                    f"[ParkingStartup] 首个停车位结果已立即应用(未对齐回退) | "
                    f"age={age*1000:.1f} ms | gap={frame_gap} | slots={len(result.polygons)}"
                )
            return

        self._parking_result_dropped_motion += 1

    def _schedule_parking_async(self, frame: np.ndarray, current_perf: float, force: bool = False) -> None:
        if self.parking_worker is None:
            return
        interval = self.cfg.parking_async_interval_s
        if self.cfg.adaptive_aux_load_shedding and self._main_fps_ema > 0 and self._main_fps_ema < self.cfg.aux_low_fps_threshold:
            interval = max(interval, self.cfg.aux_low_fps_parking_interval)
        if force or current_perf - self._last_parking_submit_perf >= interval:
            self.parking_worker.submit(frame, self.frame_idx)
            self._last_parking_submit_perf = current_perf

    def _find_state_by_identity_key(self, identity_key: int) -> Optional[VehicleTrackState]:
        for st in self.vehicles.states.values():
            if id(st) == identity_key:
                return st
        return None

    def _consume_plate_results(self, current_perf: float, now_wall: float) -> None:
        if self.plate_worker is None:
            return
        for result in self.plate_worker.pop_results():
            if result.error:
                print(f"[PlateWorker] 推理错误：{result.error}")
                continue
            age = max(0.0, current_perf - result.task.submit_perf)
            if age > self.cfg.plate_result_max_age_s:
                self._plate_result_dropped_stale += 1
                continue
            st = self._find_state_by_identity_key(result.task.identity_key)
            if st is None:
                self._plate_result_dropped_identity += 1
                continue
            obs = result.observation
            st.update_plate(
                obs.plate_no, obs.plate_color, obs.score, obs.detect_conf, obs.global_box,
                self.cfg.plate_min_hits, result.task.vehicle_box_snapshot, now_wall,
            )
            self._plate_results_applied += 1

    def _assign_parking_relations(self, active_states: Sequence[VehicleTrackState], slots: Sequence[ParkingSlot]) -> None:
        for st in active_states:
            car_poly = box_to_polygon(st.box)
            car_area = max(1.0, box_area(st.box))
            center = box_center(st.box)
            best_sid = None
            best_real_sid = ""
            best_vehicle_ratio = 0.0
            best_slot_ratio = 0.0
            best_center_inside = False
            meaningful_overlap_count = 0
            for slot in slots:
                if bbox_iou(st.box, slot.bbox) <= 0.0:
                    continue
                inter = convex_intersection_area(car_poly, slot.polygon)
                if inter <= 0.0:
                    continue
                vr = inter / car_area
                sa = max(1.0, polygon_area(slot.polygon))
                sr = inter / sa
                inside = point_in_polygon(center, slot.polygon)
                if vr >= 0.12 or sr >= 0.18:
                    meaningful_overlap_count += 1
                score = vr + (0.08 if inside else 0.0)
                current_best_score = best_vehicle_ratio + (0.08 if best_center_inside else 0.0)
                if score > current_best_score:
                    best_sid = slot.slot_id
                    best_real_sid = slot.real_id
                    best_vehicle_ratio = vr
                    best_slot_ratio = sr
                    best_center_inside = inside
            st.best_slot_id = best_sid
            st.best_real_slot_id = best_real_sid
            st.best_vehicle_overlap = float(best_vehicle_ratio)
            st.best_slot_overlap = float(best_slot_ratio)
            st.center_in_slot = bool(best_center_inside)
            st.overlapping_slot_count = int(meaningful_overlap_count)
            st.violation_type = ""
            if not st.is_stationary:
                raw = STATUS_MOVING if st.motion_score is not None else STATUS_MONITORING
            else:
                normal = st.center_in_slot and st.best_vehicle_overlap >= self.cfg.normal_min_vehicle_overlap
                violation = st.best_vehicle_overlap <= self.cfg.violation_max_vehicle_overlap
                if normal:
                    raw = STATUS_NORMAL
                elif meaningful_overlap_count >= 2:
                    raw = STATUS_VIOLATION
                    st.violation_type = "MULTI_SPACE"
                elif violation:
                    raw = STATUS_VIOLATION
                    st.violation_type = "OUTSIDE"
                else:
                    raw = STATUS_BORDERLINE
                    st.violation_type = "PARTIAL_SPACE"
            st.update_status(raw, self.cfg.status_confirm_frames)

    def _plate_candidates(self, active_states: Sequence[VehicleTrackState], now: float) -> List[VehicleTrackState]:
        candidates: List[VehicleTrackState] = []
        for st in active_states:
            bw, bh = box_wh(st.box)
            if max(bw, bh) < self.cfg.min_vehicle_box_for_plate:
                continue
            interval = self.cfg.stable_plate_refresh_seconds if st.stable_plate else self.cfg.plate_retry_seconds
            if now - st.last_plate_try_t >= interval:
                candidates.append(st)
        def priority(st: VehicleTrackState):
            finalish = st.confirmed_status in (STATUS_VIOLATION, STATUS_BORDERLINE, STATUS_NORMAL)
            bw, bh = box_wh(st.box)
            overdue = now - st.last_plate_try_t
            return (1 if finalish else 0, bw * bh, overdue)
        candidates.sort(key=priority, reverse=True)
        return candidates

    def _schedule_plate_recognition(self, frame: np.ndarray, active_states: Sequence[VehicleTrackState], now: float) -> None:
        candidates = self._plate_candidates(active_states, now)
        if not candidates:
            return

        # Under severe main-chain load, only submit one plate ROI per processed frame.
        limit = min(self.cfg.max_plate_rois_per_frame, len(candidates))
        if self.cfg.adaptive_aux_load_shedding and self._main_fps_ema > 0 and self._main_fps_ema < self.cfg.aux_low_fps_threshold:
            limit = min(limit, 1)

        if self.plate_worker is not None:
            h, w = frame.shape[:2]
            for st in candidates[:limit]:
                crop_box = expand_box(st.box, 1.08, w, h)
                x1, y1, x2, y2 = map(int, map(round, crop_box))
                if x2 - x1 < 20 or y2 - y1 < 20:
                    continue
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                st.last_plate_try_t = now
                self.plate_worker.submit(PlateAsyncTask(
                    crop=crop.copy(),
                    crop_origin=(x1, y1),
                    vehicle_box_snapshot=st.box,
                    identity_key=id(st),
                    track_id_snapshot=st.track_id,
                    submit_perf=time.perf_counter(),
                    submit_wall=now,
                ))
        else:
            for st in candidates[:limit]:
                st.last_plate_try_t = now
                obs = self._recognize_plate_in_vehicle_sync(frame, st.box)
                st.update_plate(
                    obs.plate_no, obs.plate_color, obs.score, obs.detect_conf, obs.global_box,
                    self.cfg.plate_min_hits, st.box, now,
                )

    def _compute_slot_occupancy(
        self, slots: Sequence[ParkingSlot], active_states: Sequence[VehicleTrackState]
    ) -> Dict[int, Optional[int]]:
        occupancy: Dict[int, Optional[int]] = {s.slot_id: None for s in slots}
        for slot in slots:
            sa = max(1.0, polygon_area(slot.polygon))
            best_tid, best_ratio = None, 0.0
            for st in active_states:
                if bbox_iou(st.box, slot.bbox) <= 0.0:
                    continue
                inter = convex_intersection_area(box_to_polygon(st.box), slot.polygon)
                sr = inter / sa
                if sr > best_ratio:
                    best_ratio = sr
                    best_tid = st.track_id
            if best_tid is not None and best_ratio >= self.cfg.occupied_min_slot_overlap:
                occupancy[slot.slot_id] = best_tid
        return occupancy

    def _compute_slot_cumulative_evidence(
        self, slots: Sequence[ParkingSlot], active_states: Sequence[VehicleTrackState]
    ) -> Dict[str, Optional[bool]]:
        """Return tri-state route evidence keyed by a stable parking-space identity.

        Permanent-map mode: only slots with confirmed ``real_id`` participate, so an unmapped
        temporary P-ID can never be frozen and then counted again after it later becomes A037.
        Legacy mode: P1/P2/... is used as before.
        """
        evidence: Dict[str, Optional[bool]] = {}
        require_stationary = bool(CUMULATIVE_OCCUPIED_REQUIRE_STATIONARY)
        permanent_mode = bool(self.permanent_slots.active)
        for slot in slots:
            if permanent_mode:
                if not slot.real_id:
                    continue
                key = str(slot.real_id)
            else:
                key = f"P{slot.slot_id}"

            sa = max(1.0, polygon_area(slot.polygon))
            any_overlap = False
            stationary_overlap = False
            for st in active_states:
                if bbox_iou(st.box, slot.bbox) <= 0.0:
                    continue
                inter = convex_intersection_area(box_to_polygon(st.box), slot.polygon)
                sr = inter / sa
                if sr < self.cfg.occupied_min_slot_overlap:
                    continue
                any_overlap = True
                if st.is_stationary or not require_stationary:
                    stationary_overlap = True
                    break
            if stationary_overlap:
                evidence[key] = True
            elif any_overlap:
                evidence[key] = None
            else:
                evidence[key] = False
        return evidence

    def _cumulative_eligibility_map(self, slots: Sequence[ParkingSlot], w: int, h: int) -> Dict[str, bool]:
        permanent_mode = bool(self.permanent_slots.active)
        out: Dict[str, bool] = {}
        for slot in slots:
            if permanent_mode:
                if not slot.real_id:
                    continue
                key = str(slot.real_id)
            else:
                key = f"P{slot.slot_id}"
            out[key] = bool(
                slot.bbox[0] >= w * CUMULATIVE_SLOT_EDGE_MARGIN_RATIO
                and slot.bbox[1] >= h * CUMULATIVE_SLOT_EDGE_MARGIN_RATIO
                and slot.bbox[2] <= w * (1.0 - CUMULATIVE_SLOT_EDGE_MARGIN_RATIO)
                and slot.bbox[3] <= h * (1.0 - CUMULATIVE_SLOT_EDGE_MARGIN_RATIO)
            )
        return out

    def _render(
        self,
        frame: np.ndarray,
        slots: Sequence[ParkingSlot],
        active_states: Sequence[VehicleTrackState],
        occupancy: Dict[int, Optional[int]],
        fps: float,
        camera_motion_ok: bool,
    ) -> np.ndarray:
        out = frame.copy()
        text_items: List[Tuple[str, int, int, Tuple[int, int, int], int]] = []

        # 创建填充层：用于绘制半透明填充色，最后统一混合到主图上
        fill_overlay = out.copy()

        # ==================== 1. 停车位多边形（填充 + 轮廓） ====================
        # 绿色=当前空闲；黄色=行驶车辆临时经过（不进累计）；橙色=静止车辆占用。
        state_by_tid = {st.track_id: st for st in active_states}
        for slot in slots:
            occ_tid = occupancy.get(slot.slot_id)
            if occ_tid is None:
                color = (60, 200, 60)
            else:
                occ_state = state_by_tid.get(occ_tid)
                color = (0, 130, 255) if (occ_state is not None and occ_state.is_stationary) else (0, 220, 255)
            pts = np.round(slot.polygon).astype(np.int32).reshape(-1, 1, 2)
            # 半透明填充
            cv2.fillPoly(fill_overlay, [pts], color)
            # 加粗轮廓线
            cv2.polylines(out, [pts], True, color, 3, cv2.LINE_AA)
            x1, y1, x2, y2 = slot.bbox
            label_x = int(max(2, min(out.shape[1] - 2, x1)))
            if slot.real_id:
                # v6.8.1：真实编号作为主标签单独显示，临时 P-ID 只作为调试辅助。
                main_y = int(y1 - self.cfg.roadside_real_id_font_size - 6)
                if main_y < 2:
                    main_y = int(min(out.shape[0] - self.cfg.roadside_real_id_font_size - 2, y1 + 4))
                main_y = max(2, main_y)
                text_items.append((
                    f"真实车位 {slot.real_id}",
                    label_x, main_y, color, int(self.cfg.roadside_real_id_font_size),
                ))
                if self.cfg.roadside_show_local_id:
                    local_y = int(min(
                        out.shape[0] - self.cfg.roadside_local_id_font_size - 1,
                        main_y + self.cfg.roadside_real_id_font_size + 1
                    ))
                    text_items.append((
                        f"P{slot.slot_id}",
                        label_x, max(1, local_y), (225, 225, 225), int(self.cfg.roadside_local_id_font_size),
                    ))
            else:
                # 尚未完成永久编号确认时明确提示，防止把临时 P-ID 当成现实编号。
                pending_y = int(max(2, y1 - self.cfg.roadside_pending_id_font_size - 4))
                text_items.append((
                    f"P{slot.slot_id} | 真实编号匹配中",
                    label_x, pending_y, color, int(self.cfg.roadside_pending_id_font_size),
                ))

        # ==================== 2. 车辆检测框（填充 + 轮廓） ====================
        status_colors = {
            STATUS_MONITORING: (180, 180, 180), STATUS_MOVING: (255, 190, 0),
            STATUS_NORMAL: (40, 220, 40), STATUS_VIOLATION: (0, 0, 255),
            STATUS_BORDERLINE: (0, 90, 255),
        }
        for st in active_states:
            status = st.confirmed_status
            color = status_colors.get(status, (200, 200, 200))
            x1, y1, x2, y2 = map(int, map(round, st.box))
            # 半透明填充（用状态颜色）
            cv2.rectangle(fill_overlay, (x1, y1), (x2, y2), color, -1)
            # 加粗轮廓线
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 3, cv2.LINE_AA)
            plate = st.stable_plate if st.stable_plate else "车牌识别中"
            overlap = int(round(st.best_vehicle_overlap * 100.0))
            slot_text = st.best_real_slot_id if st.best_real_slot_id else (f"P{st.best_slot_id}" if st.best_slot_id is not None else "未绑定")
            text_items.append((
                f"ID{st.track_id} {st.cls_name} | {plate} | 车位{slot_text} | {status} | 入位{overlap}%",
                max(0, x1), max(0, y1 - 25), color, 18,
            ))
            # ==================== 3. 车牌框（填充 + 轮廓） ====================
            pbox = st.current_plate_box()
            if pbox is not None:
                px1, py1, px2, py2 = map(int, map(round, pbox))
                plate_color = (255, 80, 210)
                cv2.rectangle(fill_overlay, (px1, py1), (px2, py2), plate_color, -1)
                cv2.rectangle(out, (px1, py1), (px2, py2), plate_color, 3, cv2.LINE_AA)

        # ==================== 统一混合：将填充层以 35% 透明度叠加到主图 ====================
        cv2.addWeighted(fill_overlay, 0.35, out, 0.65, 0, out)

        # ==================== 4. 信息面板 ====================
        n_slots = len(slots)
        occupied = sum(1 for v in occupancy.values() if v is not None)
        stationary_occupied = sum(
            1 for tid in occupancy.values()
            if tid is not None and tid in state_by_tid and state_by_tid[tid].is_stationary
        )
        transient_occupied = max(0, occupied - stationary_occupied)
        active_n = len(active_states)
        normal_n = sum(st.confirmed_status == STATUS_NORMAL for st in active_states)
        violation_n = sum(st.confirmed_status in (STATUS_VIOLATION, STATUS_BORDERLINE) for st in active_states)
        stationary_n = sum(st.is_stationary for st in active_states)
        cm = "OK" if camera_motion_ok else "等待/不可靠"
        p_busy = "忙" if self.parking_worker is not None and self.parking_worker.busy else "闲"
        q_plate = self.plate_worker.pending_count if self.plate_worker is not None else 0
        panel_h = 204
        overlay = out.copy()
        cv2.rectangle(overlay, (8, 8), (720, panel_h), (10, 10, 10), -1)
        cv2.addWeighted(overlay, 0.68, out, 0.32, 0, out)
        lines = [
            f"无人机实时主链 | FPS {fps:.1f} | 背景运动补偿 {cm}",
            f"当前车位 {n_slots}  静止占用 {stationary_occupied}  临时经过 {transient_occupied}  空余 {max(0, n_slots - occupied)}",
            f"当前车辆 {active_n}  静止 {stationary_n}  正常 {normal_n}  违停/压线 {violation_n}",
            f"车牌绑定: 正常 {len(self.events.normal_plates)}  违停 {len(self.events.violation_plates)}",
            f"异步辅助: 停车位worker={p_busy}  车牌队列={q_plate}",
            (f"路侧编号: 路线={self.permanent_slots.last_zone_id or '-'}  "
             f"已映射={sum(1 for s in slots if s.real_id)}/{len(slots)}  "
             f"方向={'OK' if self.permanent_slots.direction_ready else '学习中'}  "
             f"进度={self.permanent_slots.furthest_real_id or '-'}")
            if self.permanent_slots.active else "路侧永久编号: 关闭（当前使用临时P编号）",
        ]
        for i, line in enumerate(lines):
            text_items.append((line, 20, 16 + i * 29, (245, 245, 245), 18))
        return draw_text_batch(out, text_items, self.cfg.font_path)



    def process_frame(
        self,
        frame: np.ndarray,
        fps_hint: float = 0.0,
        logic_result_callback: Optional[Callable[[dict, List[dict]], None]] = None,
        capture_perf: float = 0.0,
    ) -> Tuple[np.ndarray, dict, List[dict]]:
        """Process newest frame.

        ``logic_result_callback`` is invoked after recognition/state judgement has produced a
        complete statistics snapshot but BEFORE annotation rendering. This lets runtime statistics and
        route accumulation update independently of local/TB image rendering latency.
        """
        t_all = time.perf_counter()
        now_wall = time.time()
        capture_to_process_start_ms = (
            max(0.0, (t_all - float(capture_perf)) * 1000.0) if capture_perf and capture_perf > 0 else 0.0
        )
        # v6.5.6 deployment default is non-blocking. Compatibility bootstrap runs only if explicitly enabled.
        if self.cfg.parking_first_frame_bootstrap:
            self._bootstrap_parking_first_frame(frame)
        elif not self._parking_bootstrap_done:
            self._parking_bootstrap_done = True
        self.frame_idx += 1
        h, w = frame.shape[:2]

        # v6.5.7: the first REAL frame gives parking segmentation a head start immediately.
        # This is non-blocking: vehicle detection continues at once in the main thread.
        if self._first_real_frame_perf <= 0:
            self._first_real_frame_perf = t_all
            if self.parking_worker is not None and self.cfg.parking_early_first_frame_submit:
                self._schedule_parking_async(frame, time.perf_counter(), force=True)

        # 1) Sparse background motion compensation.
        t = time.perf_counter()
        M = self.motion.update(frame, self.prev_vehicle_boxes)
        self.slots.warp(M)
        self._record_motion(self.frame_idx, M)
        motion_ms = (time.perf_counter() - t) * 1000.0

        # 2) Consume completed parking result first, but DO NOT launch a new auxiliary GPU job yet.
        # New parking jobs are submitted only after the current vehicle main-chain inference finishes.
        t = time.perf_counter()
        now_perf = time.perf_counter()
        if self.parking_worker is not None:
            self._consume_parking_result(now_perf)
        elif self.frame_idx == 1 or self.frame_idx % max(1, self.cfg.parking_every_n) == 0:
            det_slots = self._extract_parking_polygons_sync(frame)
            self.slots.update_detections(det_slots, self.frame_idx)
        parking_main_ms = (time.perf_counter() - t) * 1000.0
        # 6.8.3: if duplicate P-IDs were merged, immediately repair permanent-ID ownership/cache.
        slot_merge_aliases = self.slots.consume_merge_aliases()
        if slot_merge_aliases:
            self.permanent_slots.rebind_tracker_aliases(slot_merge_aliases, list(self.slots.slots.values()))
        visible_slots = self.slots.visible_slots(w, h)

        # 3) HIGH-PRIORITY vehicle detector + BoT-SORT on every processed newest frame.
        t = time.perf_counter()
        vehicle_dets = self._track_vehicles(frame)
        vehicle_track_ms = (time.perf_counter() - t) * 1000.0

        # Only now submit the newest parking task. The main chain never waits for it.
        t = time.perf_counter()
        if self.parking_worker is not None:
            self._schedule_parking_async(frame, time.perf_counter())
        parking_submit_ms = (time.perf_counter() - t) * 1000.0

        active_states: List[VehicleTrackState] = []
        current_boxes: List[Box] = []
        current_tracker_ids = {tid for tid, *_ in vehicle_dets}
        for tid, cid, cname, conf, box in vehicle_dets:
            current_boxes.append(box)
            st, is_new_identity, old_tid = self.vehicles.obtain_state(
                tid, cid, cname, box, conf, now_wall, self.frame_idx, M, current_tracker_ids
            )
            if not is_new_identity:
                st.update_motion(
                    box, now_wall, self.frame_idx, M,
                    self.cfg.stationary_residual_ratio, self.cfg.stationary_seconds,
                    self.cfg.stationary_fast_seconds, self.cfg.stationary_strong_residual_factor,
                )
            st.conf = conf
            active_states.append(st)
        self.prev_vehicle_boxes = current_boxes
        self.vehicles.purge(now_wall)

        # 4) Apply finished OCR results only in main thread (state mutation remains thread-safe).
        self._consume_plate_results(time.perf_counter(), now_wall)

        # 4.5) 路侧永久车位编号映射（无GPS）。只做轻量顺序/运动几何，仍放在车辆主检测之后。
        t_map = time.perf_counter()
        permanent_diag = self.permanent_slots.update(frame, visible_slots, self.frame_idx)
        permanent_map_ms = (time.perf_counter() - t_map) * 1000.0

        # 5) Cheap geometry + status judgement.
        t = time.perf_counter()
        self._assign_parking_relations(active_states, visible_slots)
        # Visual occupancy reacts immediately to any sufficiently overlapping vehicle.
        occupancy = self._compute_slot_occupancy(visible_slots, active_states)
        # Route cumulative occupancy is stricter: moving pass-through vehicles are tri-state None.
        cumulative_slot_state = self._compute_slot_cumulative_evidence(visible_slots, active_states)
        geometry_ms = (time.perf_counter() - t) * 1000.0

        # 6) Submit plate ROIs without waiting for plate engine/OCR.
        t = time.perf_counter()
        self._schedule_plate_recognition(frame, active_states, now_wall)
        plate_submit_ms = (time.perf_counter() - t) * 1000.0

        # 7) Plate-bound event generation remains in main thread.
        new_events = []
        for st in active_states:
            # In real-ID mode, a normally parked vehicle must first be bound to a permanent roadside slot.
            # Otherwise EventLogger's plate/status de-duplication could emit a P-ID event too early and prevent
            # a later corrected real-slot event from being published.
            if self.permanent_slots.active and st.confirmed_status == STATUS_NORMAL and not st.best_real_slot_id:
                continue
            event = self.events.maybe_log(st, now_wall)
            if event is not None:
                new_events.append(event)
        self.last_events = new_events

        pre_render_dt = max(1e-6, time.perf_counter() - t_all)
        pre_render_fps = 1.0 / pre_render_dt
        if self._main_fps_ema <= 0:
            self._main_fps_ema = pre_render_fps
        else:
            self._main_fps_ema = 0.85 * self._main_fps_ema + 0.15 * pre_render_fps

        # Build the recognition/state snapshot BEFORE any drawing.
        # runtime numeric statistics must not wait for annotation rendering or GUI display.
        pworker = self.parking_worker
        plworker = self.plate_worker
        stats = {
            "fps_processing": round(pre_render_fps, 2),
            "fps_processing_ema": round(self._main_fps_ema, 2),
            "processing_ms": round(pre_render_dt * 1000.0, 2),
            "capture_to_process_start_ms": round(capture_to_process_start_ms, 2),
            "startup_warmup_ms": round(float(self._startup_warmup_ms), 2),
            "startup_tracker_warmup_ms": round(float(self._startup_tracker_warmup_ms), 2),
            "startup_tracker_warmup_ok": bool(self._startup_tracker_warmup_ok),
            "vehicle_track_ms": round(vehicle_track_ms, 2),
            "camera_motion_ms": round(motion_ms, 2),
            "parking_main_thread_ms": round(parking_main_ms, 2),
            "parking_submit_ms": round(parking_submit_ms, 2),
            "roadside_slot_map_ms": round(permanent_map_ms, 2),
            "permanent_slot_map_ms": round(permanent_map_ms, 2),  # backward compatibility
            "parking_worker_infer_ms": round(float(pworker.last_infer_ms if pworker else 0.0), 2),
            "parking_result_age_ms": round(float(self._last_parking_result_age_ms), 2),
            "parking_worker_busy": bool(pworker.busy) if pworker else False,
            "parking_tasks_dropped": int(pworker.dropped_tasks) if pworker else 0,
            "parking_results_dropped_stale": int(self._parking_result_dropped_stale),
            "parking_results_dropped_motion": int(self._parking_result_dropped_motion),
            "parking_results_applied": int(self._parking_results_applied),
            "parking_results_applied_unaligned": int(self._parking_results_applied_unaligned),
            "first_parking_visible_ms": round(float(self._first_parking_visible_ms), 2),
            "plate_submit_ms": round(plate_submit_ms, 2),
            "plate_worker_infer_ms": round(float(plworker.last_infer_ms if plworker else 0.0), 2),
            "plate_worker_busy": bool(plworker.busy) if plworker else False,
            "plate_queue_pending": int(plworker.pending_count) if plworker else 0,
            "plate_tasks_dropped": int(plworker.dropped_tasks) if plworker else 0,
            "plate_results_applied": int(self._plate_results_applied),
            "plate_results_dropped_stale": int(self._plate_result_dropped_stale),
            "plate_results_dropped_identity": int(self._plate_result_dropped_identity),
            "geometry_ms": round(geometry_ms, 2),
            "render_ms": 0.0,
            "frame_idx": self.frame_idx,
            "camera_motion_ok": M is not None,
            "parking_slots_current": len(visible_slots),
            "parking_slots_occupied_current": sum(1 for v in occupancy.values() if v is not None),
            "parking_slots_free_current": sum(1 for v in occupancy.values() if v is None),
            "parking_slots_stationary_occupied_current": sum(1 for v in cumulative_slot_state.values() if v is True),
            "parking_slots_transient_current": sum(1 for v in cumulative_slot_state.values() if v is None),
            "parking_slots_free_cumulative_evidence_current": sum(1 for v in cumulative_slot_state.values() if v is False),
            "parking_slot_local_ids_seen": self.slots.total_local_ids_created,
            "parking_slot_reid_revived": self.slots.total_reidentified,
            "parking_detection_duplicates_suppressed_last": self.slots.last_detection_duplicates_suppressed,
            "parking_detection_duplicates_suppressed_total": self.slots.total_detection_duplicates_suppressed,
            "parking_duplicate_local_ids_merged_last": self.slots.last_duplicate_ids_merged,
            "parking_duplicate_local_ids_merged_total": self.slots.total_duplicate_ids_merged,
            "parking_bootstrap_success": bool(self._parking_bootstrap_success),
            "parking_bootstrap_ms": round(float(self._parking_bootstrap_ms), 2),
            "stationary_effective_seconds_avg": round(float(np.mean([st.stationary_effective_seconds for st in active_states if st.stationary_effective_seconds > 0])) if any(st.stationary_effective_seconds > 0 for st in active_states) else 0.0, 3),
            "vehicles_current": len(active_states),
            "vehicles_stationary_current": sum(st.is_stationary for st in active_states),
            "normal_parked_current": sum(st.confirmed_status == STATUS_NORMAL for st in active_states),
            "violation_current": sum(st.confirmed_status in (STATUS_VIOLATION, STATUS_BORDERLINE) for st in active_states),
            "unique_normal_plate_bound": len(self.events.normal_plates),
            "unique_violation_plate_bound": len(self.events.violation_plates),
            # Current-frame local occupancy remains keyed by visual IDs for diagnostics.
            "_slot_status_current": {f"P{sid}": (tid is not None) for sid, tid in occupancy.items()},
            # Route cumulative evidence is keyed by REAL IDs when permanent mapping is active.
            "_slot_cumulative_state": {str(sid): state for sid, state in cumulative_slot_state.items()},
            "_slot_cumulative_eligible": self._cumulative_eligibility_map(visible_slots, w, h),
            "_stable_plates_current": [
                {"plate": st.stable_plate, "color": st.stable_plate_color}
                for st in active_states if st.stable_plate
            ],
        }

        stats.update(permanent_diag)

        # Publish logical recognition results before drawing the annotated frame.
        # A slow PIL/OpenCV render can no longer delay cumulative counting or runtime statistics.
        if logic_result_callback is not None:
            try:
                logic_result_callback(stats, new_events)
            except Exception as e:
                print(f"[WARN] 识别结果回调异常（不影响画面继续处理）：{e}")

        t = time.perf_counter()
        annotated = self._render(frame, visible_slots, active_states, occupancy, self._main_fps_ema, M is not None)
        render_ms = (time.perf_counter() - t) * 1000.0
        dt = max(1e-6, time.perf_counter() - t_all)
        fps = 1.0 / dt
        self._main_fps_ema = 0.85 * self._main_fps_ema + 0.15 * fps

        # Final performance fields include rendering cost; recognition/count fields are unchanged.
        stats["fps_processing"] = round(fps, 2)
        stats["fps_processing_ema"] = round(self._main_fps_ema, 2)
        stats["processing_ms"] = round(dt * 1000.0, 2)
        stats["render_ms"] = round(render_ms, 2)
        stats["capture_to_output_ms"] = round(
            max(0.0, (time.perf_counter() - float(capture_perf)) * 1000.0), 2
        ) if capture_perf and capture_perf > 0 else 0.0
        return annotated, stats, new_events

# ---------------------- platform latest-frame adapter -----------------------


class LatestFrameInferenceRuntime:
    """Non-blocking platform adapter: keep only the newest submitted frame.

    Use this when a platform pushes frames faster than the detector can process them. submit_frame()
    never builds a FIFO backlog; an unprocessed older frame is overwritten by the newest one.
    The platform may poll get_latest_result() for the newest completed annotated frame/stats/events.
    """

    def __init__(self, engine: UAVParkingRealtime, fps_hint: float = 0.0) -> None:
        self.engine = engine
        self.fps_hint = float(fps_hint)
        self._cv = threading.Condition()
        self._pending: Optional[Tuple[np.ndarray, float, int]] = None
        self._latest_result: Optional[Tuple[np.ndarray, dict, List[dict], int]] = None
        self._stop = False
        self._thread = threading.Thread(target=self._run, name="uav-latest-frame-infer", daemon=True)
        self.submitted_frames = 0
        self.processed_frames = 0
        self.dropped_input_frames = 0
        self._thread.start()

    def submit_frame(self, frame: np.ndarray, capture_perf: float = 0.0) -> int:
        if frame is None or frame.size == 0:
            return self.submitted_frames
        ts = float(capture_perf) if capture_perf and capture_perf > 0 else time.perf_counter()
        with self._cv:
            self.submitted_frames += 1
            seq = self.submitted_frames
            if self._pending is not None:
                self.dropped_input_frames += 1
            self._pending = (frame.copy(), ts, seq)
            self._cv.notify()
            return seq

    def get_latest_result(self) -> Optional[Tuple[np.ndarray, dict, List[dict], int]]:
        with self._cv:
            if self._latest_result is None:
                return None
            frame, stats, events, seq = self._latest_result
            return frame.copy(), dict(stats), list(events), int(seq)

    def _run(self) -> None:
        while True:
            with self._cv:
                while self._pending is None and not self._stop:
                    self._cv.wait(timeout=0.1)
                if self._stop:
                    return
                frame, capture_perf, seq = self._pending
                self._pending = None
            try:
                annotated, stats, events = self.engine.process_frame(
                    frame, fps_hint=self.fps_hint, capture_perf=capture_perf
                )
                stats["platform_input_frames_submitted"] = int(self.submitted_frames)
                stats["platform_input_frames_processed"] = int(self.processed_frames + 1)
                stats["platform_input_frames_dropped"] = int(self.dropped_input_frames)
                with self._cv:
                    self.processed_frames += 1
                    self._latest_result = (annotated, stats, events, seq)
            except Exception as e:
                print(f"[LatestFrameRuntime] 推理异常：{e}")

    def close(self) -> None:
        with self._cv:
            self._stop = True
            self._cv.notify_all()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)


# -------------------------- one-click program entry -------------------------


def validate_runtime_paths(
    source: str | int,
    vehicle_model: str,
    parking_model: str,
    plate_det_model: str,
    plate_rec_model: str,
    font_path: str,
) -> None:
    """启动前一次性检查本地文件，错误直接指出具体路径。"""
    checks = [
        ("车辆检测模型", vehicle_model),
        ("停车位模型", parking_model),
        ("车牌检测模型", plate_det_model),
        ("车牌识别模型", plate_rec_model),
        ("中文字体", font_path),
    ]
    for label, path in checks:
        if not Path(path).exists():
            raise FileNotFoundError(f"{label}不存在：{path}\n请到代码顶部【一键运行配置区】修改路径。")

    if isinstance(source, str):
        low = source.lower()
        is_stream = low.startswith(("rtsp://", "rtsps://", "http://", "https://", "udp://", "tcp://"))
        if not is_stream and not Path(source).exists():
            raise FileNotFoundError(f"视频文件不存在：{source}\n请修改代码顶部 VIDEO_SOURCE。")


def print_startup_config(source: str | int, cfg: RealtimeConfig, save_video_path: str) -> None:
    print("=" * 78)
    print("无人机停车场实时巡检识别 - 一键运行")
    print(f"视频源          : {source}")
    print(f"车辆模型        : {cfg.vehicle_model}")
    print(f"停车位模型      : {cfg.parking_model}")
    print(f"车牌检测模型    : {cfg.plate_det_model}")
    print(f"车牌识别模型    : {cfg.plate_rec_model}")
    print(f"GPU             : {cfg.device}")
    print(f"停车位推理      : {'异步' if cfg.async_parking else '同步'} | 基础间隔 {cfg.parking_async_interval_s:.2f}s")
    print(f"车牌推理        : {'异步' if cfg.async_plate else '同步'} | 每主帧最多提交 {cfg.max_plate_rois_per_frame} ROI")
    print(f"结果视频        : {save_video_path if SAVE_RESULT_VIDEO else '关闭'}")
    if SAVE_RESULT_VIDEO:
        fps_text = "自动" if float(RESULT_VIDEO_FPS) <= 0 else f"{float(RESULT_VIDEO_FPS):.2f} FPS"
        print(f"录像输出FPS     : {fps_text}")
        print(f"录像真实时间轴  : {'开启' if RESULT_VIDEO_KEEP_REALTIME else '关闭'}")
    print(f"事件日志        : {cfg.event_log if cfg.event_log else '关闭'}")
    print(f"路侧永久编号    : {'开启' if cfg.roadside_slot_map_enabled else '关闭'}")
    if cfg.roadside_slot_map_enabled:
        print(f"车位顺序JSON    : {cfg.roadside_slot_map_path}")
        print(f"画面顺序模式    : {cfg.roadside_screen_order} | 起始车位 {cfg.roadside_start_real_id or '由JSON决定'}")
        print("多航段控制      : 平台调用 engine.switch_roadside_segment()；本地窗口按 N 切下一航段")
    print("按 Q 或 ESC 退出。")
    print("=" * 78)


def main() -> None:
    # 所有参数直接来自文件顶部【一键运行配置区】。
    source = _resolve_video_source(VIDEO_SOURCE)
    vehicle_model = _resolve_local_path(VEHICLE_MODEL_PATH)
    parking_model = _resolve_local_path(PARKING_MODEL_PATH)
    plate_det_model = _resolve_local_path(PLATE_DET_MODEL_PATH)
    plate_rec_model = _resolve_local_path(PLATE_REC_MODEL_PATH)
    font_path = _resolve_local_path(FONT_PATH)
    event_log = _resolve_local_path(EVENT_LOG_PATH) if EVENT_LOG_PATH else ""
    save_video_path = _resolve_local_path(RESULT_VIDEO_PATH) if RESULT_VIDEO_PATH else ""
    roadside_slot_map_path = _resolve_local_path(ROADSIDE_SLOT_MAP_PATH) if ROADSIDE_SLOT_MAP_PATH else ""

    validate_runtime_paths(
        source,
        vehicle_model,
        parking_model,
        plate_det_model,
        plate_rec_model,
        font_path,
    )

    if SAVE_RESULT_VIDEO and save_video_path:
        Path(save_video_path).parent.mkdir(parents=True, exist_ok=True)
    if event_log:
        Path(event_log).parent.mkdir(parents=True, exist_ok=True)

    cv2.setNumThreads(1)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass

    cfg = RealtimeConfig(
        vehicle_model=vehicle_model,
        parking_model=parking_model,
        plate_det_model=plate_det_model,
        plate_rec_model=plate_rec_model,
        device=DEVICE,
        vehicle_imgsz=VEHICLE_IMGSZ,
        vehicle_conf=VEHICLE_CONF,
        parking_conf=PARKING_CONF,
        plate_conf=PLATE_CONF,
        iou=IOU_THRESHOLD,
        vehicle_classes=tuple(VEHICLE_CLASSES),
        parking_every_n=max(1, int(PARKING_EVERY_N_FRAMES)),
        max_plate_rois_per_frame=max(1, int(MAX_PLATE_ROIS_PER_FRAME)),
        plate_retry_seconds=max(0.05, float(PLATE_RETRY_SECONDS)),
        plate_min_hits=max(1, int(PLATE_MIN_HITS)),
        stationary_seconds=max(0.2, float(STATIONARY_SECONDS)),
        stationary_fast_seconds=max(0.2, min(float(STATIONARY_FAST_SECONDS), float(STATIONARY_SECONDS))),
        stationary_strong_residual_factor=clamp(float(STATIONARY_STRONG_RESIDUAL_FACTOR), 0.25, 0.90),
        stationary_residual_ratio=max(0.005, float(STATIONARY_RESIDUAL_RATIO)),
        normal_min_vehicle_overlap=clamp(float(NORMAL_MIN_VEHICLE_OVERLAP), 0.05, 0.95),
        violation_max_vehicle_overlap=clamp(float(VIOLATION_MAX_VEHICLE_OVERLAP), 0.0, 0.90),
        status_confirm_frames=max(1, int(STATUS_CONFIRM_FRAMES)),
        slot_max_missed_detects=max(1, int(SLOT_MAX_MISSED_DETECTS)),
        slot_reid_retention_frames=max(1, int(SLOT_REID_RETENTION_FRAMES)),
        async_parking=bool(ASYNC_PARKING_ENABLED),
        async_plate=bool(ASYNC_PLATE_ENABLED),
        parking_async_interval_s=max(0.10, float(PARKING_ASYNC_INTERVAL_SECONDS)),
        parking_result_max_age_s=max(0.20, float(PARKING_RESULT_MAX_AGE_SECONDS)),
        parking_accept_first_fresh_unaligned=bool(PARKING_ACCEPT_FIRST_FRESH_UNALIGNED),
        parking_fresh_unaligned_fallback=bool(PARKING_FRESH_UNALIGNED_FALLBACK),
        parking_fresh_unaligned_max_age_s=max(0.05, float(PARKING_FRESH_UNALIGNED_MAX_AGE_SECONDS)),
        parking_fresh_unaligned_max_frame_gap=max(1, int(PARKING_FRESH_UNALIGNED_MAX_FRAME_GAP)),
        parking_early_first_frame_submit=bool(PARKING_EARLY_FIRST_FRAME_SUBMIT),
        parking_worker_max_width=max(0, int(PARKING_WORKER_MAX_WIDTH)),
        parking_first_frame_bootstrap=bool(PARKING_FIRST_FRAME_BOOTSTRAP),
        parking_first_frame_timeout_s=max(0.05, float(PARKING_FIRST_FRAME_TIMEOUT_SECONDS)),
        auto_warmup_on_init=bool(AUTO_WARMUP_ON_INIT),
        vehicle_warmup_passes=max(1, int(VEHICLE_WARMUP_PASSES)),
        warmup_tracker_on_init=bool(WARMUP_TRACKER_ON_INIT),
        warmup_dummy_height=max(64, int(WARMUP_DUMMY_HEIGHT)),
        warmup_dummy_width=max(64, int(WARMUP_DUMMY_WIDTH)),
        plate_async_queue_size=max(1, int(PLATE_ASYNC_QUEUE_SIZE)),
        plate_result_max_age_s=max(0.20, float(PLATE_RESULT_MAX_AGE_SECONDS)),
        serialize_aux_gpu=bool(SERIALIZE_AUX_GPU_INFERENCE),
        vehicle_half=bool(VEHICLE_USE_FP16),
        motion_process_width=max(320, int(MOTION_PROCESS_WIDTH)),
        motion_max_corners=max(80, int(MOTION_MAX_CORNERS)),
        adaptive_aux_load_shedding=bool(ADAPTIVE_AUX_LOAD_SHEDDING),
        aux_low_fps_threshold=max(0.5, float(AUX_LOW_FPS_THRESHOLD)),
        aux_low_fps_parking_interval=max(0.20, float(AUX_LOW_FPS_PARKING_INTERVAL)),
        roadside_slot_map_enabled=bool(ROADSIDE_SLOT_MAP_ENABLED),
        roadside_slot_map_path=roadside_slot_map_path,
        roadside_slot_min_hits=max(1, int(ROADSIDE_SLOT_MIN_HITS)),
        roadside_screen_order=str(ROADSIDE_SCREEN_ORDER),
        roadside_start_real_id=str(ROADSIDE_START_REAL_ID),
        roadside_direction_min_samples=max(2, int(ROADSIDE_DIRECTION_MIN_SAMPLES)),
        roadside_min_motion_pixels=max(0.2, float(ROADSIDE_MIN_MOTION_PIXELS)),
        roadside_gap_skip_ratio=clamp(float(ROADSIDE_GAP_SKIP_RATIO), 1.25, 4.0),
        roadside_max_inferred_skip=max(0, int(ROADSIDE_MAX_INFERRED_SKIP)),
        roadside_assign_edge_margin_ratio=clamp(float(ROADSIDE_ASSIGN_EDGE_MARGIN_RATIO), 0.0, 0.12),
        roadside_show_local_id=bool(ROADSIDE_SHOW_LOCAL_ID),
        roadside_real_id_font_size=max(14, int(ROADSIDE_REAL_ID_FONT_SIZE)),
        roadside_local_id_font_size=max(10, int(ROADSIDE_LOCAL_ID_FONT_SIZE)),
        roadside_pending_id_font_size=max(10, int(ROADSIDE_PENDING_ID_FONT_SIZE)),
        font_path=font_path,
        event_log=event_log or None,
    )

    print_startup_config(source, cfg, save_video_path)
    print("键盘控制：R=清空本次航线累计  Q/Esc=退出（仅在开启本地显示时有效）")
    print("[DEPLOY] 已移除平台网络传输/远程控制代码，推理链路可独立部署。")
    print("[DEPLOY] 核心接口：UAVParkingRealtime.process_frame(frame, fps_hint=...)，frame 为 OpenCV BGR 图像。")
    print("[6.8.3] 车位唯一ID去重：同帧mask去重 + 全局一对一关联 + 重复P-ID在线合并 + 真实编号唯一约束。")
    print("[6.8.2] 真实车位编号醒目显示 + tcw0914.pt 自定义增强检查点兼容已启用。")
    print(f"[COMPAT] UAVImageOnlyOcclusion={'Albumentations原类兼容' if A is not None else '最小pickle兼容'}")
    print("[6.8.2] 累计占用仅由静止车辆确认；行驶车辆经过车位不写入航线累计。")
    print("[6.7.0] 路侧无GPS映射：固定航线 + 真实编号顺序 + 视觉运动方向 + 漏检间距容错。")
    print("[6.7.0] 模型/BoT-SORT 在 READY 前预热；首帧停车位默认不阻塞，车辆主链优先。")

    engine = UAVParkingRealtime(cfg)
    cap = LatestFrameCapture(source).start()
    accumulator = PatrolAccumulator(
        occupied_confirm_frames=CUMULATIVE_OCCUPIED_CONFIRM_FRAMES,
        empty_confirm_frames=CUMULATIVE_EMPTY_CONFIRM_FRAMES,
        max_gap_seconds=CUMULATIVE_CONFIRM_MAX_GAP_SECONDS,
    )

    writer: Optional[AsyncRealTimeResultVideoWriter] = None
    last_seq = -1
    last_stats_t = 0.0
    latest_public_stats: dict = {
        "fps_processing": 0.0,
        "processing_ms": 0.0,
        "camera_motion_ok": False,
        "parking_slots_current": 0,
        "parking_slots_occupied_current": 0,
        "parking_slots_free_current": 0,
        "parking_slots_stationary_occupied_current": 0,
        "parking_slots_transient_current": 0,
        "parking_slot_local_ids_seen": 0,
        "parking_slot_reid_revived": 0,
        "parking_bootstrap_success": False,
        "parking_bootstrap_ms": 0.0,
        "stationary_effective_seconds_avg": 0.0,
        "vehicles_current": 0,
        "vehicles_stationary_current": 0,
        "normal_parked_current": 0,
        "violation_current": 0,
    }

    last_display_size: Optional[Tuple[int, int]] = None
    if DISPLAY_WINDOW:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)

    def reset_patrol() -> None:
        """Reset only local detection/tracking/cumulative state; no external control dependency."""
        nonlocal latest_public_stats
        accumulator.reset()
        engine.reset_runtime_tracking(reset_events=True, reset_slot_ids=True)
        latest_public_stats = {
            "fps_processing": 0.0,
            "processing_ms": 0.0,
            "camera_motion_ok": False,
            "parking_slots_current": 0,
            "parking_slots_occupied_current": 0,
            "parking_slots_free_current": 0,
            "parking_slots_stationary_occupied_current": 0,
            "parking_slots_transient_current": 0,
            "parking_slot_local_ids_seen": 0,
            "parking_slot_reid_revived": 0,
            "parking_bootstrap_success": False,
            "parking_bootstrap_ms": 0.0,
            "stationary_effective_seconds_avg": 0.0,
            "vehicles_current": 0,
            "vehicles_stationary_current": 0,
            "normal_parked_current": 0,
            "violation_current": 0,
        }
        print("[CONTROL] 本次航线累计/跟踪状态已重置")

    def on_logic_result_ready(stats: dict, events: List[dict]) -> None:
        """Update local cumulative results as soon as logic output is ready, before annotation rendering."""
        nonlocal latest_public_stats
        latest_public_stats = stats
        accumulator.update(stats, events, time.time())

    try:
        while True:
            frame, seq, capture_perf = cap.read_latest(last_seq)
            if frame is None:
                if cap.stopped:
                    break
                time.sleep(0.001)
                continue
            last_seq = seq

            if not engine._warmup_done:
                # Fallback only: normal deployment should already be warm before the source starts.
                engine.warmup(frame.shape[:2], warmup_tracker=True)

            annotated, stats, events = engine.process_frame(
                frame,
                fps_hint=cap.source_fps,
                logic_result_callback=on_logic_result_ready,
                capture_perf=capture_perf,
            )
            # 回调负责即时更新累计结果；这里保留最终性能统计字段并叠加航线累计。
            latest_public_stats = stats
            cumulative_snapshot = accumulator.snapshot()
            annotated = render_cumulative_overlay(annotated, cumulative_snapshot, font_path)

            if SAVE_RESULT_VIDEO and save_video_path:
                if writer is None:
                    h, w = annotated.shape[:2]
                    configured_fps = float(RESULT_VIDEO_FPS)
                    if configured_fps > 0:
                        out_fps = configured_fps
                    elif cap.source_fps > 1.0:
                        out_fps = min(float(cap.source_fps), 15.0)
                    else:
                        out_fps = 15.0

                    writer = AsyncRealTimeResultVideoWriter(
                        path=save_video_path,
                        source_frame_size=(w, h),
                        output_fps=out_fps,
                        source_fps=cap.source_fps,
                        is_file=cap.is_file,
                        clock_provider=cap.get_clock,
                        keep_realtime=RESULT_VIDEO_KEEP_REALTIME,
                        max_width=RESULT_VIDEO_MAX_WIDTH,
                    )
                    print(
                        f"[VIDEO] 结果录像已启动：{save_video_path} | "
                        f"{out_fps:.2f} FPS | 保存尺寸={writer.frame_size[0]}x{writer.frame_size[1]} | "
                        f"异步真实时间轴={'ON' if RESULT_VIDEO_KEEP_REALTIME else 'OFF'}"
                    )
                writer.submit(annotated, source_seq=seq, capture_perf=capture_perf)

            now_perf = time.perf_counter()
            if now_perf - last_stats_t >= max(0.1, float(STATS_INTERVAL_SECONDS)):
                public_stats = {k: v for k, v in latest_public_stats.items() if not str(k).startswith("_")}
                public_stats.update({
                    "progress_percent": round(cap.progress_percent, 1),
                    "cumulative_total_spaces": cumulative_snapshot.get("cumulative_total_spaces", 0),
                    "cumulative_occupied_spaces": cumulative_snapshot.get("cumulative_occupied_spaces", 0),
                    "cumulative_empty_spaces": cumulative_snapshot.get("cumulative_empty_spaces", 0),
                    "cumulative_occupancy_rate": cumulative_snapshot.get("cumulative_occupancy_rate", 0.0),
                    "cumulative_recognized_plate_count": cumulative_snapshot.get("cumulative_recognized_plate_count", 0),
                    "cumulative_violation_count": cumulative_snapshot.get("cumulative_violation_count", 0),
                })
                print(json.dumps(public_stats, ensure_ascii=False))
                for e in events:
                    print("[EVENT]", json.dumps(e, ensure_ascii=False))
                last_stats_t = now_perf

            if DISPLAY_WINDOW:
                display_frame = _fit_frame_for_display(annotated)
                dh, dw = display_frame.shape[:2]
                display_size = (dw, dh)
                if display_size != last_display_size:
                    cv2.resizeWindow(WINDOW_NAME, dw, dh)
                    last_display_size = display_size
                cv2.imshow(WINDOW_NAME, display_frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q"), ord("Q")):
                    break
                if key in (ord("r"), ord("R")):
                    reset_patrol()
                if key in (ord("n"), ord("N")):
                    engine.switch_roadside_segment()
    except KeyboardInterrupt:
        print("收到 Ctrl+C，正在退出。")
    finally:
        cap.release()
        try:
            engine.close()
        except Exception as e:
            print(f"[WARN] 关闭异步推理 worker 时出现异常：{e}")
        if writer is not None:
            duration_s = writer.duration_seconds
            frame_count = writer.frames_written
            out_fps = writer.output_fps
            writer.release()
            print(f"[VIDEO] 录像已保存：{save_video_path}")
            print(f"[VIDEO] 输出帧数={frame_count}  FPS={out_fps:.2f}  时长≈{duration_s:.2f}s")
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
