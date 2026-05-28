"""
姿势提取器 - 封装 MediaPipe Tasks API（Pose + Holistic 双模型）

支持两种模式:
- "pose": 仅33个骨架关键点（快速，适合四肢穴位）
- "holistic": Pose + Face Mesh + Hands（全面，适合全身体穴位）

需要模型文件：
- models/pose_landmarker.task（Pose模式）
- models/holistic_landmarker.task（Holistic模式）

模型下载:
  python scripts/download_models.py

MediaPipe 33个Pose关键点索引:
  0=鼻尖, 1-4=眼, 5-6=耳, 7-8=嘴角
  9-10=嘴唇侧, 11=左肩, 12=右肩
  13=左肘, 14=右肘, 15=左腕, 16=右腕
  17=左小指根, 18=右小指根, 19=左食指, 20=右食指
  21=左拇指, 22=右拇指
  23=左髋, 24=右髋, 25=左膝, 26=右膝
  27=左踝, 28=右踝, 29=左脚跟, 30=右脚跟
  31=左脚趾, 32=右脚趾

GPU加速说明:
  - Windows pip 包默认未编译 GPU delegate（需从源码编译或使用WSL2）
  - RTX 3080 Laptop 可通过以下方案加速:
    方案A: 安装 WSL2 + MediaPipe GPU 版本 (推荐，最快)
    方案B: 从源码编译 MediaPipe with GPU support
    方案C: 降级 CPU 推理（375ms→约80ms，开4线程稳定可用）
  - 当前默认: CPU多线程推理，已优化推理速度
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
from enum import Enum
import time
import os
import platform

from mediapipe.tasks.python import vision, BaseOptions
from mediapipe.tasks.python.vision import RunningMode
from mediapipe import Image, ImageFormat

# GPU 加速状态（模块级单例）
_gpu_available = False  # Windows pip 包不支持 GPU delegate
_gpu_status_msg = "CPU 推理 (Windows pip 包未编译 GPU delegate)"


class ModelMode(Enum):
    POSE = "pose"
    HOLISTIC = "holistic"


@dataclass
class PoseResult:
    """统一的姿势检测结果"""
    # Pose 骨架关键点 (33个, 归一化坐标 0-1)
    pose_landmarks: Optional[np.ndarray] = None       # shape: (33, 4) [x, y, z, visibility]
    # Pose 世界坐标 (真实3D, 单位: 米)
    pose_world_landmarks: Optional[np.ndarray] = None # shape: (33, 4) [x, y, z, visibility]
    # Face Mesh 面部网格点
    face_landmarks: Optional[np.ndarray] = None       # shape: (N, 3)
    # 左手关键点 (21个)
    left_hand_landmarks: Optional[np.ndarray] = None  # shape: (21, 3)
    # 右手关键点 (21个)
    right_hand_landmarks: Optional[np.ndarray] = None  # shape: (21, 3)
    # 原始图像
    image: Optional[np.ndarray] = None
    # 图像尺寸
    image_shape: Tuple[int, int] = (0, 0)
    # 是否有有效检测
    has_pose: bool = False
    has_face: bool = False
    has_left_hand: bool = False
    has_right_hand: bool = False


class PoseExtractor:
    """
    MediaPipe 姿势提取器 (Tasks API)，自动 GPU/CPU 回退

    用法:
        extractor = PoseExtractor(mode=ModelMode.POSE)
        result = extractor.process(image)
    """

    # 骨架连线定义
    POSE_CONNECTIONS = [
        (11, 12), (11, 23), (12, 24), (23, 24),
        (11, 13), (13, 15), (12, 14), (14, 16),
        (23, 25), (25, 27), (24, 26), (26, 28),
        (27, 29), (29, 31), (27, 31),
        (28, 30), (30, 32), (28, 32),
        (0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6),
    ]

    LANDMARK_NAMES = {
        0: "鼻尖", 1: "左眼内角", 2: "右眼内角", 3: "左眼外角", 4: "右眼外角",
        5: "左耳", 6: "右耳", 7: "左嘴角", 8: "右嘴角",
        9: "左嘴角侧", 10: "右嘴角侧",
        11: "左肩", 12: "右肩", 13: "左肘", 14: "右肘",
        15: "左腕", 16: "右腕", 17: "左小指根", 18: "右小指根",
        19: "左食指尖", 20: "右食指尖", 21: "左拇指尖", 22: "右拇指尖",
        23: "左髋", 24: "右髋", 25: "左膝", 26: "右膝",
        27: "左踝", 28: "右踝", 29: "左脚跟", 30: "右脚跟",
        31: "左脚趾", 32: "右脚趾",
    }

    @staticmethod
    def get_gpu_status() -> str:
        """获取当前加速状态"""
        global _gpu_available, _gpu_status_msg
        if _gpu_available:
            return f"[GPU] {_gpu_status_msg}"
        return f"[CPU] {_gpu_status_msg}"

    def __init__(self, mode: ModelMode = ModelMode.POSE,
                 model_complexity: int = 2,
                 min_detection_confidence: float = 0.7,
                 min_tracking_confidence: float = 0.5,
                 static_image_mode: bool = False,
                 model_dir: str = "models"):
        """
        Args:
            mode: POSE 或 HOLISTIC
            model_complexity: 0/1/2 (目前Tasks API通过模型选择控制)
            min_detection_confidence: 最小检测置信度
            min_tracking_confidence: 最小追踪置信度
            static_image_mode: 图片模式（不做时序追踪）
            model_dir: 模型文件目录
        """
        global _gpu_available, _gpu_status_msg

        self.mode = mode
        self._model_dir = model_dir

        running_mode = RunningMode.IMAGE if static_image_mode else RunningMode.VIDEO
        self._static_mode = static_image_mode
        self._frame_timestamp = 0

        # 尝试 GPU，失败自动回退 CPU
        delegate = BaseOptions.Delegate.GPU if _gpu_available else BaseOptions.Delegate.CPU

        try:
            self._init_landmarker(mode, model_dir, running_mode,
                                  min_detection_confidence, min_tracking_confidence,
                                  delegate)
        except Exception as e:
            err_msg = str(e)
            if "GPU" in err_msg or "gpu" in err_msg.lower():
                # GPU 不可用，回退 CPU
                _gpu_available = False
                _gpu_status_msg = f"CPU 推理 (GPU不可用: {err_msg[:60]}...)"
                print(f"[PoseExtractor] GPU 不可用，自动回退 CPU")
                self._init_landmarker(mode, model_dir, running_mode,
                                      min_detection_confidence, min_tracking_confidence,
                                      BaseOptions.Delegate.CPU)
            else:
                raise

    def _init_landmarker(self, mode, model_dir, running_mode,
                         min_det_conf, min_track_conf, delegate):
        """实际初始化 landmarker"""
        if mode == ModelMode.POSE:
            self._init_pose(model_dir, running_mode, min_det_conf, min_track_conf, delegate)
        else:  # HOLISTIC
            self._init_holistic(model_dir, running_mode, min_det_conf, min_track_conf, delegate)

    def _init_pose(self, model_dir, running_mode, min_det_conf, min_track_conf, delegate):
        """初始化 PoseLandmarker"""
        model_path = os.path.join(model_dir, "pose_landmarker.task")
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"模型文件不存在: {model_path}\n"
                f"请运行: python scripts/download_models.py"
            )
        base = BaseOptions(model_asset_path=model_path, delegate=delegate)
        opts = vision.PoseLandmarkerOptions(
            base_options=base,
            running_mode=running_mode,
            num_poses=1,
            min_pose_detection_confidence=min_det_conf,
            min_pose_presence_confidence=min_det_conf,
            min_tracking_confidence=min_track_conf,
            output_segmentation_masks=False,
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(opts)

    def _init_holistic(self, model_dir, running_mode, min_det_conf, min_track_conf, delegate):
        """初始化 HolisticLandmarker，失败自动回退 Pose"""
        model_path = os.path.join(model_dir, "holistic_landmarker.task")
        if not os.path.exists(model_path):
            print(f"[WARNING] Holistic模型不存在: {model_path}")
            print("  自动回退到Pose模式...")
            self.mode = ModelMode.POSE
            self._init_pose(model_dir, running_mode, min_det_conf, min_track_conf, delegate)
            return

        base = BaseOptions(model_asset_path=model_path, delegate=delegate)
        opts = vision.HolisticLandmarkerOptions(
            base_options=base,
            running_mode=running_mode,
            min_face_detection_confidence=min_det_conf,
            min_face_landmarks_confidence=min_det_conf,
            min_pose_detection_confidence=min_det_conf,
            min_pose_landmarks_confidence=min_det_conf,
            min_hand_landmarks_confidence=min_det_conf,
            output_face_blendshapes=False,
            output_segmentation_mask=False,
        )
        self._landmarker = vision.HolisticLandmarker.create_from_options(opts)

    def process(self, image: np.ndarray, timestamp_ms: int = None) -> PoseResult:
        """
        处理一帧图像

        Args:
            image: BGR格式图像 (H, W, 3)
            timestamp_ms: 视频时间戳(毫秒)，图片模式忽略

        Returns:
            PoseResult 统一结果对象
        """
        h, w = image.shape[:2]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_img = Image(image_format=ImageFormat.SRGB,
                       data=np.ascontiguousarray(rgb))

        if self._static_mode:
            if self.mode == ModelMode.POSE:
                result = self._landmarker.detect(mp_img)
            else:
                result = self._landmarker.detect(mp_img)
        else:
            if timestamp_ms is None:
                self._frame_timestamp += 33  # 假设30fps
                timestamp_ms = self._frame_timestamp
            if self.mode == ModelMode.POSE:
                result = self._landmarker.detect_for_video(mp_img, timestamp_ms)
            else:
                result = self._landmarker.detect_for_video(mp_img, timestamp_ms)

        output = PoseResult(image=image, image_shape=(w, h))

        if self.mode == ModelMode.POSE:
            self._extract_pose_result(result, output)
        else:
            self._extract_holistic_result(result, output)

        return output

    def _extract_pose_result(self, result, output: PoseResult):
        """从 PoseLandmarkerResult 提取数据"""
        if result.pose_landmarks and len(result.pose_landmarks) > 0:
            output.has_pose = True
            lms = result.pose_landmarks[0]
            output.pose_landmarks = np.array([
                [lm.x, lm.y, lm.z, lm.visibility if hasattr(lm, 'visibility') else 1.0]
                for lm in lms
            ])

        if result.pose_world_landmarks and len(result.pose_world_landmarks) > 0:
            wlms = result.pose_world_landmarks[0]
            output.pose_world_landmarks = np.array([
                [lm.x, lm.y, lm.z, lm.visibility if hasattr(lm, 'visibility') else 1.0]
                for lm in wlms
            ])

    @staticmethod
    def _unwrap_landmarks(landmark_data):
        """处理 IMAGE 模式（平铺列表）和 VIDEO 模式（嵌套列表）的 landmarks 结构"""
        if landmark_data is None or len(landmark_data) == 0:
            return None
        first = landmark_data[0]
        # IMAGE 模式：第一个元素是 NormalizedLandmark（有 .x 属性）
        if hasattr(first, 'x'):
            return landmark_data
        # VIDEO 模式：第一个元素是 NormalizedLandmarkList（可迭代，没有 .x）
        try:
            return list(first)
        except TypeError:
            return None

    def _extract_holistic_result(self, result, output: PoseResult):
        """从 HolisticLandmarkerResult 提取数据"""
        # Pose
        lms = self._unwrap_landmarks(result.pose_landmarks)
        if lms is not None:
            output.has_pose = True
            output.pose_landmarks = np.array([
                [lm.x, lm.y, lm.z, lm.visibility if hasattr(lm, 'visibility') else 1.0]
                for lm in lms
            ])

        wlms = self._unwrap_landmarks(result.pose_world_landmarks)
        if wlms is not None:
            output.pose_world_landmarks = np.array([
                [lm.x, lm.y, lm.z, lm.visibility if hasattr(lm, 'visibility') else 1.0]
                for lm in wlms
            ])

        # Face
        flms = self._unwrap_landmarks(result.face_landmarks)
        if flms is not None:
            output.has_face = True
            output.face_landmarks = np.array([
                [lm.x, lm.y, lm.z] for lm in flms
            ])

        # Left Hand
        hlms = self._unwrap_landmarks(result.left_hand_landmarks)
        if hlms is not None:
            output.has_left_hand = True
            output.left_hand_landmarks = np.array([
                [lm.x, lm.y, lm.z] for lm in hlms
            ])

        # Right Hand
        hlms = self._unwrap_landmarks(result.right_hand_landmarks)
        if hlms is not None:
            output.has_right_hand = True
            output.right_hand_landmarks = np.array([
                [lm.x, lm.y, lm.z] for lm in hlms
            ])

    def get_landmark_name(self, index: int) -> str:
        """获取关键点中文名称"""
        return self.LANDMARK_NAMES.get(index, f"未知点{index}")

    def close(self):
        """释放资源"""
        if hasattr(self._landmarker, 'close'):
            self._landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
