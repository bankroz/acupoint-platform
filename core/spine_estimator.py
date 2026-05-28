"""
虚拟脊柱构建器 - 从肩髋4个关键点推算躯干脊柱路径

原理:
  MediaPipe Pose 在躯干上仅有4个关键点（11=左肩, 12=右肩, 23=左髋, 24=右髋）。
  本模块通过肩中点和髋中点构建虚拟脊柱，模拟人体S形生理弯曲，
  从而推算 C7~L5 各椎骨的三维空间位置。

解剖基础:
  - 脊柱分为: 颈椎(C1-C7) + 胸椎(T1-T12) + 腰椎(L1-L5) = 24节活动椎骨
  - C7(大椎GV14) 在肩中点稍上方
  - T7(至阳GV9) 在肩髋中点
  - L2(命门GV4) 在髋中点稍上方
  - L4(腰阳关GV3) 在髋中点

S形弯曲参数:
  - 颈椎段: 前凸 (向前弯曲) → 正偏移
  - 胸椎段: 后凸 (向后弯曲) → 负偏移(向后)
  - 腰椎段: 前凸 (向前弯曲) → 正偏移

输出:
  - 各椎骨的3D坐标
  - 前后正中线路径
  - 用于躯干穴位定位的参考框架
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional


@dataclass
class SpineResult:
    """脊柱推算结果"""
    # 关键参考点 (3D 世界坐标 [x, y, z])
    shoulder_left: np.ndarray = field(default_factory=lambda: np.zeros(3))
    shoulder_right: np.ndarray = field(default_factory=lambda: np.zeros(3))
    shoulder_mid: np.ndarray = field(default_factory=lambda: np.zeros(3))
    hip_left: np.ndarray = field(default_factory=lambda: np.zeros(3))
    hip_right: np.ndarray = field(default_factory=lambda: np.zeros(3))
    hip_mid: np.ndarray = field(default_factory=lambda: np.zeros(3))

    # 虚拟椎骨坐标 (24节: C7 + T1-T12 + L1-L5)
    vertebrae: Dict[str, np.ndarray] = field(default_factory=dict)
    # 前正中线路径 (任脉路径，N个插值点)
    front_midline: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    # 后正中线路径 (督脉路径，N个插值点)
    back_midline: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))

    # 尺寸参考
    shoulder_width: float = 0.0    # 肩宽 (米)
    hip_width: float = 0.0         # 髋宽 (米)
    spine_length: float = 0.0      # 脊柱总长 (米，肩中→髋中)
    torso_width: float = 0.0       # 躯干宽度估算 (米)

    # 身高估算
    estimated_height: float = 0.0  # 估算身高 (米)

    # 是否有效
    is_valid: bool = False


class SpineEstimator:
    """
    虚拟脊柱构建器

    用法:
        estimator = SpineEstimator()
        spine = estimator.build(pose_world_landmarks)
        print(f"大椎(GV14): {spine.vertebrae['C7']}")
        print(f"命门(GV4):  {spine.vertebrae['L2']}")
    """

    # 椎骨标签 (C7, T1-T12, L1-L5)
    VERTEBRA_LABELS = (
        ["C7"] +
        [f"T{i}" for i in range(1, 13)] +
        [f"L{i}" for i in range(1, 6)]
    )

    # 实际椎骨节数 (C7 + T1-T12 + L1-L5 = 18，从肩部开始无法推算C1-C6)
    CERVICAL_START = 0    # C7 是第一个可推算的椎骨
    THORACIC_START = 1    # T1
    LUMBAR_START = 13     # L1
    TOTAL_VERTEBRAE = 18

    def __init__(self,
                 cervical_lordosis: float = 0.15,
                 thoracic_kyphosis: float = 0.35,
                 lumbar_lordosis: float = 0.50,
                 surface_depth: float = 0.05):
        """
        Args:
            cervical_lordosis: 颈椎前凸系数 (相对于脊柱长度的偏移比例)
            thoracic_kyphosis: 胸椎后凸系数
            lumbar_lordosis: 腰椎前凸系数
            surface_depth: 体表到脊柱的深度比例（用于前正中线计算）
        """
        self.cervical_lordosis = cervical_lordosis
        self.thoracic_kyphosis = thoracic_kyphosis
        self.lumbar_lordosis = lumbar_lordosis
        self.surface_depth = surface_depth

    def build(self, pose_world_landmarks: np.ndarray) -> SpineResult:
        """
        从 Pose world landmarks 构建虚拟脊柱

        Args:
            pose_world_landmarks: shape (33, 4) [x, y, z, visibility]
                                  来自 MediaPipe pose_world_landmarks

        Returns:
            SpineResult 包含所有推算数据
        """
        spine = SpineResult()

        # 提取关键点
        spine.shoulder_left = pose_world_landmarks[11, :3]
        spine.shoulder_right = pose_world_landmarks[12, :3]
        spine.hip_left = pose_world_landmarks[23, :3]
        spine.hip_right = pose_world_landmarks[24, :3]

        # 检查关键点可见性
        vis_left_shoulder = pose_world_landmarks[11, 3]
        vis_right_shoulder = pose_world_landmarks[12, 3]
        vis_left_hip = pose_world_landmarks[23, 3]
        vis_right_hip = pose_world_landmarks[24, 3]

        min_vis = min(vis_left_shoulder, vis_right_shoulder, vis_left_hip, vis_right_hip)
        if min_vis < 0.3:
            return spine  # 关键点不可靠，返回空结果

        spine.is_valid = True

        # 计算中点
        spine.shoulder_mid = (spine.shoulder_left + spine.shoulder_right) / 2.0
        spine.hip_mid = (spine.hip_left + spine.hip_right) / 2.0

        # 计算尺寸
        spine.shoulder_width = np.linalg.norm(
            spine.shoulder_right - spine.shoulder_left
        )
        spine.hip_width = np.linalg.norm(
            spine.hip_right - spine.hip_left
        )

        # 脊柱直线长度
        straight_spine = spine.hip_mid - spine.shoulder_mid
        spine.spine_length = np.linalg.norm(straight_spine)

        if spine.spine_length < 0.1:
            spine.is_valid = False
            return spine

        # 估算躯干宽度（取肩宽和髋宽的较大值）
        spine.torso_width = max(spine.shoulder_width, spine.hip_width)

        # 估算身高：脊柱段 + 头颈 + 下肢
        head_neck = spine.spine_length * 0.3
        legs = spine.spine_length * 0.9
        spine.estimated_height = spine.spine_length + head_neck + legs

        # 构建椎骨坐标
        spine.vertebrae = self._build_vertebrae(
            spine.shoulder_mid, spine.hip_mid, straight_spine, spine.spine_length
        )

        # 构建前后正中线
        spine.front_midline, spine.back_midline = self._build_midlines(
            spine.shoulder_mid, spine.hip_mid,
            straight_spine, spine.spine_length, spine.torso_width
        )

        return spine

    def _build_vertebrae(self, shoulder_mid: np.ndarray,
                         hip_mid: np.ndarray,
                         direction: np.ndarray,
                         length: float) -> Dict[str, np.ndarray]:
        """
        沿脊柱方向插值24节椎骨，叠加S形弯曲

        使用正弦波叠加来模拟生理弯曲：
        - 颈椎段 (C7-T1): 前凸（向前）
        - 胸椎段 (T2-T12): 后凸（向后）
        - 腰椎段 (L1-L5): 前凸（向前）
        """
        direction_norm = direction / length

        # 前后轴：用 world 坐标中的 YZ 平面法向量估算
        # MediaPipe world: Y朝上, Z朝前(人物前方), X朝右
        up = np.array([0, 1, 0])
        forward = np.cross(direction_norm, up)
        if np.linalg.norm(forward) < 0.001:
            forward = np.array([0, 0, -1])
        forward = forward / np.linalg.norm(forward)
        # 确保Z分量为正（人物前方）
        if forward[2] > 0:
            forward = -forward

        vertebrae = {}

        for i in range(self.TOTAL_VERTEBRAE):
            t = i / (self.TOTAL_VERTEBRAE - 1)  # 0=C7, 1=L5
            label = self.VERTEBRA_LABELS[i]

            # 直线插值位置
            linear_pos = shoulder_mid + t * direction

            # S形弯曲偏移
            curve_offset = self._spine_curve(t, length)

            # 最终位置 = 直线位置 + 前后方向偏移
            pos = linear_pos + forward * curve_offset
            vertebrae[label] = pos

        return vertebrae

    def _spine_curve(self, t: float, spine_length: float) -> float:
        """
        S形脊柱弯曲函数

        使用分段正弦函数:
        - 颈椎段 (t=0~0.1): 向前凸 (正偏移)
        - 胸椎段 (t=0.1~0.6): 向后凸 (负偏移)
        - 腰椎段 (t=0.6~1.0): 向前凸 (正偏移)

        Returns:
            前后偏移量 (正=向前, 负=向后)
        """
        base = spine_length * 0.02  # 基础偏移量 = 脊柱长的2%

        # 颈椎段: t=0~0.08
        cervical = 0
        if t <= 0.08:
            cervical = self.cervical_lordosis * base * np.sin(np.pi * t / 0.08)

        # 胸椎段: t=0.08~0.65
        thoracic = 0
        if 0.08 < t <= 0.65:
            local_t = (t - 0.08) / (0.65 - 0.08)
            thoracic = -self.thoracic_kyphosis * base * np.sin(np.pi * local_t)

        # 腰椎段: t=0.65~1.0
        lumbar = 0
        if t > 0.65:
            local_t = (t - 0.65) / 0.35
            lumbar = self.lumbar_lordosis * base * np.sin(np.pi * local_t)

        return cervical + thoracic + lumbar

    def _build_midlines(self, shoulder_mid, hip_mid,
                        direction, length, torso_width):
        """
        构建前正中线（任脉路径）和后正中线（督脉路径）

        前正中线 = 脊柱路径 + 向前偏移(胸廓厚度)
        后正中线 = 脊柱路径 + 向后偏移(极小，脊柱贴近背面)
        """
        n_pts = 50  # 插值点数
        front_pts = []
        back_pts = []

        direction_norm = direction / length

        # 前后方向
        up = np.array([0, 1, 0])
        forward = np.cross(direction_norm, up)
        if np.linalg.norm(forward) < 0.001:
            forward = np.array([0, 0, -1])
        forward = forward / np.linalg.norm(forward)
        if forward[2] > 0:
            forward = -forward

        # 胸廓前后径估算（用肩宽的0.4）
        chest_depth = torso_width * 0.4

        for i in range(n_pts):
            t = i / (n_pts - 1)
            linear_pos = shoulder_mid + t * direction
            curve = self._spine_curve(t, length)

            # 椎骨位置
            vertebra_pos = linear_pos + forward * curve

            # 前正中线（体表）= 椎骨 + 胸廓深度向前
            front_pos = vertebra_pos + forward * chest_depth
            front_pts.append(front_pos)

            # 后正中线（体表）= 椎骨 + 微小向后偏移
            back_pos = vertebra_pos - forward * (self.surface_depth * length)
            back_pts.append(back_pos)

        return np.array(front_pts), np.array(back_pts)

    def get_vertebra_index(self, label: str) -> Optional[int]:
        """获取椎骨在列表中的索引"""
        if label in self.VERTEBRA_LABELS:
            return self.VERTEBRA_LABELS.index(label)
        return None

    @staticmethod
    def vertebra_to_acupoint_name(level: str) -> Optional[str]:
        """
        椎骨等级 → 督脉穴位名
        C7=大椎GV14, T3=身柱GV12, T7=至阳GV9, L2=命门GV4, L4=腰阳关GV3
        """
        mapping = {
            "C7": "大椎GV14",
            "T3": "身柱GV12",
            "T7": "至阳GV9",
            "L2": "命门GV4",
            "L4": "腰阳关GV3",
        }
        return mapping.get(level)
