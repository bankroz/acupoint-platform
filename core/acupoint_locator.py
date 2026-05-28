"""
穴位定位器 - 综合 Pose 关键点 + 虚拟脊柱 + 骨度分寸法计算穴位坐标

支持四类穴位定位:
1. 躯干穴位: 基于虚拟脊柱 + 前后正中线推算
2. 四肢穴位: 基于 Pose 骨骼关键点 + 骨度比例法
3. 面部穴位: 基于 Holistic Face Mesh (468点)
4. 手部穴位: 基于 Holistic Hand Landmarks (每手21点)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import json

from .pose_extractor import PoseResult
from .spine_estimator import SpineEstimator, SpineResult
from .population_adapter import PopulationAdapter, PopulationProfile, AdaptationCoefficients


@dataclass
class AcupointPosition:
    """单个穴位定位结果"""
    id: str                          # 如 "ST36"
    name_cn: str                     # 如 "足三里"
    name_pinyin: str = ""
    meridian: str = ""               # 经络名
    meridian_code: str = ""          # 经络代码
    position_3d: np.ndarray = field(   # 3D坐标 [x, y, z] (世界坐标,米)
        default_factory=lambda: np.zeros(3))
    position_2d: Optional[np.ndarray] = None  # 2D图像坐标 [x, y] (像素)
    confidence: float = 0.5          # 置信度
    grade: str = "B"                 # 精度等级: A/B/C/D
    depth: Optional[float] = None    # 穴位深度 (cm)
    functions: List[str] = field(default_factory=list)
    indications: List[str] = field(default_factory=list)
    description: str = ""
    # 定位依据
    reference_landmarks: List[int] = field(default_factory=list)
    ratio: float = 0.0              # 骨度分寸比例


@dataclass
class AcupointResult:
    """穴位定位总结果"""
    acupoints: List[AcupointPosition] = field(default_factory=list)
    spine: Optional[SpineResult] = None
    image_shape: Tuple[int, int] = (0, 0)
    # 统计
    total_found: int = 0
    grade_counts: Dict[str, int] = field(default_factory=dict)
    meridian_counts: Dict[str, int] = field(default_factory=dict)

    def get_by_id(self, acupoint_id: str) -> Optional[AcupointPosition]:
        for ap in self.acupoints:
            if ap.id == acupoint_id:
                return ap
        return None

    def get_by_meridian(self, code: str) -> List[AcupointPosition]:
        return [ap for ap in self.acupoints if ap.meridian_code == code]

    def to_list(self) -> List[Dict]:
        """转为可序列化的字典列表"""
        result = []
        for ap in self.acupoints:
            d = {
                "id": ap.id, "name_cn": ap.name_cn, "name_pinyin": ap.name_pinyin,
                "meridian": ap.meridian, "meridian_code": ap.meridian_code,
                "x": float(ap.position_3d[0]),
                "y": float(ap.position_3d[1]),
                "z": float(ap.position_3d[2]),
                "confidence": ap.confidence, "grade": ap.grade,
                "functions": ap.functions, "indications": ap.indications,
                "description": ap.description,
            }
            if ap.position_2d is not None:
                d["px"] = float(ap.position_2d[0])
                d["py"] = float(ap.position_2d[1])
            result.append(d)
        return result


class AcupointLocator:
    """
    穴位定位器 - 主入口

    用法:
        locator = AcupointLocator()
        locator.load_database("database/acupoints_torso.json")
        result = locator.locate(pose_result)
        for ap in result.acupoints:
            print(f"{ap.name_cn}: {ap.position_3d}")
    """

    def __init__(self, config: Optional[Dict] = None,
                 profile: Optional[PopulationProfile] = None,
                 hand_flip_correction: bool = True):
        """
        Args:
            config: 配置字典（可选，从config.yaml加载）
            profile: 用户身体画像（可选，用于人群系数适配）
            hand_flip_correction: 是否启用手部镜像纠正。
                当摄像头画面经过 cv2.flip(1) 水平翻转后再传给 MediaPipe 时，
                MediaPipe 的左右手标签会与视觉方向相反。启用此选项会自动纠正。
                对于静态图片（未翻转），应设为 False。
                （默认 True，适配实时摄像头场景）
        """
        self.config = config or {}
        self.profile = profile
        self._population_adapter = PopulationAdapter() if profile else None
        self._hand_flip_correction = hand_flip_correction

        # 如果提供了画像，预计算通用适配系数
        if self._population_adapter and self.profile:
            spine_coeffs = config.get("spine", {}) if config else {}
            adapt = self._population_adapter.get_coefficients("spine", self.profile)
            self.spine_estimator = SpineEstimator(
                cervical_lordosis=spine_coeffs.get("cervical_lordosis", 0.15) * adapt.spine_curve_modifier,
                thoracic_kyphosis=spine_coeffs.get("thoracic_kyphosis", 0.35) * adapt.spine_curve_modifier,
                lumbar_lordosis=spine_coeffs.get("lumbar_lordosis", 0.50) * adapt.spine_curve_modifier,
            )
        else:
            self.spine_estimator = SpineEstimator(
                cervical_lordosis=config.get("spine", {}).get("cervical_lordosis", 0.15)
                if config else 0.15,
                thoracic_kyphosis=config.get("spine", {}).get("thoracic_kyphosis", 0.35)
                if config else 0.35,
                lumbar_lordosis=config.get("spine", {}).get("lumbar_lordosis", 0.50)
                if config else 0.50,
            )

        # 穴位数据库: id -> 定义
        self._acupoint_defs: Dict[str, dict] = {}

        # 朝向缓存（每次 locate() 调用时更新）
        self._cached_body_orientation: str = "unknown"

        # 经络颜色
        self.meridian_colors = {
            "CV": [1.0, 0.42, 0.14],    # 任脉 橙红
            "GV": [1.0, 0.84, 0.0],     # 督脉 金色
            "LU": [1.0, 1.0, 1.0],      # 肺经 白
            "LI": [0.5, 0.5, 0.5],      # 大肠经 灰
            "ST": [1.0, 0.55, 0.0],     # 胃经 深橙
            "SP": [0.6, 0.2, 0.8],      # 脾经 紫
            "HT": [1.0, 0.0, 0.0],      # 心经 红
            "SI": [1.0, 0.5, 0.5],      # 小肠经 粉红
            "BL": [0.26, 0.41, 0.88],   # 膀胱经 蓝
            "KI": [0.0, 0.8, 0.82],     # 肾经 青
            "PC": [0.8, 0.0, 0.4],      # 心包经 深粉
            "TE": [0.0, 0.5, 1.0],      # 三焦经 蓝
            "GB": [0.0, 0.6, 0.3],      # 胆经 绿
            "LR": [0.13, 0.55, 0.13],   # 肝经 深绿
        }

    # ──────────── 朝向检测 ────────────

    def detect_body_orientation(self, pose_result: PoseResult) -> str:
        """
        检测人体朝向（正面/背面）

        原理：
        - MediaPipe Pose 模型的 0~10 号关键点（鼻、眼、耳、嘴）仅在正面可见
        - 背面时后脑勺遮挡导致这些关键点不可见或置信度极低
        - Holistic 模式下 has_face 是更强的正面信号

        Returns:
            "front"  /  "back"  /  "unknown"
        """
        # 最强信号：Holistic 检测到了面部网格
        if pose_result.has_face:
            return "front"

        if pose_result.pose_landmarks is None:
            return "unknown"

        # 检查面部关键点（0=鼻, 1-4=眼, 5-6=耳, 7-8=嘴角, 9-10=唇侧）
        face_indices = list(range(0, 11))
        visibilities = []
        for idx in face_indices:
            if idx < pose_result.pose_landmarks.shape[0]:
                vis = pose_result.pose_landmarks[idx, 3] if pose_result.pose_landmarks.shape[1] > 3 else 0.0
                visibilities.append(vis)

        if not visibilities:
            return "unknown"

        # 至少3个面部关键点可见度 > 0.5 → 正面
        visible_count = sum(1 for v in visibilities if v > 0.5)
        if visible_count >= 3:
            return "front"

        # 面部关键点全不可见但有躯干检测 → 背面
        if visible_count == 0 and pose_result.has_pose:
            return "back"

        return "unknown"

    def detect_hand_orientation(self, hand_lms: np.ndarray) -> Optional[str]:
        """
        检测手掌朝向（掌心 vs 掌背对镜头）

        核心原理（与手指弯曲/伸直无关，握拳和张开均有效）：
        
          MCP骨性凸起是唯一与手指位置无关的解剖学差异。
          手背的食中无小指根部骨节（MCP关节）在任何手指姿态下
          都比手腕更靠近镜头，掌心时 MCP 与掌面齐平。
        
        辅助特征：
          - MCP连线弯曲度（手背呈凸拱桥形）
          - 四指在2D图像中的张开度（掌心自然更张开）

        Returns:
            "palm"  /  "back_of_hand"  /  None（无法判断）
        """
        if hand_lms is None or hand_lms.shape[0] < 21:
            return None

        # ── F1: MCP凸出度（金标准）──
        # 正值 = MCP比手腕更靠近镜头 → 掌背
        mcp_indices = [5, 9, 13, 17]
        mcp_zs = [hand_lms[i, 2] for i in mcp_indices]
        mcp_mean_z = np.mean(mcp_zs)
        wrist_z = hand_lms[0, 2]
        mcp_prominence = wrist_z - mcp_mean_z

        # ── F2: MCP连线弯曲度 ──
        # 比较中+无名MCP 与 index-pinky 直线插值的Z差 → 正值=凸起=掌背
        x_idx, x_pinky = hand_lms[5, 0], hand_lms[17, 0]
        z_idx, z_pinky = hand_lms[5, 2], hand_lms[17, 2]
        dx = x_pinky - x_idx
        if abs(dx) > 1e-6:
            t_mid = (hand_lms[9, 0] - x_idx) / dx
            t_ring = (hand_lms[13, 0] - x_idx) / dx
            interp_z9 = z_idx + t_mid * (z_pinky - z_idx)
            interp_z13 = z_idx + t_ring * (z_pinky - z_idx)
        else:
            interp_z9 = interp_z13 = (z_idx + z_pinky) * 0.5
        ridge_curve = ((interp_z9 - hand_lms[9, 2]) + (interp_z13 - hand_lms[13, 2])) * 0.5

        # ── F3: 四指张开度（2D角度，0=并拢, ~1=大幅张开）──
        tips_2d = hand_lms[[8, 12, 16, 20], :2]
        wrist_2d = hand_lms[0, :2]
        total_angle = 0.0
        for i in range(3):
            v1 = tips_2d[i] - wrist_2d
            v2 = tips_2d[i + 1] - wrist_2d
            cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
            total_angle += np.arccos(np.clip(cos_a, -1.0, 1.0))
        spread = np.clip(total_angle / 0.75, 0.0, 1.0)

        # ── 决策树（MCP凸出度为主，张开度为辅，弯曲度为校验）──
        # 三个阈值: 0.015(清晰), 0.005(微弱), -0.010(掌心)
        PROM_CLEAR = 0.015   # MCP明显比手腕近 → 掌背
        PROM_WEAK  = 0.005   # MCP微弱偏近 → 需辅助判断
        PROM_PALM  = -0.010  # MCP比手腕远 → 掌心

        if mcp_prominence > PROM_CLEAR:
            # MCP明显凸出 → 掌背（金标准）
            # 仅当手指极度张开且凸出度刚过阈值时可能误判，需额外校验
            if spread > 0.55 and mcp_prominence <= 0.018:
                return "palm"
            return "back_of_hand"

        elif mcp_prominence < PROM_PALM:
            # MCP比手腕远 → 掌心
            return "palm"

        elif mcp_prominence > PROM_WEAK:
            # MCP微弱偏近 → 大概率掌背，用张开度和弯曲度交叉验证
            if spread > 0.55:
                # 手指大幅张开 → 矛盾，相信张开度 → 掌心
                return "palm"
            if ridge_curve > 0.003:
                return "back_of_hand"
            if spread < 0.30:
                return "back_of_hand"
            return "back_of_hand"

        elif mcp_prominence < -0.003:
            # MCP微弱偏远 → 掌心
            if spread < 0.25 and ridge_curve > 0.003:
                # 手指并拢 + 凸脊 → 矛盾，更可能是掌背
                return "back_of_hand"
            return "palm"

        else:
            # MCP几乎与手腕齐平 → 靠辅助特征判断，需要强证据
            if spread < 0.22 and ridge_curve > 0.005:
                return "back_of_hand"
            elif spread > 0.65 and ridge_curve < 0.002:
                return "palm"
            # 其他情况不作判断，避免在侧面等歧义角度错误显示穴位
            return None

    def _resolve_hand_data(self, hand_side: str,
                            pose_result) -> Optional[np.ndarray]:
        """
        根据 hand_side 获取正确的手部关键点数据，
        自动处理摄像头镜像翻转导致的左右手标签颠倒。

        原理：
        摄像头实时画面通常通过 cv2.flip(frame, 1) 进行水平镜像翻转，
        再传给 MediaPipe。MediaPipe 在非镜像数据上训练，镜像输入会导致
        其左右手标签与视觉方向相反。启用 hand_flip_correction 后，
        自动交换 left↔right 以纠正此偏差。

        对于静态图片（不翻转输入），应设置 hand_flip_correction=False。

        Args:
            hand_side: 穴位定义中的 hand_side ("left"/"right")
            pose_result: 包含左右手关键点数据的检测结果

        Returns:
            手部关键点数组 (21,3) 或 None
        """
        if self._hand_flip_correction:
            # 镜像纠正：交换左右手数据
            target_side = "left" if hand_side == "right" else "right"
        else:
            target_side = hand_side

        if target_side == "left" and pose_result.has_left_hand:
            return pose_result.left_hand_landmarks
        elif target_side == "right" and pose_result.has_right_hand:
            return pose_result.right_hand_landmarks
        return None

    # ──────────── 数据库加载 ────────────

    def load_database(self, json_paths: List[str]):
        """加载穴位定义数据库"""
        for path in json_paths:
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                for ap in data.get("acupoints", []):
                    self._acupoint_defs[ap["id"]] = ap
                    # 手部穴位自动镜像：为 right 生成对应的 left 版本
                    rule = ap.get("location_rule", {})
                    if (rule.get("method") == "hand_landmark"
                            and rule.get("hand_side") == "right"):
                        mirrored = self._mirror_hand_acupoint(ap)
                        if mirrored is not None:
                            self._acupoint_defs[mirrored["id"]] = mirrored
                    # 躯干穴位自动镜像：为 side="right" 的旁开穴位生成左侧版本
                    if (rule.get("method") in ("spine_bone_ratio", "midline_ratio")
                            and rule.get("side") == "right"
                            and rule.get("lateral_offset_cun", 0) > 0):
                        mirrored = self._mirror_torso_acupoint(ap)
                        if mirrored is not None:
                            self._acupoint_defs[mirrored["id"]] = mirrored
                print(f"[AcupointLocator] 已加载 {len(data.get('acupoints', []))} 个穴位定义: {path}")
            except FileNotFoundError:
                print(f"[AcupointLocator] 警告: 文件不存在 - {path}")
            except Exception as e:
                print(f"[AcupointLocator] 加载失败 {path}: {e}")

    @staticmethod
    def _mirror_hand_acupoint(ap_def: dict) -> Optional[dict]:
        """为右手穴位定义生成左手镜像副本"""
        import copy
        mirror = copy.deepcopy(ap_def)
        mirror["id"] = ap_def["id"] + "_L"
        # 中文名加(左)后缀以区分
        name_cn = ap_def.get("name_cn", "")
        if "(左)" not in name_cn:
            mirror["name_cn"] = name_cn.replace("(掌心)", "(掌心-左)").replace("(掌心面)", "(掌心面-左)").replace("(手背)", "(手背-左)")
            if "name_cn" not in str(mirror["location_rule"].get("description", "")) or True:
                if mirror["name_cn"] == name_cn:
                    mirror["name_cn"] = name_cn + "(左)"
        mirror["location_rule"]["hand_side"] = "left"
        return mirror

    @staticmethod
    def _mirror_torso_acupoint(ap_def: dict) -> Optional[dict]:
        """为右侧躯干/四肢穴位生成左侧镜像副本"""
        import copy
        mirror = copy.deepcopy(ap_def)
        # ID 加 _L 后缀
        mirror["id"] = ap_def["id"] + "_L"
        # 中文名加(左)
        name_cn = ap_def.get("name_cn", "")
        if "(右)" in name_cn:
            mirror["name_cn"] = name_cn.replace("(右)", "(左)")
        elif "(左)" not in name_cn:
            mirror["name_cn"] = name_cn + "(左)"
        mirror["location_rule"]["side"] = "left"
        return mirror

    def get_acupoint_count(self) -> int:
        """获取已加载穴位总数"""
        return len(self._acupoint_defs)

    def locate(self, pose_result: PoseResult) -> AcupointResult:
        """
        主定位函数: 综合所有数据源计算穴位位置

        Args:
            pose_result: Pose 检测结果

        Returns:
            AcupointResult 包含所有定位到的穴位
        """
        result = AcupointResult()
        result.image_shape = pose_result.image_shape

        if not pose_result.has_pose:
            return result

        # Step 1: 构建虚拟脊柱（如果有肩髋关键点）
        if pose_result.pose_world_landmarks is not None:
            spine = self.spine_estimator.build(pose_result.pose_world_landmarks)
            result.spine = spine
        else:
            return result  # 没有world坐标无法计算

        # Step 2: 缓存身体朝向（避免每个穴位重复检测）
        self._cached_body_orientation = self.detect_body_orientation(pose_result)

        # Step 3: 遍历穴位定义，逐个定位
        for ap_id, ap_def in self._acupoint_defs.items():
            ap_pos = self._locate_single(ap_def, pose_result, result.spine)
            if ap_pos is not None:
                # ── 应用人群适配系数 ──
                if self._population_adapter and self.profile:
                    coeffs = self._population_adapter.get_coefficients(ap_id, self.profile)
                    # 深度调整：肥胖者穴位更深（皮下脂肪厚度）
                    if ap_pos.depth is not None:
                        ap_pos.depth *= coeffs.skin_thickness
                    # 体表偏移调整：影响脊柱定位精度
                    ap_pos.position_3d = ap_pos.position_3d * coeffs.offset_modifier
                    # 标注适配信息
                    ap_pos.description += f" [适配: skin×{coeffs.skin_thickness:.1f} off×{coeffs.offset_modifier:.2f}]"
                result.acupoints.append(ap_pos)

        # Step 4: 统计
        result.total_found = len(result.acupoints)
        for ap in result.acupoints:
            result.grade_counts[ap.grade] = result.grade_counts.get(ap.grade, 0) + 1
            mc = ap.meridian_code
            result.meridian_counts[mc] = result.meridian_counts.get(mc, 0) + 1

        return result

    def _locate_single(self, ap_def: dict, pose_result: PoseResult,
                       spine: SpineResult) -> Optional[AcupointPosition]:
        """
        定位单个穴位

        根据定位方法(location_rule.method)分派到对应函数。
        2D坐标直接从pose_landmarks（归一化图像坐标）计算，不再经过world坐标中转投影，
        避免world→image变换带来的累积误差和深度丢失问题。
        """
        rule = ap_def.get("location_rule", {})
        method = rule.get("method", "")

        ap = AcupointPosition(
            id=ap_def["id"],
            name_cn=ap_def.get("name_cn", ""),
            name_pinyin=ap_def.get("name_pinyin", ""),
            meridian=ap_def.get("meridian", ""),
            meridian_code=ap_def.get("meridian_code", ""),
            functions=ap_def.get("clinical", {}).get("functions", []),
            indications=ap_def.get("clinical", {}).get("indications", []),
            depth=ap_def.get("clinical", {}).get("needling_depth_cm"),
            description=rule.get("description", ""),
            grade=ap_def.get("validation", {}).get("grade", "B"),
        )

        try:
            # ── 计算3D世界坐标（保留供AR/3D可视化等用途） ──
            if method == "spine_bone_ratio":
                pos_3d = self._locate_spine_ratio(rule, spine)
            elif method == "bone_proportion":
                pos_3d = self._locate_bone_proportion(rule, pose_result.pose_world_landmarks)
            elif method == "midline_ratio":
                pos_3d = self._locate_midline_ratio(rule, spine)
            elif method == "face_mesh":
                pos_3d = self._locate_face_mesh(rule, pose_result)
            elif method == "hand_landmark":
                pos_3d = self._locate_hand(rule, pose_result)
            else:
                return None

            if pos_3d is None:
                return None

            ap.position_3d = pos_3d
            ap.confidence = self._estimate_confidence(ap_def, pose_result)

            # ── 直接从pose_landmarks计算2D图像坐标（核心修复） ──
            if method == "face_mesh":
                ap.position_2d = self._compute_face_2d(rule, pose_result)
            elif method == "hand_landmark":
                ap.position_2d = self._compute_hand_2d(rule, pose_result)
            elif method == "bone_proportion" and pose_result.pose_landmarks is not None:
                # 四肢穴位：直接在图像空间插值两个端点landmark
                ap.position_2d = self._compute_limb_2d(rule, pose_result.pose_landmarks,
                                                        pose_result.image_shape)
            elif method in ("spine_bone_ratio", "midline_ratio") and pose_result.pose_landmarks is not None:
                # 躯干穴位：在图像空间沿躯干轴插值
                ap.position_2d = self._compute_torso_2d(rule, pose_result.pose_landmarks,
                                                         pose_result.image_shape, spine)
            elif pose_result.pose_landmarks is not None:
                # 兜底：用改进版投影
                ap.position_2d = self._project_to_2d_improved(
                    pos_3d, pose_result.pose_world_landmarks,
                    pose_result.pose_landmarks, pose_result.image_shape
                )

            # ── 朝向检测：验证当前姿态是否匹配穴位要求的朝向 ──
            valid_ori = ap_def.get("valid_orientation")
            if valid_ori is not None:
                ori_match = self._check_orientation(valid_ori, rule, pose_result)
                if not ori_match:
                    return None  # 朝向不匹配，完全不显示此穴位

            return ap

        except Exception as e:
            print(f"[AcupointLocator] 定位 {ap.id} 失败: {e}")

        return None

    def _check_orientation(self, required: str, rule: dict,
                           pose_result: PoseResult) -> bool:
        """
        检查当前姿态朝向是否匹配穴位要求的朝向

        Args:
            required: 穴位要求的朝向 ("front"/"back"/"palm"/"back_of_hand")
            rule: 定位规则
            pose_result: 检测结果

        Returns:
            True = 朝向匹配, False = 朝向不匹配
        """
        # 人体前/后朝向（使用缓存，避免重复检测）
        if required in ("front", "back"):
            body_ori = getattr(self, "_cached_body_orientation", "unknown")
            if body_ori == "unknown":
                return True  # 不确定时不拦截
            return body_ori == required

        # 手部掌心/掌背朝向
        if required in ("palm", "back_of_hand"):
            hand_side = rule.get("hand_side", "right")
            hand_lms = self._resolve_hand_data(hand_side, pose_result)

            if hand_lms is None:
                return True  # 检测不到手时不拦截

            hand_ori = self.detect_hand_orientation(hand_lms)
            if hand_ori is None:
                # 不确定时不显示朝向敏感的穴位（宁可漏标也不错标）
                return False
            return hand_ori == required

        # 其他未知朝向类型，默认通过
        return True

    def _compute_limb_2d(self, rule: dict, pose_lms: np.ndarray,
                          image_shape: Tuple[int, int]) -> Optional[np.ndarray]:
        """
        四肢穴位2D定位：直接在图像空间插值

        与 _locate_bone_proportion 使用相同的参考landmark和比例，
        但在归一化的pose_landmarks上操作，避免world→image投影误差。
        """
        w, h = image_shape
        n_lms = len(pose_lms)

        proximal_idx = rule.get("landmark_proximal")
        distal_idx = rule.get("landmark_distal")
        ratio = rule.get("ratio", 0.5)
        offset_cun = rule.get("offset_cun", 0)
        direction = rule.get("offset_direction", "")

        if proximal_idx is None or distal_idx is None:
            return None
        if proximal_idx >= n_lms or distal_idx >= n_lms:
            return None

        # 检查可见性
        vis_p = pose_lms[proximal_idx, 3] if pose_lms.shape[1] > 3 else 1.0
        vis_d = pose_lms[distal_idx, 3] if pose_lms.shape[1] > 3 else 1.0
        if vis_p < 0.3 and vis_d < 0.3:
            return None

        # 近端和远端的归一化2D坐标
        p_prox = pose_lms[proximal_idx, :2]   # [nx, ny]
        p_dis = pose_lms[distal_idx, :2]

        # 沿线段插值
        pos_2d = p_prox + ratio * (p_dis - p_prox)

        # 偏移（在2D平面内垂直于骨骼方向）
        if offset_cun != 0 and direction:
            segment_2d = p_dis - p_prox
            segment_len_px = np.linalg.norm(segment_2d * np.array([w, h]))
            cun_per_seg = rule.get("cun_per_segment", 16)
            px_per_cun = segment_len_px / cun_per_seg if cun_per_seg > 0 else 10
            offset_px = offset_cun * px_per_cun

            offset_vec = self._get_offset_vector_2d(segment_2d, direction, w, h)
            pos_2d = pos_2d + offset_vec * (offset_px / max(w, h))

        # 转换为像素坐标
        return np.array([pos_2d[0] * w, pos_2d[1] * h])

    def _compute_torso_2d(self, rule: dict, pose_lms: np.ndarray,
                           image_shape: Tuple[int, int],
                           spine: SpineResult) -> Optional[np.ndarray]:
        """
        躯干穴位2D定位：在图像空间沿肩-髋轴插值

        改进：当髋部检测不稳定时（上半身入镜），使用肩-颈区域作为
        替代参考轴，避免所有穴位堆积到肩部。
        """
        w, h = image_shape
        n_lms = len(pose_lms)

        offset_t = rule.get("offset_t", 0.0)
        lateral_offset_cun = rule.get("lateral_offset_cun", 0.0)
        side = rule.get("side", "midline")
        use_front = rule.get("use_front", False)

        # 肩关键点的2D位置（归一化）
        shoulder_l = pose_lms[11, :2] if 11 < n_lms else None
        shoulder_r = pose_lms[12, :2] if 12 < n_lms else None
        hip_l = pose_lms[23, :2] if 23 < n_lms else None
        hip_r = pose_lms[24, :2] if 24 < n_lms else None

        if shoulder_l is None or shoulder_r is None:
            return None

        # 检查髋部可见性
        hip_vis = 0.0
        if hip_l is not None and hip_r is not None:
            vis_hl = pose_lms[23, 3] if pose_lms.shape[1] > 3 else 1.0
            vis_hr = pose_lms[24, 3] if pose_lms.shape[1] > 3 else 1.0
            hip_vis = max(vis_hl, vis_hr)

        if hip_l is None or hip_r is None or hip_vis < 0.15:
            # ── 髋部不可靠：用肩宽按比例估算髋部位置 ──
            # 典型站姿：躯干像素高度/肩宽像素 ≈ 1.0（受镜头透视影响）
            mid_shoulder = (shoulder_l + shoulder_r) / 2.0

            shoulder_w_px = np.linalg.norm((shoulder_r - shoulder_l) * np.array([w, h]))
            torso_est_px = shoulder_w_px * 1.0  # 躯干估算长度（像素）
            mid_hip_est = mid_shoulder + np.array([0.0, torso_est_px / h])

            axis_top = mid_shoulder
            axis_bottom = mid_hip_est

            # 沿估算轴插值，offset_t 保持原来的含义 (0=肩, 1=髋)
            t = min(max(offset_t, 0.0), 1.0)
            base_pos = axis_top + t * (axis_bottom - axis_top)

            if use_front:
                base_pos[1] += 0.01

        else:
            # ── 正常模式：髋部可见 ──
            vis = [pose_lms[i, 3] for i in [11, 12, 23, 24]] if pose_lms.shape[1] > 3 else [1]*4
            if sum(vis) < 1.0:
                return None

            mid_shoulder = (shoulder_l + shoulder_r) / 2.0
            mid_hip = (hip_l + hip_r) / 2.0

            t = min(max(offset_t, 0.0), 1.0)
            base_pos = mid_shoulder + t * (mid_hip - mid_shoulder)

            if use_front:
                front_bias = 0.02
                base_pos[1] += front_bias

        # 左右旁开偏移
        if lateral_offset_cun != 0 and side in ("left", "right"):
            shoulder_w_px = np.linalg.norm((shoulder_r - shoulder_l) * np.array([w, h]))
            # 改进：用躯干宽度更精确地估算每寸
            # 成人肩宽约 16 寸（同身寸），每寸 ≈ 肩宽/16
            px_per_cun = shoulder_w_px / 16.0
            lateral_px = lateral_offset_cun * px_per_cun

            lr_dir = shoulder_r - shoulder_l
            lr_len = np.linalg.norm(lr_dir)
            if lr_len > 0.001:
                lr_unit = lr_dir / lr_len
            else:
                lr_unit = np.array([1.0, 0.0])

            # 侧向偏移（新 fallback 用比例估算髋部，不再需要缩小系数）
            if side == "right":
                base_pos = base_pos + lr_unit * (lateral_px / max(w, h))
            else:
                base_pos = base_pos - lr_unit * (lateral_px / max(w, h))

        return np.array([base_pos[0] * w, base_pos[1] * h])

    def _get_offset_vector_2d(self, segment: np.ndarray,
                               direction: str,
                               img_w: int, img_h: int) -> np.ndarray:
        """在归一化2D空间获取偏移方向向量"""
        seg_norm = segment / (np.linalg.norm(segment) + 1e-8)

        # 骨骼方向的大致朝向
        # up方向（在归一化坐标中Y轴向下，所以"上"是-Y）
        up = np.array([0.0, -1.0])
        # 垂直于骨骼的横向（右侧为正）
        lateral = np.array([seg_norm[1], -seg_norm[0]])
        if np.linalg.norm(lateral) < 0.001:
            lateral = np.array([1.0, 0.0])
        lateral = lateral / np.linalg.norm(lateral)
        # 前方（垂直于骨骼和横向）
        anterior = np.array([-seg_norm[0] * seg_norm[1],
                             seg_norm[0]**2 - seg_norm[1]**2])
        if np.linalg.norm(anterior) < 0.001:
            anterior = up
        anterior = anterior / np.linalg.norm(anterior)

        direction_map = {
            "anterior": anterior,
            "anterior_lateral": self._norm2(anterior + lateral),
            "lateral": lateral,
            "posterior": -anterior,
            "posterior_lateral": self._norm2(-anterior + lateral),
            "medial": -lateral,
            "anterior_medial": self._norm2(anterior - lateral),
            "posterior_medial": self._norm2(-anterior - lateral),
        }
        return direction_map.get(direction, anterior)

    @staticmethod
    def _norm2(v: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(v)
        return v / n if n > 1e-8 else v

    def _locate_spine_ratio(self, rule: dict, spine: SpineResult) -> Optional[np.ndarray]:
        """
        基于虚拟脊柱比例定位（躯干正中线穴位）

        rule参数:
          vertebra_ref: "C7" / "T7" / "L2" 等椎骨参考
          offset_t: 沿脊柱方向的偏移比例（0=C7, 1=L5）
          lateral_offset_cun: 旁开寸数
          side: "midline" / "left" / "right"
          grade: "B" / "C"
        """
        if not spine.is_valid:
            return None

        vertebra_ref = rule.get("vertebra_ref", "")
        offset_t = rule.get("offset_t", 0.0)
        lateral_offset_cun = rule.get("lateral_offset_cun", 0.0)
        side = rule.get("side", "midline")
        use_front = rule.get("use_front", False)  # True=任脉(前), False=督脉(后)

        # 沿脊柱插值位置
        if vertebra_ref and vertebra_ref in spine.vertebrae:
            base_pos = spine.vertebrae[vertebra_ref].copy()
        else:
            # 用 offset_t 在椎骨间插值
            t = min(max(offset_t, 0.0), 1.0)
            idx = int(t * (SpineEstimator.TOTAL_VERTEBRAE - 1))
            base_pos = spine.vertebrae[SpineEstimator.VERTEBRA_LABELS[idx]].copy()

        # 前后偏移（前正中线 vs 后正中线）
        if use_front and spine.front_midline.shape[0] > 0:
            # 在前正中线上找最近点
            t = min(max(offset_t, 0.0), 1.0)
            idx = int(t * (spine.front_midline.shape[0] - 1))
            base_pos = spine.front_midline[idx]
        elif not use_front and spine.back_midline.shape[0] > 0:
            t = min(max(offset_t, 0.0), 1.0)
            idx = int(t * (spine.back_midline.shape[0] - 1))
            base_pos = spine.back_midline[idx]

        # 旁开偏移
        if lateral_offset_cun != 0:
            cun_length = self._get_cun_length(spine, rule)
            lateral_dist = lateral_offset_cun * cun_length
            # 左右方向向量
            left_dir = self._get_lateral_direction(spine)
            if side == "left":
                base_pos += left_dir * lateral_dist
            elif side == "right":
                base_pos -= left_dir * lateral_dist

        return base_pos

    def _locate_bone_proportion(self, rule: dict,
                                 world_lms: np.ndarray) -> Optional[np.ndarray]:
        """
        基于骨度比例定位（四肢穴位）

        rule参数:
          landmark_proximal: 近端关键点索引
          landmark_distal: 远端关键点索引
          ratio: 比例 (0~1)
          offset_direction: 偏移方向 "anterior"/"posterior"/"lateral"/"medial"
          offset_cun: 偏移寸数
        """
        proximal_idx = rule.get("landmark_proximal")
        distal_idx = rule.get("landmark_distal")
        ratio = rule.get("ratio", 0.5)
        offset_cun = rule.get("offset_cun", 0)
        direction = rule.get("offset_direction", "")

        if proximal_idx is None or distal_idx is None:
            return None

        vis_proximal = world_lms[proximal_idx, 3]
        vis_distal = world_lms[distal_idx, 3]

        if vis_proximal < 0.5 and vis_distal < 0.5:
            return None

        proximal = world_lms[proximal_idx, :3]
        distal = world_lms[distal_idx, :3]

        # 主位置 = 近端 + 比例 × (远端 - 近端)
        segment = distal - proximal
        pos = proximal + ratio * segment

        # 偏移
        if offset_cun != 0 and direction:
            segment_len = np.linalg.norm(segment)
            if segment_len > 0:
                cun_length = segment_len / rule.get("cun_per_segment", 16)
                offset_dist = offset_cun * cun_length
                offset_vec = self._get_offset_vector(segment, direction)
                pos = pos + offset_vec * offset_dist

        return pos

    def _locate_midline_ratio(self, rule: dict, spine: SpineResult) -> Optional[np.ndarray]:
        """
        基于前后正中线比例定位（躯干前后穴位）
        """
        if not spine.is_valid:
            return None

        use_front = rule.get("use_front", True)
        ratio = rule.get("ratio", 0.5)
        lateral_cun = rule.get("lateral_offset_cun", 0)
        side = rule.get("side", "midline")

        midline = spine.front_midline if use_front else spine.back_midline
        if midline.shape[0] == 0:
            return None

        idx = int(ratio * (midline.shape[0] - 1))
        idx = min(max(idx, 0), midline.shape[0] - 1)
        pos = midline[idx].copy()

        # 旁开
        if lateral_cun != 0 and side in ("left", "right"):
            cun_length = spine.spine_length / 50  # 躯干寸大概估算
            lateral_dist = lateral_cun * cun_length
            left_dir = self._get_lateral_direction(spine)
            if side == "left":
                pos += left_dir * lateral_dist
            elif side == "right":
                pos -= left_dir * lateral_dist

        return pos

    def _locate_face_mesh(self, rule: dict, pose_result: PoseResult) -> Optional[np.ndarray]:
        """
        基于面部网格定位（面部穴位）

        使用Holistic Face Mesh 468点中的特定索引
        """
        if not pose_result.has_face or pose_result.face_landmarks is None:
            return None

        face_indices = rule.get("face_indices", [])
        if not face_indices:
            return None

        pts = []
        for idx in face_indices:
            if idx < pose_result.face_landmarks.shape[0]:
                lm = pose_result.face_landmarks[idx]
                pts.append(lm[:2])  # 归一化坐标

        if not pts:
            return None

        # 面部位置 → 近似3D（用Pose的鼻尖作为深度参考）
        avg_pt = np.mean(pts, axis=0)
        # 使用Pose鼻尖的world坐标作为面部参考
        if pose_result.pose_world_landmarks is not None:
            nose_world = pose_result.pose_world_landmarks[0, :3].copy()
            # 面部穴位在鼻尖附近，简化处理
            pos = nose_world.copy()
            # 根据归一化偏移调整
            norm_nose = pose_result.pose_landmarks[0, :2]
            offset = (avg_pt - norm_nose) * 0.3  # 缩放到world空间
            pos[0] += offset[0]
            pos[1] -= offset[1]
            return pos

        return None

    def _locate_hand(self, rule: dict, pose_result: PoseResult) -> Optional[np.ndarray]:
        """
        基于手部关键点定位（手部穴位）

        使用Holistic Hand Landmarks (每手21点)
        """
        hand_side = rule.get("hand_side", "right")
        landmark_idx = rule.get("hand_landmark_index")

        if landmark_idx is None:
            return None

        hand_lms = self._resolve_hand_data(hand_side, pose_result)

        if hand_lms is None:
            return None

        # 手部关键点坐标（归一化）
        pt = hand_lms[landmark_idx, :2]

        # 用对应手腕的world坐标作为参考
        # 注意：Pose 手腕 landmark (15=left, 16=right) 沿用手部镜像纠正后的逻辑
        if self._hand_flip_correction:
            wrist_idx = 16 if hand_side == "left" else 15
        else:
            wrist_idx = 15 if hand_side == "left" else 16
        if pose_result.pose_world_landmarks is not None:
            wrist_world = pose_result.pose_world_landmarks[wrist_idx, :3].copy()
            # 手部穴位粗略放在手腕附近
            pos = wrist_world.copy()
            return pos

        return None

    def _get_cun_length(self, spine: SpineResult, rule: dict) -> float:
        """计算1寸对应的长度（米）"""
        if spine.spine_length > 0:
            # 躯干寸：脊柱长 / 50寸（大致）
            return spine.spine_length / 50
        return 0.01  # 默认1cm

    def _get_lateral_direction(self, spine: SpineResult) -> np.ndarray:
        """获取左右方向向量（从左肩指向右肩）"""
        direction = spine.shoulder_right - spine.shoulder_left
        norm = np.linalg.norm(direction)
        if norm > 0:
            return direction / norm
        return np.array([1, 0, 0])

    def _get_offset_vector(self, segment: np.ndarray,
                            direction: str) -> np.ndarray:
        """根据方向描述获取偏移向量"""
        # 沿骨骼方向的单位向量
        seg_norm = segment / (np.linalg.norm(segment) + 1e-8)
        # 前后方向: 用叉积估算
        up = np.array([0, 1, 0])
        lateral = np.cross(seg_norm, up)
        if np.linalg.norm(lateral) < 0.001:
            lateral = np.array([1, 0, 0])
        lateral = lateral / np.linalg.norm(lateral)
        anterior = np.cross(lateral, seg_norm)

        direction_map = {
            "anterior": anterior,
            "anterior_lateral": (anterior + lateral) / np.linalg.norm(anterior + lateral),
            "lateral": lateral,
            "posterior": -anterior,
            "posterior_lateral": (-anterior + lateral) / np.linalg.norm(-anterior + lateral),
            "medial": -lateral,
            "anterior_medial": (anterior - lateral) / np.linalg.norm(anterior - lateral),
            "posterior_medial": (-anterior - lateral) / np.linalg.norm(-anterior - lateral),
        }

        return direction_map.get(direction, anterior)

    def _project_to_2d_improved(self, world_pos: np.ndarray,
                       world_lms: np.ndarray,
                       pose_lms: np.ndarray,
                       image_shape: Tuple[int, int]) -> np.ndarray:
        """
        兜底投影：world坐标 → 2D图像（仅当直接2D计算不可用时使用）

        改进版：使用最近邻landmark加权而非全局质心
        """
        w, h = image_shape
        n_landmarks = len(world_lms)

        # 可见性过滤
        if world_lms.shape[1] > 3:
            mask = world_lms[:, 3] > 0.5
        else:
            mask = np.ones(n_landmarks, dtype=bool)

        # 优先使用躯干+四肢关键点（这些世界坐标更可靠，不包含面部）
        body_idx = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
        valid = [i for i in body_idx if i < n_landmarks and mask[i]]

        # 如果躯干点不够，回退到所有可见点
        if len(valid) < 4:
            valid = [i for i in range(n_landmarks) if mask[i]]

        if len(valid) < 2:
            return np.array([w / 2, h / 2])  # 无法投影，放中心

        # 提取参考点的世界坐标和图像坐标
        ref_world = world_lms[valid][:, :2]  # (N, 2): [x_world, y_world]
        ref_image = pose_lms[valid][:, :2] * np.array([w, h])  # (N, 2): [x_img, y_img]

        # 质心
        world_cx, world_cy = np.mean(ref_world, axis=0)
        image_cx, image_cy = np.mean(ref_image, axis=0)

        # 分轴计算缩放因子（X 和 Y 独立）
        w_range = max(np.ptp(ref_world[:, 0]), 0.03)   # 世界X: 左右
        i_range = max(np.ptp(ref_image[:, 0]), 30.0)   # 图像X: 水平
        scale_x = i_range / w_range

        w_range_y = max(np.ptp(ref_world[:, 1]), 0.03)  # 世界Y: 上下
        i_range_y = max(np.ptp(ref_image[:, 1]), 30.0)  # 图像Y: 垂直
        scale_y = i_range_y / w_range_y

        # 目标穴位相对于质心的世界偏移量
        dx = world_pos[0] - world_cx
        dy = world_pos[1] - world_cy

        # 映射到图像（Y轴翻转：world Y↑ → image Y↓）
        px = image_cx + dx * scale_x
        py = image_cy - dy * scale_y

        return np.array([px, py])

    def _compute_face_2d(self, rule: dict, pose_result: PoseResult) -> Optional[np.ndarray]:
        """
        从面部网格关键点直接计算面部穴位的2D图像坐标

        Args:
            rule: 定位规则，包含 face_indices 数组
            pose_result: Holistic 检测结果（需包含 face_landmarks）

        Returns:
            2D 像素坐标 [x, y] 或 None
        """
        face_indices = rule.get("face_indices", [])
        if not face_indices or pose_result.face_landmarks is None:
            return None

        w, h = pose_result.image_shape
        pts = []
        for idx in face_indices:
            if idx < pose_result.face_landmarks.shape[0]:
                lm = pose_result.face_landmarks[idx]
                pts.append(lm[:2])  # 归一化坐标

        if not pts:
            return None

        avg = np.mean(pts, axis=0)
        return np.array([avg[0] * w, avg[1] * h])

    def _compute_hand_2d(self, rule: dict, pose_result: PoseResult) -> Optional[np.ndarray]:
        """
        从手部关键点直接计算手部穴位的2D图像坐标

        Args:
            rule: 定位规则，包含 hand_side 和 hand_landmark_index
            pose_result: Holistic 检测结果（需包含 hand_landmarks）

        Returns:
            2D 像素坐标 [x, y] 或 None
        """
        hand_side = rule.get("hand_side", "right")
        landmark_idx = rule.get("hand_landmark_index")

        if landmark_idx is None:
            return None

        hand_lms = self._resolve_hand_data(hand_side, pose_result)

        if hand_lms is None or landmark_idx >= hand_lms.shape[0]:
            return None

        w, h = pose_result.image_shape
        pt = hand_lms[landmark_idx, :2]
        return np.array([pt[0] * w, pt[1] * h])

    def _estimate_confidence(self, ap_def: dict, pose_result: PoseResult) -> float:
        """估算穴位定位置信度"""
        rule = ap_def.get("location_rule", {})
        validation = ap_def.get("validation", {})

        base_conf = validation.get("confidence_min", 0.5)

        # 检查所需关键点可见性
        required = validation.get("visibility_required_landmarks", [])
        if required and pose_result.pose_world_landmarks is not None:
            visibilities = [pose_result.pose_world_landmarks[idx, 3] for idx in required]
            if visibilities:
                vis_conf = sum(visibilities) / len(visibilities)
                base_conf = min(base_conf, vis_conf)

        # 根据精度等级调整
        grade_adjust = {"A": 0.1, "B": 0.0, "C": -0.1, "D": -0.2}
        base_conf += grade_adjust.get(ap_def.get("validation", {}).get("grade", "B"), 0)

        return max(0.1, min(1.0, base_conf))
