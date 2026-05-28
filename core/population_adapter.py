"""
人群系数适配器 - 根据年龄/性别/BMI调整穴位定位参数

支持的人群分组:
  年龄: 0-12 / 13-17 / 18-60 / 60-75 / 75+
  性别: 男/女
  体型: 偏瘦(<18.5) / 标准(18.5-24) / 超重(24-28) / 肥胖(28+)
  体态: 正常/驼背/脊柱侧弯/骨盆前倾/O型腿/X型腿
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, List
from enum import Enum
import json


class AgeGroup(Enum):
    CHILD = "child_0_12"
    TEEN = "teen_13_17"
    ADULT = "adult_18_60"
    ELDERLY = "elderly_60_75"
    SENIOR = "senior_75_plus"

class Gender(Enum):
    MALE = "male"
    FEMALE = "female"

class BodyType(Enum):
    THIN = "thin"       # BMI < 18.5
    NORMAL = "normal"   # 18.5-24
    OVERWEIGHT = "overweight"  # 24-28
    OBESE = "obese"     # BMI > 28

class PostureType(Enum):
    NORMAL = "normal"
    KYPHOSIS = "kyphosis"          # 驼背
    SCOLIOSIS = "scoliosis"        # 脊柱侧弯
    ANTERIOR_PELVIC = "anterior_pelvic"  # 骨盆前倾
    BOW_LEGS = "bow_legs"          # O型腿
    KNOCK_KNEES = "knock_knees"    # X型腿


@dataclass
class PopulationProfile:
    """人群画像"""
    age: int = 30
    gender: Gender = Gender.MALE
    height_cm: float = 170.0
    weight_kg: float = 65.0
    bmi: float = 0.0
    posture: PostureType = PostureType.NORMAL

    def __post_init__(self):
        if self.bmi == 0 and self.height_cm > 0:
            self.bmi = self.weight_kg / ((self.height_cm / 100) ** 2)

    @property
    def age_group(self) -> AgeGroup:
        if self.age <= 12:
            return AgeGroup.CHILD
        elif self.age <= 17:
            return AgeGroup.TEEN
        elif self.age <= 60:
            return AgeGroup.ADULT
        elif self.age <= 75:
            return AgeGroup.ELDERLY
        else:
            return AgeGroup.SENIOR

    @property
    def body_type(self) -> BodyType:
        if self.bmi < 18.5:
            return BodyType.THIN
        elif self.bmi < 24:
            return BodyType.NORMAL
        elif self.bmi < 28:
            return BodyType.OVERWEIGHT
        else:
            return BodyType.OBESE

    def get_group_key(self) -> str:
        """获取人群分组键"""
        return f"{self.age_group.value}|{self.gender.value}|{self.body_type.value}"


@dataclass
class AdaptationCoefficients:
    """适配系数"""
    ratio_modifier: float = 1.0     # 骨骼比例修正
    offset_modifier: float = 1.0    # 偏移量修正
    spine_curve_modifier: float = 1.0  # 脊柱弯曲修正
    skin_thickness: float = 1.0     # 皮肤/脂肪层厚度修正


class PopulationAdapter:
    """
    人群适配器

    用法:
        adapter = PopulationAdapter()
        profile = PopulationProfile(age=25, gender=Gender.FEMALE, height_cm=160)
        coeffs = adapter.get_coefficients("ST36", profile)
        adapter.update_from_annotations("database/annotations.json")
    """

    # 默认系数表
    DEFAULT_COEFFICIENTS = {
        # 年龄组
        "child_0_12":   {"ratio_modifier": 0.90, "offset_modifier": 0.80, "skin_thickness": 0.8},
        "teen_13_17":   {"ratio_modifier": 0.95, "offset_modifier": 0.90, "skin_thickness": 0.85},
        "adult_18_60":  {"ratio_modifier": 1.00, "offset_modifier": 1.00, "skin_thickness": 1.0},
        "elderly_60_75":{"ratio_modifier": 1.02, "offset_modifier": 1.05, "skin_thickness": 0.9},
        "senior_75_plus":{"ratio_modifier": 1.03, "offset_modifier": 1.08, "skin_thickness": 0.85},
        # 体型
        "thin":       {"ratio_modifier": 0.98, "offset_modifier": 0.95, "skin_thickness": 0.7},
        "normal":     {"ratio_modifier": 1.00, "offset_modifier": 1.00, "skin_thickness": 1.0},
        "overweight": {"ratio_modifier": 1.02, "offset_modifier": 1.2,  "skin_thickness": 1.5},
        "obese":      {"ratio_modifier": 1.05, "offset_modifier": 1.4,  "skin_thickness": 2.0},
        # 体态
        "kyphosis":        {"spine_curve_modifier": 1.5},    # 驼背加大胸椎后凸
        "scoliosis":       {"spine_curve_modifier": 1.3},    # 侧弯
        "anterior_pelvic": {"spine_curve_modifier": 1.3},    # 骨盆前倾加大腰椎前凸
    }

    # 穴位级个性化系数（从标注数据学习得到）
    PER_ACUPOINT_COEFFICIENTS: Dict[str, Dict[str, float]] = {}

    def __init__(self):
        self._learned_coeffs: Dict[str, Dict[str, float]] = {}

    def get_coefficients(self, acupoint_id: str,
                         profile: PopulationProfile) -> AdaptationCoefficients:
        """获取某穴位对某人群的适配系数"""
        coeffs = AdaptationCoefficients()

        # 基础系数：年龄组
        age_key = profile.age_group.value
        if age_key in self.DEFAULT_COEFFICIENTS:
            c = self.DEFAULT_COEFFICIENTS[age_key]
            coeffs.ratio_modifier *= c.get("ratio_modifier", 1.0)
            coeffs.offset_modifier *= c.get("offset_modifier", 1.0)

        # 体型系数
        body_key = profile.body_type.value
        if body_key in self.DEFAULT_COEFFICIENTS:
            c = self.DEFAULT_COEFFICIENTS[body_key]
            coeffs.ratio_modifier *= c.get("ratio_modifier", 1.0)
            coeffs.offset_modifier *= c.get("offset_modifier", 1.0)
            coeffs.skin_thickness = c.get("skin_thickness", 1.0)

        # 体态系数
        posture_key = profile.posture.value
        if posture_key in self.DEFAULT_COEFFICIENTS:
            c = self.DEFAULT_COEFFICIENTS[posture_key]
            coeffs.spine_curve_modifier = c.get("spine_curve_modifier", 1.0)

        # 穴位个性化系数
        group_key = profile.get_group_key()
        acupoint_coeffs = self._learned_coeffs.get(acupoint_id, {})
        if group_key in acupoint_coeffs:
            learned = acupoint_coeffs[group_key]
            coeffs.ratio_modifier *= learned.get("ratio_modifier", 1.0)
            coeffs.offset_modifier *= learned.get("offset_modifier", 1.0)

        return coeffs

    def update_from_annotations(self, annotations_path: str):
        """
        从标注文件学习人群系数

        标注文件格式:
        {
            "annotations": [
                {
                    "acupoint_id": "ST36",
                    "age": 30, "gender": "male", "bmi": 22,
                    "predicted_ratio": 0.1875,
                    "annotated_ratio": 0.19,
                    ...
                }
            ]
        }

        学习算法: 最小二乘回归，拟合 ratio_modifier
        """
        try:
            with open(annotations_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except FileNotFoundError:
            print(f"[PopulationAdapter] 标注文件不存在: {annotations_path}")
            return
        except Exception as e:
            print(f"[PopulationAdapter] 加载标注文件失败: {e}")
            return

        # 按 (acupoint_id, group_key) 分组
        groups: Dict[str, List[float]] = {}
        for ann in data.get("annotations", []):
            ap_id = ann.get("acupoint_id", "")
            age = ann.get("age", 30)
            gender = ann.get("gender", "male")
            bmi = ann.get("bmi", 22)

            profile = PopulationProfile(
                age=age,
                gender=Gender.MALE if gender == "male" else Gender.FEMALE,
                bmi=bmi
            )
            group_key = profile.get_group_key()
            full_key = f"{ap_id}|{group_key}"

            if ann.get("annotated_ratio") and ann.get("predicted_ratio"):
                ratio_error = ann["annotated_ratio"] / ann["predicted_ratio"]
                if full_key not in groups:
                    groups[full_key] = []
                groups[full_key].append(ratio_error)

        # 计算每组均值作为修正系数
        for full_key, errors in groups.items():
            ap_id, group_key = full_key.split("|", 1)
            if ap_id not in self._learned_coeffs:
                self._learned_coeffs[ap_id] = {}
            # 简单平均（后续可改为加权/中位数）
            modifier = sum(errors) / len(errors)
            self._learned_coeffs[ap_id][group_key] = {
                "ratio_modifier": modifier,
                "sample_count": len(errors)
            }

        total = sum(len(v) for v in groups.values())
        print(f"[PopulationAdapter] 从 {total} 条标注数据学习了 {len(groups)} 个人群组")

    def get_learned_count(self) -> int:
        """获取已学习的人群组数量"""
        return sum(len(v) for v in self._learned_coeffs.values())

    def export_coefficients(self) -> Dict:
        """导出所有学习到的系数"""
        return {
            "version": "1.0",
            "coeffs_per_acupoint": self._learned_coeffs,
            "total_groups": self.get_learned_count(),
        }
