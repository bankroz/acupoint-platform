# MediaPipe 骨架经络穴位验证平台

> 本项目基于 MediaPipe Tasks API（Pose Landmarker + Holistic Landmarker）双模型融合，通过骨度分寸法 + 虚拟脊柱构建自动推算经络与穴位坐标，支持图片→高清相机→深度相机的渐进式验证路径。**重点覆盖躯干穴位（任脉/督脉/膀胱经/胃经等），兼顾四肢、面部和手部穴位。**

---

## 目录

- [零、MediaPipe 模型对比与选型](#零mediapipe-模型对比与选型)
- [零-B、躯干穴位定位：虚拟脊柱构建方案](#零-b躯干穴位定位虚拟脊柱构建方案)
- [一、硬件资源需求](#一硬件资源需求)
- [二、系统环境搭建（Windows）](#二系统环境搭建windows)
- [三、输入源渐进路径：图片→相机→深度相机](#三输入源渐进路径图片相机深度相机)
- [四、穴位数据库构建指南](#四穴位数据库构建指南)
- [五、3D交互展示界面实现路径](#五3d交互展示界面实现路径)
- [六、Windows 兼容性说明与障碍解决](#六windows-兼容性说明与障碍解决)
- [七、项目目录结构](#七项目目录结构)
- [八、快速开始](#八快速开始)

---

## 零、MediaPipe 模型对比与选型

### 三种模型方案对比

MediaPipe 提供三套人体关键点检测方案，本项目使用 Tasks API（`.task` 模型文件）进行混合推理：

| 特性 | **Pose Landmarker** | **Holistic Landmarker** |
|------|--------------------|-------------------------|
| API路径 | `mediapipe.tasks.python.vision` | `mediapipe.tasks.python.vision` |
| 骨骼关键点 | **33个** | **33个（Pose）** |
| 面部关键点 | 无 | **468个（Face Mesh）** |
| 手部关键点 | 无 | **每只手21个（Hand）** |
| 总计关键点 | **33** | **543** ⭐ |
| GPU推理速度(RTX3080) | ~8ms/帧 | ~25-35ms/帧 |
| CPU推理速度 | ~50-80ms/帧 | ~200ms/帧 |
| 穴位覆盖区域 | 四肢+躯干 | **四肢+面部+手部+躯干** |
| 模型文件 | `models/pose_landmarker.task` | `models/holistic_landmarker.task` |
| 安装方式 | `pip install mediapipe` | `pip install mediapipe` |

### 本项目选型策略

```
┌─────────────────────────────────────────────────┐
│               穴位定位模型分层策略                  │
├──────────┬─────────────┬──────────────────────────┤
│ 检测区域  │  使用模型     │  覆盖穴位（已录入）        │
├──────────┼─────────────┼──────────────────────────┤
│ 躯干前后  │ Pose（33点）  │ 躯干穴位（任脉+督脉+膀胱经 │
│          │ +虚拟脊柱     │  +胃经+脾经+肾经+肝经+胆经）│
├──────────┼─────────────┼──────────────────────────┤
│ 四肢     │ Pose（33点）  │ 四肢穴位（六条手经+足经）   │
├──────────┼─────────────┼──────────────────────────┤
│ 面部     │ Holistic Face │ 20个穴位（胃经+膀胱经+胆经 │
│          │ Mesh（468点） │  +督脉+三焦经+经外奇穴）   │
├──────────┼─────────────┼──────────────────────────┤
│ 手部     │ Holistic Hand │ 12个定义 → 镜像后24个    │
│          │（每手21点）   │ （肺经+心经+心包经+大肠经+ │
│          │              │  小肠经+三焦经手穴）      │
├──────────┼─────────────┼──────────────────────────┤
│ 合计     │ 双模型融合    │ JSON 86个定义（镜像后98个）│
└──────────┴─────────────┴──────────────────────────┘
```

### 为什么用 Tasks API 而非 Legacy API？

- MediaPipe Legacy API (`mp.solutions`) 已于 2025 年进入维护模式，不再添加新特性
- Tasks API 支持 `.task` 模型文件热加载，无需重新编译
- `Holoistic` 仍可同时输出 Pose + Face + Hands 的联合结果
- **实际实现中**：`PoseExtractor` 封装了 Tasks API 的 `PoseLandmarker` 和 `HolisticLandmarker`，可通过 `ModelMode.POSE` / `ModelMode.HOLISTIC` 切换

### Holistic 带来的关键能力

| 能力 | 穴位举例 |
|------|---------|
| 面部468点精确定位 | 印堂(GV29)、太阳(EX-HN5)、睛明(BL1)、鱼腰(EX-HN4)、丝竹空(TE23)、瞳子髎(GB1)、承泣(ST1)、攒竹(BL2)等20个面部穴位 |
| 手部21点独立追踪 | 合谷(LI4)、劳宫(PC8)、太渊(LU9)、神门(HT7)、少商(LU11)、中冲(PC9)、少冲(HT9)等12种穴位（镜像为24个） |
| 手指关节信息 | 可用于"一夫法"等以手指宽度为单位的取穴辅助 |
| 眉毛/眼眶标记点 | 保留眉毛关键点标注（印堂、攒竹、鱼腰、丝竹空），视觉效果清晰 |

---

## 零-B、躯干穴位定位：虚拟脊柱构建方案

> **核心挑战**：MediaPipe Pose 在躯干区域仅有4个关键点（双肩11/12 + 双髋23/24），而躯干是穴位最密集的区域（任脉24穴、督脉28穴、膀胱经67穴等）。
>
> **解决思路**：利用肩髋4点构建**虚拟脊柱模型**，结合人体解剖比例，推算躯干关键穴位位置。

### 虚拟脊柱构建原理

```
        颈静脉切迹 (胸骨上窝)
              │
    肩(11)────●────肩(12)        ← MediaPipe 已知点
              │ C7 (大椎GV14)
              │ T1
              │ T2-T6
              │ T7 (至阳GV9)
              │ T8-T11
              │ T12
              │ L1
              │ L2 (命门GV4)
              │ L3-L4 (腰阳关GV3)
    髋(23)────●────髋(24)        ← MediaPipe 已知点
              │ L5
              │ 骶骨
```

### 推算步骤

```
Step 1: 计算肩中点 (Shoulder_Mid) = (P11 + P12) / 2
Step 2: 计算髋中点 (Hip_Mid)    = (P23 + P24) / 2
Step 3: 脊柱总长度 = distance(Shoulder_Mid, Hip_Mid)
Step 4: 按解剖比例S形插值，生成虚拟椎骨位置
Step 5: 用肩宽估算胸廓宽度 → 推算旁开穴位（膀胱经、胃经等）
```

### 分经定位矩阵

| 经络 | 参考中线 | 旁开距离 | 定位方法 | 可估算穴位示例 |
|------|---------|---------|---------|--------------|
| **任脉CV** | 前正中线 | 0寸 | 沿肩髋前中线等分 | 天突CV22、膻中CV17、中脘CV12、神阙CV8、气海CV6、关元CV4、中极CV3 |
| **督脉GV** | 后正中线 | 0寸 | 虚拟脊柱沿S形曲线 | 大椎GV14、身柱GV12、至阳GV9、命门GV4、腰阳关GV3 |
| **膀胱经BL(背)** | 后正中线 | 1.5寸 | 脊柱旁1.5寸（肩宽×比例） | 肺俞BL13、心俞BL15、肝俞BL18、脾俞BL20、胃俞BL21、肾俞BL23 |
| **膀胱经BL(腰)** | 后正中线 | 1.5寸 | 同上 | 大肠俞BL25、关元俞BL26、小肠俞BL27、膀胱俞BL28 |
| **胃经ST(胸腹)** | 前正中线 | 胸4寸/腹2寸 | 旁开+垂直等分 | 不容ST19、承满ST20、梁门ST21、天枢ST25、归来ST29 |
| **脾经SP(胸腹)** | 前正中线 | 胸6寸/腹4寸 | 旁开+垂直等分 | 腹哀SP16、大横SP15 |
| **肾经KI(胸腹)** | 前正中线 | 胸2寸/腹0.5寸 | 旁开+垂直等分 | 幽门KI21、肓俞KI16、气穴KI13 |
| **肝经LV(腹)** | 前正中线 | 旁开4寸 | 旁开+垂直等分 | 期门LV14、章门LV13 |
| **胆经GB(胁肋)** | 体侧 | — | 肩髋连线侧向 + 肋弓比例 | 肩井GB21、京门GB25、带脉GB26、环跳GB30 |

### 精度分层

| 精度等级 | 区域 | 误差预估 | 条件 |
|---------|------|---------|------|
| **A级（高精度）** | 四肢穴位 | ±1-2cm | 骨架关键点充足，比例法可靠 |
| **B级（中精度）** | 躯干正中线穴位（任脉/督脉） | ±2-4cm | 依赖脊柱S形插值的准确性 |
| **C级（偏低精度）** | 躯干旁开穴位（膀胱经/胃经等） | ±3-6cm | 依赖肩宽→体宽映射，个体差异大 |
| **D级（低精度）** | 胁肋部穴位（胆经等） | ±4-8cm | 缺少肋骨参考点，依赖体态推算 |

> **重要说明**：C/D级穴位在初期仅作**粗定位参考**，需要专家标注反馈（阶段3）逐步校准。引入深度相机或额外躯干摄像头后精度可跃升。

### 改进路径

```
当前方案（阶段1-2）:
  肩髋4点 → 虚拟脊柱 → 躯干穴位（B/C/D级精度）

未来改进（阶段3-4）:
  ① 多摄像头融合（前方+侧方）→ 获取真实胸廓轮廓
  ② 深度相机 → 体表3D重建 → 肋骨/胸骨识别
  ③ 专家标注反馈 → 人群系数校准 → C/D→B级精度提升
```

---

## 一、硬件资源需求

### 最低配置（验证阶段）

| 组件 | 规格 | 备注 |
|------|------|------|
| CPU | Intel i5 / Ryzen 5，8核以上 | MediaPipe CPU推理依赖多核 |
| GPU | 可无（纯CPU推理） | 图片阶段无需GPU |
| 内存 | 8GB | 单帧图片处理够用 |
| 存储 | SSD 256GB | 避免HDD读写瓶颈 |
| 摄像头 | USB 1080p 网络摄像头 | 第二阶段引入 |
| 深度相机 | Intel RealSense D435 / Azure Kinect | 第三阶段引入 |

### 推荐配置（多人群大样本验证）

| 组件 | 规格 |
|------|------|
| CPU | Intel i7/i9 第12代以上，12核+ |
| GPU | NVIDIA RTX 3060 / 4060（CUDA 12.x） |
| 内存 | 32GB |
| 存储 | NVMe SSD 1TB |
| 深度相机 | Intel RealSense D435i（含IMU） |

---

## 一-B、GPU 加速说明

### 当前状态

**Windows pip 包 (`pip install mediapipe`) 未编译 GPU delegate**，默认使用 CPU + XNNPACK 多线程推理。代码已通过 `_setup_deps.py` 自动检测 GPU 状态并提示优化建议。

| 推理后端 | 速度 | RTX 3080 状态 |
|---------|------|-------------|
| CPU XNNPACK | ~50-80ms/帧 (Pose) | 当前可用 |
| GPU TFLite Delegate | ~8-15ms/帧 | 需额外配置 |

### 启用 GPU 加速的三种方案

```
方案A (推荐): WSL2 + MediaPipe GPU
  ├─ 安装 WSL2 Ubuntu
  ├─ pip install mediapipe (Linux版含GPU delegate)
  └─ 预计提速 5-10x

方案B: 编译 MediaPipe from source
  ├─ 安装 Bazel + CUDA Toolkit
  ├─ 编译 mediapipe with GPU support
  └─ 配置复杂，不推荐

方案C: 使用 PyTorch/ONNX Runtime GPU (替代方案)
  ├─ 将 MediaPipe 模型转 ONNX
  ├─ pip install onnxruntime-gpu
  └─ 需重写推理代码
```

> **建议**: 当前 CPU 模式 50-80ms 对实时视频（30fps=33ms延迟预算）稍慢，但可用。**方案A（WSL2）是最简单高效的 GPU 加速方案。**

---

## 二、系统环境搭建（Windows）

### Step 1：安装 Miniconda（Python环境管理）

```bash
# 下载地址：https://docs.conda.io/en/latest/miniconda.html
# 安装时勾选：Add Miniconda3 to PATH（推荐）
```

### Step 2：创建隔离 Python 环境

```bash
conda create -n acupoint-env python=3.10
conda activate acupoint-env
```

### Step 3：安装核心依赖

```bash
# 基础必装（满足所有功能）
pip install mediapipe opencv-python numpy plotly pillow
```

（`requirements.txt` 中有完整清单，含可选依赖说明）

### Step 4：GPU加速路径（可选，仅NVIDIA显卡）

```bash
# 确认显卡驱动版本 >= 535
nvidia-smi

# 安装 CUDA Toolkit（如使用 TensorFlow GPU delegate 时才需要）
# 官网下载：developer.nvidia.com/cuda-downloads
```

**注意**：当前默认 CPU 推理对图片验证完全够用，GPU 加速非必需。

### Step 5：验证 MediaPipe 安装

```bash
python -c "import mediapipe as mp; print('MediaPipe version:', mp.__version__)"
```

### Step 6：下载模型文件

```bash
python scripts/download_models.py
```

> 模型文件自动保存到 `models/` 目录。如果已存在则跳过下载。

---

## 三、输入源渐进路径：图片→相机→深度相机

### 阶段一：静态图片（当前推荐起点）

```python
from core.pose_extractor import PoseExtractor, ModelMode
import cv2

def process_image(image_path: str):
    """
    从本地图片提取骨架关键点
    支持格式：JPG / PNG / BMP
    """
    extractor = PoseExtractor(mode=ModelMode.HOLISTIC)
    image = cv2.imread(image_path)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    result = extractor.extract(rgb)
    # result.pose_landmarks：33个骨架点（归一化坐标 0-1）
    # result.pose_world_landmarks：真实 3D 坐标（米）
    # result.face_landmarks：468 个面部点（Holistic模式）
    # result.left_hand_landmarks / right_hand_landmarks：手部 21 点
    return result, image
```

> **注意**：图片来源建议：
> - 标准站姿正面/侧面照（站距肩宽，双臂自然下垂）
> - 分辨率 >= 720p，全身入框
> - 避免衣物遮挡（紧身衣为佳）

### 阶段二：高清相机（实时视频流）

```python
def open_camera():
    """
    Windows 下推荐使用 CAP_DSHOW 后端，稳定性更好
    """
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    return cap

# 实时推理使用 video 模式：
# PoseExtractor 内部使用 RunningMode.LIVE_STREAM 增加时序平滑
```

### 阶段三：深度相机（Intel RealSense D435）

```python
import pyrealsense2 as rs
import numpy as np

def open_realsense():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    profile = pipeline.start(config)

    # 获取深度传感器内参，用于深度→3D转换
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    return pipeline, depth_scale

# 深度相机能给出真实Z轴深度，解决MediaPipe Z轴精度不足的问题
# 穴位坐标从"2D+估算Z"升级为"真实3D空间坐标"
```

---

## 四、穴位数据库构建指南

### 设计原则

穴位数据库采用**三层架构**，按身体区域分文件管理：

```
第一层（静态基准）  ←→  第二层（人群动态适配）  ←→  第三层（标注反馈迭代）
JSON 区域定义           年龄/性别/体态/BMI系数        人工拖拽修正 → 参数优化
```

实际穴位定义分为 4 个 JSON 文件：

| 文件 | 区域 | 定义数 | 说明 |
|------|------|--------|------|
| `acupoints_torso.json` | 躯干 | 躯干穴位 | 任脉/督脉/膀胱经/胃经/脾经/肾经/肝经/胆经 |
| `acupoints_limbs.json` | 四肢 | 四肢穴位 | 手三阴经+手三阳经+足三阴经+足三阳经 |
| `acupoints_face.json` | 面部 | 20个 | 含眼部穴位（睛明/鱼腰/丝竹空/瞳子髎/承泣）|
| `acupoints_hands.json` | 手部 | 12个 | 运行时自动镜像为 24 个（左手+右手各12） |

### 数据格式规范（JSON）

```json
{
  "acupoints": [
    {
      "id": "ST36",
      "name_cn": "足三里",
      "name_pinyin": "ZuSanLi",
      "meridian": "足阳明胃经",
      "meridian_code": "ST",
      "index_in_meridian": 36,

      "location_rule": {
        "method": "bone_ratio",
        "landmark_proximal": 26,
        "landmark_distal": 28,
        "landmark_side": "right",
        "ratio": 0.1875,
        "offset_direction": "anterior_lateral",
        "offset_cun": 1.0,
        "cun_per_segment": 16,
        "description": "膝眼下3寸，胫骨前嵴外一横指"
      },

      "clinical": {
        "functions": ["健脾胃", "强壮要穴", "调理气血"],
        "indications": ["胃痛", "腹泻", "下肢痿痹", "保健强壮"],
        "needling_depth_cm": 1.5
      },

      "population_adjustments": {
        "child_0_12": {"ratio_modifier": 0.95, "note": "骨骼较短，比例微调"},
        "elderly_65plus": {"ratio_modifier": 1.02, "note": "骨骼退化延长"},
        "female": {"ratio_modifier": 1.0, "note": "胫骨段比例相近"},
        "obese_bmi_30plus": {"offset_modifier": 1.3, "note": "脂肪层增厚"}
      },

      "validation": {
        "confidence_min": 0.7,
        "visibility_required_landmarks": [26, 28],
        "fallback_method": "2d_projection"
      }
    }
  ]
}
```

### 数据库表结构（SQLite）

```sql
-- 标准穴位基准表（静态）
CREATE TABLE acupoints (
    id TEXT PRIMARY KEY,
    name_cn TEXT NOT NULL,
    meridian_code TEXT,
    landmark_proximal INTEGER,
    landmark_distal INTEGER,
    ratio REAL,
    offset_cun REAL,
    offset_direction TEXT,
    data_json TEXT
);

-- 人群系数表（动态，可更新）
CREATE TABLE population_coefficients (
    acupoint_id TEXT,
    group_key TEXT,          -- e.g. "child_0_12", "female", "obese"
    ratio_modifier REAL DEFAULT 1.0,
    offset_modifier REAL DEFAULT 1.0,
    updated_at TIMESTAMP,
    PRIMARY KEY (acupoint_id, group_key)
);

-- 标注记录表（用于修正模型）
CREATE TABLE annotations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    acupoint_id TEXT,
    source_image TEXT,
    subject_age INTEGER,
    subject_gender TEXT,
    subject_bmi REAL,
    predicted_x REAL, predicted_y REAL, predicted_z REAL,
    annotated_x REAL, annotated_y REAL, annotated_z REAL,
    delta_x REAL, delta_y REAL, delta_z REAL,
    annotator TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 系数优化历史（可追溯）
CREATE TABLE coefficient_history (
    acupoint_id TEXT,
    group_key TEXT,
    old_ratio REAL,
    new_ratio REAL,
    sample_count INTEGER,
    updated_at TIMESTAMP
);
```

### 人群覆盖策略

```
年龄分组（5组）：
  child_0_12（儿童）  teen_13_17（青少年）  adult_18_60（成人）
  elderly_60_75（老年）  senior_75plus（高龄）

性别（2组）：男 / 女

体型（4档，基于BMI）：
  underweight BMI<18.5  normal 18.5-24  overweight 24-28  obese 28+

适配参数通过 core/population_adapter.py 实现：
  - ratio_modifier：比例修正系数（影响骨度分寸计算）
  - offset_modifier：偏移修正系数（影响体表投影位置）
  - skin_thickness：皮肤/脂肪厚度修正（影响穴位深度估算）
```

### 用户画像输入接口

启动 `demo_verify.py` 时，程序会自动弹出交互式采集界面：

```
  请输入您的身体参数（用于穴位定位精度校准）
  性别 [1=男 / 2=女, 默认1]: 1
  年龄 [默认30]: 28
  身高(cm) [默认170]: 175
  体重(kg) [默认65]: 72

  [OK] 画像已保存: 28岁 男性  175cm/72kg  BMI=23.5
  年龄组: adult_18_60  体型: normal
```

画像自动持久化到 `database/user_profile.json`，下次启动可直接复用。

### 标注工具工作流

```
1. 输入图片 → MediaPipe 输出预测穴位坐标（绿色点）
2. 人工审核：若预测正确，点击"确认"
3. 若有偏差：拖拽绿色点到正确位置（蓝色点 = 修正后坐标）
4. 记录主体信息（年龄/性别/BMI）→ 写入 annotations 表
5. 积累 N 条同组标注后，触发"参数优化"：
   用最小二乘回归拟合 ratio_modifier = mean(annotated_ratio / base_ratio)
6. 更新 population_coefficients，重新计算该人群的穴位位置
```

---

## 五、3D交互展示界面实现路径

### 推荐方案：Open3D（Windows原生支持最佳）

Open3D 内置 GUI 窗口支持**鼠标旋转/拖拽/滚轮缩放**，无需自己实现交互逻辑。

```python
import open3d as o3d
import numpy as np

def visualize_skeleton_3d(world_landmarks, acupoints_3d: dict):
    """
    world_landmarks: MediaPipe pose_world_landmarks
    acupoints_3d: {"足三里": [x, y, z], ...}
    """
    # --- 骨架线段定义（MediaPipe 33个关键点连接关系）---
    CONNECTIONS = [
        (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
        (11, 23), (12, 24), (23, 24),
        (23, 25), (25, 27), (24, 26), (26, 28),
        (27, 29), (27, 31), (28, 30), (28, 32),
    ]

    # 提取骨架点云
    skeleton_pts = np.array([
        [lm.x, -lm.y, -lm.z]  # Y轴翻转符合3D直觉（Y朝上）
        for lm in world_landmarks.landmark
    ])

    # 创建骨架点
    skeleton_pcd = o3d.geometry.PointCloud()
    skeleton_pcd.points = o3d.utility.Vector3dVector(skeleton_pts)
    skeleton_pcd.paint_uniform_color([0.5, 0.5, 0.5])

    # 创建骨架连线
    lines = o3d.geometry.LineSet()
    lines.points = o3d.utility.Vector3dVector(skeleton_pts)
    lines.lines = o3d.utility.Vector2iVector(CONNECTIONS)
    lines.paint_uniform_color([0.7, 0.7, 0.7])

    # 穴位点（红色大球）
    acupoint_spheres = []
    meridian_colors = {
        "足阳明胃经": [1.0, 0.3, 0.0],
        "手厥阴心包经": [0.2, 0.6, 1.0],
        "足太阴脾经": [0.8, 0.2, 0.8],
        "督脉": [1.0, 0.8, 0.0],
    }
    for name, (x, y, z) in acupoints_3d.items():
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.012)
        sphere.translate([x, -y, -z])
        color = meridian_colors.get("足阳明胃经", [1.0, 0.0, 0.0])
        sphere.paint_uniform_color(color)
        acupoint_spheres.append(sphere)

    # 启动交互式3D窗口（支持旋转/平移/缩放）
    geometries = [skeleton_pcd, lines] + acupoint_spheres
    o3d.visualization.draw_geometries(
        geometries,
        window_name="骨架经络穴位 3D 视图",
        width=1280, height=800,
        mesh_show_back_face=True
    )
    # 操作方法：
    # 鼠标左键拖拽 → 旋转
    # 鼠标中键拖拽 → 平移
    # 滚轮 → 缩放
    # 键盘 R → 重置视角
```

### 备选方案对比

| 方案 | Windows 支持 | 交互性 | 适用场景 |
|------|-------------|--------|----------|
| **Open3D** | 原生支持 | 鼠标旋转/平移/缩放 | 推荐首选 |
| **PyVista** | 支持（需VTK） | 同上，API更高级 | 需要更复杂场景 |
| **Plotly（网页版）** | 浏览器运行 | 旋转/缩放/hover | 无需安装，分享方便 |
| **Three.js + Gradio** | 浏览器运行 | 完全自定义 | 需前端开发能力 |
| **Mayavi** | Windows安装复杂 | 功能强大 | 不推荐Windows首选 |

### Plotly 3D方案（无需安装额外依赖，浏览器交互）

```python
import plotly.graph_objects as go

def plotly_3d_view(skeleton_pts, acupoints_3d):
    fig = go.Figure()

    # 骨架点
    fig.add_trace(go.Scatter3d(
        x=skeleton_pts[:, 0], y=skeleton_pts[:, 1], z=skeleton_pts[:, 2],
        mode='markers', marker=dict(size=4, color='gray'),
        name='骨骼关键点'
    ))

    # 穴位点
    ax = [v[0] for v in acupoints_3d.values()]
    ay = [v[1] for v in acupoints_3d.values()]
    az = [v[2] for v in acupoints_3d.values()]
    anames = list(acupoints_3d.keys())

    fig.add_trace(go.Scatter3d(
        x=ax, y=ay, z=az,
        mode='markers+text',
        text=anames,
        marker=dict(size=8, color='red'),
        name='穴位'
    ))

    fig.update_layout(
        scene=dict(aspectmode='data'),
        title='骨架经络穴位 3D 视图'
    )
    fig.show()  # 在浏览器中打开，支持旋转/缩放/hover显示穴位名
```

---

## 六、Windows 兼容性说明与障碍解决

### 已知问题及解决方案

| 问题 | 现象 | 解决方案 |
|------|------|----------|
| 中文路径报错 | MediaPipe 模型加载失败 | 所有路径使用英文，避免中文目录 |
| 摄像头初始化失败 | cv2.VideoCapture(0) 卡住 | 改用 `cv2.VideoCapture(0, cv2.CAP_DSHOW)` |
| CUDA 找不到 | TF 不识别 GPU | 安装顺序严格：驱动→CUDA→cuDNN，版本必须匹配 |
| Open3D 窗口无法打开 | 报 OpenGL 错误 | 安装 Visual C++ Redistributable 2019+ |
| RealSense 驱动问题 | `pyrealsense2` 导入失败 | 先安装 [Intel RealSense SDK 2.0](https://github.com/IntelRealSense/librealsense/releases)，再 pip 安装 |
| MediaPipe 版本冲突 | 新旧 API 混用报错 | 统一使用新版 Tasks API：`mediapipe.tasks.python` |

### Windows 下更好的替代路径

**如果 Open3D 安装遇到障碍：**
```bash
pip install pyvista   # 备选3D库，基于VTK，Windows支持良好
# 或者直接用 Plotly，在浏览器展示，零安装障碍
pip install plotly
```

**如果 CUDA 配置复杂：**
- 纯 CPU 推理对图片验证完全够用，先跳过 GPU 配置
- MediaPipe 的 `model_complexity=2` 在 CPU 下处理单张图片约 200-500ms，完全可接受

**推荐的纯 Windows 友好技术栈（无 GPU，无深度相机阶段）：**
```
Python 3.10 (Miniconda) + MediaPipe Tasks API + OpenCV + Plotly + SQLite
```
以上组合在 Windows 下安装成功率接近 100%，无需任何额外系统组件。

---

## 七、项目目录结构

```
acupoint-platform/
│
├── README.md                        # 本文件
├── DEPLOY.md                        # 部署文档
├── config.yaml                      # 统一配置（路径/阈值/人群参数）
├── requirements.txt                 # 完整依赖清单
├── setup.bat                        # 环境安装与依赖检查脚本
├── 一键启动模型.bat                  # 一键启动（推荐入口）
│
├── core/                            # 【核心引擎层】
│   ├── pose_extractor.py            # MediaPipe Tasks API 骨架/全身提取封装
│   ├── spine_estimator.py           # 虚拟脊柱构建器（C7~L5 椎骨估算）
│   ├── acupoint_locator.py          # 穴位坐标计算引擎（骨度分寸法）
│   └── population_adapter.py        # 人群系数适配（年龄/性别/BMI）
│
├── database/                        # 【穴位数据库层】
│   ├── acupoints_torso.json         # 躯干穴位定义
│   ├── acupoints_limbs.json         # 四肢穴位定义
│   ├── acupoints_face.json          # 面部穴位定义（20个，含眼部穴位）
│   ├── acupoints_hands.json         # 手部穴位定义（12个，运行时镜像为24个）
│   ├── acupoint.db                  # SQLite 数据库（运行时生成）
│   └── db_init.py                   # 数据库初始化脚本
│
├── models/                          # 【AI 模型文件】
│   ├── pose_landmarker.task         # Pose 模型（~29MB）
│   └── holistic_landmarker.task     # Holistic 模型（~13MB）
│
├── scripts/                         # 【脚本/演示层】
│   ├── demo_verify.py               # 主验证脚本（图片+摄像头+交互式输入）
│   ├── demo_realtime.py             # 实时摄像头穴位定位演示
│   ├── demo_torso.py                # 躯干穴位定位演示
│   ├── download_models.py           # 模型下载脚本
│   └── _setup_deps.py               # setup.bat 依赖检查辅助
│
├── ui/                              # 【用户界面层】
│   ├── viewer_3d.py                 # Plotly 3D 交互展示
│   └── realtime_overlay.py          # 实时摄像头穴位叠加工具
│
├── input/                           # 【输入数据】
│   └── images/                      # 测试图片集（JPG/PNG）
│
└── output/                          # 【输出结果】
    ├── acupoints_3d.html            # 3D 穴位可视化 HTML
    ├── annotations/                 # 标注结果图片
    ├── screenshots/                 # 截图保存
    └── visualizations/              # 可视化输出
```

---

## 八、快速开始

### 一键验证（推荐）

```bash
cd d:\CodeBuddy\MediaPipe\acupoint-platform

# 方式1：双击运行 bat 脚本
一键启动模型.bat

# 方式2：命令行运行
python scripts/demo_verify.py
```

> 首次运行会进行：
> 1. **模型检查**：自动检测 `models/` 目录下的 `.task` 模型文件，如缺失则提示下载
> 2. **用户画像采集**：交互式输入身高/体重/年龄/性别，用于穴位定位的人群系数适配
> 3. **模式选择**：选择图片模式（拍照/文件）或实时摄像头模式
> 4. **穴位定位管线**：运行完整的骨架提取 → 脊柱构建 → 穴位坐标计算流程

### 实时摄像头模式

```bash
python scripts/demo_realtime.py
```

操作按键：
| 按键 | 功能 |
|------|------|
| Q | 退出 |
| S | 保存截图到 output/screenshots/ |
| G | 循环切换精度过滤 (A→B→C→D→ALL) |
| 1-4 | 直接选择精度等级 |

### 图片/躯干模式

```bash
# 放入全身站姿照到 input/images/ 目录
python scripts/demo_torso.py
```

输出：
- `output/acupoints_3d.html` — 交互式 3D 视图（Plotly，浏览器打开）

### 模型下载

```bash
# 首次运行前确保模型文件已下载
python scripts/download_models.py
```

模型文件存放于 `models/` 目录：
- `pose_landmarker.task`（Pose 骨架检测）
- `holistic_landmarker.task`（全身检测：面部+手部+骨架）

### 安装依赖（仅首次）

```bash
pip install mediapipe opencv-python numpy plotly pillow
```

### GPU 加速（可选，推荐 WSL2）

Windows pip 包默认使用 CPU。启用 GPU 可提速 5-10x：

```bash
# 方案A: WSL2（推荐）
wsl --install                    # 安装 WSL2 Ubuntu
wsl cd /mnt/d/CodeBuddy/MediaPipe/acupoint-platform
pip install mediapipe            # Linux版含 GPU delegate
python scripts/demo_verify.py    # GPU自动启用

# 方案B: 自编译（复杂，不推荐）
# 见上方 "一-B、GPU 加速说明"
```

---

## 附录：MediaPipe Pose 33个关键点索引

```
0=鼻尖  1-4=眼周  5-6=耳  7-8=嘴角
11=左肩  12=右肩  13=左肘  14=右肘  15=左腕  16=右腕
17=左小指根  18=右小指根  19=左食指尖  20=右食指尖  21=左拇指  22=右拇指
23=左髋  24=右髋  25=左膝  26=右膝  27=左踝  28=右踝
29=左脚跟  30=右脚跟  31=左脚趾  32=右脚趾

注：MediaPipe 以人物自身解剖学左右为准。
    画面做了水平镜像翻转（cv2.flip），因此：
    - 屏幕右侧显示的手 = 人物左手（MediaPipe left_hand）
    - 屏幕左侧显示的手 = 人物右手（MediaPipe right_hand）
    HUD 标注文字已根据镜像方向修正，标注的 "左/右" 为视觉方向。
```

---

*文档版本：v2.0 | 更新日期：2026-05-28*
*更新内容：同步项目架构（Tasks API）、修正穴位数量、更新目录结构与启动入口*
