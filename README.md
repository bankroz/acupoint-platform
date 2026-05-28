# MediaPipe 穴位自动定位平台

> 基于 Google MediaPipe AI 人体姿态检测，通过虚拟脊柱构建 + 骨度分寸法 + 面部/手部追踪，自动推算全身穴位坐标。支持图片、实时摄像头、面部+手部三种检测模式，输出交互式 3D 可视化和 2D 标注叠加图。

---

## 目录

- [设计意图](#设计意图)
- [模块架构](#模块架构)
- [数据流管线](#数据流管线)
- [项目结构](#项目结构)
- [快速开始](#快速开始)
- [检测模式](#检测模式)
- [穴位精度体系](#穴位精度体系)
- [踩坑记录](#踩坑记录)
- [路线图](#路线图)

---

## 设计意图

### 问题背景

中医穴位的传统定位依赖**骨度分寸法**——以人体骨骼标志为参考，按比例推算穴位位置。这个过程高度依赖医师的经验和对个体差异的感知，标准化难度大。

### 核心思路

利用计算机视觉技术，将骨度分寸法**数字化、自动化**：

```
传统定位:  医师目测 → 触摸骨骼标志 → 用手比划"寸" → 按压定位
           ↓
数字化定位: 摄像头/图片 → MediaPipe AI 骨架检测 → 虚拟脊柱构建
           → 骨度比例计算 → 穴位坐标输出 → 3D可视化验证
```

### 为什么选 MediaPipe

| 方案 | 优势 | 劣势 |
|------|------|------|
| **MediaPipe** | 免费、离线运行、33点+手部+面部、跨平台 | 躯干点少（仅4个） |
| OpenPose | 25/135点可选 | 安装复杂、需GPU |
| 纯手工标定 | 精度高 | 不可规模化 |
| **本案选择** | MediaPipe + 虚拟脊柱弥补躯干不足 | — |

### 核心创新点

1. **虚拟脊柱算法**：仅用肩髋 4 个关键点，通过正弦波叠加模拟人体 S 形生理弯曲，推算出 C7~L5 共 18 节椎骨的三维坐标
2. **前后正中线分离**：自动检测人体朝向（正面/背面），分别定位任脉（前）和督脉（后）穴位
3. **多精度分层**：A/B/C/D 四级精度，基于骨骼数据丰富度自动分级，用户可按精度过滤
4. **手部自动镜像**：右手穴位定义自动生成左手副本，无需手动维护两套数据

---

## 模块架构

```
┌─────────────────────────────────────────────────────────┐
│                        scripts/                         │
│  ┌─────────────┐ ┌──────────┐ ┌──────────────┐         │
│  │demo_verify  │ │demo_torso│ │demo_realtime │         │
│  │(主入口)      │ │(躯干演示) │ │(实时摄像头)   │         │
│  └──────┬──────┘ └────┬─────┘ └──────┬───────┘         │
│         │             │              │                  │
│         └──────────┬──┘──────────────┘                  │
│                    ▼                                    │
├─────────────────────────────────────────────────────────┤
│                      core/                              │
│  ┌──────────────────────────────────────────────────┐   │
│  │              PoseExtractor                       │   │
│  │  ┌─────────┐  ┌──────────────────┐              │   │
│  │  │  Pose   │  │    Holistic      │              │   │
│  │  │ (33点)  │  │ (33+468+21×2点)  │              │   │
│  │  └────┬────┘  └────────┬─────────┘              │   │
│  │       └────────┬───────┘                        │   │
│  │                ▼                                │   │
│  │           PoseResult                            │   │
│  │   (统一数据结构：pose/face/hands)               │   │
│  └──────────────────────┬───────────────────────────┘   │
│                         ▼                               │
│  ┌──────────────────────────────────────────────────┐   │
│  │            SpineEstimator                        │   │
│  │  ┌─────────────────────────────────────┐         │   │
│  │  │ 肩(11,12) + 髋(23,24) → 虚拟脊柱     │         │   │
│  │  │ S形正弦波弯曲 → 18节椎骨坐标          │         │   │
│  │  │ → 前后正中线(各50个插值点)             │         │   │
│  │  │ → SpineResult                        │         │   │
│  │  └─────────────────────────────────────┘         │   │
│  └──────────────────────┬───────────────────────────┘   │
│                         ▼                               │
│  ┌──────────────────────────────────────────────────┐   │
│  │           AcupointLocator                        │   │
│  │  ┌──────────────────────────────────────────┐    │   │
│  │  │ 方法分派（按 location_rule.method）:       │    │   │
│  │  │  spine_bone_ratio  → 椎骨比例定位(躯干)   │    │   │
│  │  │  bone_proportion   → 骨度比例定位(四肢)   │    │   │
│  │  │  midline_ratio     → 正中线比例(任督脉)   │    │   │
│  │  │  face_mesh         → 面部网格定位         │    │   │
│  │  │  hand_landmark     → 手部关键点定位       │    │   │
│  │  │  + 朝向校验(正面/背面/掌心/掌背)           │    │   │
│  │  │  + 2D/3D 分离计算(避免投影误差)            │    │   │
│  │  └──────────────────────────────────────────┘    │   │
│  │                    ▼                             │   │
│  │              AcupointResult                      │   │
│  │       (穴位列表 + 统计 + 经络分组)                │   │
│  └──────────────────────┬───────────────────────────┘   │
│                         ▼                               │
│  ┌──────────────────────────────────────────────────┐   │
│  │          PopulationAdapter                       │   │
│  │  年龄/性别/BMI/体态 → 人群系数 → 定位置信度修正    │   │
│  │  标注反馈 → 最小二乘回归 → 自动学习系数            │   │
│  └──────────────────────────────────────────────────┘   │
│                                                         │
├─────────────────────────────────────────────────────────┤
│                       ui/                               │
│  ┌────────────┐ ┌──────────────┐ ┌─────────────┐       │
│  │ viewer_3d  │ │  realtime_   │ │   image_    │       │
│  │ (Plotly    │ │   overlay    │ │  annotator  │       │
│  │ 交互式3D)  │ │ (实时叠加)    │ │ (人工标注)   │       │
│  └────────────┘ └──────────────┘ └─────────────┘       │
│                                                         │
├─────────────────────────────────────────────────────────┤
│                     database/                           │
│  ┌─────────────────────┐  ┌──────────────────┐         │
│  │ acupoints_torso.json│  │ acupoint.db       │         │
│  │ acupoints_limbs.json│  │ (SQLite: 穴位基准 │         │
│  │ acupoints_face.json │  │  +种群系数+标注    │         │
│  │ acupoints_hands.json│  │  +历史可追溯)      │         │
│  └─────────────────────┘  └──────────────────┘         │
└─────────────────────────────────────────────────────────┘
```

### 核心模块详解

#### 1. `core/pose_extractor.py` — 姿态提取器

封装 MediaPipe Tasks API，作为整个系统的数据入口。

```
PoseExtractor
├── ModelMode.POSE     → PoseLandmarker (33点骨架)
└── ModelMode.HOLISTIC → HolisticLandmarker (骨架+面部468+手42)

关键设计:
  - IMAGE vs VIDEO 模式自动切换
  - _unwrap_landmarks() 统一处理两种模式的数据结构差异
  - GPU 回退机制：Windows pip 包不支持GPU delegate，自动降级CPU
  - Holistic 模型缺失 → 自动回退 Pose 模式
```

**输出的 `PoseResult` 结构体**：
- `pose_landmarks`：(33,4) 归一化图像坐标 [x,y,z,visibility]
- `pose_world_landmarks`：(33,4) 真实3D世界坐标（米）
- `face_landmarks`：(N,3) 面部468点归一化坐标（仅Holistic）
- `left/right_hand_landmarks`：(21,3) 手部关键点（仅Holistic）

#### 2. `core/spine_estimator.py` — 虚拟脊柱构建器

解决 MediaPipe 躯干仅 4 个关键点的核心难题。

```
输入: 肩(11,12) + 髋(23,24) 的3D世界坐标
     ┌──────────────┐
     │ 直线路径插值   │  肩中点 → 髋中点，18等分
     │ + S形弯曲叠加  │  三段正弦波模拟生理弯曲
     │  颈椎 t=0~0.08 │  前凸(+) 
     │  胸椎 t=0.08~0.65│ 后凸(-)  
     │  腰椎 t=0.65~1.0│  前凸(+)
     └──────┬───────┘
            ▼
     18节椎骨3D坐标 (C7, T1-T12, L1-L5)
     + 前正中线(任脉路径, 50点)
     + 后正中线(督脉路径, 50点)
     + 肩宽/躯干宽/估算身高
```

**关键参数**：
- `cervical_lordosis=0.15`：颈椎前凸系数
- `thoracic_kyphosis=0.35`：胸椎后凸系数
- `lumbar_lordosis=0.50`：腰椎前凸系数
- `surface_depth=0.05`：体表到脊柱的深度比例

#### 3. `core/acupoint_locator.py` — 穴位定位器（核心算法）

遍历所有穴位定义，根据 `location_rule.method` 分派到对应计算函数：

| 方法 | 适用区域 | 依赖数据 | 精度 |
|------|---------|---------|------|
| `spine_bone_ratio` | 躯干正中/旁开(任督脉) | 虚拟脊柱椎骨坐标 | B/C |
| `bone_proportion` | 四肢 | Pose 骨骼线段 | A |
| `midline_ratio` | 躯干前后正中线 | 前后正中线路径 | B/C |
| `face_mesh` | 面部穴位 | Holistic Face Mesh 468点 | A |
| `hand_landmark` | 手部穴位 | Holistic Hand 21点 | A |

**关键设计决策**：

- **2D/3D 分离计算**：3D 坐标从 world landmarks 计算（供 AR/可视化），2D 坐标直接从归一化图像 landmarks 计算（避免 world→image 投影的累积误差）
- **朝向校验**：检测身体正面/背面、手掌心/掌背，穴位仅在匹配朝向时高精度，不匹配时降级为 D
- **手部自动镜像**：右手穴位的定义自动生成左手副本（`_mirror_hand_acupoint`）

#### 4. `core/population_adapter.py` — 人群适配器

根据受检者的年龄、性别、BMI、体态特征调整穴位定位参数。

```
PopulationProfile(age, gender, bmi)
  → get_coefficients(acupoint_id, profile)
  → AdaptationCoefficients(ratio_modifier, offset_modifier, spine_curve_modifier, skin_thickness)
  → 修正穴位计算参数
```

支持从专家标注数据中**自动学习**系数（最小二乘回归）。

---

## 数据流管线

```
图片/摄像头帧
    │
    ▼
┌─────────────────┐
│  PoseExtractor   │  BGR → RGB → MediaPipe推理
│  .process()      │  输出 PoseResult (含所有landmarks)
└────────┬────────┘
         │
    ┌────┴────────────────┐
    ▼                     ▼
┌──────────────┐   ┌──────────────────┐
│SpineEstimator│   │ AcupointLocator  │
│.build()      │   │ .locate()        │
│              │   │                  │
│虚拟脊柱 →     │   │ 遍历穴位定义 →    │
│SpineResult   │──▶│ _locate_single()  │
│              │   │                  │
│ 椎骨坐标      │   │ 5种定位方法分派    │
│ 前后正中线    │   │ + 2D/3D分离计算   │
│ 尺寸估算      │   │ + 朝向校验        │
└──────────────┘   └────────┬─────────┘
                            │
                            ▼
                   ┌─────────────────┐
                   │ AcupointResult  │
                   │ 穴位列表+统计     │
                   └────────┬────────┘
                            │
              ┌─────────────┼──────────────┐
              ▼             ▼              ▼
        ┌──────────┐ ┌──────────┐ ┌─────────────┐
        │Viewer3D  │ │  draw_   │ │  realtime_  │
        │(Plotly   │ │ acupoints│ │  overlay    │
        │ 3D交互)  │ │ (2D标注) │ │  (实时叠加)  │
        └──────────┘ └──────────┘ └─────────────┘
```

---

## 项目结构

```
acupoint-platform/
│
├── README.md                    # 本文件（设计文档）
├── DEPLOY.md                    # 部署指南
├── setup.bat                    # 一键部署脚本（双击运行）
├── 一键验证.bat                  # 一键启动检测
├── config.yaml                  # 统一配置文件
├── requirements.txt             # Python 依赖
│
├── core/                        # 核心算法模块
│   ├── __init__.py
│   ├── pose_extractor.py        # MediaPipe 封装（Pose/Holistic双模式）
│   ├── spine_estimator.py       # 虚拟脊柱构建器（S形弯曲算法）
│   ├── acupoint_locator.py      # 穴位定位器（5种方法分派+2D/3D分离）
│   └── population_adapter.py    # 人群系数适配器（年龄/性别/BMI/体态）
│
├── database/                    # 穴位数据库
│   ├── acupoints_torso.json     # 躯干穴位（170+穴）
│   ├── acupoints_limbs.json     # 四肢穴位（130+穴）
│   ├── acupoints_face.json      # 面部穴位（40+穴）
│   ├── acupoints_hands.json     # 手部穴位（30+穴）
│   └── db_init.py               # SQLite初始化
│
├── models/                      # MediaPipe模型文件
│   ├── pose_landmarker.task     # Pose模型（自动下载）
│   └── holistic_landmarker.task # Holistic模型（自动下载）
│
├── scripts/                     # 可执行脚本
│   ├── demo_verify.py           # ★ 主入口（图片/摄像头/面部手部三种模式）
│   ├── demo_torso.py            # 躯干穴位专项演示
│   ├── demo_realtime.py         # 实时摄像头模式
│   └── download_models.py       # 模型下载工具
│
├── ui/                          # 可视化模块
│   ├── viewer_3d.py             # Plotly 3D交互可视化
│   ├── realtime_overlay.py      # 实时摄像头叠加层
│   └── image_annotator.py       # 图片标注工具
│
├── input/                       # 输入数据
│   └── images/                  # 测试图片
│
└── output/                      # 输出结果
    ├── annotations/             # 标注结果
    ├── visualizations/          # 可视化文件
    └── screenshots/             # 截图
```

---

## 快速开始

### 依赖环境

- **Python 3.9~3.11**（必须是这个范围，MediaPipe 目前不支持 3.12+）
- **Windows 10/11**
- **摄像头**（可选，图片模式不需要）

### 一行命令启动

```bash
# 首次使用：双击 setup.bat 一键部署
# 部署完成后：双击 一键验证.bat 启动检测
```

### 手动启动

```bash
# 安装依赖
pip install mediapipe opencv-python numpy plotly pillow

# 下载模型文件
python scripts/download_models.py

# 启动主程序
python scripts/demo_verify.py
```

---

## 检测模式

程序启动后提供三种检测模式：

| 模式 | 说明 | 适用场景 |
|------|------|---------|
| **模式1：图片模式** | 选择本地图片（JPG/PNG/BMP），自动分析全身穴位 | 静态图片分析、批量处理 |
| **模式2：全身摄像头** | 实时摄像头预览，空格键拍照后分析全身穴位 | 需要全身入镜的精确定位 |
| **模式3：面部+手部** | 仅需要面部和手在镜头前，实时标注面部和手部穴位 | 快速演示、不需要全身 |

### 各项操作按键

| 按键 | 功能 |
|------|------|
| Q | 退出 |
| 空格 | 拍照分析（模式2）/ 选择文件（模式1） |
| S | 保存截图 |
| G | 循环切换精度过滤（A→B→C→D→ALL） |
| 1-4 | 直接选择精度等级 |
| H | 切换骨架显示 |
| R | 重置3D视图 |

---

## 穴位精度体系

| 精度等级 | 区域 | 误差预估 | 条件 |
|---------|------|---------|------|
| **A级（高精度）** | 四肢穴位 | ±1-2cm | 骨架关键点充足 |
| **B级（中精度）** | 躯干正中线 | ±2-4cm | 依赖脊柱插值准确性 |
| **C级（偏低精度）** | 躯干旁开 | ±3-6cm | 肩宽→体宽映射 |
| **D级（低精度）** | 胁肋部 | ±4-8cm | 缺少肋骨参考点 |

---

## 踩坑记录

### 坑1：IMAGE vs VIDEO 模式的数据结构差异（最隐蔽）

**现象**：
- 图片模式下穴位位置正确，摄像头模式下穴位偏移严重
- `demo_verify.py` 中的 `_unwrap_landmarks()` 处理了这个问题，但被调用方经常忽略

**原因**：MediaPipe Tasks API 的 `detect()`（IMAGE模式）返回的 landmarks 是**平铺列表**，而 `detect_for_video()`（VIDEO模式）返回的是**嵌套列表**（第一层是多人，第二层是单人的点）。

```
IMAGE:  result.pose_landmarks[0].x  ✓ (NormalizedLandmark)
VIDEO:  result.pose_landmarks[0][0].x  ✓ (需要先取第一人)
```

**修复**：统一使用 `_unwrap_landmarks()` 静态方法，在 `PoseExtractor._extract_holistic_result()` 中处理。

---

### 坑2：world坐标投影到2D图像的累计误差

**现象**：躯干穴位在图像上偏移 15-30 像素，特别是旁开穴位。

**原因**：
- 原始方案：world landmarks (米) → 计算3D穴位 → `_project_to_2d()` 投影到图像
- 中间经历了 world坐标系 和 image坐标系 之间的线性近似映射，分轴放缩误差被放大

**修复**：改为**2D/3D分离计算**：
- 3D坐标：从 world landmarks 计算（保留给 AR/3D显示）
- 2D坐标：直接从归一化 image landmarks 插值（`_compute_limb_2d`, `_compute_torso_2d`, `_compute_face_2d`, `_compute_hand_2d`）

---

### 坑3：libpng C级别的stderr警告污染控制台

**现象**：读取某些 JPG 图片时，控制台输出 `libpng warning: iCCP: known incorrect sRGB profile`

**原因**：OpenCV 底层使用 libpng 解析，某些图片的 iCCP 色彩空间配置不标准，libpng 会直接向 stderr 输出警告。

**修复**：`demo_verify.py` 中的 `_imread_quiet()` 函数，在读取图片前重定向 stderr 到 `/dev/null`（Windows 用 `os.devnull`），读取完成后恢复。

---

### 坑4：Windows下中文路径导致 MediaPipe 模型加载失败

**现象**：`FileNotFoundError` 或模型加载直接崩溃

**原因**：MediaPipe C++ 底层对非 ASCII 路径支持不完善（Windows 使用 UTF-16 路径，C++ 层使用 ASCII/UTF-8）。

**修复**：项目所有路径使用纯英文目录名，读取文件统一使用 `encoding='utf-8'`。

---

### 坑5：Windows pip包不支持GPU delegate

**现象**：
- `nvidia-smi` 显示有 GPU，但 MediaPipe 只用 CPU
- 尝试设置 `BaseOptions.Delegate.GPU` 直接报错

**原因**：`pip install mediapipe` 的 Windows 预编译包**未编译 GPU delegate**。GPU 支持仅在 Linux/macOS pip 包或源码编译时可用。

**当前方案**：硬编码 `_gpu_available = False`，纯 CPU + XNNPACK 多线程推理。Holistic 模型约 200ms/帧（4-5 FPS），Pose 模型约 50ms/帧（20 FPS）。

**进阶方案**：WSL2 安装 Linux 版 MediaPipe（含 GPU delegate），预计提速 5-10x。

---

### 坑6：MediaPipe Python版本兼容性

**现象**：Python 3.12 无法安装 mediapipe

**原因**：截至 2024 年，MediaPipe 官方只提供到 Python 3.11 的预编译包。

**修复**：`setup.bat` 和 `DEPLOY.md` 明确指定使用 Python 3.10，这是最稳定兼容的版本。

---

### 坑7：OpenCV中文文字渲染

**现象**：`cv2.putText()` 无法显示中文穴位名称，显示为 `????`

**原因**：OpenCV 的 `putText` 只支持 ASCII 字符集。

**修复**：使用 PIL（Pillow）作为中文渲染桥接：
```python
def _put_chinese_text(img_bgr, text, position, color):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)
    draw.text(position, text, font=font, fill=color)
    img_bgr[:] = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
```
字体使用 Windows 系统自带的 `msyh.ttc`（微软雅黑），`simhei.ttf`（黑体），`simsun.ttc`（宋体）作为备选。

---

### 坑8：摄像头初始化的稳定性

**现象**：`cv2.VideoCapture(0)` 在部分 Windows 机器上卡住或报错

**修复**：统一使用 `cv2.VideoCapture(idx, cv2.CAP_DSHOW)`，DirectShow 后端在 Windows 下比默认的 Media Foundation 更稳定。

---

### 坑9：身体朝向检测的边界情况

**现象**：侧面站姿时穴位定位出现大量 D 级

**原因**：朝向检测依赖面部关键点（0-10号）的可见性：≥3个可见→正面，全部不可见→背面，1-2个可见→unknown。侧面时刚好 1-2 个面部关键点可见，导致 `unknown`，穴位批量降级。

**修复**：在 `_check_orientation()` 中，当朝向为 `unknown` 时**不拦截**（默认通过），避免侧面站姿被误判。同时引入 Holistic 的 `has_face` 作为正面**强信号**。

---

## 路线图

- [x] Pose 33点骨架检测
- [x] 虚拟脊柱算法（18节椎骨推算）
- [x] 躯干穴位定位（任脉/督脉/膀胱经/胃经等）
- [x] 四肢穴位定位（骨度比例法）
- [x] Holistic 面部468点检测
- [x] 面部穴位定位（太阳、印堂、迎香等40+穴）
- [x] Holistic 手部42点检测
- [x] 手部穴位定位（合谷、劳宫等30+穴）
- [x] 身体朝向自动检测（正面/背面）
- [x] 手掌朝向检测（掌心/掌背）
- [x] 穴位精度分层（A/B/C/D 四级）
- [x] 2D/3D分离计算（避免投影误差）
- [x] Plotly 交互式3D可视化
- [x] 可缩放/拖拽的2D结果查看器
- [x] 人群系数适配（年龄/性别/BMI/体态）
- [ ] 多摄像头融合（前方+侧方，获取真实胸廓轮廓）
- [ ] 深度相机支持（RealSense D435，真实3D精度提升）
- [ ] 标注反馈闭环（专家标注→参数回归→自动优化）
- [ ] WSL2 GPU 加速方案
- [ ] Gradio Web UI界面
- [ ] 批量图片处理
- [ ] 移动端适配（MediaPipe Android/iOS）

---

*文档版本：v2.0 | 更新日期：2026-05-27*
