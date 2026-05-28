# 部署指南 — MediaPipe 穴位定位平台

> 目标：在任意一台 Windows 电脑上，**双击一个 bat 文件即可完成全部部署**。

---

## 前置条件

| 条件 | 要求 | 备注 |
|------|------|------|
| 操作系统 | Windows 10 / 11 (64位) | 不支持 32 位 |
| Python | **3.9 ~ 3.11** | 必须是这个范围！3.12+ 不支持 |
| 网络 | 可访问 PyPI 和 Google CDN | 用于下载依赖包和模型文件 |
| 磁盘 | 约 2GB 空闲 | 含 Python + 依赖 + 模型 |

> ⚠️ **最重要**：Python 版本必须 ≤ 3.11！MediaPipe 目前不提供 3.12+ 的预编译包。

---

## 一键部署（推荐）

### 如果你已经装了 Python 3.9~3.11

```
第一步：双击 setup.bat
第二步：等待 3-5 分钟
第三步：双击 一键验证.bat 启动
```

就这么简单。`setup.bat` 会自动：
1. 检测 Python 环境（版本是否符合要求）
2. 升级 pip 到最新版
3. 安装 mediapipe、opencv-python、numpy、plotly、pillow
4. 下载 Pose 和 Holistic 两个 MediaPipe 模型文件
5. 创建 input/output/models 目录
6. 验证安装完整性

### 如果你还没装 Python

1. 打开 [python.org/downloads](https://www.python.org/downloads/)
2. 下载 **Python 3.10.x**（推荐 3.10.11）
3. 运行安装程序，**务必勾选 "Add Python to PATH"**（页面底部复选框）
4. 安装完成后，双击 `setup.bat`

### 如果 Python 版本不对（3.12+）

你需要先卸载现有 Python，再安装 3.10：

```
1. 设置 → 应用 → 找到 Python 3.12 → 卸载
2. 下载安装 Python 3.10.11
3. 双击 setup.bat
```

或者使用 `py` 启动器指定版本（如果你同时安装了多个 Python）：

```bash
# 先确认有 3.10
py -3.10 --version

# 用特定版本安装
py -3.10 -m pip install mediapipe opencv-python numpy plotly pillow
py -3.10 scripts/download_models.py
```

---

## 手动部署（如果 bat 脚本出问题）

### 第一步：安装 Python 依赖

```bash
# 确保在 acupoint-platform 目录下
cd acupoint-platform

# 安装核心依赖（5个包）
pip install mediapipe opencv-python numpy plotly pillow
```

你也可以一次性从 requirements.txt 安装：
```bash
pip install -r requirements.txt
```

### 第二步：下载 MediaPipe 模型

```bash
python scripts/download_models.py
```

这会从 Google CDN 下载两个文件放到 `models/` 目录：
- `pose_landmarker.task`（~15MB，必需）
- `holistic_landmarker.task`（~12MB，推荐，用于面部和手部穴位）

### 第三步：验证安装

```bash
python -c "import mediapipe; print('OK:', mediapipe.__version__)"
python -c "import cv2; print('OK:', cv2.__version__)"
python -c "import numpy; print('OK:', numpy.__version__)"
python -c "import plotly; print('OK:', plotly.__version__)"
python -c "from PIL import Image; print('Pillow OK')"
```

全部输出 `OK` 即安装成功。

### 第四步：运行验证

```bash
python scripts/demo_verify.py
```

---

## 验证部署是否成功

运行以下测试命令，每一行都不应该报错：

```bash
# 1. 基础导入测试
python -c "from core.pose_extractor import PoseExtractor, ModelMode; print('✓ 核心模块')"

# 2. 虚拟脊柱测试
python -c "from core.spine_estimator import SpineEstimator; print('✓ 脊柱估算')"

# 3. 穴位定位器测试
python -c "from core.acupoint_locator import AcupointLocator; print('✓ 穴位定位')"

# 4. 模型文件检查
python -c "import os; print('Pose模型:', os.path.exists('models/pose_landmarker.task')); print('Holistic模型:', os.path.exists('models/holistic_landmarker.task'))"
```

期望输出：
```
✓ 核心模块
✓ 脊柱估算
✓ 穴位定位
Pose模型: True
Holistic模型: True
```

---

## 常见问题排查

### 问题1：`pip install mediapipe` 报错

```
ERROR: Could not find a version that satisfies the requirement mediapipe
```

**原因**：Python 版本太高（≥ 3.12）或太低（< 3.9）。

**解决**：安装 Python 3.10.x。

---

### 问题2：模型下载失败

```
urllib.error.URLError: <urlopen error ...>
```

**原因**：网络无法访问 Google CDN。

**解决方案A**：使用代理
```bash
set HTTP_PROXY=http://127.0.0.1:7890
set HTTPS_PROXY=http://127.0.0.1:7890
python scripts/download_models.py
```

**解决方案B**：手动下载
1. 打开浏览器访问：
   - https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task
   - https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task
2. 将下载的文件放到 `acupoint-platform/models/` 目录
3. 重命名为 `pose_landmarker.task` 和 `holistic_landmarker.task`

**解决方案C**：从其他可上网的电脑拷贝 `models/` 整个目录。

---

### 问题3：`opencv-python` 安装报错

```
ERROR: Failed building wheel for opencv-python
```

**原因**：pip 版本过旧或缺少 Visual C++ 运行库。

**解决**：
```bash
# 先升级 pip
pip install --upgrade pip

# 再重试
pip install opencv-python
```

如果仍然失败，安装 [Visual C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe)。

---

### 问题4：摄像头打不开

```
[!] Cannot open camera #0
```

**检查清单**：
1. 摄像头是否物理连接（USB 是否有灯亮）
2. Windows 设置 → 隐私 → 相机 → 允许应用访问相机
3. 是否有其他应用（如微信/Zoom）占用了摄像头，关闭它们
4. 尝试在命令行运行 `python scripts/demo_verify.py`，选择图片模式先验证基本功能

---

### 问题5：运行时 `ImportError: No module named 'tkinter'`

**原因**：Python 安装时未勾选 tkinter。

**解决**：图片模式需要 tkinter 来弹出文件选择对话框。如果不需要图片模式的文件选择功能，可以跳过。

安装 tkinter：
- 重新运行 Python 安装程序，选择 Modify → 勾选 tcl/tk and IDLE
- 或者改用摄像头模式（不需要 tkinter）

---

### 问题6：控制台乱码

**原因**：Windows 默认编码不是 UTF-8。

**解决**：
1. 双击 bat 脚本运行（脚本内已设置 `chcp 65001`）
2. 或在命令行中先执行 `chcp 65001`
3. 或在 PowerShell 中执行 `[Console]::OutputEncoding = [Text.Encoding]::UTF8`

---

## 依赖包说明

| 包名 | 版本要求 | 用途 | 大小 |
|------|---------|------|------|
| mediapipe | ≥0.10.0 | Google AI 人体姿态检测引擎 | ~500MB |
| opencv-python | ≥4.8.0 | 摄像头采集、图像处理 | ~50MB |
| numpy | ≥1.24.0 | 数组计算、线性代数 | ~20MB |
| plotly | ≥5.15.0 | 交互式 3D 可视化（浏览器） | ~15MB |
| pillow | ≥10.0.0 | 中文文字渲染到图像 | ~5MB |

总计约 **600MB**（主要是 MediaPipe 的体积）。

---

## 跨机器迁移

如果要在新电脑上使用，无需重新配置：

### 方法1：拷贝整个项目目录

```
1. 将整个 acupoint-platform 文件夹拷贝到新电脑
2. 在新电脑上安装 Python 3.10
3. 双击 setup.bat
```

`setup.bat` 会自动跳过已下载的模型文件，只安装依赖包。

### 方法2：Git 克隆（如果有 Git）

```bash
git clone <仓库地址>
cd acupoint-platform
setup.bat
```

---

## 目录说明（部署后）

```
acupoint-platform/
├── setup.bat           ← 一键部署脚本
├── 一键验证.bat         ← 一键启动检测
├── config.yaml         ← 配置文件（可修改参数）
├── requirements.txt    ← 依赖列表
│
├── core/               ← 核心算法（不需要修改）
├── database/           ← 穴位数据库（JSON格式，可自定义）
├── models/             ← MediaPipe模型（部署后自动下载）
├── scripts/            ← 可执行脚本
├── ui/                 ← 可视化模块
│
├── input/images/       ← 放测试图片
└── output/             ← 输出结果（自动生成）
```

---

*最后更新：2026-05-27*
