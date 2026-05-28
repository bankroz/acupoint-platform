@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

cd /d "%~dp0"

echo.
echo ============================================================
echo   MediaPipe 穴位定位平台 - 一键部署脚本
echo ============================================================
echo.
echo   ----------------------------------------------------------
echo   !!!  重要提示：部分步骤需要科学上网（梯子/VPN）  !!!
echo   ----------------------------------------------------------
echo.
echo   原因：模型文件存放于 storage.googleapis.com（Google CDN），
echo         国内网络直连无法访问，必须通过代理下载。
echo.
echo   无需梯子的部分：pip 安装 Python 依赖包
echo   必须梯子的部分：下载两个 .task 模型文件（约 27MB）
echo.
echo   ----------------------------------------------------------
echo.
echo   如果你已经有梯子，请确保代理已开启，然后继续。
echo   如果你暂时无法使用梯子：
echo     - 可以跳过模型下载，从其他电脑拷贝 models/ 文件夹
echo     - 文件夹中包含 pose_landmarker.task 和
echo       holistic_landmarker.task 两个文件即可
echo.
echo ============================================================
echo.
echo   本脚本将自动完成以下操作:
echo     1. 检测 Python 环境
echo     2. 配置 pip 国内镜像加速（可选）
echo     3. 安装 Python 依赖包
echo     4. 下载 MediaPipe 模型文件（需要梯子）
echo     5. 创建必要目录
echo.
echo ============================================================
echo.
echo   是否继续安装?
choice /c YN /n /m "  [Y] 继续 (需要梯子)   [N] 退出  : "
if errorlevel 2 exit /b 0
echo.

REM ==========================================
REM Step 1: 检测 Python 环境
REM ==========================================
echo [Step 1/5] 检测 Python 环境...

set PYTHON_CMD=
set PYTHON_VERSION=

REM 尝试 python3 和 python
python --version >nul 2>&1
if %errorlevel% equ 0 (
    set PYTHON_CMD=python
    for /f "tokens=2" %%v in ('python --version 2^>^&1') do set PYTHON_VERSION=%%v
) else (
    python3 --version >nul 2>&1
    if %errorlevel% equ 0 (
        set PYTHON_CMD=python3
        for /f "tokens=2" %%v in ('python3 --version 2^>^&1') do set PYTHON_VERSION=%%v
    )
)

if "%PYTHON_CMD%"=="" (
    echo.
    echo   [提示] 未检测到 Python 环境
    echo.
    echo   ┌──────────────────────────────────────────────────┐
    echo   │  本脚本可以帮你自动下载 Python 3.10 安装包          │
    echo   │                                                    │
    echo   │  接下来会问你两个问题:                               │
    echo   │    1. 是否自动下载 Python 3.10 安装包?               │
    echo   │    2. 是否启动安装程序?                              │
    echo   │                                                    │
    echo   │  安装完成后，重新双击此脚本即可继续部署              │
    echo   └──────────────────────────────────────────────────┘
    echo.
    echo   是否自动下载 Python 3.10 安装包?
    echo     [Y] 是，帮我下载
    echo     [N] 否，我自己安装
    choice /c YN /n /m "  "
    if errorlevel 2 goto :MANUAL_INSTALL

    REM 下载 Python 3.10 安装包
    echo.
    echo   [下载] Python 3.10.11 (64-bit, 约 25MB)...
    echo   提示: 如果下载慢，可以 Ctrl+C 取消，手动去官网下载
    echo.
    
    REM 优先使用国内镜像（华为云），更快
    set PYTHON_URL=https://mirrors.huaweicloud.com/python/3.10.11/python-3.10.11-amd64.exe
    set PYTHON_INSTALLER=%TEMP%\python-3.10.11-amd64.exe
    
    echo   尝试从华为云镜像下载...
    powershell -Command "& {[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri '%PYTHON_URL%' -OutFile '%PYTHON_INSTALLER%' -UseBasicParsing}" 2>nul
    
    if exist "%PYTHON_INSTALLER%" (
        echo   [OK] 下载完成: %PYTHON_INSTALLER%
    ) else (
        echo   [警告] 华为云镜像下载失败，尝试从官网下载...
        set PYTHON_URL=https://www.python.org/ftp/python/3.10.11/python-3.10.11-amd64.exe
        powershell -Command "& {[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri '%PYTHON_URL%' -OutFile '%PYTHON_INSTALLER%' -UseBasicParsing}" 2>nul
        if exist "%PYTHON_INSTALLER%" (
            echo   [OK] 下载完成
        ) else (
            echo   [错误] 下载失败，请手动安装 Python 3.10
            goto :MANUAL_INSTALL
        )
    )
    
    echo.
    echo   ┌──────────────────────────────────────────────────┐
    echo   │  Python 安装包已下载，即将启动安装程序               │
    echo   │                                                    │
    echo   │  ！！！重要！！！                                    │
    echo   │  安装时务必勾选底部 "Add Python 3.10 to PATH"        │
    echo   │                                                    │
    echo   │  安装完成后，重新双击 setup.bat 即可                 │
    echo   └──────────────────────────────────────────────────┘
    echo.
    echo   是否现在启动 Python 安装程序?
    choice /c YN /n /m "  [Y] 启动安装   [N] 稍后手动安装  : "
    if errorlevel 2 goto :END
    
    echo   正在启动安装程序...
    start /wait "" "%PYTHON_INSTALLER%"
    
    echo.
    echo   Python 安装程序已退出。
    echo   如果已安装成功，请重新双击 setup.bat 继续部署。
    echo.
    pause
    exit /b 0

:MANUAL_INSTALL
    echo.
    echo   ┌──────────────────────────────────────────────────┐
    echo   │  请手动安装 Python 3.10 (64-bit)                   │
    echo   │                                                    │
    echo   │  国内用户推荐从华为云镜像下载（速度快）:              │
    echo   │  https://mirrors.huaweicloud.com/python/            │
    echo   │  找到 3.10.11/python-3.10.11-amd64.exe              │
    echo   │                                                    │
    echo   │  官网下载（可能较慢）:                               │
    echo   │  https://www.python.org/downloads/                  │
    echo   │                                                    │
    echo   │  ！！！重要！！！                                    │
    echo   │  安装时勾选 "Add Python to PATH"                    │
    echo   │                                                    │
    echo   │  安装完成后重新双击此脚本                            │
    echo   └──────────────────────────────────────────────────┘
    echo.
    pause
    exit /b 1

:END
    echo.
    echo   已跳过安装。准备好后重新双击 setup.bat 即可。
    echo.
    pause
    exit /b 0
)

echo   [OK] 检测到 Python: %PYTHON_VERSION%

REM 检查 Python 版本
for /f "tokens=1,2 delims=." %%a in ("%PYTHON_VERSION%") do (
    set MAJOR=%%a
    set MINOR=%%b
)

if !MAJOR! lss 3 (
    echo   [错误] Python 版本过低，需要 3.9+
    pause
    exit /b 1
)
if !MAJOR! equ 3 if !MINOR! lss 9 (
    echo   [错误] Python 版本过低: %PYTHON_VERSION%，需要 3.9+
    pause
    exit /b 1
)
if !MAJOR! equ 3 if !MINOR! gtr 11 (
    echo   [警告] Python %PYTHON_VERSION% 可能不被 MediaPipe 完整支持
    echo   推荐使用 Python 3.10 或 3.11
    echo.
    echo   是否继续安装? (Y/N)
    choice /c YN /n /m "  "
    if errorlevel 2 exit /b 1
)

echo.

REM ==========================================
REM Step 2: 配置 pip 国内镜像（可选）
REM ==========================================
echo [Step 2/5] 配置 pip 国内镜像源...
echo.
echo   国内用户建议使用镜像加速，否则 pip 安装可能很慢。
echo.
echo   选择镜像源:
echo     [1] 清华源 (推荐，速度快)
echo     [2] 阿里源
echo     [3] 中科大源
echo     [4] 不使用镜像 (默认，需要梯子或等待较久)
echo.
choice /c 1234 /n /m "  请输入数字 (1-4): "
set MIRROR_CHOICE=%errorlevel%

if %MIRROR_CHOICE% equ 1 (
    set PIP_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple
    echo   [OK] 使用清华源
)
if %MIRROR_CHOICE% equ 2 (
    set PIP_INDEX=https://mirrors.aliyun.com/pypi/simple/
    echo   [OK] 使用阿里源
)
if %MIRROR_CHOICE% equ 3 (
    set PIP_INDEX=https://pypi.mirrors.ustc.edu.cn/simple/
    echo   [OK] 使用中科大源
)
if %MIRROR_CHOICE% equ 4 (
    set PIP_INDEX=
    echo   [OK] 使用默认源 (PyPI)
)

echo.

REM ==========================================
REM Step 3: 升级 pip
REM ==========================================
echo [Step 3/5] 升级 pip 到最新版...
if "%PIP_INDEX%"=="" (
    %PYTHON_CMD% -m pip install --upgrade pip --quiet
) else (
    %PYTHON_CMD% -m pip install --upgrade pip -i %PIP_INDEX% --quiet
)
if %errorlevel% neq 0 (
    echo   [警告] pip 升级失败，继续使用当前版本
) else (
    echo   [OK] pip 升级完成
)
echo.

REM ==========================================
REM Step 4: 安装依赖
REM ==========================================
echo [Step 4/5] 安装 Python 依赖包...
echo   这可能需要 3-5 分钟，请耐心等待...
echo.

REM 核心依赖（必须安装）
echo   [1/5] 安装 mediapipe (AI 人体检测引擎)...
if "%PIP_INDEX%"=="" (
    %PYTHON_CMD% -m pip install mediapipe --quiet
) else (
    %PYTHON_CMD% -m pip install mediapipe -i %PIP_INDEX% --quiet
)
if %errorlevel% neq 0 (
    echo   [错误] mediapipe 安装失败!
    echo   可能原因: Python版本不兼容 (仅支持 3.9-3.11)
    echo   请安装 Python 3.10 后重试
    pause
    exit /b 1
)
echo         [OK]

echo   [2/5] 安装 opencv-python (图像处理 + 摄像头)...
if "%PIP_INDEX%"=="" (
    %PYTHON_CMD% -m pip install opencv-python --quiet
) else (
    %PYTHON_CMD% -m pip install opencv-python -i %PIP_INDEX% --quiet
)
if %errorlevel% neq 0 (
    echo   [错误] opencv-python 安装失败!
    pause
    exit /b 1
)
echo         [OK]

echo   [3/5] 安装 numpy (数值计算)...
if "%PIP_INDEX%"=="" (
    %PYTHON_CMD% -m pip install numpy --quiet
) else (
    %PYTHON_CMD% -m pip install numpy -i %PIP_INDEX% --quiet
)
if %errorlevel% neq 0 (
    echo   [错误] numpy 安装失败!
    pause
    exit /b 1
)
echo         [OK]

echo   [4/5] 安装 plotly (交互式3D可视化)...
if "%PIP_INDEX%"=="" (
    %PYTHON_CMD% -m pip install plotly --quiet
) else (
    %PYTHON_CMD% -m pip install plotly -i %PIP_INDEX% --quiet
)
if %errorlevel% neq 0 (
    echo   [错误] plotly 安装失败!
    pause
    exit /b 1
)
echo         [OK]

echo   [5/5] 安装 pillow (中文文字渲染)...
if "%PIP_INDEX%"=="" (
    %PYTHON_CMD% -m pip install pillow --quiet
) else (
    %PYTHON_CMD% -m pip install pillow -i %PIP_INDEX% --quiet
)
if %errorlevel% neq 0 (
    echo   [错误] pillow 安装失败!
    pause
    exit /b 1
)
echo         [OK]

echo.
echo   [OK] 所有依赖安装完成!
echo.

REM ==========================================
REM Step 5: 下载模型文件 + 创建目录
REM ==========================================
echo [Step 5/5] 下载 MediaPipe 模型文件...
echo.
echo   !!! 模型下载需要梯子（访问 storage.googleapis.com）!!!
echo   如果下载失败，可从已有环境的 models/ 文件夹拷贝过来。
echo.

REM 创建必要目录
if not exist "models" mkdir models
if not exist "input\images" mkdir input\images
if not exist "output\annotations" mkdir output\annotations
if not exist "output\visualizations" mkdir output\visualizations
if not exist "output\screenshots" mkdir output\screenshots

REM 检查模型文件是否已存在
set POSE_MODEL=models\pose_landmarker.task
set HOLISTIC_MODEL=models\holistic_landmarker.task

if exist "%POSE_MODEL%" (
    echo   [SKIP] pose_landmarker.task 已存在
) else (
    echo   [下载] pose_landmarker.task (约 15MB)...
    %PYTHON_CMD% -c "import urllib.request; urllib.request.urlretrieve('https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task', 'models/pose_landmarker.task')"
    if %errorlevel% neq 0 (
        echo   [警告] Pose 模型下载失败，稍后可用 python scripts/download_models.py 重试
    ) else (
        echo         [OK]
    )
)

if exist "%HOLISTIC_MODEL%" (
    echo   [SKIP] holistic_landmarker.task 已存在
) else (
    echo   [下载] holistic_landmarker.task (约 12MB)...
    %PYTHON_CMD% -c "import urllib.request; urllib.request.urlretrieve('https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task', 'models/holistic_landmarker.task')"
    if %errorlevel% neq 0 (
        echo   [警告] Holistic 模型下载失败，稍后可用 python scripts/download_models.py 重试
    ) else (
        echo         [OK]
    )
)

echo.

REM ==========================================
REM 验证安装
REM ==========================================
echo ============================================================
echo   验证安装...
echo ============================================================
echo.

%PYTHON_CMD% -c "import mediapipe; print('  MediaPipe版本:', mediapipe.__version__)" 2>nul
%PYTHON_CMD% -c "import cv2; print('  OpenCV版本:', cv2.__version__)" 2>nul
%PYTHON_CMD% -c "import numpy; print('  NumPy版本:', numpy.__version__)" 2>nul
%PYTHON_CMD% -c "import plotly; print('  Plotly版本:', plotly.__version__)" 2>nul
%PYTHON_CMD% -c "from PIL import Image; print('  Pillow: 已安装')" 2>nul

echo.

REM 检查模型文件完整性
set ALL_MODELS_OK=1
if not exist "%POSE_MODEL%" (
    echo   [警告] Pose 模型缺失，请运行: python scripts/download_models.py
    set ALL_MODELS_OK=0
)
if not exist "%HOLISTIC_MODEL%" (
    echo   [警告] Holistic 模型缺失，请运行: python scripts/download_models.py
    set ALL_MODELS_OK=0
)
if %ALL_MODELS_OK% equ 1 (
    echo   [OK] 模型文件完整
)

echo.
echo ============================================================
echo   部署完成!
echo ============================================================
echo.
echo   ┌──────────────────────────────────────────────────────┐
echo   │  启动方式:                                             │
echo   │                                                       │
echo   │  方式1: 双击 "一键验证.bat" 启动主程序                   │
echo   │  方式2: 命令行运行                                     │
echo   │    %PYTHON_CMD% scripts/demo_verify.py                  │
echo   │                                                       │
echo   │  提示:                                                 │
echo   │  - 图片模式: 将全身照放到 input/images/test.jpg          │
echo   │  - 摄像头模式: 确保摄像头已连接                          │
echo   │  - 面部+手部模式: 不需要全身入镜                         │
echo   └──────────────────────────────────────────────────────┘
echo.
echo   如需详细了解功能，请查看 README.md
echo.
pause
