@echo off
cd /d "%~dp0"

echo(
echo ============================================================
echo   MediaPipe 穴位定位平台 - 一键部署脚本
echo ============================================================
echo(
echo(
echo   +-------------------------------------------------------+
echo   ^|                                                       ^|
echo   ^|     ^>^>^>  老于：此处注意！需要梯子！  ^<^<^<              ^|
echo   ^|                                                       ^|
echo   ^|  模型文件存放于 Google CDN (storage.googleapis.com)    ^|
echo   ^|  国内直连无法访问，必须开代理下载模型                  ^|
echo   ^|                                                       ^|
echo   ^|  无需梯子：pip 安装依赖包                              ^|
echo   ^|  必须梯子：下载 .task 模型文件 (约 27MB)               ^|
echo   ^|                                                       ^|
echo   ^|  没有梯子？可以从其他电脑拷贝 models/ 文件夹过来      ^|
echo   ^|                                                       ^|
echo   +-------------------------------------------------------+
echo(
echo(
echo   本脚本将自动完成以下操作:
echo(
echo     Step 1/5  - 检测 Python 环境
echo     Step 2/5  - 配置 pip 国内镜像源 (可选)
echo     Step 3/5  - 安装 Python 依赖包 (5个)
echo     Step 4/5  - 下载模型文件 (需梯子)
echo     Step 5/5  - 创建目录 + 验证安装
echo(
echo ============================================================
echo(

choice /c YN /n /m "  [Y] 开始部署   [N] 退出  : "
if errorlevel 2 exit /b 0
echo(

REM ==========================================
REM Step 1: 检测 Python 环境
REM ==========================================
echo(
echo   ================================
echo     Step 1 / 5  -  检测 Python 环境
echo   ================================
echo(
echo   [..] 正在检测 python 命令...

set PYTHON_CMD=
set PYTHON_VERSION=

python --version >nul 2>&1
if %ERRORLEVEL% equ 0 (
    set PYTHON_CMD=python
    goto :GOT_PYTHON
)

python3 --version >nul 2>&1
if %ERRORLEVEL% equ 0 (
    set PYTHON_CMD=python3
    goto :GOT_PYTHON
)

REM ---- 未找到 Python ----
echo(
echo   +-------------------------------------------------------+
echo   ^|                                                       ^|
echo   ^|     ^>^>^>  老于：此处注意！未检测到 Python  ^<^<^<         ^|
echo   ^|                                                       ^|
echo   ^|  这台电脑还没有装 Python，本脚可以帮你一键下载        ^|
echo   ^|                                                       ^|
echo   +-------------------------------------------------------+
echo(
choice /c YN /n /m "  [Y] 自动下载并安装 Python 3.10   [N] 我自己装  : "
if errorlevel 2 goto :MANUAL_INSTALL

echo(
echo   [..] 准备从华为云镜像下载 Python 3.10.11 ...
echo       文件大小约 25MB
echo       如果网速慢可以 Ctrl+C 取消后手动安装
echo(

set PYTHON_URL=https://mirrors.huaweicloud.com/python/3.10.11/python-3.10.11-amd64.exe
set PYTHON_INSTALLER=%TEMP%\python-3.10.11-amd64.exe

echo   [..] 正在连接华为云镜像...
powershell -Command "& {[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Write-Host 'DOWNLOADING'; Invoke-WebRequest -Uri '%PYTHON_URL%' -OutFile '%PYTHON_INSTALLER%' -UseBasicParsing}" >nul 2>&1

if exist "%PYTHON_INSTALLER%" (
    echo   [OK] Python 安装包下载完成
) else (
    echo   [!!] 华为云失败，尝试官网...
    set PYTHON_URL=https://www.python.org/ftp/python/3.10.11/python-3.10.11-amd64.exe
    powershell -Command "& {[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Write-Host 'DOWNLOADING'; Invoke-WebRequest -Uri '%PYTHON_URL%' -OutFile '%PYTHON_INSTALLER%' -UseBasicParsing}" >nul 2>&1
    if exist "%PYTHON_INSTALLER%" (
        echo   [OK] 从官网下载完成
    ) else (
        echo   [错误] 下载失败，请手动安装
        goto :MANUAL_INSTALL
    )
)
echo(
echo   +-------------------------------------------------------+
echo   ^|                                                       ^|
echo   ^|     ^>^>^>  老于：此处注意！安装选项  ^<^<^<                 ^|
echo   ^|                                                       ^|
echo   ^|  即将弹出 Python 安装界面                             ^|
echo   ^|  *** 务必勾选底部的 "Add Python to PATH" ***          ^|
echo   ^|  然后点 Install Now                                  ^|
echo   ^|                                                       ^|
echo   ^|  装完后关闭此窗口，重新双击 setup.bat 继续            ^|
echo   ^|                                                       ^|
echo   +-------------------------------------------------------+
echo(
choice /c YN /n /m "  [Y] 立即启动安装程序   [N] 稍后再装  : "
if errorlevel 2 goto :END_PYTHON

echo   [..] 正在启动安装程序...
start /wait "" "%PYTHON_INSTALLER%"
echo(
echo   安装程序已退出。
echo   如果已成功安装，请重新双击 setup.bat 继续后续步骤。
echo(
pause
exit /b 0

:MANUAL_INSTALL
echo(
echo   手动安装方法：
echo(
echo   推荐国内镜像（速度快）:
echo   https://mirrors.huaweicloud.com/python/
echo   找到: 3.10.11 / python-3.10.11-amd64.exe
echo(
echo   官网（可能较慢）:
echo   https://www.python.org/downloads/
echo(
echo   *** 安装时务必勾选 "Add Python to PATH" ***
echo(
pause
exit /b 1

:END_PYTHON
echo(
echo   已跳过。准备好后重新双击 setup.bat。
echo(
pause
exit /b 0

REM ---- 找到 Python，读取版本 ----
:GOT_PYTHON
echo   [..] 正在读取 Python 版本...

for /f "tokens=2 delims= " %%a in ('%PYTHON_CMD% --version 2^>^&1') do set PYTHON_VERSION=%%a

echo   [OK] Python %PYTHON_VERSION% 已就绪

REM 版本检查（简化版，只做基本判断）
echo   [..] 校验版本兼容性...

%PYTHON_CMD% -c "import sys; v=sys.version_info; exit(0 if (v.major==3 and 9<=v.minor<=11) or (v.major==3 and v.minor>11) else 1)" >nul 2>&1
if errorlevel 1 (
    echo   [警告] Python %PYTHON_VERSION% 可能不被 MediaPipe 支持
    echo          推荐使用 Python 3.10 或 3.11
    choice /c YN /n /m "  [Y] 强制继续   [N] 退出  : "
    if errorlevel 2 exit /b 1
)
echo   [OK] 版本校验通过
echo(
echo   ======== Step 1 完成 ========
echo(


REM ==========================================
REM Step 2: 配置 pip 国内镜像
REM ==========================================
echo(
echo   ================================
echo     Step 2 / 5  -  配置 pip 镜像源
echo   ================================
echo(
echo   国内用户建议选择镜像源，pip 安装会快很多：
echo(
echo     [1] 清华源     pypi.tuna.tsinghua.edu.cn  (推荐)
echo     [2] 阿里源     mirrors.aliyun.com
echo     [3] 中科大源   mirrors.ustc.edu.cn
echo     [4] 不用镜像   直接连 PyPI (可能很慢)
echo(
choice /c 1234 /n /m "  请选择 (1-4): "
set MIRROR_CHOICE=%ERRORLEVEL%

if %MIRROR_CHOICE% equ 1 (
    set PIP_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple
    echo   [OK] 已选择清华源
)
if %MIRROR_CHOICE% equ 2 (
    set PIP_INDEX=https://mirrors.aliyun.com/pypi/simple/
    echo   [OK] 已选择阿里源
)
if %MIRROR_CHOICE% equ 3 (
    set PIP_INDEX=https://pypi.mirrors.ustc.edu.cn/simple/
    echo   [OK] 已选择中科大源
)
if %MIRROR_CHOICE% equ 4 (
    set PIP_INDEX=
    echo   [OK] 使用默认 PyPI 源
)
echo(
echo   ======== Step 2 完成 ========
echo(


REM ==========================================
REM Step 3: 升级 pip + 安装依赖
REM ==========================================
echo(
echo   ================================
echo     Step 3 / 5  -  安装 Python 依赖包
echo   ================================
echo(
echo   共需检查 6 个包 (含 pip 升级)，请耐心等待...
echo(
echo   +-------------------------------------------------------+
echo   ^|  已有依赖将自动跳过，只安装缺失的包                  ^|
echo   ^|  安装过程中屏幕会显示进度条和下载信息                ^|
echo   +-------------------------------------------------------+
echo(

REM 调用 Python 辅助脚本统一处理: 检查 → 安装 → 报告
REM 脚本会生成 TEMP 下的 _deps_result.txt 供 Step 5 使用
%PYTHON_CMD% scripts\_setup_deps.py
if errorlevel 1 (
    echo(
    echo   [警告] 部分依赖安装失败，请检查上方错误信息
    echo(
    pause
    exit /b 1
)
echo(


REM ==========================================
REM Step 4: 下载模型文件
REM ==========================================
echo(
echo   ================================
echo     Step 4 / 5  -  下载模型文件
echo   ================================
echo(
echo   +-------------------------------------------------------+
echo   ^|                                                       ^|
echo   ^|     ^>^>^>  老于：此处注意！这一步需要梯子！  ^<^<^<        ^|
echo   ^|                                                       ^|
echo   ^|  模型文件存储在 Google CDN                            ^|
echo   ^|  请确保你的代理/VPN 已经开启                          ^|
echo   ^|                                                       ^|
echo   +-------------------------------------------------------+
echo(

REM 创建目录
if not exist "models" mkdir models
if not exist "input\images" mkdir input\images
if not exist "output\annotations" mkdir output\annotations
if not exist "output\visualizations" mkdir output\visualizations
if not exist "output\screenshots" mkdir output\screenshots

set POSE_MODEL=models\pose_landmarker.task
set HOLISTIC_MODEL=models\holistic_landmarker.task

REM --- Pose 模型 ---
if exist "%POSE_MODEL%" (
    echo   [SKIP] pose_landmarker.task (已存在)
) else (
    echo   [下载 1/2] pose_landmarker.task (~15MB) ...
    echo           请耐心等待，不要关闭窗口...
    echo(
    %PYTHON_CMD% -c "import urllib.request,sys; print('正在连接 Google CDN...'); urllib.request.urlretrieve('https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task', 'models/pose_landmarker.task'); print('下载完成')" 2>&1
    if %ERRORLEVEL% neq 0 (
        echo   [警告] Pose 模型下载失败 (可能是网络问题)
        echo          可稍后运行: python scripts/download_models.py
    ) else (
        echo   [OK]
    )
)
echo(

REM --- Holistic 模型 ---
if exist "%HOLISTIC_MODEL%" (
    echo   [SKIP] holistic_landmarker.task (已存在)
) else (
    echo   [下载 2/2] holistic_landmarker.task (~12MB) ...
    echo           请耐心等待，不要关闭窗口...
    echo(
    %PYTHON_CMD% -c "import urllib.request,sys; print('正在连接 Google CDN...'); urllib.request.urlretrieve('https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task', 'models/holistic_landmarker.task'); print('下载完成')" 2>&1
    if %ERRORLEVEL% neq 0 (
        echo   [警告] Holistic 模型下载失败 (可能是网络问题)
        echo          可稍后运行: python scripts/download_models.py
    ) else (
        echo   [OK]
    )
)
echo(
echo   ======== Step 4 完成 ========
echo(


REM ==========================================
REM Step 5: 验证安装 + 总结报告
REM ==========================================
echo(
echo   ================================
echo     Step 5 / 5  -  验证安装 ^& 总结
echo   ================================
echo(
echo   [..] 正在检查所有组件，请稍候...
echo(

REM ---- 读取 Step 3 生成的依赖安装结果 ----
set DEPS_FILE=%TEMP%\_deps_result.txt
set PYTHON_VER=未知
set SKIP_INSTALL_COUNT=0
set NEW_INSTALL_COUNT=0

set PKG_MEDIAPIPE_VER=未知
set PKG_MEDIAPIPE_STATUS=未知
set PKG_OPENCV_PYTHON_VER=未知
set PKG_OPENCV_PYTHON_STATUS=未知
set PKG_NUMPY_VER=未知
set PKG_NUMPY_STATUS=未知
set PKG_PLOTLY_VER=未知
set PKG_PLOTLY_STATUS=未知
set PKG_PILLOW_VER=未知
set PKG_PILLOW_STATUS=未知

if exist "%DEPS_FILE%" (
    for /f "tokens=1,* delims==" %%a in ("%DEPS_FILE%") do (
        if "%%a"=="SKIP_COUNT"            set SKIP_INSTALL_COUNT=%%b
        if "%%a"=="NEW_COUNT"             set NEW_INSTALL_COUNT=%%b
        if "%%a"=="PYTHON_VER"            set PYTHON_VER=%%b
        if "%%a"=="PKG_MEDIAPIPE_VER"     set PKG_MEDIAPIPE_VER=%%b
        if "%%a"=="PKG_MEDIAPIPE_STATUS"  set PKG_MEDIAPIPE_STATUS=%%b
        if "%%a"=="PKG_OPENCV_PYTHON_VER" set PKG_OPENCV_PYTHON_VER=%%b
        if "%%a"=="PKG_OPENCV_PYTHON_STATUS" set PKG_OPENCV_PYTHON_STATUS=%%b
        if "%%a"=="PKG_NUMPY_VER"         set PKG_NUMPY_VER=%%b
        if "%%a"=="PKG_NUMPY_STATUS"      set PKG_NUMPY_STATUS=%%b
        if "%%a"=="PKG_PLOTLY_VER"        set PKG_PLOTLY_VER=%%b
        if "%%a"=="PKG_PLOTLY_STATUS"     set PKG_PLOTLY_STATUS=%%b
        if "%%a"=="PKG_PILLOW_VER"        set PKG_PILLOW_VER=%%b
        if "%%a"=="PKG_PILLOW_STATUS"     set PKG_PILLOW_STATUS=%%b
    )
    del "%DEPS_FILE%" >nul 2>&1
)

REM ---- 检查模型文件 ----
set POSE_MODEL_OK=0
set POSE_MODEL_SIZE=0
set HOLI_MODEL_OK=0
set HOLI_MODEL_SIZE=0

if exist "models\pose_landmarker.task" (
    set POSE_MODEL_OK=1
    for %%A in ("models\pose_landmarker.task") do set POSE_MODEL_SIZE=%%~zA
)
if exist "models\holistic_landmarker.task" (
    set HOLI_MODEL_OK=1
    for %%A in ("models\holistic_landmarker.task") do set HOLI_MODEL_SIZE=%%~zA
)

REM ---- 格式化显示 ----
REM 依赖包状态 (OK / 失败)
if "%PKG_MEDIAPIPE_STATUS%"=="fail"  (set _MP_S=[失败]) else (set _MP_S=[ OK ])
if "%PKG_OPENCV_PYTHON_STATUS%"=="fail" (set _CV_S=[失败]) else (set _CV_S=[ OK ])
if "%PKG_NUMPY_STATUS%"=="fail"     (set _NP_S=[失败]) else (set _NP_S=[ OK ])
if "%PKG_PLOTLY_STATUS%"=="fail"    (set _PL_S=[失败]) else (set _PL_S=[ OK ])
if "%PKG_PILLOW_STATUS%"=="fail"    (set _PW_S=[失败]) else (set _PW_S=[ OK ])

REM 安装状态标签 (已有/新装/失败)
if "%PKG_MEDIAPIPE_STATUS%"=="skip"  (set _MP_T=[已有]) else if "%PKG_MEDIAPIPE_STATUS%"=="new" (set _MP_T=[新装]) else (set _MP_T=[失败])
if "%PKG_OPENCV_PYTHON_STATUS%"=="skip" (set _CV_T=[已有]) else if "%PKG_OPENCV_PYTHON_STATUS%"=="new" (set _CV_T=[新装]) else (set _CV_T=[失败])
if "%PKG_NUMPY_STATUS%"=="skip"     (set _NP_T=[已有]) else if "%PKG_NUMPY_STATUS%"=="new" (set _NP_T=[新装]) else (set _NP_T=[失败])
if "%PKG_PLOTLY_STATUS%"=="skip"    (set _PL_T=[已有]) else if "%PKG_PLOTLY_STATUS%"=="new" (set _PL_T=[新装]) else (set _PL_T=[失败])
if "%PKG_PILLOW_STATUS%"=="skip"    (set _PW_T=[已有]) else if "%PKG_PILLOW_STATUS%"=="new" (set _PW_T=[新装]) else (set _PW_T=[失败])

REM 模型状态
if %POSE_MODEL_OK% equ 1 (set _PS_S=[ OK ]) else (set _PS_S=[缺失])
if %HOLI_MODEL_OK% equ 1 (set _HS_S=[ OK ]) else (set _HS_S=[缺失])

echo   [OK] 检查完成
echo(
echo ============================================================
echo(
echo           ***************************************
echo           *                                     *
echo           *          部署完成!  总结报告         *
echo           *                                     *
echo           ***************************************
echo(
echo ============================================================
echo(
echo   +-------------------------------------------------------+
echo   ^|                                                       ^|
echo   ^|  【系统环境】                                          ^|
echo   ^|                                                       ^|
echo   ^|    Python  %PYTHON_VER%                                  ^|
echo   ^|                                                       ^|
echo   ^|  【Python 依赖包】 已有=%SKIP_INSTALL_COUNT% 新装=%NEW_INSTALL_COUNT%                              ^|
echo   ^|                                                       ^|
echo   ^|    %_MP_S% mediapipe         %PKG_MEDIAPIPE_VER%    %_MP_T%      ^|
echo   ^|    %_CV_S% opencv-python     %PKG_OPENCV_PYTHON_VER%    %_CV_T%      ^|
echo   ^|    %_NP_S% numpy             %PKG_NUMPY_VER%    %_NP_T%      ^|
echo   ^|    %_PL_S% plotly            %PKG_PLOTLY_VER%    %_PL_T%      ^|
echo   ^|    %_PW_S% pillow            %PKG_PILLOW_VER%    %_PW_T%      ^|
echo   ^|                                                       ^|
echo   ^|  【AI 模型文件】                                        ^|
echo   ^|                                                       ^|
echo   ^|    %_PS_S% pose_landmarker.task      (~15MB)            ^|
echo   ^|    %_HS_S% holistic_landmarker.task  (~12MB)            ^|
echo   ^|                                                       ^|
echo   ^|  【安装统计】                                           ^|
echo   ^|                                                       ^|
echo   ^|    已有: %SKIP_INSTALL_COUNT% / 5   新安装: %NEW_INSTALL_COUNT% / 5                         ^|
echo   ^|                                                       ^|
echo   ^|  【启动方式】                                           ^|
echo   ^|                                                       ^|
echo   ^|    方式1: 双击主程序 bat 文件启动                       ^|
echo   ^|    方式2: 命令行运行                                    ^|
echo   ^|      %PYTHON_CMD% scripts/demo_verify.py                 ^|
echo   ^|                                                       ^|
echo   ^|    图片模式: 放照片到 input/images/test.jpg            ^|
echo   ^|    摄像头:   确保摄像头已连接                            ^|
echo   ^|                                                       ^|
echo   +-------------------------------------------------------+
echo(
echo   详细说明请看 README.md
echo(
pause
exit /b 0
