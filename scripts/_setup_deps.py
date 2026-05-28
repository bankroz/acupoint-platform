#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
setup.bat 的依赖检查与安装辅助脚本
由 setup.bat 调用，负责：检查已安装包 → 安装缺失包 → 输出版本报告
"""

import subprocess
import sys
import importlib
import os

# pip 安装时使用的镜像源（由 bat 通过环境变量传入）
PIP_INDEX = os.environ.get("PIP_INDEX", "")

# 需要检查的包：{pip包名: (导入名, 描述)}
PACKAGES = {
    "mediapipe":      ("mediapipe", "AI人体检测引擎"),
    "opencv-python":  ("cv2",       "图像处理+摄像头"),
    "numpy":          ("numpy",     "数值计算库"),
    "plotly":         ("plotly",    "3D交互式可视化"),
    "pillow":         ("PIL",       "中文文字渲染"),
}

# 输出文件路径
RESULT_FILE = os.path.join(os.environ.get("TEMP", "."), "_deps_result.txt")


def get_installed_version(import_name):
    """尝试导入模块并获取版本号"""
    try:
        mod = importlib.import_module(import_name)
        return getattr(mod, "__version__", "已安装")
    except Exception:
        return None


def run_pip_install(pkg_name):
    """执行 pip install 并返回 (success, already_satisfied)"""
    cmd = [sys.executable, "-m", "pip", "install", pkg_name]
    if PIP_INDEX:
        cmd += ["-i", PIP_INDEX]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
        )
        output = result.stdout + result.stderr
        already = "already satisfied" in output.lower()
        success = result.returncode == 0
        return success, already
    except Exception:
        return False, False


def upgrade_pip():
    """升级 pip"""
    cmd = [sys.executable, "-m", "pip", "install", "--upgrade", "pip"]
    if PIP_INDEX:
        cmd += ["-i", PIP_INDEX]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        return result.returncode == 0
    except Exception:
        return False


def main():
    results = {}   # pkg_name -> {status, version, desc}
    new_count = 0
    skip_count = 0
    all_passed = True

    # ---- 0: 升级 pip ----
    print("  [1/6] 升级 pip ...", flush=True)
    if upgrade_pip():
        print("  [OK] pip 升级完成", flush=True)
    else:
        print("  [警告] pip 升级失败，继续使用当前版本", flush=True)
    print(flush=True)

    item = 2
    for pkg_name, (import_name, desc) in PACKAGES.items():
        # 检查是否已安装
        current_ver = get_installed_version(import_name)

        if current_ver is not None:
            # 已安装 → 跳过
            print(f"  [{item}/6] {pkg_name} ({desc}) ... [已有] 已安装，跳过", flush=True)
            results[pkg_name] = {"status": "skip", "version": current_ver, "desc": desc}
            skip_count += 1
        else:
            # 未安装 → pip install
            print(f"  [{item}/6] 安装 {pkg_name} ({desc}) ...", flush=True)
            success, already = run_pip_install(pkg_name)

            if success:
                # 安装成功后读取版本
                new_ver = get_installed_version(import_name) or "已安装"
                if already:
                    print(f"  [OK] 已有，版本: {new_ver}", flush=True)
                    tag = "skip"
                    skip_count += 1
                else:
                    print(f"  [OK] 新安装，版本: {new_ver}", flush=True)
                    tag = "new"
                    new_count += 1
                results[pkg_name] = {"status": tag, "version": new_ver, "desc": desc}
            else:
                print(f"  [错误] {pkg_name} 安装失败!", flush=True)
                if pkg_name == "mediapipe":
                    print("         可能原因: Python版本不兼容 (仅支持 3.9-3.11)", flush=True)
                all_passed = False
                results[pkg_name] = {"status": "fail", "version": "失败", "desc": desc}

        print(flush=True)
        item += 1

    # ---- 汇总 ----
    if not all_passed:
        print("  ======== Step 3 完成 (有安装失败，请检查) ========", flush=True)
    elif skip_count == len(PACKAGES):
        print(f"  ======== Step 3 完成 ({len(PACKAGES)}/{len(PACKAGES)} 包已安装，无需重新安装) ========", flush=True)
    else:
        print(f"  ======== Step 3 完成 ({skip_count} 已有, {new_count} 新装) ========", flush=True)
    print(flush=True)

    # ---- 写入结果文件供 Step 5 使用 ----
    with open(RESULT_FILE, "w", encoding="utf-8") as f:
        f.write(f"SKIP_COUNT={skip_count}\n")
        f.write(f"NEW_COUNT={new_count}\n")
        f.write(f"PYTHON_VER={sys.version.split()[0]}\n")
        for pkg_name, info in results.items():
            key = pkg_name.upper().replace("-", "_")
            f.write(f"PKG_{key}_VER={info['version']}\n")
            f.write(f"PKG_{key}_STATUS={info['status']}\n")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
