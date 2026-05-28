"""
躯干穴位定位演示脚本

快速验证虚拟脊柱 + 躯干穴位定位功能
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import time

from core.pose_extractor import PoseExtractor, ModelMode, PoseResult
from core.spine_estimator import SpineEstimator
from core.acupoint_locator import AcupointLocator, AcupointResult
from ui.viewer_3d import Viewer3D


def generate_test_image():
    """如果没有测试图片，生成一个简单的人体示意图"""
    h, w = 720, 480
    img = np.ones((h, w, 3), dtype=np.uint8) * 240

    # 画简易人体
    center_x, center_y = w // 2, h // 2

    # 头
    cv2.circle(img, (center_x, 100), 40, (200, 180, 160), -1)
    # 身体
    cv2.line(img, (center_x, 140), (center_x, 380), (200, 180, 160), 15)
    # 手臂
    cv2.line(img, (center_x, 170), (center_x - 80, 280), (200, 180, 160), 8)
    cv2.line(img, (center_x, 170), (center_x + 80, 280), (200, 180, 160), 8)
    # 腿
    cv2.line(img, (center_x, 380), (center_x - 40, 580), (200, 180, 160), 10)
    cv2.line(img, (center_x, 380), (center_x + 40, 580), (200, 180, 160), 10)

    # 标注文字
    cv2.putText(img, "[测试人体示意图]", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)
    cv2.putText(img, "请替换为真实全身站姿照片效果更佳", (10, 55),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

    os.makedirs("input/images", exist_ok=True)
    path = "input/images/test.jpg"
    cv2.imwrite(path, img)
    print(f"[Demo] 已生成测试图片: {path}")
    return path


def main():
    print("=" * 60)
    print("  躯干穴位定位 - 虚拟脊柱验证演示")
    print("=" * 60)

    # 1. 检查/生成测试图片
    image_path = "input/images/test.jpg"
    if not os.path.exists(image_path):
        image_path = generate_test_image()
        print("\n[提示] MediaPipe可能无法在示意图上检测到人体。")
        print("        建议替换为真实的全身站姿照片以获得最佳效果。")
        print("        照片要求：正面站姿、全身入框、分辨率>=720p\n")

    image = cv2.imread(image_path)
    if image is None:
        print(f"错误：无法读取图片 {image_path}")
        return

    print(f"图片尺寸: {image.shape[1]}x{image.shape[0]}")

    # 2. 运行Pose检测（GPU加速）
    print("\n[Step 1] 运行 MediaPipe Pose 检测...")
    t0 = time.time()
    extractor = PoseExtractor(mode=ModelMode.POSE, model_complexity=2)
    print(f"   {PoseExtractor.get_gpu_status()}")
    pose_result = extractor.process(image)
    print(f"   推理耗时: {(time.time() - t0)*1000:.1f}ms")

    if not pose_result.has_pose:
        print("[!] 未检测到人体！")
        print("  可能原因:")
        print("  1. 测试图片不包含清晰的人体")
        print("  2. 图片质量过低")
        print("  3. 人体距离太远或光线不足")
        print("\n  请将真实的全身照片放到 input/images/test.jpg 后重试")
        extractor.close()
        return

    print("[OK] 成功检测到人体骨架")

    # 3. 构建虚拟脊柱
    print("\n[Step 2] 构建虚拟脊柱...")
    spine = None
    if pose_result.pose_world_landmarks is not None:
        spine_estimator = SpineEstimator()
        spine = spine_estimator.build(pose_result.pose_world_landmarks)

        if spine.is_valid:
            print(f"[OK] 虚拟脊柱构建成功")
            print(f"   - 脊柱长度: {spine.spine_length*100:.1f}cm")
            print(f"   - 肩宽: {spine.shoulder_width*100:.1f}cm")
            print(f"   - 估算身高: {spine.estimated_height*100:.0f}cm")
            print(f"   - 躯干宽度: {spine.torso_width*100:.1f}cm")

            # 关键椎骨坐标
            for v in ["C7", "T7", "L2", "L4"]:
                if v in spine.vertebrae:
                    pos = spine.vertebrae[v]
                    name = spine.vertebra_to_acupoint_name(v) or v
                    print(f"   - {name}: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")
        else:
            print("[!]  虚拟脊柱构建失败（关键点不可靠）")
            extractor.close()
            return
    else:
        print("[!]  缺少世界坐标，无法构建虚拟脊柱")
        extractor.close()
        return

    # 4. 加载穴位数据库并定位
    print("\n[Step 3] 穴位定位...")
    locator = AcupointLocator()
    locator.load_database([
        "database/acupoints_torso.json",
        "database/acupoints_limbs.json",
        "database/acupoints_hands.json",
    ])
    print(f"   已加载穴位定义: {locator.get_acupoint_count()} 个")

    acupoint_result = locator.locate(pose_result)
    print(f"\n[OK] 成功定位 {acupoint_result.total_found} 个穴位")

    # 按经络分组统计
    for mc, cnt in sorted(acupoint_result.meridian_counts.items()):
        meridian_names = {
            "CV": "任脉", "GV": "督脉", "BL": "膀胱经", "ST": "胃经",
            "KI": "肾经", "SP": "脾经", "LR": "肝经", "GB": "胆经",
            "LU": "肺经", "LI": "大肠经", "PC": "心包经", "HT": "心经",
        }
        name = meridian_names.get(mc, mc)
        print(f"   {name}({mc}): {cnt}穴")

    # 精度统计
    print(f"\n   精度分布: A:{acupoint_result.grade_counts.get('A',0)} "
          f"B:{acupoint_result.grade_counts.get('B',0)} "
          f"C:{acupoint_result.grade_counts.get('C',0)} "
          f"D:{acupoint_result.grade_counts.get('D',0)}")

    # 列出前10个穴位
    print("\n   前10个定位穴位:")
    for ap in acupoint_result.acupoints[:10]:
        grade_icon = {"A": "[A]", "B": "[B]", "C": "[C]", "D": "[D]"}.get(ap.grade, "?")
        print(f"   {grade_icon} {ap.name_cn}({ap.id}) [{ap.grade}级] "
              f"conf={ap.confidence:.0%} | {ap.meridian}")

    # 5. 生成3D可视化
    print("\n[Step 4] 生成3D可视化...")
    viewer = Viewer3D(width=1280, height=800)
    fig = viewer.build(acupoint_result, pose_result, spine)

    output_html = "output/acupoints_torso_3d.html"
    os.makedirs("output", exist_ok=True)
    viewer.save_html(fig, output_html)

    # 也生成2D投影
    fig_2d = viewer.show_2d_overlay_data(acupoint_result, pose_result)
    output_2d = "output/acupoints_2d.html"
    viewer.save_html(fig_2d, output_2d)

    print(f"\n{'='*60}")
    print(f"  演示完成!")
    print(f"  3D视图: file:///{os.path.abspath(output_html)}")
    print(f"  2D投影: file:///{os.path.abspath(output_2d)}")
    print(f"\n  在浏览器中打开HTML文件查看交互式3D视图")
    print(f"  鼠标操作: 左键拖拽=旋转 | 滚轮=缩放 | 右键拖拽=平移")
    print(f"  悬停穴位 = 查看详细气泡信息")
    print(f"{'='*60}")

    # 尝试在浏览器中打开3D视图
    try:
        import webbrowser
        webbrowser.open(f"file:///{os.path.abspath(output_html)}")
    except:
        pass

    extractor.close()


if __name__ == "__main__":
    main()
