"""
实时摄像头穴位定位演示（Holistic 模式：面部 + 双手 + 姿态）

按键：
  Q = 退出
  S = 保存截图
  G = 切换精度过滤
  1-4 = 选择精度等级
  H = 切换骨架显示
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2, time, numpy as np
from typing import Optional

from core.pose_extractor import PoseExtractor, ModelMode
from core.acupoint_locator import AcupointLocator
from core.population_adapter import PopulationProfile, Gender, PopulationAdapter

# ── 用户画像持久化 ──
_PROFILE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "database", "user_profile.json")

def _load_profile():
    """加载保存的用户画像"""
    import json
    if not os.path.exists(_PROFILE_PATH):
        return None
    try:
        with open(_PROFILE_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return PopulationProfile(
            age=data.get("age", 30),
            gender=Gender.MALE if data.get("gender", "male") == "male" else Gender.FEMALE,
            height_cm=data.get("height_cm", 170.0),
            weight_kg=data.get("weight_kg", 65.0),
        )
    except Exception:
        return None

# ── 绘制常量 ──
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),(5,9),(9,13),(13,17),
]
POSE_CONNECTIONS = [
    (11,12),(11,23),(12,24),(23,24),(11,13),(13,15),(12,14),(14,16),
    (23,25),(25,27),(24,26),(26,28),(27,29),(29,31),(27,31),(28,30),(30,32),(28,32),
]
BGR_COLORS = {
    "CV":(53,107,255),"GV":(0,215,255),"LU":(232,232,232),"LI":(176,176,176),
    "ST":(0,140,255),"SP":(204,50,153),"BL":(225,105,65),"KI":(209,206,0),
    "PC":(60,20,220),"TE":(255,144,30),"GB":(34,139,34),"LR":(34,205,50),
}


def main():
    from mediapipe.tasks.python import vision, BaseOptions
    from mediapipe.tasks.python.vision import RunningMode
    from mediapipe import Image as MPImage, ImageFormat

    print("=" * 60)
    print("  实时摄像头 - Holistic 穴位定位演示")
    print("=" * 60)
    print("  Q=退出  S=截图  G=精度过滤  H=切换骨架  1-4=精度等级")
    print()

    # ── 初始化 Holistic Landmarker ──
    model_path = os.path.join("models", "holistic_landmarker.task")
    if not os.path.exists(model_path):
        print("[!] Holistic模型未找到，回退Pose")
        model_path = os.path.join("models", "pose_landmarker.task")
        holistic = False
    else:
        holistic = True

    base = BaseOptions(model_asset_path=model_path)
    if holistic:
        opts = vision.HolisticLandmarkerOptions(
            base_options=base, running_mode=RunningMode.VIDEO,
            min_face_detection_confidence=0.5, min_face_landmarks_confidence=0.5,
            min_pose_detection_confidence=0.5, min_pose_landmarks_confidence=0.5,
            min_hand_landmarks_confidence=0.5,
            output_face_blendshapes=False, output_segmentation_mask=False,
        )
        landmarker = vision.HolisticLandmarker.create_from_options(opts)
        print("  模式: Holistic (面部 + 双手 + 姿态)")
    else:
        opts = vision.PoseLandmarkerOptions(
            base_options=base, running_mode=RunningMode.VIDEO,
            num_poses=1, min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5, min_tracking_confidence=0.5,
            output_segmentation_masks=False,
        )
        landmarker = vision.PoseLandmarker.create_from_options(opts)
        print("  模式: Pose (仅姿态)")

    # ── 穴位定位器 ──
    profile = _load_profile()
    if profile:
        print(f"  [人群画像] {'男' if profile.gender == Gender.MALE else '女'} {profile.age}岁 "
              f"BMI={profile.bmi:.1f} {profile.body_type.value}")
    locator = AcupointLocator(profile=profile)
    locator.load_database(["database/acupoints_torso.json", "database/acupoints_limbs.json", "database/acupoints_hands.json"])
    print(f"  穴位: {locator.get_acupoint_count()} 个")

    # ── PoseExtractor（用于穴位分析）──
    extractor = PoseExtractor(mode=ModelMode.POSE)
    print(f"  {PoseExtractor.get_gpu_status()}")
    print("\n启动中...\n")

    # ── 摄像头 ──
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    if not cap.isOpened():
        print("错误: 无法打开摄像头")
        return

    show_skeleton = True
    grade_filter = "D"
    grade_order = {"A":0,"B":1,"C":2,"D":3,"ALL":9}
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        display = frame.copy()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = MPImage(image_format=ImageFormat.SRGB, data=frame_rgb)

        t0 = time.time()
        ts = int(time.time() * 1000)
        result = landmarker.detect_for_video(mp_image, ts)
        t_ms = (time.time() - t0) * 1000
        frame_count += 1

        # ── 绘制 Holistic ──
        if result is not None and show_skeleton:
            # 面部
            if holistic:
                face_lms = getattr(result, 'face_landmarks', None)
                if face_lms and len(face_lms) > 0:
                    for fl in face_lms:
                        pts = [(int(lm.x*w), int(lm.y*h)) for lm in fl]
                        for i in range(17):
                            j = (i+1) % 17
                            cv2.line(display, pts[i], pts[j], (255,180,100), 1, cv2.LINE_AA)
                        cv2.circle(display, pts[1], 4, (50,200,255), -1)
                        cv2.circle(display, pts[159], 2, (255,255,255), -1)
                        cv2.circle(display, pts[386], 2, (255,255,255), -1)

            # 双手
            if holistic:
                for hand_key, color in [('left_hand_landmarks', (100,255,100)),
                                         ('right_hand_landmarks', (255,150,50))]:
                    hls = getattr(result, hand_key, None)
                    if hls and len(hls) > 0:
                        for hl in hls:
                            pts = [(int(lm.x*w), int(lm.y*h)) for lm in hl]
                            for a,b in HAND_CONNECTIONS:
                                cv2.line(display, pts[a], pts[b], color, 2, cv2.LINE_AA)
                            for i,p in enumerate(pts):
                                r = 5 if i in (4,8,12,16,20) else 3
                                cv2.circle(display, p, r, color, -1)

            # 姿态
            pl = getattr(result, 'pose_landmarks', None)
            if pl and len(pl) > 0:
                pts = [(int(lm.x*w), int(lm.y*h)) for lm in pl[0]]
                for a,b in POSE_CONNECTIONS:
                    if a < len(pts) and b < len(pts):
                        cv2.line(display, pts[a], pts[b], (120,180,255), 2, cv2.LINE_AA)
                for i,p in enumerate(pts):
                    vis = pl[0][i].visibility if hasattr(pl[0][i], 'visibility') else 1.0
                    if vis > 0.5:
                        cv2.circle(display, p, 4, (255,220,100), -1)

        # ── 穴位定位（从 PoseExtractor）──
        pose_result = extractor.process(frame)
        if pose_result.has_pose:
            acu = locator.locate(pose_result)
            if acu and acu.acupoints:
                for ap in acu.acupoints:
                    if grade_order.get(ap.grade, 9) > grade_order.get(grade_filter, 9):
                        continue
                    if ap.position_2d is not None:
                        px, py = int(ap.position_2d[0]), int(ap.position_2d[1])
                        if 0 <= px < w and 0 <= py < h:
                            color = BGR_COLORS.get(ap.meridian_code, (0,0,255))
                            r = 10 if ap.grade=="A" else 7 if ap.grade=="B" else 4
                            cv2.circle(display, (px,py), r, color, -1)
                            cv2.circle(display, (px,py), r, (255,255,255), 1)
                            cv2.putText(display, ap.name_cn, (px+6,py-6),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)

        # ── HUD ──
        fps = 1000/t_ms if t_ms > 0 else 0
        cv2.putText(display, f"FPS:{fps:.0f} | {t_ms:.0f}ms | Holistic | 过滤:{grade_filter}级 | H=骨架",
                   (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,255), 1, cv2.LINE_AA)

        cv2.imshow('perfess mediapipe test', display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            break
        elif key == ord('h') or key == ord('H'):
            show_skeleton = not show_skeleton
        elif key == ord('s') or key == ord('S'):
            ts_str = time.strftime("%Y%m%d_%H%M%S")
            os.makedirs("output/screenshots", exist_ok=True)
            cv2.imwrite(f"output/screenshots/acupoint_{ts_str}.png", display)
            print(f"  截图已保存")
        elif key == ord('g') or key == ord('G'):
            grades = ["A","B","C","D","ALL"]
            idx = grades.index(grade_filter) if grade_filter in grades else 3
            grade_filter = grades[(idx+1)%len(grades)]
            print(f"  精度过滤: {grade_filter}级")
        elif key == ord('1'): grade_filter = "A"
        elif key == ord('2'): grade_filter = "B"
        elif key == ord('3'): grade_filter = "C"
        elif key == ord('4'): grade_filter = "D"

    landmarker.close()
    extractor.close()
    cap.release()
    cv2.destroyAllWindows()
    print(f"已停止. 总帧数: {frame_count}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n用户中断.")
