"""
最小验证环境 - 一键测试脚本

流程：
1. 摄像头选择
2. 实时 Holistic 检测预览（面部 + 双手 + 姿态骨架）
   ── 你能看到检测框、面部网格、手部关键点、姿态骨架
3. 空格键拍照 → 运行穴位定位管线 → 浏览器 3D 视图
4. 按 Q 或 Ctrl+C 退出
"""
import sys, os, time, json, cv2, numpy as np
from datetime import datetime
from typing import Optional


def _imread_quiet(path):
    """读取图片并抑制 libpng C 级别的 stderr 警告"""
    devnull = os.open(os.devnull, os.O_WRONLY)
    saved = os.dup(2)  # 备份原始 stderr
    os.dup2(devnull, 2)
    os.close(devnull)
    img = cv2.imread(path)
    os.dup2(saved, 2)  # 恢复 stderr
    os.close(saved)
    return img

# ── PIL 中文渲染支持 ──
from PIL import Image, ImageDraw, ImageFont

# 尝试加载系统中文字体（Windows优先）
_CN_FONT = None
for _fn in [
    "C:/Windows/Fonts/msyh.ttc",    # 微软雅黑
    "C:/Windows/Fonts/simhei.ttf",   # 黑体
    "C:/Windows/Fonts/simsun.ttc",   # 宋体
]:
    if os.path.exists(_fn):
        try:
            _CN_FONT = ImageFont.truetype(_fn, 20)
            break
        except Exception:
            continue


def _put_chinese_text(img_bgr, text, position, color, font_size=20):
    """在 OpenCV 图像上绘制中文文字（使用PIL）"""
    global _CN_FONT
    if not _CN_FONT or not text.strip():
        return
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)
    # 动态调整字号
    if font_size != 20 and _CN_FONT is not None:
        try:
            fnt = ImageFont.truetype(_CN_FONT.path if hasattr(_CN_FONT, 'path')
                                      else "C:/Windows/Fonts/msyh.ttc", font_size)
        except Exception:
            fnt = _CN_FONT
    else:
        fnt = _CN_FONT
    bgr = (int(color[2]), int(color[1]), int(color[0]))
    draw.text(position, text, font=fnt, fill=bgr)
    img_bgr[:] = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.pose_extractor import PoseExtractor, ModelMode
from core.spine_estimator import SpineEstimator
from core.acupoint_locator import AcupointLocator
from core.population_adapter import PopulationAdapter, PopulationProfile, Gender

# ── 用户画像持久化 ──────────────────────────────────────────
_PROFILE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "database", "user_profile.json")


def _load_user_profile() -> Optional[PopulationProfile]:
    """从本地文件加载用户画像"""
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


def _save_user_profile(profile: PopulationProfile):
    """保存用户画像到本地文件"""
    os.makedirs(os.path.dirname(_PROFILE_PATH), exist_ok=True)
    data = {
        "age": profile.age,
        "gender": profile.gender.value,
        "height_cm": profile.height_cm,
        "weight_kg": profile.weight_kg,
        "bmi": round(profile.bmi, 1),
        "age_group": profile.age_group.value,
        "body_type": profile.body_type.value,
    }
    with open(_PROFILE_PATH, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _prompt_user_profile() -> PopulationProfile:
    """
    交互式采集用户身体参数（身高/体重/年龄/性别）。
    如果已有保存的画像则直接加载，允许用户修改。
    """
    existing = _load_user_profile()
    if existing is not None:
        print("\n" + "=" * 50)
        print(f"  [已有用户画像]")
        print(f"    年龄: {existing.age}岁  |  性别: {'男' if existing.gender == Gender.MALE else '女'}")
        print(f"    身高: {existing.height_cm}cm  |  体重: {existing.weight_kg}kg")
        print(f"    BMI: {existing.bmi:.1f}  |  体型: {existing.body_type.value}")
        print(f"    年龄组: {existing.age_group.value}")
        print("=" * 50)
        ans = input("  是否使用此画像? [Y=是 / N=重新输入]: ").strip().lower()
        if ans != 'n':
            return existing

    print("\n" + "=" * 50)
    print("    请输入您的身体参数（用于穴位定位精度校准）")
    print("=" * 50)

    # 性别
    while True:
        gender_str = input("  性别 [1=男 / 2=女, 默认1]: ").strip()
        if gender_str in ('', '1'):
            gender = Gender.MALE
            break
        elif gender_str == '2':
            gender = Gender.FEMALE
            break
        else:
            print("    请输入 1 或 2")

    # 年龄
    while True:
        try:
            age_str = input("  年龄 [默认30]: ").strip()
            age = 30 if age_str == '' else int(age_str)
            if 1 <= age <= 120:
                break
            else:
                print("    请输入 1-120 之间的数字")
        except ValueError:
            print("    请输入有效数字")

    # 身高
    while True:
        try:
            h_str = input("  身高(cm) [默认170]: ").strip()
            height = 170.0 if h_str == '' else float(h_str)
            if 50 <= height <= 250:
                break
            else:
                print("    请输入 50-250 之间的数字")
        except ValueError:
            print("    请输入有效数字")

    # 体重
    while True:
        try:
            w_str = input("  体重(kg) [默认65]: ").strip()
            weight = 65.0 if w_str == '' else float(w_str)
            if 20 <= weight <= 300:
                break
            else:
                print("    请输入 20-300 之间的数字")
        except ValueError:
            print("    请输入有效数字")

    profile = PopulationProfile(age=age, gender=gender, height_cm=height, weight_kg=weight)
    _save_user_profile(profile)

    print(f"\n  [OK] 画像已保存:")
    print(f"    {age}岁 {'男' if gender == Gender.MALE else '女'}性")
    print(f"    {height}cm / {weight}kg  BMI={profile.bmi:.1f}")
    print(f"    年龄组: {profile.age_group.value}  体型: {profile.body_type.value}")
    print(f"    适配系数: ratio×{profile.age_group.value} + body×{profile.body_type.value}")
    return profile


def _format_profile_hud(profile: PopulationProfile) -> str:
    """格式化用户画像为 HUD 显示字符串"""
    gender_label = "男" if profile.gender == Gender.MALE else "女"
    return f"{gender_label} {profile.age}岁 {profile.height_cm}cm {profile.weight_kg}kg BMI={profile.bmi:.1f}"


# ── Landmark 解包 ──────────────────────────────────────────
def _unwrap_landmarks(landmark_data):
    """处理 IMAGE 模式（平铺列表）和 VIDEO 模式（嵌套列表）的 landmarks 结构"""
    if landmark_data is None or len(landmark_data) == 0:
        return None
    first = landmark_data[0]
    # IMAGE 模式：第一个元素是 NormalizedLandmark（有 .x 属性），直接返回
    if hasattr(first, 'x'):
        return landmark_data
    # VIDEO 模式：第一个元素是 NormalizedLandmarkList，取其内容
    try:
        return list(first)
    except TypeError:
        return None


# ── 绘制常量 ──────────────────────────────────────────────

# 面部轮廓（环绕脸部一圈的 17 个点）
FACE_OVAL = list(range(17))

# 左手关键点连线 (21 点)
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),          # 拇指
    (0,5),(5,6),(6,7),(7,8),          # 食指
    (0,9),(9,10),(10,11),(11,12),     # 中指
    (0,13),(13,14),(14,15),(15,16),   # 无名指
    (0,17),(17,18),(18,19),(19,20),   # 小指
    (5,9),(9,13),(13,17),             # MCP 横向
]

# 姿态骨架连线
POSE_CONNECTIONS = [
    (11,12),(11,23),(12,24),(23,24),            # 躯干
    (11,13),(13,15),(12,14),(14,16),            # 上臂
    (23,25),(25,27),(24,26),(26,28),            # 大腿
    (27,29),(29,31),(27,31),                    # 小腿+脚
    (28,30),(30,32),(28,32),
]

# 检测状态颜色
COLOR_OK   = (0, 200, 0)
COLOR_WARN = (50, 150, 255)
COLOR_OFF  = (80, 80, 80)
COLOR_FACE = (255, 180, 100)
COLOR_HAND = (100, 255, 100)
COLOR_POSE = (120, 180, 255)


# ── 统一的身体朝向检测 ──────────────────────────────────────

def _detect_body_orientation(pose_lms_np, has_face=False):
    """
    检测人体朝向（正面/背面）。
    
    与 AcupointLocator.detect_body_orientation() 使用完全相同的算法，
    确保静态图片和动态视频得到一致的朝向判断结果。
    
    Args:
        pose_lms_np: numpy (33, 4) 或 None
        has_face: 是否检测到面部网格（Holistic模式强信号）
    
    Returns: "front" / "back" / "unknown"
    """
    if has_face:
        return "front"
    if pose_lms_np is None:
        return "unknown"

    face_indices = list(range(0, 11))
    visibilities = []
    for idx in face_indices:
        if idx < pose_lms_np.shape[0]:
            vis = pose_lms_np[idx, 3] if pose_lms_np.shape[1] > 3 else 0.0
            visibilities.append(vis)

    if not visibilities:
        return "unknown"

    visible_count = sum(1 for v in visibilities if v > 0.5)
    if visible_count >= 3:
        return "front"
    if visible_count == 0:
        return "back"
    return "unknown"


def draw_body_indicators(image, pose_lms_np, orientation=None, has_face=False):
    """
    在图像上绘制躯干朝向指示标记（统一函数，静态和视频共用）。
    
    正面：左右乳头标记 + 肚脐十字标记
    背面：脊柱椎骨标记点（C7~L4）
    
    Args:
        image: BGR 图像 (H, W, 3) — 原地修改
        pose_lms_np: numpy (33, N) 归一化pose坐标
        orientation: "front"/"back"/"unknown" — 如为None则自动检测
        has_face: 是否有面部检测结果
    """
    h, w = image.shape[:2]
    if pose_lms_np is None or pose_lms_np.shape[0] < 25:
        return
    
    # 自动检测朝向（如未传入）
    if orientation is None:
        orientation = _detect_body_orientation(pose_lms_np, has_face)
    
    # 肩、髋关键点
    shoulder_l = pose_lms_np[11, :2]
    shoulder_r = pose_lms_np[12, :2]
    hip_l = pose_lms_np[23, :2]
    hip_r = pose_lms_np[24, :2]
    
    mid_shoulder = (shoulder_l + shoulder_r) / 2.0
    mid_hip = (hip_l + hip_r) / 2.0
    
    if orientation == "front":
        # ── 乳头（T4水平, offset_t≈0.17, 旁开25%半肩宽） ──
        nipple_t = 0.17
        nipple_ratio = 0.25
        mid_y_nipple = float(mid_shoulder[1] + nipple_t * (mid_hip[1] - mid_shoulder[1]))
        mid_x_nipple = float(mid_shoulder[0] + nipple_t * (mid_hip[0] - mid_shoulder[0]))
        half_span = float(abs(shoulder_r[0] - shoulder_l[0]) * nipple_ratio)
        
        for (px, py) in [
            (int((mid_x_nipple - half_span) * w), int(mid_y_nipple * h)),
            (int((mid_x_nipple + half_span) * w), int(mid_y_nipple * h)),
        ]:
            if 0 <= px < w and 0 <= py < h:
                cv2.circle(image, (px, py), 8, (180, 80, 255), 2)
                cv2.circle(image, (px, py), 3, (180, 80, 255), -1)
        
        # ── 肚脐（L1水平, offset_t≈0.54） ──
        navel_t = 0.54
        navel_x = float(mid_shoulder[0] + navel_t * (mid_hip[0] - mid_shoulder[0]))
        navel_y = float(mid_shoulder[1] + navel_t * (mid_hip[1] - mid_shoulder[1]))
        nv = (int(navel_x * w), int(navel_y * h))
        if 0 <= nv[0] < w and 0 <= nv[1] < h:
            cv2.circle(image, nv, 10, (0, 210, 210), 2)
            cv2.circle(image, nv, 4, (0, 210, 210), -1)
            cv2.line(image, (nv[0]-7, nv[1]-7), (nv[0]+7, nv[1]+7), (0, 210, 210), 1)
            cv2.line(image, (nv[0]-7, nv[1]+7), (nv[0]+7, nv[1]-7), (0, 210, 210), 1)
            _put_chinese_text(image, "肚脐", (nv[0] - 12, nv[1] + 12),
                            (0, 210, 210), font_size=12)
    
    elif orientation == "back":
        # ── 脊柱标记（C7→T3→T5→T7→T11→L2→L4） ──
        vertebra = [
            (0.00, "C7"), (0.12, "T3"), (0.21, "T5"),
            (0.29, "T7"), (0.46, "T11"), (0.58, "L2"), (0.67, "L4"),
        ]
        spine_points = []
        for t_ratio, name in vertebra:
            vx = float(mid_shoulder[0] + t_ratio * (mid_hip[0] - mid_shoulder[0]))
            vy = float(mid_shoulder[1] + t_ratio * (mid_hip[1] - mid_shoulder[1]))
            px, py = int(vx * w), int(vy * h)
            if 0 <= px < w and 0 <= py < h:
                cv2.circle(image, (px, py), 5, (100, 200, 255), -1)
                cv2.circle(image, (px, py), 6, (150, 120, 50), 1)
                _put_chinese_text(image, name, (px + 8, py - 14),
                                (100, 200, 255), font_size=11)
                spine_points.append((px, py))
        # 绘制脊柱连线
        if len(spine_points) >= 2:
            for i in range(len(spine_points) - 1):
                cv2.line(image, spine_points[i], spine_points[i+1],
                        (80, 160, 200), 2, cv2.LINE_AA)
    
    # ── 朝向标签（放在右上角，不遮挡HUD和穴位列表） ──
    label = {"front": "【正面】", "back": "【背面】", "unknown": "【朝向未识别】"}
    label_c = {"front": (0, 255, 0), "back": (100, 220, 255), "unknown": (150, 150, 150)}
    _put_chinese_text(image, label.get(orientation, "?"),
                     (w - 140, 12), label_c.get(orientation, (150, 150, 150)),
                     font_size=22)


# ── 摄像头扫描 / 选择 ─────────────────────────────────────

def scan_cameras(max_index: int = 9):
    available = []
    for idx in range(max_index + 1):
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            continue
        ret, frame = cap.read()
        if ret and frame is not None:
            h, w = frame.shape[:2]
            name = f"摄像头 #{idx}  ({w}x{h})"
            available.append({"id": idx, "name": name, "width": w, "height": h})
        cap.release()
    return available


def _draw_help_panel(img, x, y, w_panel, lines):
    """在画面上绘制半透明帮助面板"""
    overlay = img.copy()
    h_panel = 20 + len(lines) * 28
    cv2.rectangle(overlay, (x, y), (x + w_panel, y + h_panel), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
    for i, (text, color) in enumerate(lines):
        cv2.putText(img, text, (x + 12, y + 25 + i * 28),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)


def select_camera():
    print("\n[Camera Scan] Scanning available cameras...")
    cameras = scan_cameras()
    if not cameras:
        print("  [!] No camera detected!")
        return None
    print(f"  Found {len(cameras)} camera(s):")
    for c in cameras:
        print(f"    #{c['id']}: {c['name']}")
    if len(cameras) == 1:
        c = cameras[0]
        print(f"\n  >> Auto-select: #{c['id']} ({c['name']})\n")
        return c["id"]

    print("\n  [Camera Selector] Use LEFT/RIGHT to switch, ENTER to confirm, Q to quit\n")

    current_idx = 0
    caps = []
    selected = None
    for c in cameras:
        cap = cv2.VideoCapture(c["id"], cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        caps.append(cap)

    try:
        while True:
            ret, frame = caps[current_idx].read()
            if not ret or frame is None:
                current_idx = (current_idx + 1) % len(cameras)
                continue
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            display = frame.copy()

            # 顶部标题栏
            cam = cameras[current_idx]
            cv2.putText(display, f"Camera Selector  |  Current: #{cam['id']} ({cam['width']}x{cam['height']})",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)

            # ── 中央大号提示面板 ──
            help_lines = [
                ("  LEFT / RIGHT   or   A / D   =   Switch Camera", (255, 255, 255)),
                ("  ENTER   =   Confirm & Continue", (0, 255, 100)),
                ("  Q       =   Quit Program", (100, 100, 255)),
            ]
            _draw_help_panel(display, 20, h//2 - 60, 420, help_lines)

            # ── 底部摄像头指示器 ──
            indicator_y = h - 40
            dot_spacing = min(80, (w - 60) // max(len(cameras), 1))
            for i in range(len(cameras)):
                ix = 40 + i * dot_spacing
                if i == current_idx:
                    cv2.circle(display, (ix, indicator_y), 12, (0, 255, 0), -1)
                    cv2.rectangle(display, (ix - 14, indicator_y + 14),
                                 (ix + 14, indicator_y + 22), (0, 0, 0), -1)
                    cv2.putText(display, f"#{cameras[i]['id']}", (ix - 14, indicator_y + 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
                else:
                    cv2.circle(display, (ix, indicator_y), 10, (80, 80, 80), 1)

            cv2.imshow('perfess mediapipe test', display)

            key = cv2.waitKey(50) & 0xFF
            if key == 13:  # Enter
                selected = cameras[current_idx]["id"]
                break
            elif key == ord('q') or key == ord('Q'):
                selected = None
                break
            elif key == 81 or key == ord('a'):  # Left arrow or A
                current_idx = (current_idx - 1) % len(cameras)
            elif key == 83 or key == ord('d'):  # Right arrow or D
                current_idx = (current_idx + 1) % len(cameras)
    finally:
        for cap in caps:
            cap.release()
        cv2.destroyWindow('perfess mediapipe test')

    if selected is not None:
        cam = next(c for c in cameras if c["id"] == selected)
        print(f"\n  Selected: #{cam['id']} {cam['name']}\n")
    else:
        print("\n  [Cancelled]\n")
    return selected


# ── 实时 Holistic 检测预览 ────────────────────────────────

def live_detection_preview(camera_id: int):
    """
    实时 MediaPipe Holistic 检测预览。
    
    画面中实时绘制：
    - 面部网格（橙色轮廓）
    - 左手关键点（绿色骨架）
    - 右手关键点（绿色骨架）
    - 姿态骨架（蓝色连线）
    - 右侧状态面板（实时显示各模块检测结果）
    
    按键：
      空格 = 拍照 → 运行穴位分析管线
      Q    = 退出
      G    = 切换姿态显示
    """
    from mediapipe.tasks.python import vision, BaseOptions
    from mediapipe.tasks.python.vision import RunningMode
    from mediapipe import Image as MPImage, ImageFormat

    model_path = os.path.join("models", "holistic_landmarker.task")
    if not os.path.exists(model_path):
        # 回退：只检测姿态
        model_path = os.path.join("models", "pose_landmarker.task")
        holistic_mode = False
        print("  [!] Holistic model not found, falling back to Pose mode")
    else:
        holistic_mode = True

    print(f"\n[Live Preview] Starting camera #{camera_id}...")

    cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    if not cap.isOpened():
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera #{camera_id}")

    # 创建 Holistic / Pose Landmarker（VIDEO 模式，支持逐帧 detect_for_video）
    base = BaseOptions(model_asset_path=model_path)
    if holistic_mode:
        opts = vision.HolisticLandmarkerOptions(
            base_options=base,
            running_mode=RunningMode.VIDEO,
            min_face_detection_confidence=0.5,
            min_face_landmarks_confidence=0.5,
            min_pose_detection_confidence=0.5,
            min_pose_landmarks_confidence=0.5,
            min_hand_landmarks_confidence=0.5,
            output_face_blendshapes=False,
            output_segmentation_mask=False,
        )
        landmarker = vision.HolisticLandmarker.create_from_options(opts)
    else:
        opts = vision.PoseLandmarkerOptions(
            base_options=base,
            running_mode=RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_segmentation_masks=False,
        )
        landmarker = vision.PoseLandmarker.create_from_options(opts)

    # 状态变量
    show_pose = True
    frame_ms = 0
    captured_frame = None
    latest_result = None

    print("  Controls: [SPACE] Capture & Analyze  [G] Toggle Skeleton  [Q] Quit\n")
    print("  Place your face + hands in front of the camera to see live detection\n")

    while True:
        ret, frame_bgr = cap.read()
        if not ret or frame_bgr is None:
            continue

        # 镜像
        frame_bgr = cv2.flip(frame_bgr, 1)
        h, w = frame_bgr.shape[:2]
        display = frame_bgr.copy()
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = MPImage(image_format=ImageFormat.SRGB, data=frame_rgb)

        # ── MediaPipe 推理 ──
        t0 = time.time()
        frame_ms_timestamp = int(time.time() * 1000)
        if holistic_mode:
            latest_result = landmarker.detect_for_video(mp_image, frame_ms_timestamp)
        else:
            latest_result = landmarker.detect_for_video(mp_image, frame_ms_timestamp)
        frame_ms = (time.time() - t0) * 1000

        # ── 状态检测变量 ──
        face_ok   = False
        left_ok   = False
        right_ok  = False
        pose_ok   = False

        # ── 绘制 ──
        if latest_result is not None:
            # 1. 面部网格
            if holistic_mode:
                face_lm = _unwrap_landmarks(getattr(latest_result, 'face_landmarks', None))
                if face_lm:
                    face_ok = True
                    pts = [(int(lm.x * w), int(lm.y * h)) for lm in face_lm]
                    # 脸部轮廓
                    for i in range(17):
                        j = (i + 1) % 17
                        cv2.line(display, pts[i], pts[j], COLOR_FACE, 1, cv2.LINE_AA)
                    # 左眉
                    for i in [63, 105, 66, 107, 55]:
                        cv2.circle(display, pts[i], 1, COLOR_FACE, -1)
                    # 右眉
                    for i in [296, 334, 293, 336, 285]:
                        cv2.circle(display, pts[i], 1, COLOR_FACE, -1)
                    # 眼睛中心
                    cv2.circle(display, pts[159], 3, (255,255,255), -1)  # 左眼
                    cv2.circle(display, pts[386], 3, (255,255,255), -1)  # 右眼
                    # 鼻尖
                    cv2.circle(display, pts[1], 4, (50,200,255), -1)
                    # 嘴角
                    cv2.circle(display, pts[61], 2, (200,150,200), -1)
                    cv2.circle(display, pts[291], 2, (200,150,200), -1)

            # 2. 双手
            if holistic_mode:
                # 左手
                hand_lm = _unwrap_landmarks(getattr(latest_result, 'left_hand_landmarks', None))
                if hand_lm:
                    left_ok = True
                    pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_lm]
                    for (a, b) in HAND_CONNECTIONS:
                        cv2.line(display, pts[a], pts[b], COLOR_HAND, 2, cv2.LINE_AA)
                    for i, p in enumerate(pts):
                        r = 5 if i in (4,8,12,16,20) else 3
                        cv2.circle(display, p, r, (50,255,50), -1)

                # 右手
                hand_lm = _unwrap_landmarks(getattr(latest_result, 'right_hand_landmarks', None))
                if hand_lm:
                    right_ok = True
                    pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_lm]
                    for (a, b) in HAND_CONNECTIONS:
                        cv2.line(display, pts[a], pts[b], (255,150,50), 2, cv2.LINE_AA)
                    for i, p in enumerate(pts):
                        r = 5 if i in (4,8,12,16,20) else 3
                        cv2.circle(display, p, r, (255,180,50), -1)

            # 3. 姿态骨架
            pose = _unwrap_landmarks(getattr(latest_result, 'pose_landmarks', None))
            if pose and show_pose:
                pose_ok = True
                pts = [(int(lm.x * w), int(lm.y * h)) for lm in pose]
                for (a, b) in POSE_CONNECTIONS:
                    if a < len(pts) and b < len(pts):
                        cv2.line(display, pts[a], pts[b], COLOR_POSE, 3, cv2.LINE_AA)
                for i, p in enumerate(pts):
                    vis = pose[i].visibility if hasattr(pose[i], 'visibility') else 1.0
                    if vis > 0.5:
                        r = 6 if i in (0,7,8,11,12,23,24) else 4
                        cv2.circle(display, p, r, (255,220,100), -1)

            # ── 3.5. 躯干朝向实时指示（与静态图片模式使用同一套算法） ──
            pose_raw = _unwrap_landmarks(getattr(latest_result, 'pose_landmarks', None))
            if pose_raw and len(pose_raw) >= 25:
                # 转换为 numpy array（与 PoseExtractor 输出格式一致）
                pose_np = np.array([
                    [lm.x, lm.y, lm.z,
                     lm.visibility if hasattr(lm, 'visibility') else 1.0]
                    for lm in pose_raw
                ])
                has_face_preview = bool(
                    _unwrap_landmarks(getattr(latest_result, 'face_landmarks', None)))
                draw_body_indicators(display, pose_np,
                                    orientation=None, has_face=has_face_preview)

        # ── 右侧状态面板 ──
        panel_x = w - 200
        panel_y = 20
        panel_items = [
            ("Face   ", face_ok),
            ("L-Hand ", left_ok),
            ("R-Hand ", right_ok),
            ("Pose   ", pose_ok),
        ]
        for i, (label, ok) in enumerate(panel_items):
            y = panel_y + i * 30
            color = COLOR_OK if ok else COLOR_OFF
            cv2.putText(display, label, (panel_x, y + 22),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)
            # 状态指示圆点
            cv2.circle(display, (panel_x - 10, y + 17), 6, color, -1 if ok else 1)

        # 检测到人数
        total_detected = sum([face_ok, left_ok, right_ok, pose_ok])
        status_text = "Waiting..." if total_detected == 0 else f"Detected {total_detected}/4"
        status_color = COLOR_OFF if total_detected == 0 else COLOR_OK
        cv2.putText(display, status_text, (panel_x, panel_y + 150),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.55, status_color, 1, cv2.LINE_AA)

        # ── 顶部 HUD ──
        fps = 1000 / frame_ms if frame_ms > 0 else 0
        cv2.putText(display, f"FPS:{fps:.0f} | Latency:{frame_ms:.0f}ms | SPACE=Capture | G=Skeleton | Q=Quit",
                   (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1, cv2.LINE_AA)

        # ── 底部提示 ──
        cv2.putText(display, "Place face + hands in view to see live detection",
                   (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180,180,180), 1, cv2.LINE_AA)

        cv2.imshow('perfess mediapipe test', display)

        # ── 按键处理 ──
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            captured_frame = frame_bgr.copy()
            break
        elif key == ord('q') or key == ord('Q'):
            captured_frame = None
            break
        elif key == ord('g') or key == ord('G'):
            show_pose = not show_pose
            print(f"  Skeleton display: {'ON' if show_pose else 'OFF'}")

    # 清理
    landmarker.close()
    cap.release()
    cv2.destroyAllWindows()

    return captured_frame


# ── 穴位绘制辅助函数 ──────────────────────────────────────

# 经络颜色 (BGR格式用于OpenCV)
MERIDIAN_BGR = {
    "CV": (53, 107, 255),    # 任脉 橙红
    "GV": (0, 215, 255),     # 督脉 金色
    "LU": (232, 232, 232),   # 肺经 白
    "LI": (176, 176, 176),   # 大肠经 灰
    "ST": (0, 140, 255),     # 胃经 深橙
    "SP": (204, 50, 153),    # 脾经 紫
    "HT": (0, 0, 255),       # 心经 红
    "SI": (180, 105, 255),   # 小肠经 粉红
    "BL": (225, 105, 65),    # 膀胱经 蓝
    "KI": (209, 206, 0),     # 肾经 青
    "PC": (60, 20, 220),     # 心包经 深粉
    "TE": (255, 144, 30),    # 三焦经 蓝
    "GB": (34, 139, 34),     # 胆经 绿
    "LR": (34, 205, 50),     # 肝经 深绿
    "EX": (180, 180, 180),   # 经外奇穴
}


def _show_zoomable_result(annotated: np.ndarray):
    """
    可缩放/拖拽的结果查看器。

    操作：
      鼠标滚轮 = 缩放（以鼠标位置为中心）
      左键拖拽 = 平移
      R 键     = 重置视图
      +/- 键   = 缩放
      WASD键   = 平移
      Q/ESC    = 关闭查看器

    穴位标注已固定在原始图像像素上，缩放只是改变观察视角，
    不会触发重新识别。
    """
    H, W = annotated.shape[:2]      # 源图像尺寸
    canvas_h, canvas_w = H, W        # 画布尺寸（与源图一致）

    # ── 视口状态：view_x/view_y = 源图像中视口左上角坐标 ──
    scale = 1.0                      # 缩放倍率 (≥1)
    min_scale, max_scale = 1.0, 8.0
    view_x, view_y = 0.0, 0.0       # 源图像坐标系

    # 拖拽状态
    dragging = False
    drag_start = (0, 0)
    drag_view_start = (0.0, 0.0)

    win_name = "Acupoint Viewer [Scroll=Zoom | Drag=Pan | R=Reset | Q=Quit]"

    def clamp_view():
        """限制视口不超出源图像范围"""
        nonlocal view_x, view_y
        vw = canvas_w / scale   # 视口在源图像中的宽度
        vh = canvas_h / scale
        margin = 5.0 / scale    # 允许少量越界（5像素）
        view_x = max(-margin, min(W - vw + margin, view_x))
        view_y = max(-margin, min(H - vh + margin, view_y))

    def render():
        nonlocal view_x, view_y
        clamp_view()

        vw = canvas_w / scale
        vh = canvas_h / scale

        # 源图像中待裁剪区域（浮点→整数）
        sx1 = max(0, int(view_x))
        sy1 = max(0, int(view_y))
        sx2 = min(W, int(np.ceil(view_x + vw)))
        sy2 = min(H, int(np.ceil(view_y + vh)))

        canvas = np.full((canvas_h, canvas_w, 3), 35, dtype=np.uint8)

        if sx1 < sx2 and sy1 < sy2:
            crop = annotated[sy1:sy2, sx1:sx2]
            # crop 在画布上的目标矩形
            dx1 = max(0, int((sx1 - view_x) * scale))
            dy1 = max(0, int((sy1 - view_y) * scale))
            # 预计算未裁剪的目标尺寸（浮点）
            dw_raw = crop.shape[1] * scale
            dh_raw = crop.shape[0] * scale
            # 裁剪到画布内
            dx2 = min(dx1 + int(dw_raw + 0.5), canvas_w)
            dy2 = min(dy1 + int(dh_raw + 0.5), canvas_h)
            target_w = dx2 - dx1
            target_h = dy2 - dy1
            if target_w > 0 and target_h > 0:
                # 直接 resize 到精确的目标尺寸，避免 shape 不匹配
                resized = cv2.resize(crop, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
                canvas[dy1:dy2, dx1:dx2] = resized

        # 底部状态栏
        cv2.rectangle(canvas, (0, canvas_h - 26), (canvas_w, canvas_h), (0, 0, 0), -1)
        info = f"Zoom: {scale:.1f}x  |  滚轮=缩放 | 拖拽=平移 | R=重置 | Q/ESC=退出"
        cv2.putText(canvas, info, (8, canvas_h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)

        cv2.imshow(win_name, canvas)

    def _safe_render():
        """安全渲染，捕获任何 shape 不匹配等异常"""
        try:
            render()
        except Exception as e:
            # 静默跳过单帧错误，避免整个查看器崩溃
            pass

    def on_mouse(event, x, y, flags, param):
        nonlocal scale, view_x, view_y, dragging, drag_start, drag_view_start

        if event == cv2.EVENT_MOUSEWHEEL:
            # 先将鼠标位置映射到源图像坐标
            src_x = view_x + x / scale
            src_y = view_y + y / scale

            old_scale = scale
            if flags > 0:
                scale = min(max_scale, scale * 1.2)
            elif flags < 0:
                scale = max(min_scale, scale / 1.2)
            else:
                return

            if abs(scale - old_scale) < 0.001:
                return

            # 缩放后保持鼠标下的源图像点不动
            view_x = src_x - x / scale
            view_y = src_y - y / scale
            _safe_render()

        elif event == cv2.EVENT_LBUTTONDOWN:
            dragging = True
            drag_start = (x, y)
            drag_view_start = (view_x, view_y)

        elif event == cv2.EVENT_MOUSEMOVE and dragging:
            dx = x - drag_start[0]
            dy = y - drag_start[1]
            # canvas 像素增量 → 源图像坐标增量
            view_x = drag_view_start[0] - dx / scale
            view_y = drag_view_start[1] - dy / scale
            _safe_render()

        elif event == cv2.EVENT_LBUTTONUP:
            dragging = False

    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(win_name, canvas_w, canvas_h)
    cv2.setMouseCallback(win_name, on_mouse)

    print("\n  [交互查看器] 滚轮缩放 | 拖拽平移 | R=重置 | Q=关闭")
    _safe_render()

    while True:
        key = cv2.waitKey(30) & 0xFF

        if key in (ord('q'), ord('Q'), 27):
            break
        elif key in (ord('r'), ord('R')):
            scale = 1.0
            view_x, view_y = 0.0, 0.0
            _safe_render()
        elif key == ord('+') or key == ord('='):
            scale = min(max_scale, scale * 1.2)
            # 居中缩放
            view_x = (view_x + canvas_w / scale / 2) - canvas_w / (2 * scale)
            view_y = (view_y + canvas_h / scale / 2) - canvas_h / (2 * scale)
            _safe_render()
        elif key == ord('-'):
            scale = max(min_scale, scale / 1.2)
            _safe_render()
        # 键盘平移（WASD）
        elif key in (ord('w'), ord('W')):
            view_y -= 40 / scale
            _safe_render()
        elif key in (ord('s'), ord('S')):
            view_y += 40 / scale
            _safe_render()
        elif key in (ord('a'), ord('A')):
            view_x -= 40 / scale
            _safe_render()
        elif key in (ord('d'), ord('D')):
            view_x += 40 / scale
            _safe_render()

        # 检查窗口关闭
        if cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.setMouseCallback(win_name, lambda *args: None)
    print("  Viewer closed.")


def draw_acupoints_on_image(image: np.ndarray, acu, pose_result,
                            body_orientation: str = None) -> np.ndarray:
    """在图片上绘制穴位标注和骨架"""
    h, w = image.shape[:2]
    overlay = image.copy()

    # ── 1. 绘制姿态骨架 ──
    if pose_result.has_pose and pose_result.pose_landmarks is not None:
        pts = pose_result.pose_landmarks[:, :2] * np.array([w, h])
        pts_i = pts.astype(int)
        for (a, b) in POSE_CONNECTIONS:
            if a < len(pts_i) and b < len(pts_i):
                pa, pb = pts_i[a], pts_i[b]
                p_vis_a = pose_result.pose_landmarks[a, 3] if pose_result.pose_landmarks.shape[1] > 3 else 1.0
                p_vis_b = pose_result.pose_landmarks[b, 3] if pose_result.pose_landmarks.shape[1] > 3 else 1.0
                if p_vis_a > 0.5 and p_vis_b > 0.5:
                    cv2.line(overlay, tuple(pa), tuple(pb), (128, 128, 128), 2, cv2.LINE_AA)
        for i, (px, py) in enumerate(pts):
            vis = pose_result.pose_landmarks[i, 3] if pose_result.pose_landmarks.shape[1] > 3 else 1.0
            if vis > 0.5:
                r = 5 if i in (0, 7, 8, 11, 12, 23, 24) else 3
                cv2.circle(overlay, (int(px), int(py)), r, (200, 200, 200), -1)

    # ── 2. 绘制面部网格 ──
    if pose_result.has_face and pose_result.face_landmarks is not None:
        for i in range(17):
            j = (i + 1) % 17
            pi = (int(pose_result.face_landmarks[i][0] * w), int(pose_result.face_landmarks[i][1] * h))
            pj = (int(pose_result.face_landmarks[j][0] * w), int(pose_result.face_landmarks[j][1] * h))
            cv2.line(overlay, pi, pj, (255, 180, 100), 1, cv2.LINE_AA)

    # ── 3. 绘制手部关键点 ──
    for hand_side, hand_lms, color in [
        ("right", pose_result.right_hand_landmarks, (255, 150, 50)),
        ("left", pose_result.left_hand_landmarks, (100, 255, 100)),
    ]:
        if hand_lms is not None:
            pts = [(int(lm[0] * w), int(lm[1] * h)) for lm in hand_lms]
            for (a, b) in HAND_CONNECTIONS:
                if a < len(pts) and b < len(pts):
                    cv2.line(overlay, pts[a], pts[b], color, 2, cv2.LINE_AA)
            for i, p in enumerate(pts):
                r = 5 if i in (4, 8, 12, 16, 20) else 3
                cv2.circle(overlay, p, r, color, -1)

    # ── 3.5. 躯干朝向指示（统一算法：正面=乳头+肚脐, 背面=脊柱标记） ──
    if pose_result.has_pose and pose_result.pose_landmarks is not None:
        if body_orientation is None:
            body_orientation = _detect_body_orientation(
                pose_result.pose_landmarks, pose_result.has_face)
        draw_body_indicators(overlay, pose_result.pose_landmarks,
                            orientation=body_orientation,
                            has_face=pose_result.has_face)

    # ── 4. 绘制穴位标注 ──
    grade_colors = {"A": (0, 255, 0), "B": (0, 255, 255), "C": (0, 165, 255), "D": (0, 0, 255)}
    for ap in acu.acupoints:
        if ap.position_2d is not None:
            px, py = int(ap.position_2d[0]), int(ap.position_2d[1])
            if 0 <= px < w and 0 <= py < h:
                color = MERIDIAN_BGR.get(ap.meridian_code, (0, 0, 255))
                radius = 6 if ap.grade == "A" else 5 if ap.grade == "B" else 4
                # 外圈
                cv2.circle(overlay, (px, py), radius + 1, (40, 40, 40), 1)
                cv2.circle(overlay, (px, py), radius, color, -1)
                cv2.circle(overlay, (px, py), radius, (255, 255, 255), 1)
                # 中文标签（使用PIL渲染）
                label = f"{ap.name_cn}"
                _put_chinese_text(overlay, label,
                                  (px - len(label) * 3, py - radius - 10),
                                  color, font_size=14)

    # ── 5. 右侧图例面板 ──
    legend_x = w - 185
    legend_y = 10
    overlay_bg = overlay.copy()
    cv2.rectangle(overlay_bg, (legend_x - 10, legend_y),
                 (w - 5, legend_y + 30 + len(acu.acupoints) * 22), (0, 0, 0), -1)
    cv2.addWeighted(overlay_bg, 0.55, overlay, 0.45, 0, overlay)

    cv2.putText(overlay, f"Acupoints: {acu.total_found}", (legend_x, legend_y + 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    for i, ap in enumerate(acu.acupoints[:25]):
        y = legend_y + 42 + i * 20
        color = MERIDIAN_BGR.get(ap.meridian_code, (255, 255, 255))
        cv2.circle(overlay, (legend_x + 3, y - 3), 4, color, -1)
        _put_chinese_text(overlay, f"{ap.name_cn} [{ap.grade}]",
                          (legend_x + 12, y - 14),
                          (220, 220, 220), font_size=14)

    # ── 6. 底部统计栏 ──
    stats_text = f"Total: {acu.total_found}  |  A:{acu.grade_counts.get('A',0)}  B:{acu.grade_counts.get('B',0)}  C:{acu.grade_counts.get('C',0)}  D:{acu.grade_counts.get('D',0)}"
    cv2.rectangle(overlay, (0, h - 30), (w, h), (0, 0, 0), -1)
    cv2.putText(overlay, stats_text, (10, h - 8),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

    return overlay


# ── 穴位定位管线 ──────────────────────────────────────────

def run_detection_pipeline(image, profile: Optional[PopulationProfile] = None):
    """运行完整的穴位定位管线，返回成功与否"""
    h, w = image.shape[:2]
    print(f"  Image size: {w}x{h}")

    # ── 人群画像信息 ──
    if profile is not None:
        adapter = PopulationAdapter()
        coeffs = adapter.get_coefficients("full_body", profile)
        print(f"  [人群画像] {_format_profile_hud(profile)}")
        print(f"             ratio×{coeffs.ratio_modifier:.2f} offset×{coeffs.offset_modifier:.2f}")


    # 1. Pose detection
    print("\n[Step 1] MediaPipe Pose detection...")
    t_start = time.time()
    extractor = PoseExtractor(mode=ModelMode.HOLISTIC)
    pose_result = extractor.process(image)
    t_infer = time.time() - t_start
    print(f"  Inference: {t_infer*1000:.0f}ms")

    if not pose_result.has_pose:
        print("\n  [!] No human body detected!")
        print("  Tips: stand upright, arms at sides, 1.5-3m from camera, good lighting")
        extractor.close()
        return False

    print(f"  [OK] Body skeleton detected (33 keypoints)")

    # 2. Virtual spine
    print("\n[Step 2] Virtual spine estimation...")
    if pose_result.pose_world_landmarks is None:
        print("[!] Missing world coordinates")
        extractor.close()
        return False

    spine_estimator = SpineEstimator()
    spine = spine_estimator.build(pose_result.pose_world_landmarks)

    if not spine.is_valid:
        print("[!] Spine estimation failed (unreliable keypoints)")
        extractor.close()
        return False

    print(f"  [OK] Spine length: {spine.spine_length*100:.1f}cm")
    print(f"  Shoulder width: {spine.shoulder_width*100:.1f}cm")
    print(f"  Estimated height: {spine.estimated_height*100:.0f}cm")

    # 3. Warm-up
    print("\n  [Benchmark] 5-frame warm inference...")
    times = []
    for _ in range(5):
        t0 = time.time()
        _ = extractor.process(image)
        times.append((time.time() - t0) * 1000)
    avg = sum(times) / len(times)
    print(f"  Avg: {avg:.0f}ms | FPS: {1000/avg:.1f}")

    # 4. Acupoint localization
    print("\n[Step 3] Acupoint localization...")
    locator = AcupointLocator(profile=profile)
    locator.load_database([
        "database/acupoints_torso.json",
        "database/acupoints_limbs.json",
        "database/acupoints_hands.json",
        "database/acupoints_face.json",
    ])
    print(f"  Loaded {locator.get_acupoint_count()} acupoint definitions")

    acu = locator.locate(pose_result)
    body_ori = locator._cached_body_orientation
    print(f"  [OK] Located {acu.total_found} acupoints")
    print(f"  Body orientation: {body_ori}")
    print(f"  Grade distribution: A:{acu.grade_counts.get('A',0)} B:{acu.grade_counts.get('B',0)} "
          f"C:{acu.grade_counts.get('C',0)} D:{acu.grade_counts.get('D',0)}")

    print("\n  Top acupoints:")
    for ap in acu.acupoints[:15]:
        icon = {"A":"[A]","B":"[B]","C":"[C]","D":"[D]"}.get(ap.grade,"?")
        print(f"    {icon} {ap.name_cn}({ap.id}) [{ap.meridian}] conf={ap.confidence:.0%}")

    # 5. Draw acupoints on image
    print("\n[Step 4] Drawing acupoint overlay on image...")
    annotated = draw_acupoints_on_image(image, acu, pose_result,
                                        body_orientation=body_ori)

    os.makedirs("output/annotations", exist_ok=True)
    output_path = "output/annotations/acupoints_result.jpg"
    cv2.imwrite(output_path, annotated)

    print(f"\n{'='*60}")
    print(f"  Verification successful!")
    print(f"  Annotated image: {os.path.abspath(output_path)}")
    print(f"{'='*60}")

    # Show result window — 可缩放查看细节
    _show_zoomable_result(annotated)
    # 保持窗口，由外层调用者管理

    extractor.close()
    return True


# ── 面部+手部实时穴位检测 ──────────────────────────────────

def live_face_hands_acupoints(camera_id: int,
                              profile: Optional[PopulationProfile] = None):
    """
    实时面部+手部穴位检测模式。

    仅使用 Holistic 模型的面部网格和手部关键点数据，
    不依赖姿态/脊柱，直接在视频画面上标注面部和手部穴位。

    优点：
    - 不需要全身入镜，只需面部+手在摄像头前
    - 2D坐标直接从 Holistic landmarks 计算，精度高
    - 实时帧率 ≈ 15-25 FPS

    按键：
      Q = 退出
      S = 截图保存
      G = 切换精度过滤
    """
    from mediapipe.tasks.python import vision, BaseOptions
    from mediapipe.tasks.python.vision import RunningMode
    from mediapipe import Image as MPImage, ImageFormat

    model_path = os.path.join("models", "holistic_landmarker.task")
    if not os.path.exists(model_path):
        print("\n[!] Holistic model not found!")
        print("  Please place holistic_landmarker.task in models/ directory")
        return

    # ── 加载面部+手部穴位数据库（通过 AcupointLocator，确保与其他模式使用同一套穴位定义） ──
    locator = AcupointLocator()
    locator.load_database([
        "database/acupoints_face.json",
        "database/acupoints_hands.json",
    ])

    # 将穴位定义按方法分类（face_mesh / hand_landmark）
    face_defs = {}   # id -> ap_def
    hand_defs = {}   # id -> ap_def
    for ap_id, ap_def in locator._acupoint_defs.items():
        rule = ap_def.get("location_rule", {})
        method = rule.get("method", "")
        if method == "face_mesh":
            face_defs[ap_id] = ap_def
        elif method == "hand_landmark":
            hand_defs[ap_id] = ap_def

    print(f"\n  Face acupoint definitions: {len(face_defs)}")
    print(f"  Hand acupoint definitions (incl. left mirror): {len(hand_defs)}")

    # ── 打开摄像头 ──
    cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    if not cap.isOpened():
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print(f"[!] Cannot open camera #{camera_id}")
            return

    # ── 创建 Holistic Landmarker ──
    base = BaseOptions(model_asset_path=model_path)
    opts = vision.HolisticLandmarkerOptions(
        base_options=base,
        running_mode=RunningMode.VIDEO,
        min_face_detection_confidence=0.5,
        min_face_landmarks_confidence=0.5,
        min_pose_detection_confidence=0.3,
        min_pose_landmarks_confidence=0.3,
        min_hand_landmarks_confidence=0.5,
        output_face_blendshapes=False,
        output_segmentation_mask=False,
    )
    landmarker = vision.HolisticLandmarker.create_from_options(opts)

    # ── 状态变量 ──
    grade_filter = "D"
    grade_order = {"A": 0, "B": 1, "C": 2, "D": 3}

    print("\n  [Face+Hands Acupoint Mode]")
    print("  Controls: [Q] Quit  [S] Screenshot  [G] Grade Filter\n")
    print("  Place your face and hands in front of the camera")

    # ── 计算人群适配系数 ──
    _adapter = PopulationAdapter()
    coeffs = _adapter.get_coefficients("default", profile) if profile \
        else _adapter.get_coefficients("default", PopulationProfile())
    if profile:
        print(f"  [人群适配] {_format_profile_hud(profile)}")
        print(f"             ratio×{coeffs.ratio_modifier:.2f} offset×{coeffs.offset_modifier:.2f} "
              f"skin×{coeffs.skin_thickness:.2f}")

    while True:
        ret, frame_bgr = cap.read()
        if not ret or frame_bgr is None:
            continue

        frame_bgr = cv2.flip(frame_bgr, 1)
        h, w = frame_bgr.shape[:2]
        display = frame_bgr.copy()
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = MPImage(image_format=ImageFormat.SRGB, data=frame_rgb)

        t0 = time.time()
        frame_ms = int(time.time() * 1000)
        result = landmarker.detect_for_video(mp_image, frame_ms)
        latency = (time.time() - t0) * 1000

        face_ok = False
        left_ok = False
        right_ok = False
        detected_acupoints = []
        face_points = []  # (x, y, name, id, meridian_code, grade)
        hand_points = []  # (x, y, name, id, meridian_code, grade, side)

        if result is not None:
            # ── 面部检测 ──
            flms = _unwrap_landmarks(getattr(result, 'face_landmarks', None))
            if flms:
                face_ok = True

                # 绘制面部轮廓
                pts = [(int(lm.x * w), int(lm.y * h)) for lm in flms]
                for i in range(17):
                    j = (i + 1) % 17
                    cv2.line(display, pts[i], pts[j], COLOR_FACE, 1, cv2.LINE_AA)

                # 关键面部点
                for idx, r, color in [
                    (1, 5, (50, 200, 255)), (4, 4, (255, 200, 100)),
                    (159, 3, (255, 255, 255)), (386, 3, (255, 255, 255)),
                    (61, 2, (200, 150, 200)), (291, 2, (200, 150, 200)),
                ]:
                    if idx < len(pts):
                        cv2.circle(display, pts[idx], r, color, -1)

                # 计算面部穴位2D位置（使用统一的 AcupointLocator 穴位定义）
                for ap_id, ap in face_defs.items():
                    rule = ap.get("location_rule", {})
                    face_indices = rule.get("face_indices", [])
                    if not face_indices:
                        continue
                    fpts = []
                    for idx in face_indices:
                        if idx < len(flms):
                            fpts.append(np.array([flms[idx].x * w, flms[idx].y * h]))
                    if fpts:
                        avg = np.mean(fpts, axis=0)
                        grade = ap.get("validation", {}).get("grade", "B")
                        face_points.append((
                            int(avg[0]), int(avg[1]),
                            ap.get("name_cn", ""), ap["id"],
                            ap.get("meridian_code", "EX"), grade
                        ))
                        detected_acupoints.append(ap)

            # ── 手部检测 ──
            # 注意：画面经过 cv2.flip 水平镜像，MediaPipe 的解剖学左右
            # 与屏幕视觉左右相反，因此 HUD 标签需交换显示
            for hand_side, hand_lms_attr, color, side_label in [
                ("left", "left_hand_landmarks", COLOR_HAND, "Right"),
                ("right", "right_hand_landmarks", (255, 150, 50), "Left"),
            ]:
                hlms = _unwrap_landmarks(getattr(result, hand_lms_attr, None))
                if hlms:
                    if hand_side == "left":
                        left_ok = True
                    else:
                        right_ok = True
                    pts = [(int(lm.x * w), int(lm.y * h)) for lm in hlms]

                    # 绘制手部骨架
                    for (a, b) in HAND_CONNECTIONS:
                        if a < len(pts) and b < len(pts):
                            cv2.line(display, pts[a], pts[b], color, 2, cv2.LINE_AA)
                    for i, p in enumerate(pts):
                        r = 5 if i in (4, 8, 12, 16, 20) else 3
                        cv2.circle(display, p, r, (50, 255, 50) if hand_side == "left" else (255, 180, 50), -1)

                    # 计算手部穴位2D位置（使用统一的 AcupointLocator 穴位定义，含左右手镜像）
                    for ap_id, ap in hand_defs.items():
                        rule = ap.get("location_rule", {})
                        ap_side = rule.get("hand_side", "right")
                        if ap_side != hand_side:
                            continue
                        lm_idx = rule.get("hand_landmark_index")
                        if lm_idx is None or lm_idx >= len(hlms):
                            continue
                        px = int(hlms[lm_idx].x * w)
                        py = int(hlms[lm_idx].y * h)
                        grade = ap.get("validation", {}).get("grade", "B")
                        hand_points.append((
                            px, py,
                            ap.get("name_cn", ""), ap["id"],
                            ap.get("meridian_code", "EX"), grade, side_label
                        ))
                        detected_acupoints.append(ap)

        # ── 绘制穴位标记 ──
        # 面部穴位
        for (px, py, name, aid, meridian, grade) in face_points:
            if grade_order.get(grade, 9) > grade_order.get(grade_filter, 9):
                continue
            color = MERIDIAN_BGR.get(meridian, (0, 255, 255))
            radius = 12 if grade == "A" else 9 if grade == "B" else 6
            cv2.circle(display, (px, py), radius + 2, (0, 0, 0), 1)
            cv2.circle(display, (px, py), radius, color, -1)
            cv2.circle(display, (px, py), radius, (255, 255, 255), 1)
            _put_chinese_text(display, name,
                              (px - len(name) * 6 // 2, py - radius - 8),
                              color, font_size=16)

        # 手部穴位
        for (px, py, name, aid, meridian, grade, side) in hand_points:
            if grade_order.get(grade, 9) > grade_order.get(grade_filter, 9):
                continue
            color = MERIDIAN_BGR.get(meridian, (0, 255, 255))
            radius = 11 if grade == "A" else 8 if grade == "B" else 5
            cv2.circle(display, (px, py), radius + 2, (0, 0, 0), 1)
            cv2.circle(display, (px, py), radius, color, -1)
            cv2.circle(display, (px, py), radius, (255, 255, 255), 1)
            _put_chinese_text(display, f"{name}({side[0]})",
                              (px + radius + 4, py - 6),
                              color, font_size=14)

        # ── 右侧状态面板 ──
        panel_x = w - 200
        panel_y = 20
        # 注意：画面已镜像，状态面板左右也需交换
        for i, (label, ok) in enumerate([
            ("Face    ", face_ok), ("L-Hand  ", right_ok), ("R-Hand  ", left_ok),
        ]):
            y = panel_y + i * 30
            color = COLOR_OK if ok else COLOR_OFF
            cv2.putText(display, label, (panel_x, y + 22),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)
            cv2.circle(display, (panel_x - 10, y + 17), 6, color, -1 if ok else 1)

        # 穴位计数
        visible = sum(1 for pt in face_points + hand_points
                      if grade_order.get(pt[5] if len(pt) > 5 else "D", 9) <= grade_order.get(grade_filter, 9))
        cv2.putText(display, f"Acupoints: {visible}", (panel_x, panel_y + 105),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_OK if visible > 0 else COLOR_OFF, 1, cv2.LINE_AA)

        # ── 用户画像 HUD（右侧面板下方） ──
        if profile is not None:
            profile_texts = [
                f"{'男' if profile.gender == Gender.MALE else '女'} {profile.age}岁",
                f"{profile.height_cm:.0f}cm {profile.weight_kg:.0f}kg",
                f"BMI={profile.bmi:.1f} {profile.body_type.value}",
                f"ratio×{coeffs.ratio_modifier:.2f} off×{coeffs.offset_modifier:.2f}",
            ]
            for j, text in enumerate(profile_texts):
                y = panel_y + 140 + j * 20
                cv2.putText(display, text, (panel_x, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 200), 1, cv2.LINE_AA)

        # ── 顶部 HUD ──
        fps = 1000 / latency if latency > 0 else 0
        cv2.putText(display, f"Face+Hands Mode | FPS:{fps:.0f} | Latency:{latency:.0f}ms | Grade:{grade_filter} | Q=Quit S=Shot G=Filter",
                   (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

        # ── 底部提示 ──
        cv2.putText(display, "Place face + hands in view for acupoint detection",
                   (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1, cv2.LINE_AA)

        cv2.imshow('perfess mediapipe test', display)

        # ── 按键处理 ──
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            break
        elif key == ord('s') or key == ord('S'):
            os.makedirs("output/screenshots", exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            fname = f"output/screenshots/face_hands_{ts}.png"
            cv2.imwrite(fname, display)
            print(f"  [Screenshot] {fname}")
        elif key == ord('g') or key == ord('G'):
            grades = ["A", "B", "C", "D"]
            try:
                idx = grades.index(grade_filter)
                grade_filter = grades[(idx + 1) % len(grades)]
            except ValueError:
                grade_filter = "D"
            print(f"  Grade filter: {grade_filter}")

    landmarker.close()
    cap.release()
    cv2.destroyAllWindows()
    print(f"\n  Face+Hands mode ended.\n")


# ── 图片模式引导界面 ───────────────────────────────────────

def _make_guide_image(width=800, height=600):
    """生成图片模式的引导画面"""
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    # 背景渐变（深蓝灰）
    for i in range(height):
        ratio = i / height
        b = int(35 + ratio * 20)
        g = int(35 + ratio * 15)
        r = int(50 + ratio * 15)
        canvas[i, :] = (b, g, r)

    # 顶部装饰线
    cv2.rectangle(canvas, (60, 80), (width - 60, 84), (0, 180, 200), -1)

    # 标题
    _put_chinese_text(canvas, "穴位检测 - 图片模式",
                      (width // 2 - 140, 100),
                      (0, 220, 255), font_size=32)

    # 中央提示框
    box_x, box_y = 100, 180
    box_w, box_h = width - 200, height - 280
    overlay = canvas.copy()
    cv2.rectangle(overlay, (box_x, box_y), (box_x + box_w, box_y + box_h), (30, 40, 50), -1)
    cv2.addWeighted(overlay, 0.7, canvas, 0.3, 0, canvas)
    cv2.rectangle(canvas, (box_x, box_y), (box_x + box_w, box_y + box_h), (60, 140, 160), 1)

    # 操作说明
    instructions = [
        ("操作指南", 24, (0, 220, 255)),
        ("", 16, (0, 0, 0)),
        ("[D] — 打开文件对话框，选择图片进行分析", 18, (200, 200, 200)),
        ("[Q] — 退出程序", 18, (200, 200, 200)),
        ("", 16, (0, 0, 0)),
        ("提示", 22, (0, 200, 150)),
        ("图片中人物需正面站立，全身或大半身入镜", 16, (170, 170, 170)),
        ("光照充足、背景简洁时检测效果最佳", 16, (170, 170, 170)),
        ("支持的格式: JPG, PNG, BMP, WebP", 16, (170, 170, 170)),
    ]

    y = box_y + 40
    for text, size, color in instructions:
        if text == "":
            y += 10
            continue
        _put_chinese_text(canvas, text, (box_x + 30, y), color, font_size=size)
        y += size + 10

    # 底部状态栏
    cv2.rectangle(canvas, (0, height - 35), (width, height), (20, 30, 40), -1)
    _put_chinese_text(canvas, "就绪 - 等待操作...",
                      (15, height - 30), (0, 180, 200), font_size=16)

    return canvas


# ── 主流程 ────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  MediaPipe Acupoint Detection - Verification")
    print("=" * 60)
    print(f"\n  {PoseExtractor.get_gpu_status()}\n")

    # ── 采集用户身体参数（用于人群加权适配） ──
    profile = _prompt_user_profile()
    adapter = PopulationAdapter()
    coeffs = adapter.get_coefficients("default", profile)
    print(f"\n  [人群适配] ratio×{coeffs.ratio_modifier:.2f}  "
          f"offset×{coeffs.offset_modifier:.2f}  "
          f"skin×{coeffs.skin_thickness:.2f}")

    # ── 选择检测来源和质量 ──
    image_path = "input/images/test.jpg"
    has_existing = os.path.exists(image_path)

    if has_existing:
        print("=" * 50)
        print("  Choose detection mode:")
        print("    [1] Use existing image: input/images/test.jpg (full body)")
        print("    [2] Use camera - live capture (full body)")
        print("    [3] Face + Hands only (real-time)")
        print("=" * 50)
        while True:
            choice = input("  Enter 1, 2 or 3: ").strip()
            if choice == "1":
                use_image = True
                face_hands_only = False
                break
            elif choice == "2":
                use_image = False
                face_hands_only = False
                break
            elif choice == "3":
                use_image = False
                face_hands_only = True
                break
            else:
                print("  [!] Invalid input, please enter 1, 2 or 3")
    else:
        print("  No existing image found.")
        print("=" * 50)
        print("  Choose detection mode:")
        print("    [1] Use camera - live capture (full body)")
        print("    [2] Face + Hands only (real-time)")
        print("=" * 50)
        while True:
            choice = input("  Enter 1 or 2: ").strip()
            if choice == "1":
                use_image = False
                face_hands_only = False
                break
            elif choice == "2":
                use_image = False
                face_hands_only = True
                break
            else:
                print("  [!] Invalid input, please enter 1 or 2")

    # ── 面部+手部实时模式 ──
    if face_hands_only:
        camera_id = select_camera()
        if camera_id is None:
            print("No camera selected. Exiting.")
            return
        live_face_hands_acupoints(camera_id, profile=profile)
        return

    # ── 图片模式（循环选择） ──
    if use_image:
        # 先处理默认图片
        image = _imread_quiet(image_path)
        if image is not None:
            run_detection_pipeline(image, profile=profile)

        print("\n" + "=" * 60)
        print("  [图片模式] 选择图片即可自动解析穴位")
        print("  操作: [D] 选择文件 | [Q] 退出")
        print("=" * 60)

        # 显示引导界面
        guide = _make_guide_image()
        cv2.imshow('perfess mediapipe test', guide)

        while True:
            key = cv2.waitKey(100) & 0xFF
            if key == ord('q') or key == ord('Q'):
                break
            elif key == ord('d') or key == ord('D'):
                # 临时创建 Tk 打开文件对话框，用完立即销毁
                from tkinter import Tk, filedialog
                root = Tk()
                try:
                    root.withdraw()
                    root.attributes('-topmost', True)
                    root.update()
                    file_paths = filedialog.askopenfilenames(
                        title="选择图片文件",
                        filetypes=[
                            ("图片文件", "*.jpg *.jpeg *.png *.bmp *.webp"),
                            ("所有文件", "*.*")
                        ],
                        parent=root
                    )
                finally:
                    root.destroy()

                for fp in file_paths:
                    img = _imread_quiet(fp)
                    if img is not None:
                        print(f"\n  Analyzing: {os.path.basename(fp)}")
                        run_detection_pipeline(img, profile=profile)
                        # 处理完后重新显示引导界面
                        guide = _make_guide_image()
                        cv2.imshow('perfess mediapipe test', guide)

        cv2.destroyAllWindows()
        return

    # ── 摄像头模式 ──
    camera_id = select_camera()
    if camera_id is None:
        print("No camera selected. Exiting.")
        return

    # 主循环：实时预览 → 拍照分析 → 失败则回到预览重试
    attempt = 0
    while True:
        attempt += 1
        print(f"\n{'─'*40}")
        print(f"  Attempt #{attempt}")
        print(f"{'─'*40}")

        try:
            captured_frame = live_detection_preview(camera_id)
        except RuntimeError as e:
            print(f"\n[!] {e}")
            return

        if captured_frame is None:
            print("\n  User chose to exit. Goodbye!")
            return

        # 保存拍照帧
        os.makedirs("input/images", exist_ok=True)
        image_path = "input/images/test.jpg"
        cv2.imwrite(image_path, captured_frame)
        print(f"  [OK] Photo saved: {image_path}")

        # 运行分析管线
        success = run_detection_pipeline(captured_frame, profile=profile)

        if success:
            return
        else:
            print(f"\n  Retrying - adjust your pose and try again...")
            print(f"  Press Ctrl+C to quit anytime\n")
            time.sleep(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n  Interrupted by user. Exiting.")
