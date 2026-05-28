"""
实时摄像头穴位叠加显示

功能:
- 打开摄像头实时捕获视频
- 逐帧运行 MediaPipe 检测
- 实时叠加骨架和穴位标注
- 显示FPS和穴位计数
- 按键控制：Q=退出, S=截图, G=切换精度过滤
"""

import cv2
import numpy as np
import time
from typing import Optional, Dict
from datetime import datetime

from core.pose_extractor import PoseExtractor, PoseResult, ModelMode
from core.acupoint_locator import AcupointLocator, AcupointResult


# 经络颜色 (BGR格式用于OpenCV)
BGR_COLORS = {
    "CV": (53, 107, 255),    # 任脉
    "GV": (0, 215, 255),     # 督脉
    "LU": (232, 232, 232),   # 肺经
    "LI": (176, 176, 176),   # 大肠经
    "ST": (0, 140, 255),     # 胃经
    "SP": (204, 50, 153),    # 脾经
    "BL": (225, 105, 65),    # 膀胱经
    "KI": (209, 206, 0),     # 肾经
    "PC": (60, 20, 220),     # 心包经
    "TE": (255, 144, 30),    # 三焦经
    "GB": (34, 139, 34),     # 胆经
    "LR": (34, 205, 50),     # 肝经
}


class RealtimeOverlay:
    """
    实时摄像头穴位叠加

    用法:
        overlay = RealtimeOverlay()
        overlay.run()
    """

    # MediaPipe Pose骨架连线
    POSE_CONNECTIONS = [
        (11, 12), (11, 23), (12, 24), (23, 24),
        (11, 13), (13, 15), (12, 14), (14, 16),
        (23, 25), (25, 27), (24, 26), (26, 28),
        (27, 29), (29, 31), (27, 31),
        (28, 30), (30, 32), (28, 32),
    ]

    def __init__(self, camera_id: int = 0, width: int = 1280, height: int = 720,
                 fps: int = 30, mode: ModelMode = ModelMode.POSE,
                 model_complexity: int = 2):
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.target_fps = fps
        self.mode = mode
        self.model_complexity = model_complexity

        # 状态
        self._running = False
        self._current_grade_filter = "D"
        self._grade_order = {"A": 0, "B": 1, "C": 2, "D": 3, "ALL": 9}

        # 性能统计
        self._fps_history = []
        self._frame_count = 0

        self.extractor: Optional[PoseExtractor] = None
        self.locator: Optional[AcupointLocator] = None

    def setup(self, json_paths: list = None):
        """初始化模型和穴位数据库"""
        if json_paths is None:
            json_paths = [
                "database/acupoints_torso.json",
                "database/acupoints_limbs.json",
                "database/acupoints_hands.json",
                "database/acupoints_face.json",
            ]

        print(f"[RealtimeOverlay] 加载模式: {self.mode.value}")
        self.extractor = PoseExtractor(
            mode=self.mode,
            model_complexity=self.model_complexity,
            static_image_mode=False,
        )
        print(f"[RealtimeOverlay] {PoseExtractor.get_gpu_status()}")

        self.locator = AcupointLocator()
        self.locator.load_database(json_paths)
        print(f"[RealtimeOverlay] 加载穴位总数: {self.locator.get_acupoint_count()}")

    def run(self):
        """主循环"""
        if self.extractor is None or self.locator is None:
            self.setup()

        # 打开摄像头
        cap = cv2.VideoCapture(self.camera_id, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.target_fps)

        if not cap.isOpened():
            print("[RealtimeOverlay] 错误: 无法打开摄像头!")
            return

        self._running = True
        print("[RealtimeOverlay] 开始实时检测...")
        print("  按键: Q=退出 | S=截图 | G=切换精度 | 1-4=选择精度(A/B/C/D)")

        last_result: Optional[PoseResult] = None
        last_acupoints: Optional[AcupointResult] = None

        while self._running:
            ret, frame = cap.read()
            if not ret:
                break

            # 水平翻转（镜像）
            frame = cv2.flip(frame, 1)

            t_start = time.time()

            # MediaPipe 推理
            pose_result = self.extractor.process(frame)

            # 穴位定位
            acupoint_result = None
            if pose_result.has_pose:
                acupoint_result = self.locator.locate(pose_result)

            t_inference = time.time() - t_start

            # 绘制叠加层
            display = self._draw_overlay(frame, pose_result, acupoint_result, t_inference)

            # 显示
            cv2.imshow('perfess mediapipe test', display)

            # 按键处理
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                self._running = False
            elif key == ord('s') or key == ord('S'):
                self._save_screenshot(display)
            elif key == ord('g') or key == ord('G'):
                self._cycle_grade_filter()
            elif key == ord('1'): self._current_grade_filter = "A"
            elif key == ord('2'): self._current_grade_filter = "B"
            elif key == ord('3'): self._current_grade_filter = "C"
            elif key == ord('4'): self._current_grade_filter = "D"

            # 更新
            self._frame_count += 1

        cap.release()
        cv2.destroyAllWindows()
        print(f"[RealtimeOverlay] 已停止. 总帧数: {self._frame_count}")

    def _draw_overlay(self, frame: np.ndarray, pose_result: PoseResult,
                      acupoint_result: Optional[AcupointResult],
                      inference_time: float) -> np.ndarray:
        """绘制骨架+穴位叠加"""
        h, w = frame.shape[:2]
        overlay = frame.copy()

        if pose_result.has_pose and pose_result.pose_landmarks is not None:
            pts = pose_result.pose_landmarks[:, :2] * np.array([w, h])

            # 绘制骨架连线
            for conn in self.POSE_CONNECTIONS:
                i, j = conn
                pi = pts[i].astype(int)
                pj = pts[j].astype(int)
                cv2.line(overlay, tuple(pi), tuple(pj), (128, 128, 128), 2)

            # 绘制骨架关键点
            for i, (px, py, vis) in enumerate(pose_result.pose_landmarks):
                if vis > 0.5:
                    pi = (int(px * w), int(py * h))
                    cv2.circle(overlay, pi, 5, (200, 200, 200), -1)

        # 绘制穴位
        if acupoint_result and acupoint_result.acupoints:
            visible_count = 0
            for ap in acupoint_result.acupoints:
                # 精度过滤
                if self._grade_order.get(ap.grade, 9) > self._grade_order.get(self._current_grade_filter, 9):
                    continue

                if ap.position_2d is not None:
                    px, py = int(ap.position_2d[0]), int(ap.position_2d[1])
                    # 边界检查
                    if 0 <= px < w and 0 <= py < h:
                        color = BGR_COLORS.get(ap.meridian_code, (0, 0, 255))
                        radius = 12 if ap.grade == "A" else 8 if ap.grade == "B" else 5
                        cv2.circle(overlay, (px, py), radius, color, -1)
                        cv2.circle(overlay, (px, py), radius, (255, 255, 255), 1)
                        cv2.putText(overlay, ap.name_cn, (px + 8, py - 8),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
                        visible_count += 1

            # 穴位计数
            grade_info = acupoint_result.grade_counts
            info_text = f"穴位: {visible_count} | A:{grade_info.get('A',0)} B:{grade_info.get('B',0)} C:{grade_info.get('C',0)} D:{grade_info.get('D',0)}"
            cv2.putText(overlay, info_text, (10, h - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # 顶部HUD
        fps = 1.0 / inference_time if inference_time > 0 else 0
        gpu_label = "[GPU]" if getattr(self.extractor, '_use_gpu', True) else "[CPU]"
        hud_y = 25
        cv2.putText(overlay, f"FPS: {fps:.0f} | 延迟: {inference_time*1000:.0f}ms | {gpu_label} {self.mode.value} | 精度过滤: {self._current_grade_filter}级",
                   (10, hud_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

        # 脊柱信息
        if acupoint_result and acupoint_result.spine and acupoint_result.spine.is_valid:
            spine = acupoint_result.spine
            hud_y2 = hud_y + 20
            cv2.putText(overlay, f"脊柱: {spine.spine_length*100:.0f}cm | 肩宽: {spine.shoulder_width*100:.0f}cm | 身高~{spine.estimated_height*100:.0f}cm",
                       (10, hud_y2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

        return overlay

    def _cycle_grade_filter(self):
        """循环切换精度过滤"""
        grades = ["A", "B", "C", "D", "ALL"]
        try:
            idx = grades.index(self._current_grade_filter)
            self._current_grade_filter = grades[(idx + 1) % len(grades)]
        except ValueError:
            self._current_grade_filter = "D"
        print(f"[RealtimeOverlay] 精度过滤: {self._current_grade_filter}级")

    def _save_screenshot(self, frame: np.ndarray):
        """保存截图"""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"output/screenshots/acupoint_{ts}.png"
        import os
        os.makedirs("output/screenshots", exist_ok=True)
        cv2.imwrite(filename, frame)
        print(f"[RealtimeOverlay] 截图已保存: {filename}")

    def stop(self):
        """停止运行"""
        self._running = False


def main():
    """直接启动实时检测"""
    overlay = RealtimeOverlay(
        camera_id=0,
        width=1280, height=720,
        mode=ModelMode.POSE,  # 使用Pose模式确保实时性
        model_complexity=2,
    )
    overlay.setup([
        "database/acupoints_torso.json",
        "database/acupoints_limbs.json",
        "database/acupoints_hands.json",
        "database/acupoints_face.json",
    ])
    overlay.run()


if __name__ == "__main__":
    main()
