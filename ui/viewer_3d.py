"""
3D交互展示界面 - Plotly 实现

功能:
- 3D骨架+穴位点云可视化
- 鼠标悬停气泡显示穴位详情（名称、经络、功能主治）
- 鼠标旋转/缩放/平移
- 按经络颜色区分穴位
- 精度等级筛选
- 支持导出HTML报告
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple
import json

from core.pose_extractor import PoseResult
from core.spine_estimator import SpineResult, SpineEstimator
from core.acupoint_locator import AcupointResult, AcupointPosition


# 经络颜色方案
MERIDIAN_COLORS = {
    "CV": "#FF6B35",   # 任脉 - 橙红
    "GV": "#FFD700",   # 督脉 - 金色
    "LU": "#E8E8E8",   # 肺经 - 银白
    "LI": "#B0B0B0",   # 大肠经 - 灰
    "ST": "#FF8C00",   # 胃经 - 深橙
    "SP": "#9932CC",   # 脾经 - 紫
    "HT": "#E6194B",   # 心经 - 深红
    "SI": "#FABEBE",   # 小肠经 - 粉
    "BL": "#4169E1",   # 膀胱经 - 蓝
    "KI": "#00CED1",   # 肾经 - 青
    "PC": "#DC143C",   # 心包经 - 深粉
    "TE": "#1E90FF",   # 三焦经 - 蓝
    "GB": "#228B22",   # 胆经 - 绿
    "LR": "#32CD32",   # 肝经 - 浅绿
    "EX": "#FF4500",   # 经外奇穴 - 橙红
}

GRADE_COLORS = {
    "A": "#00FF00",  # 高精度 - 绿
    "B": "#FFD700",  # 中精度 - 金
    "C": "#FF8C00",  # 偏低 - 橙
    "D": "#FF4500",  # 低精度 - 红橙
}

GRADE_SIZES = {
    "A": 10, "B": 8, "C": 6, "D": 4,
}


class Viewer3D:
    """
    3D穴位可视化器

    用法:
        viewer = Viewer3D()
        fig = viewer.build(acupoint_result, pose_result, spine_result)
        fig.show()  # 浏览器中打开
        viewer.save_html(fig, "output.html")
    """

    # 骨架连线
    POSE_CONNECTIONS = [
        # 躯干
        (11, 12), (11, 23), (12, 24), (23, 24),
        # 面部简化
        (0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6),
        (5, 7), (6, 8), (7, 9), (8, 10),
        # 左臂
        (11, 13), (13, 15),
        # 右臂
        (12, 14), (14, 16),
        # 左腿
        (23, 25), (25, 27),
        # 右腿
        (24, 26), (26, 28),
        # 脚
        (27, 29), (29, 31), (27, 31),
        (28, 30), (30, 32), (28, 32),
    ]

    def __init__(self, width: int = 1280, height: int = 800,
                 show_spine: bool = True, show_bubbles: bool = True,
                 min_grade: str = "D"):
        self.width = width
        self.height = height
        self.show_spine = show_spine
        self.show_bubbles = show_bubbles
        self.min_grade = min_grade
        self._grade_order = {"A": 0, "B": 1, "C": 2, "D": 3}

    def build(self, acupoint_result: AcupointResult,
              pose_result: PoseResult,
              spine_result: Optional[SpineResult] = None) -> go.Figure:
        """
        构建完整的3D可视化

        Args:
            acupoint_result: 穴位定位结果
            pose_result: Pose检测结果
            spine_result: 虚拟脊柱结果(可选)

        Returns:
            Plotly Figure 对象
        """
        fig = go.Figure()

        # 1. 骨架点云
        if pose_result.pose_world_landmarks is not None:
            self._add_skeleton(fig, pose_result.pose_world_landmarks)

        # 2. 虚拟脊柱
        if self.show_spine and spine_result and spine_result.is_valid:
            self._add_spine(fig, spine_result)

        # 3. 穴位点（带气泡）
        if acupoint_result.acupoints:
            self._add_acupoints(fig, acupoint_result)

        # 4. 布局设置
        fig.update_layout(
            title={
                'text': f'<b>骨架经络穴位 3D 视图</b><br>'
                        f'<sub>已定位 {acupoint_result.total_found} 个穴位 | '
                        f'A:{acupoint_result.grade_counts.get("A",0)} '
                        f'B:{acupoint_result.grade_counts.get("B",0)} '
                        f'C:{acupoint_result.grade_counts.get("C",0)} '
                        f'D:{acupoint_result.grade_counts.get("D",0)}</sub>',
                'x': 0.5, 'xanchor': 'center'
            },
            scene=dict(
                xaxis=dict(title='X (左右)米', showgrid=True, gridcolor='rgba(128,128,128,0.2)',
                          showbackground=True, backgroundcolor='rgba(20,20,30,0.9)'),
                yaxis=dict(title='Y (上下)米', showgrid=True, gridcolor='rgba(128,128,128,0.2)',
                          showbackground=True, backgroundcolor='rgba(20,20,30,0.9)'),
                zaxis=dict(title='Z (前后)米', showgrid=True, gridcolor='rgba(128,128,128,0.2)',
                          showbackground=True, backgroundcolor='rgba(20,20,30,0.9)'),
                aspectmode='data',
                camera=dict(
                    eye=dict(x=1.5, y=1.2, z=1.5),
                    center=dict(x=0, y=-0.3, z=0)
                ),
            ),
            width=self.width,
            height=self.height,
            showlegend=True,
            legend=dict(
                x=0.01, y=0.99,
                bgcolor='rgba(30,30,40,0.8)',
                font=dict(color='white', size=10),
                title=dict(text='图例', font=dict(color='white')),
            ),
            paper_bgcolor='rgba(10,10,20,1)',
            font=dict(color='white'),
            hovermode='closest',
        )

        return fig

    def _add_skeleton(self, fig: go.Figure, world_lms: np.ndarray):
        """添加骨架点云和连线"""
        pts = world_lms[:, :3].copy()
        # Y轴翻转（MediaPipe Y朝下 → 3D直觉Y朝上）
        pts[:, 1] = -pts[:, 1]

        # 骨架点
        hover_texts = []
        for i in range(len(pts)):
            vis = world_lms[i, 3] if world_lms.shape[1] > 3 else 1.0
            name = self._get_landmark_name(i)
            hover_texts.append(f"{name}[{i}] 可见度:{vis:.2f}")

        fig.add_trace(go.Scatter3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
            mode='markers',
            marker=dict(size=4, color='#A0A0A0', opacity=0.8),
            text=hover_texts,
            hoverinfo='text',
            name='骨骼关键点(33)',
            showlegend=True,
        ))

        # 骨架连线
        for conn in self.POSE_CONNECTIONS:
            i, j = conn
            fig.add_trace(go.Scatter3d(
                x=[pts[i, 0], pts[j, 0]],
                y=[pts[i, 1], pts[j, 1]],
                z=[pts[i, 2], pts[j, 2]],
                mode='lines',
                line=dict(color='rgba(160,160,160,0.5)', width=2),
                hoverinfo='none',
                showlegend=False,
            ))

    def _add_spine(self, fig: go.Figure, spine: SpineResult):
        """添加虚拟脊柱路径"""
        # 后正中线（督脉路径）
        if spine.back_midline.shape[0] > 1:
            pts = spine.back_midline.copy()
            pts[:, 1] = -pts[:, 1]
            fig.add_trace(go.Scatter3d(
                x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                mode='lines',
                line=dict(color='rgba(255,215,0,0.4)', width=3, dash='dash'),
                name='虚拟脊柱(督脉路径)',
                hoverinfo='name',
            ))

        # 前正中线（任脉路径）
        if spine.front_midline.shape[0] > 1:
            pts = spine.front_midline.copy()
            pts[:, 1] = -pts[:, 1]
            fig.add_trace(go.Scatter3d(
                x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                mode='lines',
                line=dict(color='rgba(255,107,53,0.4)', width=3, dash='dash'),
                name='虚拟前正中线(任脉路径)',
                hoverinfo='name',
            ))

        # 椎骨关键点
        key_vertebrae = ["C7", "T3", "T7", "L2", "L4"]
        v_pts = []
        v_labels = []
        for v in key_vertebrae:
            if v in spine.vertebrae:
                pt = spine.vertebrae[v].copy()
                pt[1] = -pt[1]
                v_pts.append(pt)
                label = SpineEstimator.vertebra_to_acupoint_name(v) or v
                v_labels.append(label)

        if v_pts:
            v_pts = np.array(v_pts)
            fig.add_trace(go.Scatter3d(
                x=v_pts[:, 0], y=v_pts[:, 1], z=v_pts[:, 2],
                mode='markers+text',
                marker=dict(size=5, color='#FFD700', symbol='diamond'),
                text=v_labels,
                textposition='top center',
                textfont=dict(size=10, color='#FFD700'),
                hoverinfo='text',
                name='椎骨关键参考点',
            ))

    def _add_acupoints(self, fig: go.Figure, result: AcupointResult):
        """添加穴位点（按经络分组，带气泡信息）"""
        # 筛选精度等级
        filtered = [ap for ap in result.acupoints
                    if self._grade_order.get(ap.grade, 9) <= self._grade_order.get(self.min_grade, 9)]

        # 按经络分组
        meridian_groups: Dict[str, List[AcupointPosition]] = {}
        for ap in filtered:
            mc = ap.meridian_code or "OTHER"
            if mc not in meridian_groups:
                meridian_groups[mc] = []
            meridian_groups[mc].append(ap)

        # 已添加的经络图例标记
        added_meridians = set()

        for mc, aps in meridian_groups.items():
            xs, ys, zs = [], [], []
            hover_texts = []
            sizes = []

            for ap in aps:
                xs.append(ap.position_3d[0])
                ys.append(-ap.position_3d[1])  # Y翻转
                zs.append(ap.position_3d[2])
                sizes.append(GRADE_SIZES.get(ap.grade, 6))

                # 构建悬停气泡内容
                bubble = self._build_bubble(ap)
                hover_texts.append(bubble)

            color = MERIDIAN_COLORS.get(mc, '#FF0000')
            meridian_name = self._get_meridian_name(mc)
            show_legend = mc not in added_meridians
            added_meridians.add(mc)

            fig.add_trace(go.Scatter3d(
                x=xs, y=ys, z=zs,
                mode='markers+text' if len(xs) <= 20 else 'markers',
                marker=dict(
                    size=sizes,
                    color=color,
                    opacity=0.9,
                    line=dict(width=1, color='white'),
                    symbol='circle',
                ),
                text=[ap.name_cn for ap in aps] if len(xs) <= 20 else None,
                textposition='top center',
                textfont=dict(size=9, color=color),
                hovertext=hover_texts,
                hoverinfo='text',
                hoverlabel=dict(
                    bgcolor='rgba(20,20,30,0.95)',
                    font=dict(size=12, color='white'),
                    bordercolor=color,
                    align='left',
                ),
                name=f'{meridian_name}({len(aps)}穴)',
                showlegend=show_legend,
            ))

    def _build_bubble(self, ap: AcupointPosition) -> str:
        """构建穴位悬停气泡HTML内容"""
        parts = [
            f"<b style='font-size:16px;'>{ap.name_cn}({ap.id})</b>",
            f"<span style='color:#AAA;'>{ap.name_pinyin}</span>" if ap.name_pinyin else "",
            f"<br><b>经络:</b> {ap.meridian}",
            f"<b>精度:</b> <span style='color:{GRADE_COLORS.get(ap.grade,'#FFF')};'>{ap.grade}级</span>",
            f"<b>置信度:</b> {ap.confidence:.0%}",
        ]

        if ap.functions:
            parts.append(f"<br><b>功能:</b> {'、'.join(ap.functions)}")
        if ap.indications:
            parts.append(f"<b>主治:</b> {'、'.join(ap.indications[:5])}{'...' if len(ap.indications)>5 else ''}")
        if ap.depth:
            parts.append(f"<b>针刺深度:</b> {ap.depth}cm")
        if ap.description:
            parts.append(f"<br><span style='color:#888;font-style:italic;'>{ap.description}</span>")

        return "<br>".join(parts) if self.show_bubbles else ap.name_cn

    @staticmethod
    def _get_landmark_name(idx: int) -> str:
        names = {
            0:"鼻尖", 11:"左肩", 12:"右肩", 13:"左肘", 14:"右肘",
            15:"左腕", 16:"右腕", 23:"左髋", 24:"右髋",
            25:"左膝", 26:"右膝", 27:"左踝", 28:"右踝",
            29:"左脚跟", 30:"右脚跟", 31:"左脚趾", 32:"右脚趾",
        }
        return names.get(idx, f"P{idx}")

    @staticmethod
    def _get_meridian_name(code: str) -> str:
        names = {
            "CV":"任脉", "GV":"督脉", "LU":"肺经", "LI":"大肠经",
            "ST":"胃经", "SP":"脾经", "HT":"心经", "SI":"小肠经",
            "BL":"膀胱经", "KI":"肾经", "PC":"心包经", "TE":"三焦经",
            "GB":"胆经", "LR":"肝经", "EX":"经外奇穴",
        }
        return names.get(code, code)

    def save_html(self, fig: go.Figure, filepath: str):
        """保存为独立HTML文件"""
        fig.write_html(filepath, include_plotlyjs='cdn')
        print(f"[Viewer3D] 已保存: {filepath}")

    def show_2d_overlay_data(self, acupoint_result: AcupointResult,
                             pose_result: PoseResult) -> go.Figure:
        """
        生成2D叠加数据（用于图像标注视图）
        """
        fig = go.Figure()

        # 骨架2D点
        if pose_result.pose_landmarks is not None:
            w, h = pose_result.image_shape
            pts = pose_result.pose_landmarks[:, :2] * np.array([w, h])

            fig.add_trace(go.Scatter(
                x=pts[:, 0], y=pts[:, 1],
                mode='markers',
                marker=dict(size=6, color='gray'),
                name='骨骼关键点',
            ))

            # 连线
            for conn in self.POSE_CONNECTIONS:
                i, j = conn
                fig.add_trace(go.Scatter(
                    x=[pts[i, 0], pts[j, 0]],
                    y=[pts[i, 1], pts[j, 1]],
                    mode='lines',
                    line=dict(color='rgba(128,128,128,0.4)', width=2),
                    showlegend=False,
                ))

        # 穴位2D点
        filtered = [ap for ap in acupoint_result.acupoints
                    if self._grade_order.get(ap.grade, 9) <= self._grade_order.get(self.min_grade, 9)]

        for ap in filtered:
            if ap.position_2d is not None:
                color = MERIDIAN_COLORS.get(ap.meridian_code, '#FF0000')
                bubble = self._build_bubble(ap)
                fig.add_trace(go.Scatter(
                    x=[ap.position_2d[0]], y=[ap.position_2d[1]],
                    mode='markers+text',
                    marker=dict(size=10, color=color, line=dict(width=1, color='white')),
                    text=[ap.name_cn],
                    textposition='top center',
                    textfont=dict(size=10, color=color),
                    hovertext=[bubble],
                    hoverinfo='text',
                    name=f'{ap.name_cn}',
                    showlegend=False,
                ))

        fig.update_layout(
            title='穴位定位 2D 投影',
            xaxis=dict(scaleanchor="y", scaleratio=1),
            yaxis=dict(autorange='reversed'),  # 图像坐标系
            width=self.width,
            height=self.height,
        )

        return fig


def main():
    """测试3D可视化"""
    import sys
    sys.path.insert(0, '..')
    from core.pose_extractor import PoseExtractor, ModelMode
    from core.spine_estimator import SpineEstimator
    from core.acupoint_locator import AcupointLocator
    import cv2

    # 需要一张测试图片
    image_path = "input/images/test.jpg"
    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"请将测试图片放到 {image_path}")
            print("或运行: python scripts/demo_torso.py")
            return

        extractor = PoseExtractor(mode=ModelMode.POSE)
        pose_result = extractor.process(image)

        locator = AcupointLocator()
        locator.load_database(["database/acupoints_torso.json", "database/acupoints_hands.json", "database/acupoints_face.json"])

        acu_result = locator.locate(pose_result)
        print(f"定位到 {acu_result.total_found} 个穴位")

        viewer = Viewer3D()
        fig = viewer.build(acu_result, pose_result, acu_result.spine)
        viewer.save_html(fig, "output/acupoints_3d.html")
        print("已生成 output/acupoints_3d.html，在浏览器中打开查看")
        fig.show()

    except Exception as e:
        print(f"测试失败: {e}")


if __name__ == "__main__":
    main()
