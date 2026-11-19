# -*- coding: utf-8 -*-
"""演示脚本：完整运行两次 Agent 会话并保存交互记录。

- 会话A(顺利流程): 冷启动问答 -> 分群定位 -> 打分反馈 -> 协同过滤推荐 -> 约束 -> 换一批 -> 退出
- 会话B(边界与失败处理): 连续无效输入 -> 大众回退 -> 全部没看过 -> 约束 -> 评分不足提示 -> 退出

输出:
  demo/demo_session_A.txt / demo_session_B.txt  完整交互记录
  demo/demo_session_A.png                       终端样式截图（会话A）
运行: python demo/run_demo.py
交互模式: python -m src.run_interactive  (或 python src/run_interactive.py)
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.agent import IO, MovieAgent
from src.clustering import UserClusterer
from src.data_loader import MovieLensData

DEMO_DIR = os.path.join(ROOT, "demo")

PERSONA_A = [
    "喜欢 科幻 动作",
    "1 5",
    "3 4",
    "5 4.5",
    "推荐",
    "不要恐怖",
    "推荐",
    "换一批",
    "退出",
]

PERSONA_B = [
    "我随便看看",
    "不知道，你们有什么推荐",
    "无所谓",
    "没看过",
    "不要爱情",
    "1 2",
    "推荐",
    "退出",
]


def render_terminal_png(lines: list[str], path: str, title: str) -> None:
    """把交互记录渲染成终端样式 PNG（用于报告演示截图）。"""
    from PIL import Image, ImageDraw, ImageFont

    font = ImageFont.truetype(r"C:\Windows\Fonts\msyh.ttc", 16)
    title_font = ImageFont.truetype(r"C:\Windows\Fonts\msyh.ttc", 13)
    char_w = 9
    margin, pad = 18, 14
    line_h = 26
    max_cols = 78
    wrapped: list[tuple[str, bool]] = []
    for ln in lines:
        is_input = ln.startswith("> ")
        text = ln[2:] if is_input else ln
        # 简单按显示宽度换行（中文按2列计）
        cur, width = "", 0
        for ch in text:
            w = 2 if ord(ch) > 127 else 1
            if width + w > max_cols - 2:
                wrapped.append((cur, is_input))
                cur, width = "    ", 4
            cur += ch
            width += w
        wrapped.append((cur, is_input))
    height = pad * 2 + 22 + line_h * len(wrapped)
    img = Image.new("RGB", (margin * 2 + char_w * max_cols, height), (24, 26, 32))
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, img.width, 26], fill=(44, 48, 58))
    draw.text((margin, 5), title, font=title_font, fill=(200, 200, 200))
    y = 40
    for text, is_input in wrapped:
        color = (255, 213, 128) if is_input else (222, 226, 230)
        prefix = "你 > " if is_input else "影荐 | "
        draw.text((margin, y), prefix + text, font=font, fill=color)
        y += line_h
    img.save(path)


def main() -> None:
    os.makedirs(DEMO_DIR, exist_ok=True)
    print("加载数据并训练分群模型 ...")
    data = MovieLensData()
    clusterer = UserClusterer(k_min=2, k_max=12)
    clusterer.select_k(data.user_genre_matrix())
    print(f"分群 K={clusterer.k}，开始演示会话 ...")

    sessions = [("A", PERSONA_A, "顺利流程：冷启动→分群→反馈→协同过滤→约束"), ("B", PERSONA_B, "边界处理：无效输入→回退→低置信度提示")]
    for tag, script, title in sessions:
        io = IO(scripted=script)
        agent = MovieAgent(data, clusterer)
        agent.run(io)
        txt_path = os.path.join(DEMO_DIR, f"demo_session_{tag}.txt")
        io.save(txt_path)
        print(f"[saved] {txt_path}  ({len(io.lines)} 行)")
        if tag == "A":
            png_path = os.path.join(DEMO_DIR, "demo_session_A.png")
            render_terminal_png(io.lines, png_path, f"影荐 Agent 演示 · {title}")
            print(f"[saved] {png_path}")


if __name__ == "__main__":
    main()
