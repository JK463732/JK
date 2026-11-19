# -*- coding: utf-8 -*-
"""交互模式入口：python -m src.run_interactive"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent import IO, MovieAgent
from src.clustering import UserClusterer
from src.data_loader import MovieLensData


def main() -> None:
    data = MovieLensData()
    print("正在训练分群模型，请稍候 ...")
    clusterer = UserClusterer(k_min=2, k_max=12)
    clusterer.select_k(data.user_genre_matrix())
    agent = MovieAgent(data, clusterer)
    agent.run(IO())
    print("会话结束。")


if __name__ == "__main__":
    main()
