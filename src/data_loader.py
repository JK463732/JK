# -*- coding: utf-8 -*-
"""数据加载与预处理模块：加载 MovieLens ml-latest-small 数据集，
构建用户-类型(genre)偏好画像，并划分训练/测试集。

数据来源: MovieLens ml-latest-small (GroupLens, University of Minnesota)
许可: 仅供研究与教学使用 (https://files.grouplens.org/datasets/movielens/)
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "ml-latest-small")


class MovieLensData:
    """MovieLens 小型数据集的加载器与预处理工具。"""

    def __init__(self, data_dir: str = DATA_DIR):
        self.data_dir = data_dir
        self.ratings = pd.read_csv(
            os.path.join(data_dir, "ratings.csv"),
            dtype={"userId": np.int32, "movieId": np.int32, "rating": np.float32, "timestamp": np.int64},
        )
        self.movies = pd.read_csv(
            os.path.join(data_dir, "movies.csv"),
            dtype={"movieId": np.int32, "title": str, "genres": str},
        )
        # 全部类型清单（排除无类型标记的电影占位符）
        self.genre_list = sorted(
            {g for s in self.movies["genres"] for g in str(s).split("|") if g != "(no genres listed)"}
        )
        self.genre_index = {g: i for i, g in enumerate(self.genre_list)}
        # movieId -> 类型集合 的映射
        self.movie_genres: dict[int, set[str]] = {}
        for row in self.movies.itertuples(index=False):
            gs = {g for g in str(row.genres).split("|") if g != "(no genres listed)"}
            self.movie_genres[int(row.movieId)] = gs
        self.movie_title: dict[int, str] = dict(zip(self.movies["movieId"], self.movies["title"]))
        self.users = sorted(self.ratings["userId"].unique().tolist())
        self.user_index = {u: i for i, u in enumerate(self.users)}
        self.items = sorted(self.ratings["movieId"].unique().tolist())
        self.item_index = {m: i for i, m in enumerate(self.items)}

    # ---------- 偏好画像 ----------
    def user_genre_matrix(self, ratings: pd.DataFrame | None = None) -> pd.DataFrame:
        """构建 用户x类型 偏好矩阵。

        对每个用户先减去其评分均值（去除打分宽严差异），再对每个类型
        取其评价过的该类型电影的平均中心化评分，得到偏好画像。
        ratings 缺省用全量评分；也可传入截断的评分（冷启动模拟用）。
        返回 DataFrame: index=userId, columns=genre_list。
        """
        df = (self.ratings if ratings is None else ratings).merge(
            self.movies[["movieId", "genres"]], on="movieId", how="left"
        )
        df = df[df["genres"].notna()]
        user_mean = df.groupby("userId")["rating"].transform("mean")
        df = df.assign(centered=df["rating"] - user_mean)
        rows: list[dict] = []
        exploded = df.assign(genre=df["genres"].str.split("|")).explode("genre")
        exploded = exploded[exploded["genre"] != "(no genres listed)"]
        prof = exploded.groupby(["userId", "genre"])["centered"].mean().reset_index()
        mat = pd.DataFrame(
            0.0, index=pd.Index(self.users, name="userId"), columns=self.genre_list
        )
        for row in prof.itertuples(index=False):
            mat.loc[int(row.userId), str(row.genre)] = float(row.centered)  # type: ignore[index]
        return mat

    # ---------- 数据划分 ----------
    def split(self, like_threshold: float = 4.0) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """留一法划分（按时间留出）：

        - test_rating: 每个用户时间戳上最后一条评分（用于 RMSE/MAE 评分预测评估）
        - test_pos:    每个用户最后一条评分 >= like_threshold 的记录
                       （视为"用户喜欢的电影"，用于 Top-N 排序评估）
        - train:       其余评分，用于训练
        """
        df = self.ratings.sort_values(["userId", "timestamp"])
        last_idx = df.groupby("userId").tail(1).index
        test_rating = df.loc[last_idx].copy()
        train = df.drop(index=last_idx).copy()
        # 每个用户最后一条"喜欢"(>=4.0)的评分
        liked = df[df["rating"] >= like_threshold]
        last_liked_idx = liked.groupby("userId").tail(1).index
        test_pos = df.loc[last_liked_idx][["userId", "movieId", "rating"]].copy()
        # 从训练集中剔除排序评估的正样本，保证评估无泄漏
        drop_idx = test_pos.index.intersection(train.index)
        train = train.drop(index=drop_idx)
        return train.reset_index(drop=True), test_rating.reset_index(drop=True), test_pos.reset_index(drop=True)

    # ---------- 流行度统计 ----------
    def bayesian_scores(self, ratings: pd.DataFrame, prior: float = 3.0, min_votes: int = 5) -> pd.Series:
        """贝叶斯加权平均评分（IMDB 式），用于热门度排序，避免小样本高分。"""
        stats = ratings.groupby("movieId")["rating"].agg(["count", "mean"])
        stats = stats.reindex(self.items).fillna({"count": 0.0, "mean": 0.0})
        all_mean = float(ratings["rating"].mean())
        score = (stats["count"] / (stats["count"] + float(min_votes))) * stats["mean"] + (
            float(min_votes) / (stats["count"] + float(min_votes))
        ) * (all_mean if all_mean == all_mean else prior)
        return pd.Series(score.values, index=self.items, name="score")

    def popularity_counts(self, ratings: pd.DataFrame) -> pd.Series:
        cnt = ratings.groupby("movieId")["rating"].count().reindex(self.items).fillna(0.0)
        return pd.Series(cnt.values, index=self.items, name="count")
