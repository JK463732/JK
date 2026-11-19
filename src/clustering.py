# -*- coding: utf-8 -*-
"""用户分群模块：基于 K-Means 的电影口味分群。

- 输入: 用户x类型 偏好画像（见 data_loader.MovieLensData.user_genre_matrix）
- 用轮廓系数(Silhouette)在候选 K 中选择簇数
- 每个簇给出可解释的口味标签（偏好/回避的类型 Top-N）
- 支持把新用户（Agent 冷启动画像）分配到最近的簇
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


class UserClusterer:
    """K-Means 用户口味分群器。"""

    def __init__(self, k_min: int = 2, k_max: int = 12, random_state: int = 42):
        self.k_min = k_min
        self.k_max = k_max
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model: KMeans | None = None
        self.k: int = 0
        self.silhouette: float = 0.0
        self.labels: dict[int, int] = {}
        self.centroids_scaled: np.ndarray | None = None
        self.genre_list: list[str] = []

    # ---------- 模型选择 ----------
    def select_k(self, profile: pd.DataFrame) -> list[dict]:
        """在 [k_min, k_max] 内用轮廓系数选择 K，返回逐 K 记录。"""
        x = self.scaler.fit_transform(profile.values)
        records: list[dict] = []
        best_k, best_s = self.k_min, -1.0
        rng = np.random.RandomState(self.random_state)
        sample_idx = rng.choice(len(x), size=min(len(x), 500), replace=False)
        for k in range(self.k_min, self.k_max + 1):
            km = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            lab = km.fit_predict(x)
            s = float(silhouette_score(x[sample_idx], lab[sample_idx]))
            inert = float(km.inertia_)
            records.append({"k": k, "silhouette": s, "inertia": inert})
            if s > best_s:
                best_k, best_s = k, s
        self.k = best_k
        self.silhouette = best_s
        self.genre_list = list(profile.columns)
        self.model = KMeans(n_clusters=best_k, random_state=self.random_state, n_init=10)
        scaled_labels = self.model.fit_predict(x)
        self.centroids_scaled = self.model.cluster_centers_
        self.labels = {int(u): int(l) for u, l in zip(profile.index, scaled_labels)}
        self.profile_ = profile
        return records

    def fit(self, profile: pd.DataFrame, k: int) -> None:
        """用指定 K 直接拟合（用于对照实验）。"""
        self.k = k
        self.genre_list = list(profile.columns)
        x = self.scaler.fit_transform(profile.values)
        self.model = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
        scaled_labels = self.model.fit_predict(x)
        self.centroids_scaled = self.model.cluster_centers_
        self.labels = {int(u): int(l) for u, l in zip(profile.index, scaled_labels)}
        self.profile_ = profile

    # ---------- 解释 ----------
    def cluster_summary(self, top_n: int = 3) -> dict[int, dict]:
        """每个簇的口味标签：簇内用户画像的均值上最偏好/最回避的类型。"""
        summary: dict[int, dict] = {}
        prof = self.profile_
        for c in range(self.k):
            members = [u for u, lab in self.labels.items() if lab == c]
            mean_vec = prof.loc[members].mean(axis=0)
            top_likes = mean_vec.sort_values(ascending=False).head(top_n).index.tolist()
            top_dislikes = mean_vec.sort_values(ascending=True).head(top_n).index.tolist()
            summary[c] = {
                "size": len(members),
                "members": members,
                "top_likes": top_likes,
                "top_dislikes": top_dislikes,
                "mean_profile": mean_vec,
            }
        return summary

    def describe_cluster(self, c: int) -> str:
        """生成某个簇的自然语言描述（用于 Agent 的可解释输出）。"""
        s = self.cluster_summary(1)[c]
        likes = "、".join(s["top_likes"])
        return f"第{c + 1}组（偏好{likes}，共{s['size']}名相似观众）"

    # ---------- 新用户分配 ----------
    def assign_user(self, profile_vector: dict[str, float]) -> int:
        """把新用户的类型偏好画像分配到最近的簇（Agent 冷启动定位）。"""
        vec = np.array([float(profile_vector.get(g, 0.0)) for g in self.genre_list])
        vec_scaled = self.scaler.transform(vec.reshape(1, -1))
        dists = np.linalg.norm(self.centroids_scaled - vec_scaled, axis=1)
        return int(np.argmin(dists))
