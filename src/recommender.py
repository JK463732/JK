# -*- coding: utf-8 -*-
"""推荐算法模块：基线方法与协同过滤方法。

所有推荐器实现统一接口:
    fit(train_ratings, ...) -> None
    recommend(user_id, n, exclude_seen) -> list[(movieId, score)]
支持按约束（排除类型、最低评分、年代）过滤候选——Agent 的工具组件之一。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD


class BaseRecommender:
    """推荐器基类：维护公共状态与候选过滤工具。"""

    name = "base"

    def __init__(self, data):
        self.data = data
        # 各推荐器在 fit 时填充:
        self.seen: dict[int, set[int]] = {}
        self.min_movie_rating: pd.Series | None = None
        self.movie_rating_count: pd.Series | None = None

    # ---------- 公共工具 ----------
    def _build_seen(self, ratings: pd.DataFrame) -> None:
        self.seen = (
            ratings.groupby("userId")["movieId"].agg(set).to_dict()
        )

    def _movie_quality(self, ratings: pd.DataFrame) -> None:
        """电影的全局平均分与评分人数（供约束过滤使用）。"""
        self.min_movie_rating = (
            ratings.groupby("movieId")["rating"].mean().reindex(self.data.items).fillna(0.0)
        )
        self.movie_rating_count = (
            ratings.groupby("movieId")["rating"].count().reindex(self.data.items).fillna(0.0)
        )

    def filter_candidates(
        self,
        candidates: list[tuple[int, float]],
        exclude_genres: set[str] | None = None,
        min_avg_rating: float = 0.0,
        min_votes: int = 0,
        year_from: int | None = None,
        year_to: int | None = None,
    ) -> list[tuple[int, float]]:
        """按用户约束过滤候选电影（Agent 的检索/过滤工具）。

        - exclude_genres: 排除包含任一指定类型的电影
        - min_avg_rating / min_votes: 电影最低平均分 / 最少评分人数
        - year_from / year_to: 电影发行年代区间（从片名解析年份）
        """
        out: list[tuple[int, float]] = []
        for movie_id, score in candidates:
            genres = self.data.movie_genres.get(movie_id, set())
            if exclude_genres and (genres & exclude_genres):
                continue
            if self.min_movie_rating is not None and float(self.min_movie_rating.get(movie_id, 0.0)) < min_avg_rating:
                continue
            if self.movie_rating_count is not None and float(self.movie_rating_count.get(movie_id, 0.0)) < min_votes:
                continue
            if year_from is not None or year_to is not None:
                title = self.data.movie_title.get(movie_id, "")
                year = _parse_year(title)
                if year is None:
                    continue
                if year_from is not None and year < year_from:
                    continue
                if year_to is not None and year > year_to:
                    continue
            out.append((movie_id, score))
        return out

    # ---------- 接口 ----------
    def fit(self, ratings: pd.DataFrame) -> None:  # pragma: no cover
        raise NotImplementedError

    def recommend(self, user_id: int, n: int, exclude_seen: bool = True) -> list[tuple[int, float]]:  # pragma: no cover
        raise NotImplementedError

    def top_from_list(self, user_id: int, scored: list[tuple[int, float]], n: int, exclude_seen: bool) -> list[tuple[int, float]]:
        seen = self.seen.get(user_id, set()) if exclude_seen else set()
        items = [(m, s) for m, s in scored if m not in seen]
        items.sort(key=lambda t: float(t[1]), reverse=True)
        return items[:n]


def _parse_year(title: str) -> int | None:
    """从 MovieLens 片名 'Movie Name (1995)' 中解析年份。"""
    if not title.endswith(")"):
        return None
    inner = title[-5:-1]
    if inner.isdigit():
        return int(inner)
    return None


class PopularityRecommender(BaseRecommender):
    """基线1: 全局热门推荐（贝叶斯加权评分排序）。"""

    name = "全局热门(Popularity)"

    def fit(self, ratings: pd.DataFrame) -> None:
        self._build_seen(ratings)
        self._movie_quality(ratings)
        self.scores = self.data.bayesian_scores(ratings)
        self.ranked = sorted(self.scores.items(), key=lambda t: float(t[1]), reverse=True)

    def recommend(self, user_id: int, n: int, exclude_seen: bool = True) -> list[tuple[int, float]]:
        return self.top_from_list(user_id, self.ranked, n, exclude_seen)


class RandomRecommender(BaseRecommender):
    """基线2: 随机推荐（下界对照）。"""

    name = "随机(Random)"

    def __init__(self, data, seed: int = 42):
        super().__init__(data)
        self.rng = np.random.RandomState(seed)

    def fit(self, ratings: pd.DataFrame) -> None:
        self._build_seen(ratings)
        self._movie_quality(ratings)

    def recommend(self, user_id: int, n: int, exclude_seen: bool = True) -> list[tuple[int, float]]:
        seen = self.seen.get(user_id, set()) if exclude_seen else set()
        pool = [m for m in self.data.items if m not in seen]
        chosen = self.rng.choice(len(pool), size=min(n, len(pool)), replace=False)
        return [(pool[i], 0.0) for i in chosen]


class ClusterPopRecommender(BaseRecommender):
    """基线3: 仅分群热门——把用户定位到 K-Means 簇，推荐簇内成员共同高分电影。

    体现"只用聚类不用协同过滤"的消融版本，也是 Agent 冷启动阶段的回退策略。
    """

    name = "分群热门(Cluster-Pop)"

    def __init__(self, data, labels: dict[int, int]):
        super().__init__(data)
        self.user_labels = labels

    def fit(self, ratings: pd.DataFrame) -> None:
        self._build_seen(ratings)
        self._movie_quality(ratings)
        df = ratings.copy()
        df["cluster"] = df["userId"].map(self.user_labels)
        df = df[df["cluster"].notna()]
        df["cluster"] = df["cluster"].astype(int)
        self.cluster_scores: dict[int, list[tuple[int, float]]] = {}
        for c, group in df.groupby("cluster"):
            s = self.data.bayesian_scores(group)
            ranked = sorted(s.items(), key=lambda t: float(t[1]), reverse=True)
            self.cluster_scores[int(c)] = ranked

    def recommend(self, user_id: int, n: int, exclude_seen: bool = True) -> list[tuple[int, float]]:
        c = self.user_labels.get(user_id)
        if c is None or c not in self.cluster_scores:
            return []
        return self.top_from_list(user_id, self.cluster_scores[c], n, exclude_seen)


class UserCFRecommender(BaseRecommender):
    """基于用户的协同过滤（课程方法: 协同过滤 + 相似度度量）。

    评分矩阵按用户均值中心化后用余弦相似度找 Top-K 近邻，
    候选电影得分 = 近邻中心化评分的相似度加权和。
    """

    name = "用户协同过滤(UserCF)"

    def __init__(self, data, k_neighbors: int = 30, min_support: int = 5, normalize: bool = False):
        super().__init__(data)
        self.k_neighbors = k_neighbors
        # min_support: 候选电影至少要被多少位近邻评过（过滤低支持度噪声）
        # normalize: True 为初版按 |相似度| 归一化的打分（实验证明会稀释协同信号，
        #            保留用于消融对照）；False 为最终采用的近邻加权求和打分
        self.min_support = min_support
        self.normalize = normalize

    def fit(self, ratings: pd.DataFrame) -> None:
        self._build_seen(ratings)
        self._movie_quality(ratings)
        n_users = len(self.data.users)
        n_items = len(self.data.items)
        # 中心化评分矩阵（未评分 = 0，与余弦相似度的稀疏假设一致）
        self.mat = np.zeros((n_users, n_items), dtype=np.float32)
        user_mean = ratings.groupby("userId")["rating"].mean()
        for row in ratings.itertuples(index=False):
            ui = self.data.user_index[int(row.userId)]
            ii = self.data.item_index[int(row.movieId)]
            self.mat[ui, ii] = float(row.rating) - float(user_mean.loc[int(row.userId)])
        # 余弦相似度矩阵
        norms = np.linalg.norm(self.mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normed = self.mat / norms
        self.sim = normed @ normed.T
        np.fill_diagonal(self.sim, 0.0)
        # 供 predict_rating 使用的均值缓存
        self._train_for_mean = user_mean
        self._global_mean = float(ratings["rating"].mean())

    def _top_neighbors(self, user_id: int, restrict: set[int] | None = None) -> list[tuple[int, float]]:
        ui = self.data.user_index[user_id]
        sims = self.sim[ui].copy()
        for j in range(len(sims)):
            other = self.data.users[j]
            if restrict is not None and other not in restrict:
                sims[j] = 0.0
        order = np.argsort(-sims)[: self.k_neighbors]
        return [(self.data.users[j], float(sims[j])) for j in order if sims[j] > 0]

    def _score_candidates(self, user_id: int, neighbors: list[tuple[int, float]], exclude_seen: bool) -> list[tuple[int, float]]:
        """近邻中心化评分的加权打分。

        最终版本: score(i) = sum_{v in N, i in R_v} sim(u,v) * centered_r(v,i)
        并要求至少 min_support 位近邻评过该电影（低支持度物品不参与排序）。
        normalize=True 时除以 |sim| 之和（初版打分，仅用于消融对照）。
        """
        if not neighbors:
            return []
        seen = self.seen.get(user_id, set()) if exclude_seen else set()
        scores: dict[int, float] = {}
        weights: dict[int, float] = {}
        support: dict[int, int] = {}
        for other, s in neighbors:
            uj = self.data.user_index[other]
            row = self.mat[uj]
            for ii in np.nonzero(row)[0]:
                movie_id = self.data.items[int(ii)]
                if movie_id in seen:
                    continue
                scores[movie_id] = scores.get(movie_id, 0.0) + s * float(row[ii])
                weights[movie_id] = weights.get(movie_id, 0.0) + abs(s)
                support[movie_id] = support.get(movie_id, 0) + 1
        ranked: list[tuple[int, float]] = []
        for m, sc in scores.items():
            if weights[m] <= 0 or support[m] < self.min_support:
                continue
            value = sc / weights[m] if self.normalize else sc
            ranked.append((m, value))
        ranked.sort(key=lambda t: float(t[1]), reverse=True)
        return ranked

    def recommend(self, user_id: int, n: int, exclude_seen: bool = True) -> list[tuple[int, float]]:
        neighbors = self._top_neighbors(user_id)
        ranked = self._score_candidates(user_id, neighbors, exclude_seen)
        return ranked[:n]

    def predict_rating(self, user_id: int, movie_id: int) -> float:
        """预测评分（用于 RMSE/MAE 评估），近邻均值中心化还原。"""
        train = self._train_for_mean
        if user_id in train.index:
            base = float(train.loc[user_id])
        else:
            base = float(train.mean())
        neighbors = self._top_neighbors(user_id)
        if not neighbors or movie_id not in self.data.item_index:
            return base
        ii = self.data.item_index[movie_id]
        num = den = 0.0
        for other, s in neighbors:
            uj = self.data.user_index[other]
            v = float(self.mat[uj, ii])
            if v != 0.0:
                num += s * v
                den += abs(s)
        if den == 0:
            return base
        return float(np.clip(base + num / den, 0.5, 5.0))


class ClusterUserCFRecommender(UserCFRecommender):
    """消融方法: 聚类约束的协同过滤——近邻只在本簇内选取。

    簇内近邻不足 min_neighbors 时回退到全局近邻（混合策略），
    该行为会在报告中作为设计选择说明。
    """

    name = "分群协同过滤(Cluster-UserCF)"

    def __init__(self, data, labels: dict[int, int], k_neighbors: int = 30, min_neighbors: int = 5):
        super().__init__(data, k_neighbors)
        self.user_labels = labels
        self.min_neighbors = min_neighbors
        self.fallback_count = 0

    def fit(self, ratings: pd.DataFrame) -> None:
        super().fit(ratings)

    def recommend(self, user_id: int, n: int, exclude_seen: bool = True) -> list[tuple[int, float]]:
        c = self.user_labels.get(user_id)
        if c is not None:
            members = {u for u, lab in self.user_labels.items() if lab == c}
            neighbors = self._top_neighbors(user_id, restrict=members)
            if len(neighbors) >= self.min_neighbors:
                return self._score_candidates(user_id, neighbors, exclude_seen)[:n]
            self.fallback_count += 1
        neighbors = self._top_neighbors(user_id)
        return self._score_candidates(user_id, neighbors, exclude_seen)[:n]


class SVDRecommender(BaseRecommender):
    """对照方法: 矩阵分解(TruncatedSVD)——课程方法: 表示学习/矩阵分解。"""

    name = "矩阵分解(SVD)"

    def __init__(self, data, n_components: int = 20):
        super().__init__(data)
        self.n_components = n_components

    def fit(self, ratings: pd.DataFrame) -> None:
        self._build_seen(ratings)
        self._movie_quality(ratings)
        n_users = len(self.data.users)
        n_items = len(self.data.items)
        mat = np.zeros((n_users, n_items), dtype=np.float32)
        user_mean = ratings.groupby("userId")["rating"].mean()
        self.user_mean = user_mean
        for row in ratings.itertuples(index=False):
            ui = self.data.user_index[int(row.userId)]
            ii = self.data.item_index[int(row.movieId)]
            mat[ui, ii] = float(row.rating) - float(user_mean.loc[int(row.userId)])
        self.svd = TruncatedSVD(n_components=self.n_components, random_state=42)
        self.user_factors = self.svd.fit_transform(mat)
        self.item_factors = self.svd.components_
        self.mat = mat

    def _scores(self, user_id: int) -> np.ndarray:
        ui = self.data.user_index[user_id]
        base = float(self.user_mean.loc[user_id]) if user_id in self.user_mean.index else 3.5
        recon = self.user_factors[ui] @ self.item_factors
        return np.clip(recon + base, 0.5, 5.0)

    def recommend(self, user_id: int, n: int, exclude_seen: bool = True) -> list[tuple[int, float]]:
        scores = self._scores(user_id)
        seen = self.seen.get(user_id, set()) if exclude_seen else set()
        ranked = [
            (self.data.items[i], float(scores[i]))
            for i in range(len(scores))
            if self.data.items[i] not in seen
        ]
        ranked.sort(key=lambda t: float(t[1]), reverse=True)
        return ranked[:n]

    def predict_rating(self, user_id: int, movie_id: int) -> float:
        if movie_id not in self.data.item_index:
            return 3.5
        return float(self._scores(user_id)[self.data.item_index[movie_id]])


class GlobalMeanPredictor(BaseRecommender):
    """评分预测基线: 全局均值。"""

    name = "全局均值(GlobalMean)"

    def fit(self, ratings: pd.DataFrame) -> None:
        self._build_seen(ratings)
        self._movie_quality(ratings)
        self.global_mean = float(ratings["rating"].mean())

    def recommend(self, user_id: int, n: int, exclude_seen: bool = True) -> list[tuple[int, float]]:
        return []

    def predict_rating(self, user_id: int, movie_id: int) -> float:
        return self.global_mean
