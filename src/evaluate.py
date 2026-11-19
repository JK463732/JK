# -*- coding: utf-8 -*-
"""实验评估模块：排序指标、评分预测指标、覆盖率/新颖性、误差分析。

评估协议（报告中"实验设计"部分的实现）:
- 排序评估: 留一法——每个用户时间上最后一条"喜欢"(>=4.0)的评分作为测试正样本，
  在全部未看过电影上做完整排序，看正样本是否落入 Top-N（无泄漏）。
- 评分评估: 每个用户最后一条评分作为测试，预测并计算 RMSE/MAE。
- 覆盖率: 至少被一个用户的 Top-N 推荐覆盖的电影占全部电影的比例。
- 新颖性: 推荐电影的平均自信息量 -log2(流行度占比)，越高越"长尾"。
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd


def rank_metrics(
    recommenders: dict,
    data,
    test_pos: pd.DataFrame,
    n: int = 10,
) -> pd.DataFrame:
    """对多个推荐器计算 Top-N 排序指标（HR/NDCG/Precision/覆盖率/新颖性）。"""
    # 只评估训练集中出现过(>=2条评分)且测试正样本有效的用户
    valid = test_pos[test_pos["movieId"].isin(data.item_index)]
    pop_counts = data.popularity_counts(data.ratings)
    total_ratings = float(pop_counts.sum())
    pop_p = {m: c / total_ratings for m, c in pop_counts.items()}
    rows: list[dict] = []
    for name, rec in recommenders.items():
        hits, ndcgs, precs = 0.0, 0.0, 0.0
        covered: set[int] = set()
        novelty_sum, novelty_cnt = 0.0, 0
        user_hits: dict[int, int] = {}
        for row in valid.itertuples(index=False):
            uid, mid = int(row.userId), int(row.movieId)
            top = rec.recommend(uid, n)
            top_ids = [m for m, _ in top]
            hit = 1 if mid in top_ids else 0
            hits += hit
            user_hits[uid] = hit
            if hit:
                ndcgs += 1.0 / math.log2(top_ids.index(mid) + 2)
            precs += hit / n
            covered.update(top_ids)
            for m in top_ids:
                novelty_sum += -math.log2(max(pop_p.get(m, 1e-9), 1e-9))
                novelty_cnt += 1
        n_users = len(valid)
        rows.append(
            {
                "方法": name,
                f"HR@{n}": round(hits / n_users, 4),
                f"NDCG@{n}": round(ndcgs / n_users, 4),
                f"Precision@{n}": round(precs / n_users, 4),
                f"覆盖率@{n}": round(len(covered) / len(data.items), 4),
                f"新颖性@{n}": round(novelty_sum / max(novelty_cnt, 1), 4),
            }
        )
    return pd.DataFrame(rows)


def rating_metrics(predictors: dict, data, test_rating: pd.DataFrame) -> pd.DataFrame:
    """评分预测指标（RMSE/MAE），predictor 需实现 predict_rating(uid, mid)。"""
    rows: list[dict] = []
    test = test_rating[test_rating["movieId"].isin(data.item_index)]
    for name, pred in predictors.items():
        se, ae = 0.0, 0.0
        for row in test.itertuples(index=False):
            p = pred.predict_rating(int(row.userId), int(row.movieId))
            se += (p - float(row.rating)) ** 2
            ae += abs(p - float(row.rating))
        rows.append(
            {
                "方法": name,
                "RMSE": round(math.sqrt(se / len(test)), 4),
                "MAE": round(ae / len(test), 4),
                "测试样本数": len(test),
            }
        )
    return pd.DataFrame(rows)


def error_analysis(rec, data, test_pos: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """误差分析：命中/未命中正样本按电影流行度分桶。"""
    valid = test_pos[test_pos["movieId"].isin(data.item_index)]
    pop_counts = data.popularity_counts(data.ratings)
    rows: list[dict] = []
    for row in valid.itertuples(index=False):
        uid, mid = int(row.userId), int(row.movieId)
        top_ids = [m for m, _ in rec.recommend(uid, n)]
        hit = 1 if mid in top_ids else 0
        rows.append({"userId": uid, "movieId": mid, "hit": hit, "pop": int(pop_counts.get(mid, 0))})
    df = pd.DataFrame(rows)
    bins = [0, 5, 10, 25, 50, 100, 10**9]
    labels = ["1-4次", "5-9次", "10-24次", "25-49次", "50-99次", ">=100次"]
    df["流行度桶"] = pd.cut(df["pop"], bins=bins, labels=labels, right=False)
    grouped = df.groupby("流行度桶", observed=True)["hit"].agg(["mean", "count"]).reset_index()
    grouped.columns = ["测试电影被看次数", "命中率", "样本数"]
    grouped["命中率"] = grouped["命中率"].round(4)
    return grouped


def activity_analysis(rec, data, test_pos: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """误差分析：按用户活跃度(训练期评分数)分层的命中率。"""
    valid = test_pos[test_pos["movieId"].isin(data.item_index)]
    cnt = data.ratings.groupby("userId")["rating"].count()
    rows: list[dict] = []
    for row in valid.itertuples(index=False):
        uid, mid = int(row.userId), int(row.movieId)
        top_ids = [m for m, _ in rec.recommend(uid, n)]
        rows.append({"userId": uid, "hit": 1 if mid in top_ids else 0, "activity": int(cnt.get(uid, 0))})
    df = pd.DataFrame(rows)
    q = df["activity"].quantile([0.25, 0.5, 0.75]).tolist()
    bins = [0, q[0], q[1], q[2], 10**9]
    labels = ["低活跃(≤25%)", "中低(25-50%)", "中高(50-75%)", "高活跃(>75%)"]
    df["活跃度"] = pd.cut(df["activity"], bins=bins, labels=labels, right=False)
    grouped = df.groupby("活跃度", observed=True)["hit"].agg(["mean", "count"]).reset_index()
    grouped.columns = ["用户活跃度分层", "命中率", "样本数"]
    grouped["命中率"] = grouped["命中率"].round(4)
    return grouped


def cold_start_eval(factories: dict, data, test_pos: pd.DataFrame, known_ns: list[int], n: int = 10) -> pd.DataFrame:
    """冷启动模拟（固定测试协议）：

    测试正样本固定为每个用户时间上最后一条"喜欢"(>=4.0)的评分（与主实验 E2 相同），
    变化的只是训练阶段每个用户可见的历史评分前缀条数 known_n。
    这样直接度量"历史信息量对推荐质量的影响"，避免"测试任务随 n 变化"的混淆。

    factories: {方法名: factory(known_ratings) -> 已拟合的推荐器}
    """
    df = data.ratings.sort_values(["userId", "timestamp"])
    total = df.groupby("userId").size()
    rows: list[dict] = []
    for known_n in known_ns:
        head_idx = df.groupby("userId").head(known_n).index
        known = df.loc[head_idx].reset_index(drop=True)
        # 仅评估: 总评分数 > known_n、测试电影在候选库中、且测试正样本不在已知前缀中的用户
        valid_users = set(total[total > known_n].index)
        known_pairs = set(zip(known["userId"].tolist(), known["movieId"].tolist()))
        tset = test_pos[
            test_pos["userId"].isin(valid_users) & test_pos["movieId"].isin(data.item_index)
        ].copy()
        keep = [(int(u), int(m)) not in known_pairs for u, m in zip(tset["userId"], tset["movieId"])]
        tset = tset[keep]
        for name, factory in factories.items():
            rec = factory(known)
            hits = 0
            for row in tset.itertuples(index=False):
                top_ids = [m for m, _ in rec.recommend(int(row.userId), n)]
                if int(row.movieId) in top_ids:
                    hits += 1
            rows.append(
                {
                    "已知评分数": known_n,
                    "方法": name,
                    "测试用户数": len(tset),
                    f"HR@{n}": round(hits / max(len(tset), 1), 4),
                }
            )
    return pd.DataFrame(rows)
