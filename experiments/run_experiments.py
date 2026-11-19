# -*- coding: utf-8 -*-
"""实验主脚本：一键复现全部实验。

实验内容:
  E1. K-Means 分群: K 选择(轮廓系数) + 簇画像
  E2. 排序评估: 随机/全局热门/分群热门/用户协同过滤/分群协同过滤/矩阵分解
      指标: HR@10, NDCG@10, Precision@10, 覆盖率@10, 新颖性@10 (留一法, 完整排序, 无泄漏)
  E3. 评分预测: 全局均值 vs 用户协同过滤 vs 矩阵分解 (RMSE/MAE)
  E4. 消融: UserCF 近邻数 K = 10/30/50/100
  E5. 冷启动: 已知评分 3/5/10/20 条时各方法 HR@10
  E6. 误差分析: 命中率 vs 测试电影流行度 / 用户活跃度

输出: experiments/results/*.csv, report/figures/*.png
"""
from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.clustering import UserClusterer
from src.data_loader import MovieLensData
from src.evaluate import (
    activity_analysis,
    cold_start_eval,
    error_analysis,
    rank_metrics,
    rating_metrics,
)
from src.recommender import (
    ClusterPopRecommender,
    ClusterUserCFRecommender,
    GlobalMeanPredictor,
    PopularityRecommender,
    RandomRecommender,
    SVDRecommender,
    UserCFRecommender,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "experiments", "results")
FIGURES = os.path.join(ROOT, "report", "figures")
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def save_csv(df: pd.DataFrame, name: str) -> None:
    path = os.path.join(RESULTS, name)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"[saved] {path}")


def save_fig(name: str) -> None:
    path = os.path.join(FIGURES, name)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[saved] {path}")


def main() -> None:
    os.makedirs(RESULTS, exist_ok=True)
    os.makedirs(FIGURES, exist_ok=True)

    print("=" * 60)
    print("加载数据 ...")
    data = MovieLensData()
    print(f"评分 {len(data.ratings)} 条 | 用户 {len(data.users)} 位 | 电影 {len(data.items)} 部 | 类型 {len(data.genre_list)} 种")
    train, test_rating, test_pos = data.split()
    print(f"训练 {len(train)} 条 | 排序测试正样本 {len(test_pos)} 条 | 评分测试 {len(test_rating)} 条")

    # ---------- E1: K-Means 分群 ----------
    print("=" * 60)
    print("E1. K-Means 分群与 K 选择 ...")
    profile = data.user_genre_matrix()
    clusterer = UserClusterer(k_min=2, k_max=12)
    k_records = clusterer.select_k(profile)
    k_df = pd.DataFrame(k_records)
    save_csv(k_df, "e1_k_selection.csv")
    print(f"轮廓系数最优 K = {clusterer.k} (silhouette = {clusterer.silhouette:.4f})")
    summary = clusterer.cluster_summary(top_n=3)
    rows = []
    for c, s in summary.items():
        rows.append(
            {
                "簇": c + 1,
                "人数": s["size"],
                "偏好类型Top3": "、".join(s["top_likes"]),
                "回避类型Top3": "、".join(s["top_dislikes"]),
            }
        )
    cluster_df = pd.DataFrame(rows)
    save_csv(cluster_df, "e1_cluster_summary.csv")
    print(cluster_df.to_string(index=False))

    # 图1: K 选择曲线
    plt.figure(figsize=(7, 4))
    plt.plot(k_df["k"], k_df["silhouette"], "o-", color="#2b6cb0")
    plt.axvline(clusterer.k, color="red", linestyle="--", label=f"选定 K={clusterer.k}")
    plt.xlabel("簇数 K")
    plt.ylabel("轮廓系数 Silhouette")
    plt.title("图1 K-Means 簇数选择（轮廓系数）")
    plt.legend()
    save_fig("fig1_k_selection.png")

    # 图2: 簇画像热力图（簇 x 类型 的质心偏好）
    mat = pd.DataFrame(
        [summary[c]["mean_profile"].values for c in range(clusterer.k)],
        index=[f"第{c+1}组" for c in range(clusterer.k)],
        columns=profile.columns,
    )
    plt.figure(figsize=(10, max(3, 0.5 * clusterer.k + 1.5)))
    plt.imshow(mat.values, cmap="RdBu_r", aspect="auto", vmin=-0.6, vmax=0.6)
    plt.colorbar(label="中心化偏好分")
    plt.xticks(range(len(mat.columns)), mat.columns, rotation=45, ha="right", fontsize=8)
    plt.yticks(range(len(mat.index)), mat.index, fontsize=9)
    plt.title("图2 各口味分群的类型偏好画像（红=偏好, 蓝=回避）")
    save_fig("fig2_cluster_profiles.png")

    labels = clusterer.labels

    # ---------- E2: 排序评估 ----------
    print("=" * 60)
    print("E2. Top-10 排序评估（留一法, 完整排序）...")
    recs = {
        "随机(Random)": RandomRecommender(data, seed=42),
        "全局热门(Popularity)": PopularityRecommender(data),
        "分群热门(Cluster-Pop)": ClusterPopRecommender(data, labels),
        "用户协同过滤-初版(Norm-UserCF)": UserCFRecommender(data, k_neighbors=30, min_support=1, normalize=True),
        "用户协同过滤(UserCF)": UserCFRecommender(data, k_neighbors=30, min_support=5, normalize=False),
        "分群协同过滤(Cluster-UserCF)": ClusterUserCFRecommender(data, labels, k_neighbors=30),
        "矩阵分解(SVD)": SVDRecommender(data, n_components=20),
    }
    for r in recs.values():
        r.fit(train)
    rank_df = rank_metrics(recs, data, test_pos, n=10)
    save_csv(rank_df, "e2_ranking_metrics.csv")
    print(rank_df.to_string(index=False))

    # 图3: 排序指标对比
    metrics_cols = [c for c in rank_df.columns if c != "方法"]
    fig, axes = plt.subplots(1, len(metrics_cols), figsize=(16, 3.6))
    colors = ["#a0aec0", "#ecc94b", "#63b3ed", "#f6ad55", "#48bb78", "#4299e1", "#805ad5"]
    for ax, col in zip(axes, metrics_cols):
        ax.bar(rank_df["方法"], rank_df[col], color=colors[: len(rank_df)])
        ax.set_title(col)
        ax.tick_params(axis="x", rotation=55, labelsize=7)
        for j, v in enumerate(rank_df[col]):
            ax.text(j, v, f"{v:.3f}", ha="center", va="bottom", fontsize=7)
    fig.suptitle("图3 六种方法的 Top-10 排序指标对比")
    save_fig("fig3_ranking_metrics.png")

    # ---------- E3: 评分预测 ----------
    print("=" * 60)
    print("E3. 评分预测评估 (RMSE/MAE)...")
    predictors = {
        "全局均值(GlobalMean)": GlobalMeanPredictor(data),
        "用户协同过滤(UserCF)": recs["用户协同过滤(UserCF)"],
        "矩阵分解(SVD)": recs["矩阵分解(SVD)"],
    }
    predictors["全局均值(GlobalMean)"].fit(train)
    rating_df = rating_metrics(predictors, data, test_rating)
    save_csv(rating_df, "e3_rating_metrics.csv")
    print(rating_df.to_string(index=False))

    # ---------- E4: 消融实验 ----------
    print("=" * 60)
    print("E4. 消融: UserCF 近邻数 K（改进版打分: 加权和, min_support=5）...")
    ablation_rows = []
    for k in (10, 20, 30, 50, 100):
        cf = UserCFRecommender(data, k_neighbors=k, min_support=5, normalize=False)
        cf.fit(train)
        one = rank_metrics({"UserCF-K%d" % k: cf}, data, test_pos, n=10)
        row = {"近邻数K": k, "HR@10": one.loc[0, "HR@10"], "覆盖率@10": one.loc[0, "覆盖率@10"], "新颖性@10": one.loc[0, "新颖性@10"]}
        ablation_rows.append(row)
        print(row)
    abl_df = pd.DataFrame(ablation_rows)
    save_csv(abl_df, "e4_ablation_neighbors.csv")

    plt.figure(figsize=(7, 4))
    plt.plot(abl_df["近邻数K"], abl_df["HR@10"], "o-", color="#2b6cb0", label="HR@10")
    plt.plot(abl_df["近邻数K"], abl_df["覆盖率@10"], "s--", color="#48bb78", label="覆盖率@10")
    plt.xscale("log")
    plt.xticks(abl_df["近邻数K"], [str(k) for k in abl_df["近邻数K"]])
    plt.xlabel("近邻数 K")
    plt.ylabel("指标值")
    plt.title("图4 消融实验：UserCF 近邻数 K 的影响")
    plt.legend()
    save_fig("fig4_ablation.png")

    # ---------- E5: 冷启动 ----------
    print("=" * 60)
    print("E5. 冷启动实验（固定测试正样本, 变化已知评分前缀条数）...")
    def make_usercf(known: pd.DataFrame):
        r = UserCFRecommender(data, k_neighbors=30, min_support=5, normalize=False)
        r.fit(known)
        return r

    def make_pop(known: pd.DataFrame):
        r = PopularityRecommender(data)
        r.fit(known)
        return r

    def make_clusterpop(known: pd.DataFrame):
        cl = UserClusterer()
        cl.fit(data.user_genre_matrix(known), k=clusterer.k)
        r = ClusterPopRecommender(data, cl.labels)
        r.fit(known)
        return r

    cold_df = cold_start_eval(
        {
            "全局热门(Popularity)": make_pop,
            "分群热门(Cluster-Pop)": make_clusterpop,
            "用户协同过滤(UserCF)": make_usercf,
        },
        data,
        test_pos,
        known_ns=[3, 5, 10, 20, 50],
        n=10,
    )
    save_csv(cold_df, "e5_cold_start.csv")
    print(cold_df.to_string(index=False))

    plt.figure(figsize=(8, 4.5))
    color_map = {"全局热门(Popularity)": "#ecc94b", "分群热门(Cluster-Pop)": "#48bb78", "用户协同过滤(UserCF)": "#4299e1"}
    marker_map = {"全局热门(Popularity)": "s", "分群热门(Cluster-Pop)": "^", "用户协同过滤(UserCF)": "o"}
    for m, grp in cold_df.groupby("方法"):
        grp = grp.sort_values("已知评分数")
        plt.plot(grp["已知评分数"], grp["HR@10"], marker_map[m] + "-", color=color_map[m], label=m)
    plt.xscale("log")
    plt.xticks([3, 5, 10, 20, 50], ["3", "5", "10", "20", "50"])
    plt.xlabel("新用户已知评分条数（对数轴）")
    plt.ylabel("HR@10")
    plt.title("图5 冷启动：已知评分条数对命中率的影响（测试正样本固定）")
    plt.legend(fontsize=8)
    save_fig("fig5_cold_start.png")

    # ---------- E6: 误差分析 ----------
    print("=" * 60)
    print("E6. 误差分析 ...")
    cf_main = recs["用户协同过滤(UserCF)"]
    pop_err = error_analysis(cf_main, data, test_pos, n=10)
    pop_err.insert(0, "方法", "用户协同过滤(UserCF)")
    act_err = activity_analysis(cf_main, data, test_pos, n=10)
    save_csv(pop_err, "e6_error_by_popularity.csv")
    save_csv(act_err, "e6_error_by_activity.csv")
    print("--- 按测试电影流行度 ---")
    print(pop_err.to_string(index=False))
    print("--- 按用户活跃度 ---")
    print(act_err.to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes[0].bar(pop_err["测试电影被看次数"].astype(str), pop_err["命中率"], color="#4299e1")
    axes[0].set_title("图6a 命中率 vs 测试电影被看次数（UserCF）")
    axes[0].set_xlabel("测试正样本电影在全库被评次数")
    axes[0].set_ylabel("HR@10")
    axes[0].tick_params(axis="x", rotation=30, labelsize=8)
    for j, v in enumerate(pop_err["命中率"]):
        axes[0].text(j, v, f"{v:.2f}", ha="center", va="bottom", fontsize=7)
    axes[1].bar(act_err["用户活跃度分层"].astype(str), act_err["命中率"], color="#48bb78")
    axes[1].set_title("图6b 命中率 vs 用户活跃度（UserCF）")
    axes[1].set_xlabel("用户历史评分数分位")
    axes[1].set_ylabel("HR@10")
    axes[1].tick_params(axis="x", rotation=30, labelsize=8)
    for j, v in enumerate(act_err["命中率"]):
        axes[1].text(j, v, f"{v:.2f}", ha="center", va="bottom", fontsize=7)
    save_fig("fig6_error_analysis.png")

    print("=" * 60)
    print("全部实验完成。结果见 experiments/results/，图表见 report/figures/")


if __name__ == "__main__":
    main()
