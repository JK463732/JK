# -*- coding: utf-8 -*-
"""影荐 Agent：基于用户分群与协同过滤的多轮交互式电影推荐助手。

Agent 工作流（状态机）:
  冷启动问答 -> 类型偏好画像 -> K-Means 分群定位 -> 分群热门推荐(冷启动回退)
  -> 收集用户对推荐电影的多轮反馈 -> 反馈修正画像
  -> 相似观众(协同过滤)推荐 + 可解释理由 -> 约束过滤(类型/年代/评分)
  -> 置信度提示与失败处理 -> 继续推荐 / 退出

设计要点:
- 规则式意图解析（关键词匹配），不依赖大模型 API，完全离线可复现;
- 每步输出都携带理由（分群标签、近邻数量、相似度、类型依据）;
- 无相似观众/反馈过少时回退到分群热门并明确提示（人机协作边界）。
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .clustering import UserClusterer
from .data_loader import MovieLensData

GENRE_ZH: dict[str, str] = {
    "科幻": "Sci-Fi", "动作": "Action", "喜剧": "Comedy", "爱情": "Romance",
    "恐怖": "Horror", "动画": "Animation", "儿童": "Children", "剧情": "Drama",
    "犯罪": "Crime", "战争": "War", "西部": "Western", "悬疑": "Mystery",
    "惊悚": "Thriller", "冒险": "Adventure", "奇幻": "Fantasy", "音乐": "Music",
    "音乐剧": "Musical", "黑色": "Film-Noir", "纪录片": "Documentary", "纪录": "Documentary",
    "imax": "IMAX",
}
GENRE_EN = {v: k for k, v in GENRE_ZH.items()}


class IO:
    """输入输出抽象：支持交互 stdin 与脚本回放（演示/测试）。"""

    def __init__(self, scripted: list[str] | None = None, transcript_path: str | None = None):
        self.scripted = list(scripted) if scripted else None
        self.pos = 0
        self.lines: list[str] = []

    def read(self, prompt: str) -> str:
        self.write(prompt, nl=False)
        if self.scripted is not None:
            if self.pos >= len(self.scripted):
                raw = "退出"
            else:
                raw = self.scripted[self.pos]
            self.pos += 1
            self.log(f"> {raw}")
        else:
            raw = input()
            self.log(f"> {raw}")
        return raw.strip()

    def write(self, text: str, nl: bool = True) -> None:
        print(text, end="\n" if nl else "", flush=True)
        logged = text
        if not nl:
            # 提示语末尾的换行+提示符 "> " 不写入交互记录
            logged = text.rstrip()
            if logged.endswith(">"):
                logged = logged[:-1].rstrip()
        self.log(logged)

    def log(self, line: str) -> None:
        self.lines.append(line)

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(self.lines) + "\n")


class MovieAgent:
    """影荐 Agent 主类。"""

    INTRO = [
        "==========================================================",
        "  影荐 —— 基于用户分群与协同过滤的电影推荐 Agent",
        "  我会先了解你的口味，再为你找到口味相近的观众，",
        "  用他们的真实观影评分给你有依据的推荐。",
        "  随时输入: 帮助 | 不要<类型> | <序号> <1-5分> | 换一批 | 退出",
        "==========================================================",
    ]

    def __init__(self, data: MovieLensData, clusterer: UserClusterer, k_neighbors: int = 30):
        self.data = data
        self.clusterer = clusterer
        self.k_neighbors = k_neighbors
        # 预计算: 全量数据的用户均值/画像/簇内热门，作为 Agent 的知识库
        self.user_mean = data.ratings.groupby("userId")["rating"].mean()
        self.profile = data.user_genre_matrix()
        self.cluster_pop = self._cluster_pop_scores()
        self.user_ratings: dict[int, float] = {}  # 新用户对电影的评分
        self.last_recs: list[dict] = []
        self.command_quit = False
        self.command_recommend = False
        self.command_next = False
        self.io: IO | None = None
        self.exclude_genres: set[str] = set()
        self.min_avg_rating = 0.0
        self.year_from: int | None = None
        self.likes: set[str] = set()
        self.dislikes: set[str] = set()
        self.cluster_id: int | None = None
        self.confidence = "low"

    # ---------- 工具 ----------
    def _cluster_pop_scores(self) -> dict[int, list[tuple[int, float]]]:
        df = self.data.ratings.copy()
        df["cluster"] = df["userId"].map(self.clusterer.labels)
        out: dict[int, list[tuple[int, float]]] = {}
        for c, group in df.groupby("cluster"):
            s = self.data.bayesian_scores(group)
            out[int(c)] = sorted(s.items(), key=lambda t: float(t[1]), reverse=True)
        return out

    def _parse_genres(self, text: str) -> tuple[set[str], set[str]]:
        """从自然语言中解析喜欢/不喜欢的电影类型（规则式意图解析）。"""
        likes: set[str] = set()
        dislikes: set[str] = set()
        negated = False
        tokens = [t for t in text.replace("，", " ").replace(",", " ").replace("、", " ").split() if t]
        for tok in tokens:
            low = tok.lower()
            if ("不要" in tok) or ("不喜欢" in tok) or ("讨厌" in tok) or ("排除" in tok):
                negated = True
                for zh, en in GENRE_ZH.items():
                    if zh in tok:
                        dislikes.add(en)
                continue
            if ("喜欢" in tok) or ("爱看" in tok) or ("想看" in tok):
                negated = False
                for zh, en in GENRE_ZH.items():
                    if zh in tok:
                        likes.add(en)
                continue
            matched = False
            for zh, en in GENRE_ZH.items():
                if zh in low or low == en.lower() or en.lower() in low:
                    matched = True
                    if negated:
                        dislikes.add(en)
                    else:
                        likes.add(en)
                    break
            if not matched:
                for zh, en in GENRE_ZH.items():
                    if zh in tok:
                        (dislikes if negated else likes).add(en)
        return likes - dislikes, dislikes

    def _pseudo_profile(self) -> dict[str, float]:
        """综合冷启动问答与后续评分反馈，构建新用户的类型偏好画像。"""
        vec: dict[str, float] = {g: 0.0 for g in self.data.genre_list}
        for g in self.likes:
            if g in vec:
                vec[g] += 0.8
        for g in self.dislikes:
            if g in vec:
                vec[g] -= 0.8
        # 反馈修正: 用户实际打分电影的类型按评分偏差加权
        if self.user_ratings:
            base = 3.5
            for mid, r in self.user_ratings.items():
                delta = (r - base) / 2.0
                for g in self.data.movie_genres.get(mid, set()):
                    if g in vec:
                        vec[g] += 0.4 * delta
        return vec

    def _similar_users(self) -> list[tuple[int, float]]:
        """按类型画像余弦相似度+评分重合度，找口味最接近的真实观众。"""
        vec = self._pseudo_profile()
        v = np.array([vec.get(g, 0.0) for g in self.data.genre_list])
        nv = np.linalg.norm(v)
        if nv == 0:
            return []
        sims: list[tuple[int, float]] = []
        for uid, row in self.profile.iterrows():
            u = row.values.astype(float)
            nu = np.linalg.norm(u)
            if nu == 0:
                continue
            s = float(np.dot(v, u) / (nv * nu))
            sims.append((int(uid), s))
        # 命中用户已评电影的相似观众加权（把新用户评分并入 CF 打分）
        if self.user_ratings:
            global_mean = float(self.data.ratings["rating"].mean())
            bonus: dict[int, float] = {}
            for mid, r in self.user_ratings.items():
                who = self.data.ratings[self.data.ratings["movieId"] == mid]
                for row in who.itertuples(index=False):
                    uid = int(row.userId)
                    bonus[uid] = bonus.get(uid, 0.0) + (1.0 if abs(float(row.rating) - r) <= 1.0 else -0.5)
            sims = [(uid, s + 0.05 * bonus.get(uid, 0.0)) for uid, s in sims]
        sims.sort(key=lambda t: t[1], reverse=True)
        # 优先同簇近邻；不足时全局近邻（失败回退，计入低置信度）
        if self.cluster_id is not None:
            in_cluster = [(u, s) for u, s in sims if self.clusterer.labels.get(u) == self.cluster_id]
            if len(in_cluster) >= 5:
                return in_cluster[: self.k_neighbors]
        return sims[: self.k_neighbors]

    def _cf_candidates(self, neighbors: list[tuple[int, float]], offset: int = 0, n: int = 5) -> list[dict]:
        """用近邻的真实评分做协同过滤打分，产出可解释的候选推荐。

        打分与实验最优配置一致: 近邻中心化评分的相似度加权和（不归一化），
        且候选电影至少被 min_support 位近邻评过（过滤低支持度噪声）。
        """
        min_support = 3
        scores: dict[int, float] = {}
        raw_ratings: dict[int, list[float]] = {}
        seen = set(self.user_ratings.keys())
        for uid, s in neighbors:
            um = float(self.user_mean.get(uid, 3.5))
            rows = self.data.ratings[self.data.ratings["userId"] == uid]
            for row in rows.itertuples(index=False):
                mid = int(row.movieId)
                if mid in seen:
                    continue
                scores[mid] = scores.get(mid, 0.0) + s * (float(row.rating) - um)
                raw_ratings.setdefault(mid, []).append(float(row.rating))
        cand = [
            (mid, sc, len(raw_ratings[mid]), raw_ratings[mid])
            for mid, sc in scores.items()
            if len(raw_ratings[mid]) >= min_support and sc > 0
        ]
        cand.sort(key=lambda t: t[1], reverse=True)
        # 约束过滤
        out: list[dict] = []
        avg_rating = self.data.ratings.groupby("movieId")["rating"].mean()
        for mid, score, nvoters, raws in cand:
            if self.exclude_genres and (self.data.movie_genres.get(mid, set()) & self.exclude_genres):
                continue
            if self.min_avg_rating and float(avg_rating.get(mid, 0.0)) < self.min_avg_rating:
                continue
            if self.year_from is not None:
                title = self.data.movie_title.get(mid, "")
                year = int(title[-5:-1]) if title.endswith(")") and title[-5:-1].isdigit() else None
                if year is None or year < self.year_from:
                    continue
            out.append(
                {
                    "movieId": mid,
                    "score": score,
                    "neighbors": nvoters,
                    "neighbor_avg_rating": round(sum(raws) / len(raws), 1),
                    "genres": sorted(self.data.movie_genres.get(mid, set())),
                }
            )
            if len(out) >= offset + n:
                break
        return out[offset : offset + n]

    def _cluster_candidates(self, offset: int = 0, n: int = 5) -> list[dict]:
        """分群热门候选（冷启动/回退策略），带解释信息。"""
        if self.cluster_id is None:
            return []
        avg_rating = self.data.ratings.groupby("movieId")["rating"].mean()
        cnt = self.data.ratings.groupby("movieId")["rating"].count()
        ranked = self.cluster_pop.get(self.cluster_id, [])
        out: list[dict] = []
        for mid, score in ranked:
            if mid in self.user_ratings:
                continue
            if self.exclude_genres and (self.data.movie_genres.get(mid, set()) & self.exclude_genres):
                continue
            if self.min_avg_rating and float(avg_rating.get(mid, 0.0)) < self.min_avg_rating:
                continue
            if self.year_from is not None:
                title = self.data.movie_title.get(mid, "")
                year = int(title[-5:-1]) if title.endswith(")") and title[-5:-1].isdigit() else None
                if year is None or year < self.year_from:
                    continue
            out.append(
                {
                    "movieId": mid,
                    "score": float(score),
                    "neighbors": int(cnt.get(mid, 0)),
                    "neighbor_avg_rating": round(float(avg_rating.get(mid, 0.0)), 1),
                    "genres": sorted(self.data.movie_genres.get(mid, set())),
                }
            )
            if len(out) >= offset + n:
                break
        return out[offset : offset + n]

    def _print_recs(self, recs: list[dict], reason: str) -> None:
        if not recs:
            return
        self.io.write(f"【{reason}】为你推荐 {len(recs)} 部电影：")
        for i, r in enumerate(recs, 1):
            genres = "/".join(GENRE_EN.get(g, g) for g in r["genres"])
            self.io.write(
                f"  {i}. 《{self.data.movie_title[r['movieId']]}》 "
                f"[{genres}] —— 口味相近的 {r['neighbors']} 位观众平均打了 {r['neighbor_avg_rating']} 分"
            )
        self.io.write("给看过的电影打个分(输入: 序号 1-5分) 或回复“没看过”，也可以直接说“换一批”或提出新要求。")

    # ---------- 主循环 ----------
    def run(self, io: IO) -> None:
        self.io = io
        for line in self.INTRO:
            io.write(line)
        # 1) 冷启动问答
        retries = 0
        while not self.likes and not self.dislikes:
            raw = io.read("先聊聊你的口味吧：你喜欢什么类型的电影？(可一次说多个，如: 科幻, 动作；也可以说 不要恐怖)\n> ")
            self._handle_commands(raw)
            if self.command_quit:
                return
            likes, dislikes = self._parse_genres(raw)
            if not likes and not dislikes:
                retries += 1
                if retries >= 3:
                    io.write("连续三次未能识别你的口味，先按“大众好评”为你推荐，欢迎随时补充偏好。")
                    self.likes = {"Drama"}
                    break
                io.write(f"没听懂“{raw}”里的类型，试试这些关键词: {('、'.join(list(GENRE_ZH)[:12]))} ...")
                continue
            self.likes, self.dislikes = likes, dislikes
            break
        # 2) 分群定位
        vec = self._pseudo_profile()
        self.cluster_id = self.clusterer.assign_user(vec)
        desc = self.clusterer.describe_cluster(self.cluster_id)
        like_str = "、".join(GENRE_EN.get(g, g) for g in sorted(self.likes)) if self.likes else "大众"
        dislike_str = (
            "，回避" + "、".join(GENRE_EN.get(g, g) for g in sorted(self.dislikes)) if self.dislikes else ""
        )
        io.write(f"明白了：你偏好{like_str}{dislike_str}。")
        io.write(f"口味定位: 你和{desc}最接近。")
        recs = self._cluster_candidates(n=5)
        if not recs:
            recs = self._cluster_candidates(n=5)
        self.last_recs = recs
        self._print_recs(recs, f"{desc}·群体好评榜")
        # 3) 反馈循环与协同过滤推荐
        offset = 5
        while True:
            raw = io.read("> ")
            if raw in ("退出", "quit", "exit", "q"):
                io.write("感谢使用影荐！你的口味档案已保存思路：越多反馈，推荐越准。再见！")
                break
            if not self._handle_commands(raw):
                self._handle_feedback(raw)
                continue
            if self.command_quit:
                break
            if self.command_recommend:
                self.command_recommend = False
                neighbors = self._similar_users()
                if len(self.user_ratings) >= 3 and neighbors:
                    mean_sim = float(np.mean([s for _, s in neighbors]))
                    self.confidence = "high" if (mean_sim >= 0.15 and len(neighbors) >= 8) else "medium"
                    recs = self._cf_candidates(neighbors, offset=0, n=5)
                    reason = f"协同过滤·{len(neighbors)}位口味相近的观众"
                    if self.confidence != "high":
                        io.write("提示: 相似观众较少或相似度不高，以下推荐仅供参考，欢迎继续打分修正。")
                else:
                    self.confidence = "low"
                    recs = self._cluster_candidates(offset=0, n=5)
                    reason = f"{self.clusterer.describe_cluster(self.cluster_id)}·群体好评榜"
                    io.write("提示: 你已评分的电影还不足 3 部，暂时先按群体口味推荐；评分越多，个性化越强。")
                self.last_recs = recs
                self._print_recs(recs, reason)
                if not recs:
                    io.write("当前约束下没有找到合适的电影，已自动放宽限制。")
                    self.exclude_genres, self.min_avg_rating, self.year_from = set(), 0.0, None
                    recs = self._cluster_candidates(offset=0, n=5)
                    self.last_recs = recs
                    self._print_recs(recs, f"{self.clusterer.describe_cluster(self.cluster_id)}·群体好评榜")
            if self.command_next:
                self.command_next = False
                neighbors = self._similar_users()
                if len(self.user_ratings) >= 3 and neighbors:
                    recs = self._cf_candidates(neighbors, offset=offset, n=5)
                    reason = f"协同过滤·换一批"
                else:
                    recs = self._cluster_candidates(offset=offset, n=5)
                    reason = "群体好评榜·换一批"
                offset += 5
                self.last_recs = recs if recs else self.last_recs
                self._print_recs(recs if recs else self.last_recs, reason)

    # ---------- 命令解析 ----------
    def _handle_commands(self, raw: str) -> bool:
        """处理全局命令；返回 True 表示消费了该输入。"""
        self.command_quit = False
        self.command_recommend = False
        self.command_next = False
        low = raw.lower()
        if raw in ("退出", "quit", "exit", "q"):
            self.command_quit = True
            return True
        if low in ("帮助", "help"):
            self.io.write("命令: 不要<类型> | 年代<年>后 | 高分(平均≥3.5) | 清除约束 | 换一批 | 推荐 | 退出\n给电影打分: 序号 + 空格 + 1-5分；没看过直接说“没看过 序号”。")
            return True
        if ("不要" in raw) or ("排除" in raw) or ("不喜欢" in raw and "类型" in raw):
            likes, dislikes = self._parse_genres(raw)
            self.exclude_genres |= dislikes
            if dislikes:
                self.dislikes |= dislikes
                self.io.write("已加入回避清单: " + "、".join(GENRE_EN.get(g, g) for g in sorted(dislikes)) + "。")
            else:
                self.io.write("没有识别到要排除的类型，试试: 不要恐怖。")
            return True
        if "清除约束" in raw or "清空约束" in raw:
            self.exclude_genres, self.min_avg_rating, self.year_from = set(), 0.0, None
            self.io.write("已清除全部约束，按你的原始口味继续。")
            return True
        if "高分" in raw:
            self.min_avg_rating = 3.5
            self.io.write("好，之后只推荐平均分不低于 3.5 的电影。")
            return True
        if "年代" in raw or ("后" in raw and any(ch.isdigit() for ch in raw)):
            digits = "".join(ch for ch in raw if ch.isdigit())
            if digits:
                self.year_from = int(digits)
                self.io.write(f"好，之后只推荐 {self.year_from} 年以后上映的电影。")
                return True
        if raw in ("换一批", "下一个", "再来一批", "更多"):
            self.command_next = True
            return True
        if raw in ("推荐", "继续推荐", "给我推荐", "再推荐"):
            self.command_recommend = True
            return True
        if raw in ("没看过", "都没看过", "全没看过"):
            self.io.write("没关系，那我们换一批；看到想看的随时打分！")
            self.command_next = True
            return True
        return False

    def _handle_feedback(self, raw: str) -> None:
        """解析用户对推荐列表的反馈（打分/没看过）。"""
        tokens = raw.replace("，", " ").split()
        rated = False
        # 模式1: "没看过 1 2" / 模式2: "1 5" (序号+分数) / "1 喜欢"
        i = 0
        seq: list[int] = []
        while i < len(tokens):
            tok = tokens[i]
            if tok in ("没看过", "看过没", "没"):
                j = i + 1
                while j < len(tokens) and tokens[j].isdigit() and 1 <= int(tokens[j]) <= 9:
                    seq.append(int(tokens[j]))
                    j += 1
                i = j
                continue
            if tok.isdigit() and 1 <= int(tok) <= 9 and int(tok) <= len(getattr(self, "last_recs", [])):
                idx = int(tok)
                j = i + 1
                score: float | None = None
                if j < len(tokens):
                    nxt = tokens[j]
                    if nxt.replace(".", "").isdigit():
                        val = float(nxt)
                        if 0.5 <= val <= 5.0:
                            score = val
                            j += 1
                    elif "喜欢" in nxt:
                        score = 4.5
                        j += 1
                    elif "不喜欢" in nxt or "难看" in nxt:
                        score = 2.0
                        j += 1
                if score is not None and idx <= len(getattr(self, "last_recs", [])):
                    mid = self.last_recs[idx - 1]["movieId"]
                    self.user_ratings[mid] = score
                    rated = True
                    self.io.write(
                        f"已记录: 《{self.data.movie_title[mid]}》 {score} 分。"
                        f"(累计已评 {len(self.user_ratings)} 部)"
                    )
                    if len(self.user_ratings) >= 3:
                        io = self.io
                        io.write("评分已足够启动个性化推荐，输入“推荐”看看口味相近的观众都在看什么。")
                i = j
                continue
            i += 1
        if seq:
            for idx in seq:
                if idx <= len(self.last_recs):
                    mid = self.last_recs[idx - 1]["movieId"]
                    self.user_ratings.setdefault(mid, 3.0)
            self.io.write(f"了解，{len(seq)} 部都没看过，已从你的已看清单中排除。")
            return
        if not rated:
            self.io.write(f"没理解“{raw}”。可以: 给电影打分(如 1 5)、说“换一批”、“不要<类型>”，或输入“帮助”。")
