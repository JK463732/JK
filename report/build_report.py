# -*- coding: utf-8 -*-
"""期末报告生成脚本：以课程提供的封面模板为基础填充字段，
追加 3000-5000 字正文（含图表、参考文献、附录），输出 docx。

运行: python report/build_report.py
（生成后再用 Word/WPS 另存为 PDF，或由外部命令转换）
"""
from __future__ import annotations

import os

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Pt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORT = os.path.join(ROOT, "report")
FIG = os.path.join(REPORT, "figures")
RESULTS = os.path.join(ROOT, "experiments", "results")
COVER = os.path.join(REPORT, "封面模板.docx")
OUT = os.path.join(REPORT, "028123270_王筠凯_影荐Agent期末报告.docx")


# ---------------- 基础排版工具 ----------------
def style_run(run, size=12, bold=False, east="宋体", ascii_font="Times New Roman"):
    run.font.name = ascii_font
    run.font.size = Pt(size)
    run.bold = bold
    rPr = run._element.get_or_add_rPr()
    rFonts = rPr.get_or_add_rFonts()
    rFonts.set(qn("w:eastAsia"), east)


def para(doc, text, size=12, bold=False, east="宋体", indent=True, align=None, space_after=6, line=1.4):
    p = doc.add_paragraph()
    r = p.add_run(text)
    style_run(r, size=size, bold=bold, east=east)
    pf = p.paragraph_format
    if indent:
        pf.first_line_indent = Pt(size * 2)
    pf.space_after = Pt(space_after)
    pf.line_spacing = line
    if align is not None:
        pf.alignment = align
    return p


def heading(doc, text, level=1):
    sizes = {1: 16, 2: 14}
    return para(doc, text, size=sizes.get(level, 13), bold=True, east="黑体", indent=False, space_after=8)


def add_table(doc, header, rows, widths=None, size=10.5):
    t = doc.add_table(rows=len(rows) + 1, cols=len(header))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    _set_table_borders(t)
    _set_fixed_layout(t)
    # 表头：居中加粗 + 浅灰底纹
    for j, h in enumerate(header):
        cell = t.rows[0].cells[j]
        _shade_cell(cell, "D9D9D9")
        _write_cell(cell, str(h), size=size, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER)
    # 数据行：短文本居中，长文本左对齐
    for i, row in enumerate(rows, start=1):
        for j, v in enumerate(row):
            align = WD_ALIGN_PARAGRAPH.CENTER if len(str(v)) <= 10 else WD_ALIGN_PARAGRAPH.LEFT
            _write_cell(t.rows[i].cells[j], str(v), size=size, align=align)
    if widths:
        for j, w in enumerate(widths):
            for row in t.rows:
                row.cells[j].width = Cm(w)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return t


def _set_table_borders(t):
    """给表格加全边框（单线 0.5pt）。"""
    tblPr = t._tbl.tblPr
    old = tblPr.find(qn("w:tblBorders"))
    if old is not None:
        tblPr.remove(old)
    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:val"), "single")
        el.set(qn("w:sz"), "4")
        el.set(qn("w:space"), "0")
        el.set(qn("w:color"), "000000")
        borders.append(el)
    tblPr.append(borders)


def _set_fixed_layout(t):
    layout = OxmlElement("w:tblLayout")
    layout.set(qn("w:type"), "fixed")
    t._tbl.tblPr.append(layout)
    t.autofit = False


def _shade_cell(cell, color):
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:fill"), color)
    cell._tc.get_or_add_tcPr().append(shd)


def _write_cell(cell, text, size=10.5, bold=False, align=WD_ALIGN_PARAGRAPH.CENTER):
    p = cell.paragraphs[0]
    p.alignment = align
    pf = p.paragraph_format
    pf.space_before = Pt(2)
    pf.space_after = Pt(2)
    pf.line_spacing = 1.15
    pf.first_line_indent = Pt(0)
    for r in list(p.runs):
        r._element.getparent().remove(r._element)
    r = p.add_run(text)
    style_run(r, size=size, bold=bold, east="宋体")


def add_figure(doc, filename, caption, width=5.9):
    doc.add_picture(os.path.join(FIG, filename), width=Inches(width))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    para(doc, caption, size=10.5, indent=False, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=10)


def table_caption(doc, text):
    para(doc, text, size=10.5, bold=True, indent=False, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=4)


def code_block(doc, lines, size=9):
    """代码摘录：等宽字体、无首行缩进、浅灰底纹。"""
    for ln in lines:
        p = doc.add_paragraph()
        r = p.add_run(ln if ln else " ")
        run_font(r, size)
        pf = p.paragraph_format
        pf.first_line_indent = Pt(0)
        pf.space_after = Pt(0)
        pf.line_spacing = 1.05
        pf.left_indent = Cm(0.5)
        _shade_paragraph(p)


def _shade_paragraph(p, color="F2F2F2"):
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:fill"), color)
    p._p.get_or_add_pPr().append(shd)


def set_default_style(doc):
    """统一全文默认字体：正文宋体小四（西文 Times New Roman）。"""
    normal = doc.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(12)
    rPr = normal.element.get_or_add_rPr()
    rFonts = rPr.get_or_add_rFonts()
    rFonts.set(qn("w:eastAsia"), "宋体")


def run_font(run, size):
    style_run(run, size=size, east="宋体", ascii_font="Consolas")


# ---------------- 封面填充 ----------------
def fill_cover(doc: Document):
    title_tbl = doc.tables[0]
    cell = title_tbl.rows[0].cells[1]
    p = cell.paragraphs[0]
    r = p.add_run("影荐——基于用户分群与协同过滤的电影推荐Agent")
    style_run(r, size=18, east="宋体")
    info = doc.tables[1]
    values = ["电子电气工程学院", "计算机科学与技术", "028123270", "王筠凯", "邱禧荷", "2026年11月19日"]
    for i, v in enumerate(values):
        cell = info.rows[i].cells[1]
        # 学院一格模板已有文字，替换为统一格式
        cell.paragraphs[0].clear() if hasattr(cell.paragraphs[0], "clear") else None
        for run in list(cell.paragraphs[0].runs):
            run._element.getparent().remove(run._element)
        r = cell.paragraphs[0].add_run(v)
        style_run(r, size=14, east="宋体")


# ---------------- 正文 ----------------
def build_body(doc: Document):
    doc.add_page_break()

    heading(doc, "一、需求与目标", 1)
    para(doc, "选题来自身边反复出现的真实场景：晚上想看一部电影，打开平台面对上万部片单却无从下手；热门榜单千人一面，并不贴合个人口味；主流流媒体平台的推荐结果是黑盒，既说不出“为什么推这部”，也不接受“今天不要看恐怖片”这类约束；新注册用户没有历史数据时推荐几乎不可用。这类“选择困难”在大学生群体中尤其普遍——时间碎片化、口味差异大、又习惯即时满足。")
    para(doc, "本项目面向上述痛点构建“影荐”——一个多轮交互式电影推荐 Agent。目标用户是有明确口味偏好、希望获得可解释推荐的普通观众。与现有做法相比，影荐的差异化在于：（1）每条推荐都附带依据（口味相近观众的规模与平均评分、用户所属口味分群标签），而不是黑盒输出；（2）接受自然语言约束（排除类型、年代、最低评分），并可随时打分反馈修正画像；（3）针对冷启动设计了“类型问答→分群定位→分群好评榜”的显式过渡策略，而非直接依赖不存在的评分历史。")
    para(doc, "项目范围界定为完全离线的命令行原型：数据使用公开的 MovieLens ml-latest-small 数据集（100,836 条评分、610 位用户、9,742 部电影）[1]，不调用任何外部大模型 API，保证任何人都能一键复现。成功标准在动手前预先定义：（1）排序指标 HR@10、NDCG@10 显著优于全局热门与随机基线；（2）评分预测 RMSE 优于全局均值基线；（3）冷启动阶段（评分少于 20 条）不劣于热门基线；（4）每条推荐均输出理由，Agent 在失败情形（无效输入、无相似观众、约束过强）下有明确处置而非崩溃或编造。评估方式为固定协议的留一法实验（详见第五节），测试集为每个用户时间上最后一条“喜欢”的评分，共 609 条，不参与任何训练环节。")

    heading(doc, "二、方案与系统设计", 1)
    para(doc, "影荐采用五阶段状态机工作流，模型、数据与人工确认点在各阶段各司其职，如表1所示。")
    table_caption(doc, "表1 影荐 Agent 工作流")
    add_table(
        doc,
        ["阶段", "用户输入", "Agent 行为", "输出"],
        [
            ["① 冷启动问答", "喜欢什么类型（自然语言）", "规则式意图解析提取类型偏好；连续3次无效则回退大众口味并明示", "类型偏好画像"],
            ["② 分群定位", "—", "将画像映射到 K-Means 口味分群（最近质心）", "分群标签与群体描述"],
            ["③ 群体好评榜", "对推荐打分（1-5）/没看过", "贝叶斯加权评分排序展示群体高分片；收集反馈修正画像", "Top-5 推荐与评分记录"],
            ["④ 协同过滤推荐", "“推荐”/约束指令", "评分≥3部后，检索口味相近的真实观众，用其评分加权打分（UserCF），过滤约束", "带理由的 Top-5 个性化推荐"],
            ["⑤ 反馈循环", "继续打分/换一批/退出", "更新画像与已看清单，低置信度时提示“仅供参考”", "持续修正的推荐"],
        ],
        widths=[2.6, 3.2, 6.2, 3.6],
    )
    para(doc, "系统分四层：数据层（MovieLens CSV 与影片元数据检索工具）、表示层（用户×类型中心化偏好画像、K-Means 分群）、决策层（分群热门、UserCF、候选约束过滤器）、交互层（意图解析、状态机、置信度提示）。任务入口明确：用户提供口味描述与可选约束，Agent 返回带理由的电影推荐；任务成功的判定即第一节的成功标准。")
    para(doc, "关键设计选择与备选方案：（1）冷启动策略——备选是直接用 UserCF，但新用户无评分时相似度无法计算；本方案以分群好评榜作为过渡，第五节冷启动实验证实评分少于20条时群体策略确实更稳。（2）意图解析——备选是接入大模型做自然语言理解，本方案选择规则式关键词匹配：完全离线可复现、行为可枚举可测试，代价是语言泛化能力有限（第六节讨论）。（3）切换阈值——收集满3部评分后切换到 CF 推荐，同时保留“评分还不足，仅供参考”的低置信度提示，属于在个性化与稳健性之间的折中。")
    para(doc, "失败与安全处理覆盖了课程要求的各情形：无法识别口味（连续3次重试后回退并告知）；无相似观众或相似度低（回退分群好评榜并明示“推荐仅供参考”）；约束过滤后候选为空（自动放宽约束并告知）；输入异常（重新提问，不静默忽略）；高影响结果（Agent 只给建议和依据，不代替用户决策，评分权始终在用户）。")

    heading(doc, "三、机器学习方法", 1)
    para(doc, "数据处理与划分。以 MovieLens 评分表为主数据源，电影表提供类型标注。划分协议采用按时间的留一法：每个用户时间上最后一条评分作为评分预测的测试样本；最后一条评分≥4.0 的记录视为“用户喜欢的电影”，作为排序评估的正样本；训练集为其余 99,980 条，并显式剔除排序正样本以防泄漏。全部电影仅出现于训练集的项目保留在候选库中（9,724 部）。")
    para(doc, "用户画像与 K-Means 分群。对每个用户，先将其全部评分减去个人均值（消除打分宽严差异），再对 19 个电影类型分别求中心化评分均值，得到 19 维类型偏好向量；经 StandardScaler 标准化后用 K-Means 聚类，K 在 2–12 间用轮廓系数选取（k-means++ 初始化，n_init=10，random_state=42 固定保证复现）[5][6]。分群结果既是冷启动的定位依据，也产出“分群热门”这一可解释的回退策略。")
    para(doc, "基于用户的协同过滤（UserCF）是 Agent 的核心预测组件，其思想源于 GroupLens 开创的协同过滤研究[2]。构造用户×电影的中心化评分矩阵，用余弦相似度找 Top-K 近邻；候选电影得分为近邻中心化评分的相似度加权和，并要求至少 min_support=5 位近邻评过该电影（过滤低支持度噪声，其必要性由第五节消融实验证明）。评分预测时以近邻加权残差加回用户均值并截断到 [0.5, 5]。")
    para(doc, "对照方法与基线：随机推荐（下界）、全局热门（贝叶斯加权评分，即 IMDB 式先验平滑）、分群热门（簇内贝叶斯加权）、矩阵分解（TruncatedSVD，20 维隐因子，代表表示学习方法）[3]、以及聚类约束的 UserCF（近邻限制在本簇内，不足5人则回退全局）。UserCF 近邻数 K 经消融取 30。")

    heading(doc, "四、Skills 与组件复用", 1)
    para(doc, "本项目遵循“复用开源组件、自己编写胶水与业务逻辑”的原则。全部复用组件的名称、版本、许可证与用途见表2，完整清单与许可证说明另见附录E。除表中组件外，Agent 框架、评估协议、UserCF 实现、实验脚本与全部测试样例均由本人独立编写。")
    table_caption(doc, "表2 复用组件清单")
    add_table(
        doc,
        ["组件", "版本", "许可证", "用途"],
        [
            ["MovieLens ml-latest-small", "2018-09", "仅限研究教育用途（GroupLens 条款）[1][7]", "评分与电影数据源"],
            ["scikit-learn", "1.9.1", "BSD-3-Clause [5]", "KMeans、TruncatedSVD、轮廓系数、StandardScaler"],
            ["pandas / numpy", "3.1.1 / 2.4", "BSD-3-Clause", "数据处理与数值计算"],
            ["matplotlib / Pillow", "3.10 / 11", "Matplotlib License / MIT-CMU", "实验图表、演示截图渲染"],
            ["python-docx", "1.1.2", "MIT", "本报告的自动化生成"],
        ],
        widths=[4.2, 2.0, 4.4, 5.0],
    )
    para(doc, "深入分析一个复用组件：scikit-learn 的 KMeans。它解决的子任务是把 610 个用户的标准化类型画像无监督地划分为若干口味群体；输入是数值矩阵，输出是簇标签与质心坐标；依赖纯本地内存计算，不需要任何外部服务或权限。其限制在于：需要预先指定 K（我以轮廓系数曲线代替盲选）；对初始中心敏感（采用 k-means++ 与多次初始化缓解）；隐含假设球形等方差簇——当口味空间实际连续时轮廓系数会明显偏低，这在本项目中被如实观测到（最优 K=2，轮廓系数仅 0.087），并作为发现写进误差分析而非掩盖。行为核验方面：我检查了各簇规模（228/382）、簇内用户画像的 Top 偏好类型与质心方向是否一致、固定 random_state 后多次运行结果是否完全一致，并用小样本（偏好“战争、悬疑”的模拟用户）人工验证分配结果符合直觉。")
    para(doc, "集成过程中的适配工作：把 KMeans 的输出标签转为 user_id→簇号的映射供分群热门与 Agent 使用；将 StandardScaler 的拟合参数持久化在分群器内部，保证 Agent 冷启动阶段新用户的画像向量经过与训练完全相同的标准化变换（否则最近质心计算会系统偏差）；为 assign_user 编写独立的最近质心推理路径，使其不依赖训练样本索引。复用组件的实际贡献由消融实验量化（第五节）：关闭聚类（用全局热门替代分群热门）后冷启动阶段 HR@10 从 0.0361 降至 0.0181；替换 UserCF 打分方式（归一化版）HR@10 从 0.0821 跌至 0.0049。")

    heading(doc, "五、实验设计与结果", 1)
    para(doc, "实验共六组：E1 分群与 K 选择；E2 六方法 Top-10 排序对比（随机、全局热门、分群热门、UserCF 初版、UserCF、聚类约束 UserCF、SVD）；E3 评分预测（全局均值、UserCF、SVD）；E4 近邻数 K 消融；E5 冷启动模拟；E6 误差分析。排序评估在全部未看电影上完整排序（不做负采样），指标为 HR@10、NDCG@10、Precision@10、目录覆盖率@10 与新颖性@10（推荐物品流行度自信息量均值，越高越“长尾”）[4]。")
    para(doc, "E1 分群结果：轮廓系数在 K=2 时最大（0.0866），两簇规模 228/382。第1组偏好战争、悬疑、犯罪而回避儿童、动画；第2组偏好动画、音乐剧。图1、图2 给出 K 选择曲线与簇画像热力图。值得注意的是轮廓系数绝对值偏低，说明用户口味空间连续性强、自然分簇弱——分群在本项目中的正确定位是“粗粒度冷启动锚点”，而非精细人格刻画，这一边界在设计与结论中均有体现。")
    add_figure(doc, "fig1_k_selection.png", "图1 K-Means 簇数选择（轮廓系数）", 4.6)
    add_figure(doc, "fig2_cluster_profiles.png", "图2 各口味分群的类型偏好画像（红=偏好，蓝=回避）", 6.0)
    para(doc, "E2 排序结果见表3与图3。最终版 UserCF 的 HR@10=0.0821、NDCG@10=0.0454，在全部方法中最优，约为全局热门基线（0.0181）的 4.5 倍；SVD 与聚类约束 UserCF 次之（0.0673）。所有个性化方法的覆盖率都远低于随机（0.46）——个性化与目录覆盖存在天然张力。")
    table_caption(doc, "表3 Top-10 排序指标对比（留一法，完整排序，609 个测试正样本）")
    add_table(
        doc,
        ["方法", "HR@10", "NDCG@10", "P@10", "覆盖率@10", "新颖性@10"],
        [
            ["随机(Random)", "0.0000", "0.0000", "0.0000", "0.4630", "14.84"],
            ["全局热门(Popularity)", "0.0181", "0.0099", "0.0018", "0.0041", "11.36"],
            ["分群热门(Cluster-Pop)", "0.0361", "0.0184", "0.0036", "0.0073", "10.17"],
            ["UserCF 初版(归一化)", "0.0049", "0.0021", "0.0005", "0.1142", "12.82"],
            ["UserCF（最终版）", "0.0821", "0.0454", "0.0082", "0.0347", "9.30"],
            ["分群约束 UserCF", "0.0673", "0.0383", "0.0067", "0.0333", "9.27"],
            ["矩阵分解(SVD)", "0.0673", "0.0327", "0.0067", "0.0527", "9.50"],
        ],
        widths=[4.6, 1.9, 2.1, 1.9, 2.2, 2.2],
    )
    add_figure(doc, "fig3_ranking_metrics.png", "图3 六种方法的 Top-10 排序指标对比", 6.2)
    para(doc, "这里记录一次有证据支持的迭代改进：初版 UserCF 按相似度绝对值之和归一化候选得分，HR@10 仅 0.0049，甚至低于热门基线。误差分析发现，归一化使“仅1位极端近邻给出高残差”的冷门物品得分虚高，协同信号被稀释。改为近邻加权求和并加入最低支持度过滤（≥5位近邻评过）后，HR@10 提升至 0.0821（约16.7倍），两版本在表3中并列呈现作为消融证据。")
    para(doc, "E3 评分预测：全局均值 RMSE=1.1172，UserCF 1.0224，SVD 0.9968。两个机器学习方法均显著优于基线，SVD 略优——与排序实验中 UserCF 略优形成互补，说明两类方法捕捉的模式不同。")
    para(doc, "E4 消融：UserCF 近邻数 K=10/20/30/50/100 对应 HR@10 = 0.0493/0.0837/0.0821/0.0821/0.0805，K 过小则证据不足、过大则引入不相似用户，K≈20–30 为平稳最优区（图4）。")
    add_figure(doc, "fig4_ablation.png", "图4 消融实验：UserCF 近邻数 K 的影响", 4.8)
    para(doc, "E5 冷启动（固定测试正样本，变化每个用户训练可见的评分前缀条数，图5）：已知3–10条时，全局热门与分群热门（HR≈0.02–0.025）明显优于评分向量 UserCF（0.003–0.015），因为少量评分不足以形成可靠近邻；已知20条后 UserCF（0.0311）反超热门（0.0260），50条时优势扩大（0.0322 对 0.0188）。这为 Agent 的分阶段设计提供了直接依据：冷启动期靠“类型画像→分群→群体好评榜”过渡（Agent 的近邻检索基于19维类型画像而非稀疏评分向量，进一步缓解了稀疏问题），反馈积累后再切换 CF。")
    add_figure(doc, "fig5_cold_start.png", "图5 冷启动：已知评分数对 HR@10 的影响（测试正样本固定）", 5.2)
    para(doc, "E6 误差分析与失败案例（图6）：按测试电影流行度分桶，UserCF 在被看≥100次的头部电影上 HR@10=0.3094，但被看不足25次的长尾电影上几乎全部失败（1-24次区间命中率≈0）。典型案例：用户89 的测试正样本《Frankenstein 90 (1984)》全库仅被看1次，没有任何近邻评过它，协同过滤在数学上就不可能将其排入候选——这是基于近邻方法的结构性盲区。反例是用户69 的《Cast Away (2000)》被看100次仍未命中，属于用户口味偏移（其历史偏爱与该片的合家欢基调不一致），提示偏好漂移是另一类错误来源。按用户活跃度分层则显示低活跃用户命中率（0.1111）反而高于高活跃用户（0.0584）：活跃用户的口味更分散、最后喜欢电影更冷门，可预测性更低。")
    add_figure(doc, "fig6_error_analysis.png", "图6 误差分析：命中率随测试电影流行度与用户活跃度的变化", 6.2)
    para(doc, "完整运行演示见 demo 目录：会话A覆盖“冷启动问答→分群定位→3次打分→协同过滤推荐→排除恐怖片→换一批”全流程（demo_session_A.txt 与终端截图 demo_session_A.png）；会话B演示连续无效输入、全部没看过、约束与低置信度提示等边界处理（demo_session_B.txt）。")

    heading(doc, "六、讨论与改进", 1)
    para(doc, "真实使用价值：影荐证明了“课程方法 + 工具组织 + 人工确认”的 Agent 范式在小数据上也能落地——推荐不是模型黑盒输出，而是带群体依据、可追问、可否决的建议。风险与边界：MovieLens 数据集许可仅限研究与教育用途，不可用于商业服务；推荐结果是统计意义上的建议而非客观事实，Agent 在低置信度时明示“仅供参考”，把最终判断留给用户，符合课程“避免无依据自动决策”的要求。")
    para(doc, "局限主要有三：其一，610 位用户的小数据集使近邻稀疏，长尾电影几乎无法召回（E6），这不是调参能解决的结构问题；其二，K=2 的分群粒度粗，聚类约束的 UserCF（0.0673）反而低于无约束版本（0.0821），说明当前分群信息与评分相似度部分冗余；其三，规则式意图解析只能理解预定义说法，无法泛化到任意自然语言。")
    para(doc, "有证据支持的后续改进：针对长尾盲区，引入电影内容特征（标题、类型的 TF-IDF 表示）做内容回补——当近邻支持度不足时用相似内容电影兜底，可望改善被看次数<25 的区间；针对意图理解，可在交互层外接大模型，但其输出必须落在与当前相同的结构化意图集合上，保证评估协议不变。尚未解决的问题：用户偏好随时间漂移（用户69案例）、以及推荐多样性—准确性权衡，需要在线实验才能进一步验证。")

    heading(doc, "七、参考文献", 1)
    refs = [
        "[1] Harper, F. M., & Konstan, J. A. (2015). The MovieLens Datasets: History and Context. ACM Transactions on Interactive Intelligent Systems, 5(4), Article 19.",
        "[2] Sarwar, B., Karypis, G., Konstan, J., & Riedl, J. (2001). Item-based Collaborative Filtering Recommendation Algorithms. Proceedings of the 10th International Conference on World Wide Web (WWW '01), 285–295.",
        "[3] Koren, Y., Bell, R., & Volinsky, C. (2009). Matrix Factorization Techniques for Recommender Systems. Computer, 42(8), 30–37.",
        "[4] Herlocker, J. L., Konstan, J. A., Terveen, L. G., & Riedl, J. T. (2004). Evaluating Collaborative Filtering Recommender Systems. ACM Transactions on Information Systems, 22(1), 5–53.",
        "[5] Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research, 12, 2825–2830.",
        "[6] MacQueen, J. (1967). Some Methods for Classification and Analysis of Multivariate Observations. Proceedings of the Fifth Berkeley Symposium on Mathematical Statistics and Probability, 281–297.",
        "[7] GroupLens Research. MovieLens Dataset Usage License. https://grouplens.org/datasets/movielens/",
    ]
    for r in refs:
        para(doc, r, size=10.5, indent=False, space_after=4)

    heading(doc, "八、附录", 1)
    para(doc, "按课程要求，附录列出主要代码（附A）、运行方法（附B）、依赖清单（附C）、测试样例（附D）与复用组件清单及许可证来源说明（附E）。完整项目代码随报告一并提交，目录结构如下：", size=10.5)
    code_block(doc, [
        "影荐Agent/",
        "├── README.md                  项目说明（含目录结构与复现步骤）",
        "├── requirements.txt           Python 依赖清单（附C）",
        "├── data/ml-latest-small/      MovieLens 数据集（ratings/movies/tags/links）",
        "├── src/                       源码（附A）",
        "│   ├── data_loader.py         数据加载、偏好画像、训练/测试划分",
        "│   ├── clustering.py          K-Means 分群（选K、簇画像、新用户定位）",
        "│   ├── recommender.py         6 种推荐算法 + 候选约束过滤工具",
        "│   ├── evaluate.py            评估协议（留一法排序指标、RMSE/MAE、冷启动、误差分析）",
        "│   ├── agent.py               影荐 Agent（多轮对话状态机）",
        "│   └── run_interactive.py     交互模式入口",
        "├── experiments/",
        "│   ├── run_experiments.py     一键复现全部实验（E1–E6）",
        "│   └── results/*.csv          实验结果表（报告表3、图1–6的数据来源）",
        "├── demo/",
        "│   ├── run_demo.py            演示脚本（两个剧本会话）",
        "│   └── demo_session_A/B.*     完整运行记录与终端截图（附D）",
        "└── report/                    本报告与图表（figures/fig1–fig6.png）",
    ], size=9)

    heading(doc, "附录A 主要代码", 2)
    para(doc, "以下摘录四个核心模块的关键实现（完整代码见随报告提交的代码仓库）。", size=10.5)
    para(doc, "A.1 数据划分——按时间的留一法（src/data_loader.py，防泄漏版本）", size=10.5, indent=False, bold=True)
    code_block(doc, [
        "def split(self, like_threshold=4.0):",
        "    df = self.ratings.sort_values(['userId', 'timestamp'])",
        "    last_idx = df.groupby('userId').tail(1).index",
        "    test_rating = df.loc[last_idx].copy()",
        "    train = df.drop(index=last_idx).copy()",
        "    liked = df[df['rating'] >= like_threshold]",
        "    test_pos = liked.groupby('userId').tail(1)",
        "    train = train.drop(index=test_pos.index)",
        "    return train, test_rating, test_pos",
    ], size=9)
    para(doc, "A.2 K-Means 分群与新用户定位（src/clustering.py）", size=10.5, indent=False, bold=True)
    code_block(doc, [
        "def select_k(self, profile):",
        "    x = self.scaler.fit_transform(profile.values)",
        "    for k in range(self.k_min, self.k_max + 1):",
        "        km = KMeans(n_clusters=k, random_state=42, n_init=10)",
        "        s = silhouette_score(x_sample, km.fit_predict(x)[sample_idx])",
        "        ...",
        "",
        "def assign_user(self, profile_vector):",
        "    vec = np.array([profile_vector.get(g, 0.0) for g in self.genre_list])",
        "    dists = np.linalg.norm(self.centroids_scaled",
        "                            - self.scaler.transform(vec.reshape(1, -1)), axis=1)",
        "    return int(np.argmin(dists))",
    ], size=9)
    para(doc, "A.3 UserCF 加权和打分（src/recommender.py，最终版；初版归一化打分作为消融对照保留）", size=10.5, indent=False, bold=True)
    code_block(doc, [
        "def _score_candidates(self, user_id, neighbors, exclude_seen):",
        "    for other, s in neighbors:",
        "        row = self.mat[self.data.user_index[other]]",
        "        for ii in np.nonzero(row)[0]:",
        "            scores[m] += s * row[ii]",
        "            support[m] += 1",
        "    ranked = [(m, v) for m, v in scores.items()",
        "              if support[m] >= self.min_support]",
        "    return sorted(ranked, key=lambda t: -t[1])",
    ], size=9)
    para(doc, "A.4 Agent 状态机主循环骨架（src/agent.py）", size=10.5, indent=False, bold=True)
    code_block(doc, [
        "def run(self, io):",
        "    self._cold_start_ask()",
        "    self.cluster_id = self.clusterer.assign_user(self._pseudo_profile())",
        "    self.last_recs = self._cluster_candidates(n=5)",
        "    while True:",
        "        raw = io.read('> ')",
        "        if self._handle_commands(raw):",
        "            continue",
        "        self._handle_feedback(raw)",
    ], size=9)

    heading(doc, "附录B 运行方法", 2)
    para(doc, "环境：Python ≥ 3.10（开发环境 Python 3.15 / scikit-learn 1.9 / pandas 3.1，Windows）。依次执行以下命令：安装依赖；复现全部实验（约3–5分钟，生成 experiments/results/*.csv 与 report/figures/fig1–fig6.png）；运行演示（生成两个会话记录与终端截图）；交互式体验 Agent；由封面模板与实验结果重新生成本报告 docx。", size=10.5)
    code_block(doc, [
        "pip install -r requirements.txt",
        "python experiments/run_experiments.py",
        "python demo/run_demo.py",
        "python -m src.run_interactive",
        "python report/build_report.py",
    ], size=9)

    heading(doc, "附录C 依赖清单", 2)
    para(doc, "即仓库根目录 requirements.txt（版本为开发验证版本，下界兼容）：", size=10.5)
    code_block(doc, [
        "scikit-learn>=1.5",
        "pandas>=2.0",
        "numpy>=1.26",
        "matplotlib>=3.8",
        "pillow>=10.0",
        "python-docx>=1.1",
    ], size=9)

    heading(doc, "附录D 测试样例", 2)
    para(doc, "排序评估的测试集为 609 个用户级正样本（每个用户时间上最后一条“喜欢”的评分，构造规则见附A.1），不参与训练；评分预测测试集 610 条。Agent 功能测试使用两个固定剧本（demo/run_demo.py 内置，输入序列驱动完整状态机并保存交互记录）：", size=10.5)
    para(doc, "剧本A（顺利流程，输出见 demo/demo_session_A.txt 与终端截图 demo_session_A.png）：", size=10.5, indent=False, bold=True)
    code_block(doc, [
        "喜欢 科幻 动作      → 解析偏好、K-Means 定位到第2组，展示群体好评榜 Top-5",
        "1 5 / 3 4 / 5 4.5  → 对3部电影打分，累计≥3部后提示可切换个性化推荐",
        "推荐               → 输出 UserCF Top-5，每部附“N位口味相近观众平均打了X分”",
        "不要恐怖 / 推荐    → 加入回避类型，重新过滤后推荐",
        "换一批 / 退出      → 翻页推荐、正常结束会话",
    ], size=9)
    para(doc, "剧本B（边界与失败处理，输出见 demo/demo_session_B.txt）：", size=10.5, indent=False, bold=True)
    code_block(doc, [
        "我随便看看 / 不知道，你们有什么推荐 / 无所谓   → 连续3次无效输入，回退大众口味并明示",
        "没看过                                           → 全部未看过 → 换一批",
        "不要爱情 / 1 2 / 推荐                            → 约束生效；评分不足3部时明示“先按群体口味推荐”",
    ], size=9)

    heading(doc, "附录E 复用组件清单与许可证/来源说明", 2)
    para(doc, "复用组件的名称、版本、许可证与用途见正文表2，来源均为官方发布渠道：scikit-learn/pandas/numpy/matplotlib/Pillow/python-docx 取自 PyPI 官方仓库，MovieLens 数据集取自 GroupLens 官网并遵循其使用条款（仅限研究教育用途）。本项目未复用任何 Agent 框架或他人密钥；KMeans/TruncatedSVD 的深入分析与行为核验见正文第四节。", size=10.5)


def main() -> None:
    doc = Document(COVER)
    set_default_style(doc)
    fill_cover(doc)
    build_body(doc)
    doc.save(OUT)
    print(f"[saved] {OUT}")


if __name__ == "__main__":
    main()
