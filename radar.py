from __future__ import annotations

import datetime as dt
import html
import json
import os
import re
from pathlib import Path
from typing import Any

import feedparser
import requests
import yaml
from bs4 import BeautifulSoup


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config.yaml"
STATE_PATH = ROOT / "state.json"
TRENDING_TERMS_PATH = ROOT / "state" / "trending_terms.json"


def load_config() -> dict[str, Any]:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_state() -> dict[str, Any]:
    if not STATE_PATH.exists():
        return {"seen": []}
    with STATE_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_state(state: dict[str, Any]) -> None:
    with STATE_PATH.open("w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
        f.write("\n")


def load_trending_terms() -> dict[str, Any]:
    if not TRENDING_TERMS_PATH.exists():
        return {"tier1": [], "tier2": [], "terms": []}
    try:
        with TRENDING_TERMS_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        print(f"Failed to load weekly trending terms; continuing without trend boost. Error: {exc}")
        return {"tier1": [], "tier2": [], "terms": []}


def today_local() -> dt.date:
    # GitHub runners have zoneinfo available on Python 3.11+.
    try:
        from zoneinfo import ZoneInfo

        tz = ZoneInfo(load_config().get("timezone", "UTC"))
        return dt.datetime.now(tz).date()
    except Exception:
        return dt.date.today()


def get_json(url: str, params: dict[str, Any] | None = None) -> Any:
    headers = {"User-Agent": "ai-agent-radar/0.1"}
    resp = requests.get(url, params=params, headers=headers, timeout=30)
    resp.raise_for_status()
    return resp.json()


def get_text(url: str, params: dict[str, Any] | None = None) -> str:
    headers = {"User-Agent": "ai-agent-radar/0.1"}
    resp = requests.get(url, params=params, headers=headers, timeout=30)
    resp.raise_for_status()
    return resp.text


def normalize(text: str | None) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", html.unescape(text)).strip()


def item_id(item: dict[str, Any]) -> str:
    return f"{item.get('source')}::{item.get('url') or item.get('title')}"


def score_item(item: dict[str, Any], config: dict[str, Any], trends: dict[str, Any] | None = None) -> tuple[float, list[str]]:
    title = normalize(item.get("title")).lower()
    summary = normalize(item.get("summary")).lower()
    text = f"{title} {summary}"
    score = 0.0
    reasons: list[str] = []

    for kw in config["keywords"]:
        if kw.lower() in text:
            score += 1.2
            reasons.append(f"keyword: {kw}")

    for term in config["high_value_terms"]:
        if term.lower() in text:
            score += 1.8
            reasons.append(f"high-value: {term}")

    for term in config.get("ranking", {}).get("downrank_terms", []):
        if term.lower() in text:
            score -= 1.5
            reasons.append(f"downrank: {term}")

    trend_config = config.get("ranking", {}).get("weekly_trend_boost", {})
    if trend_config.get("enabled", True):
        trends = trends or {"tier1": [], "tier2": [], "terms": []}
        max_total = float(trend_config.get("max_total", 4.0))
        trend_boost = 0.0
        generic_terms = {"agent", "agents", "ai", "model", "models", "benchmark", "framework"}
        for term in trends.get("tier1", []):
            term_l = str(term).lower()
            if term_l in generic_terms:
                continue
            if term_l in text and trend_boost < max_total:
                add = min(float(trend_config.get("tier1", 2.0)), max_total - trend_boost)
                score += add
                trend_boost += add
                reasons.append(f"weekly-trend tier1: {term}")
        for term in trends.get("tier2", []):
            term_l = str(term).lower()
            if term_l in generic_terms:
                continue
            if term_l in text and trend_boost < max_total:
                add = min(float(trend_config.get("tier2", 1.0)), max_total - trend_boost)
                score += add
                trend_boost += add
                reasons.append(f"weekly-trend tier2: {term}")

    source = item.get("source", "")
    if source == "hf_daily_papers":
        score += 2.5
        reasons.append("appeared on HF Daily Papers")
    if source == "hf_space":
        score += 1.8
        reasons.append("Hugging Face Space")
    if source == "hf_competition":
        score += 2.0
        reasons.append("Hugging Face competition")
    if source == "arxiv":
        score += 1.0
        reasons.append("new arXiv paper")

    likes = item.get("likes") or 0
    downloads = item.get("downloads") or 0
    if likes:
        score += min(float(likes) / 50.0, 2.0)
        reasons.append(f"HF likes: {likes}")
    if downloads:
        score += min(float(downloads) / 10000.0, 2.0)
        reasons.append(f"HF downloads: {downloads}")

    cited_by = item.get("cited_by_count") or 0
    if cited_by:
        score += min(float(cited_by) / 25.0, 2.0)
        reasons.append(f"OpenAlex citations: {cited_by}")

    return round(max(0.0, min(score, 10.0)), 1), reasons[:6]


def compact_for_ai(item: dict[str, Any], max_summary_chars: int) -> dict[str, Any]:
    summary = normalize(item.get("summary"))
    if len(summary) > max_summary_chars:
        summary = summary[:max_summary_chars].rsplit(" ", 1)[0] + "..."
    return {
        "title": item.get("title"),
        "source": item.get("source"),
        "url": item.get("url"),
        "score": item.get("score"),
        "reasons": item.get("reasons", []),
        "authors": item.get("authors"),
        "published": item.get("published"),
        "summary": summary,
    }


def build_ai_prompt(day: dt.date, items: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, str]]:
    ai_config = config.get("ai", {})
    compact_items = [
        compact_for_ai(item, int(ai_config.get("max_summary_chars_per_item", 1200)))
        for item in items[: int(ai_config.get("max_items", 15))]
    ]
    system = (
        "You are a research assistant for AI agents and multi-agent systems. "
        "Your job is to curate a daily research radar for a technical user. "
        "Prioritize benchmarks, leaderboards, evaluations, datasets, open-source frameworks, "
        "coding agents, web agents, computer-use agents, tool use, planning, memory, and multi-agent collaboration. "
        "Use only the supplied items and URLs. Do not invent links, papers, authors, or facts. "
        "If evidence is weak, say so."
    )
    user = f"""
请根据下面的候选条目，生成一份中文 Markdown 日报。

日期：{day.isoformat()}
语言：{ai_config.get("language", "zh-CN")}

输出结构必须是：

# AI Agent Radar - {day.isoformat()}

## 今日结论
用 3-5 句话概括今天最值得注意的趋势。

## 必看 Top 5
每条包含：
- 类型：paper / benchmark / leaderboard / competition / framework / dataset / space / other
- 重要性：A / B / C
- 为什么重要：
- 和 agent / multi-agent 的关系：
- 建议动作：
- 链接：

## 值得扫一眼
列出次重要条目，说明一句理由。

## 可以跳过或低优先级
列出看起来相关但价值较低的条目，并解释为什么。

## 今日概念
提取 3-6 个值得后续关注的概念。

## 后续行动
给出 3-5 个待办事项。

硬性要求：
- 不要编造输入中没有的链接。
- 不要声称已经阅读全文。
- 如果摘要不足以判断，请标注“证据不足”。
- 保留原始 URL。

候选条目 JSON：
{json.dumps(compact_items, ensure_ascii=False, indent=2)}
""".strip()
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_ai_prompt_v2(day: dt.date, items: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, str]]:
    ai_config = config.get("ai", {})
    compact_items = [
        compact_for_ai(item, int(ai_config.get("max_summary_chars_per_item", 1200)))
        for item in items[: int(ai_config.get("max_items", 15))]
    ]
    system = (
        "You are a skeptical research assistant for AI agents and multi-agent systems. "
        "Do not treat every new item as useful. Curate selectively for a technical user. "
        "Prioritize benchmarks, leaderboards, evaluations, datasets, open-source frameworks, "
        "coding agents, web agents, computer-use agents, tool use, planning, memory, "
        "long-horizon tasks, and multi-agent collaboration. "
        "Use only the supplied items and URLs. Do not invent links, papers, authors, dates, metrics, or facts. "
        "If evidence is weak, say so explicitly."
    )
    user = f"""
Generate a Markdown daily digest in Simplified Chinese.

# AI Agent Radar - {day.isoformat()}

Required structure:

## Today conclusion
Write 3-5 concise Chinese sentences about the most important signals today.

## Must-read Top 5
For each useful item, include:
- Basic info: type, source, authors if available, date if available, URL
- Priority: A / B / C
- Why it matters: explain the practical research value
- Innovation point: infer only from the supplied title/summary; if unclear, write "evidence insufficient"
- Agent / multi-agent relevance: explain the connection
- Recommended action: what the user should do next

## Worth scanning
List secondary items with one-line Chinese reasons.

## Low priority or skippable
List items that look new but are probably not useful enough, and explain why.

## Background knowledge you should know
Write one compact Chinese primer that helps the user understand today's most important items.
Focus on one foundational concept, benchmark family, or evaluation method that appears in the candidates.
Keep it practical: definition, why it matters, how to judge good work in this area.

## Concepts to track
Extract 3-6 Chinese concept bullets.

## Follow-up actions
Give 3-5 actionable todos.

Hard requirements:
- Output Simplified Chinese Markdown.
- Do not invent links, papers, authors, dates, claims, or metrics.
- Do not claim you read full papers.
- If the evidence is weak, explicitly say "证据不足".
- Keep original URLs.
- Be selective: not every new item deserves attention.

Candidate items JSON:
{json.dumps(compact_items, ensure_ascii=False, indent=2)}
""".strip()
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_ai_prompt_zh(day: dt.date, items: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, str]]:
    ai_config = config.get("ai", {})
    compact_items = [
        compact_for_ai(item, int(ai_config.get("max_summary_chars_per_item", 1200)))
        for item in items[: int(ai_config.get("max_items", 15))]
    ]
    system = (
        "You are a skeptical research assistant for AI agents and multi-agent systems. "
        "Your job is to curate a daily research radar for a technical user. "
        "Do not treat every new item as useful. Be selective, practical, and evidence-based. "
        "Prioritize agent benchmarks, leaderboards, evaluation methods, datasets, open-source frameworks, "
        "coding agents, web agents, computer-use agents, tool use, planning and memory, long-horizon tasks, "
        "and multi-agent collaboration. "
        "Use only the supplied candidate items and URLs. Do not invent links, papers, authors, dates, metrics, claims, or project status. "
        'If the title and summary are not enough to judge an item, explicitly say "证据不足". '
        "Important output rule: The final answer must be written entirely in Simplified Chinese. "
        "Section headings, bullet labels, explanations, priority judgments, and todos must all be in Simplified Chinese. "
        'Do not output English section headings such as "Today conclusion", "Must-read Top 5", or "Follow-up actions".'
    )
    user = f"""
请根据下面的候选条目，生成一份简体中文 Markdown 日报。

日期：{day.isoformat()}

你必须严格使用下面的中文结构和中文标题：

# AI Agent Radar - {day.isoformat()}

## 今日结论
用 3-5 句中文总结今天最值得注意的变化。不要泛泛而谈，要指出哪些方向更值得关注，哪些只是噪音。

## 必看 Top 5
只选择真正值得看的条目，不一定非要凑满 5 条。

每条必须包含：

### 1. 标题
- 基础信息：类型、来源、作者、日期、链接。如果输入里没有作者或日期，写“未提供”。
- 优先级：A / B / C
- 为什么重要：说明它的实际研究价值或工程价值。
- 可能的创新点：只能基于标题和摘要判断；如果信息不足，写“证据不足”。
- 和 agent / multi-agent 的关系：说明它为什么属于或影响 agent / multi-agent 方向。
- 建议动作：告诉我下一步应该精读、扫一眼、收藏、跟踪，还是暂时跳过。

## 值得扫一眼
列出次重要条目。每条用一到两句中文说明为什么值得快速看。

## 低优先级或可跳过
列出看起来相关但价值较低的条目，并说明原因。比如：
- 只是关键词相关，但和 agent 关系弱
- 更像教程、营销、合集
- 缺少 benchmark、代码、实验或清晰贡献
- 摘要信息不足

## 你需要知道的基础知识
补充一段中文基础知识，帮助我理解今天最重要的内容。

要求：
- 选择一个今天候选条目里反复出现、或者对理解 Top 5 最关键的概念。
- 解释它是什么。
- 解释它为什么重要。
- 解释如何判断这个方向的工作是否有价值。
- 不要写成百科，要和今天的候选内容相关。

## 值得追踪的概念
提取 3-6 个中文概念，并用一句话解释每个概念为什么值得追踪。

## 后续行动
给出 3-5 个中文待办事项，格式使用 Markdown checkbox。

硬性要求：
- 全文必须使用简体中文。
- Markdown 标题必须使用上面给出的中文标题。
- 不要输出英文标题。
- 不要编造输入中没有的链接、作者、日期、指标或事实。
- 不要声称你已经阅读全文。
- 如果证据不足，明确写“证据不足”。
- 保留每个重要条目的原始 URL。
- 保持选择性：不是所有新内容都有价值。
- 如果今天没有足够有价值的内容，要直接说“今天高价值新增内容较少”。

候选条目 JSON：
{json.dumps(compact_items, ensure_ascii=False, indent=2)}
""".strip()
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def call_deepseek(day: dt.date, items: list[dict[str, Any]], config: dict[str, Any]) -> str | None:
    ai_config = config.get("ai", {})
    if not ai_config.get("enabled", False):
        return None
    if ai_config.get("provider") != "deepseek":
        return None

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("DEEPSEEK_API_KEY is not set; using rule-based Markdown.")
        return None

    if not items:
        return None

    base_url = str(ai_config.get("base_url", "https://api.deepseek.com")).rstrip("/")
    model = ai_config.get("model", "deepseek-v4-flash")
    payload = {
        "model": model,
        "messages": build_ai_prompt_zh(day, items, config),
        "temperature": 0.2,
        "max_tokens": 3500,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "User-Agent": "ai-agent-radar/0.2",
    }

    try:
        response = requests.post(f"{base_url}/chat/completions", headers=headers, json=payload, timeout=90)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception as exc:
        print(f"DeepSeek call failed; using rule-based Markdown. Error: {exc}")
        return None


def fetch_hf_daily_papers(config: dict[str, Any], day: dt.date) -> list[dict[str, Any]]:
    data = get_json("https://huggingface.co/api/daily_papers", {"date": day.isoformat()})
    papers = data if isinstance(data, list) else data.get("papers", [])
    items = []
    for p in papers[: config["max_items_per_source"]]:
        paper = p.get("paper") if isinstance(p, dict) and "paper" in p else p
        title = normalize(paper.get("title") or paper.get("paperTitle"))
        paper_id = paper.get("id") or paper.get("arxivId") or paper.get("paperId")
        summary = normalize(paper.get("summary") or paper.get("abstract"))
        if not title:
            continue
        url = f"https://huggingface.co/papers/{paper_id}" if paper_id else "https://huggingface.co/papers"
        items.append({"source": "hf_daily_papers", "title": title, "summary": summary, "url": url})
    return items


def fetch_hf_spaces(config: dict[str, Any]) -> list[dict[str, Any]]:
    queries = config.get("search", {}).get(
        "hf_space_queries",
        ["agent leaderboard", "multi-agent benchmark", "web agent", "agent challenge"],
    )
    items: list[dict[str, Any]] = []
    for q in queries:
        try:
            data = get_json(
                "https://huggingface.co/api/spaces",
                {"search": q, "sort": "lastModified", "direction": "-1", "limit": 8, "full": "true"},
            )
        except Exception:
            continue
        for s in data if isinstance(data, list) else []:
            sid = s.get("id")
            if not sid:
                continue
            items.append(
                {
                    "source": "hf_space",
                    "title": sid,
                    "summary": normalize(s.get("cardData", {}).get("title") if isinstance(s.get("cardData"), dict) else ""),
                    "url": f"https://huggingface.co/spaces/{sid}",
                    "likes": s.get("likes"),
                    "downloads": s.get("downloads"),
                }
            )
    return dedupe(items)


def fetch_hf_competitions(config: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        text = get_text("https://huggingface.co/competitions")
    except Exception:
        return []
    soup = BeautifulSoup(text, "html.parser")
    rows = soup.find_all("tr")
    items: list[dict[str, Any]] = []
    for row in rows[1 : config["max_items_per_source"] + 1]:
        cells = [normalize(c.get_text(" ")) for c in row.find_all(["td", "th"])]
        if not cells or len(cells) < 2:
            continue
        title = cells[0]
        summary = " | ".join(cells[1:])
        if any(kw.lower() in f"{title} {summary}".lower() for kw in config["keywords"]):
            items.append(
                {
                    "source": "hf_competition",
                    "title": title,
                    "summary": summary,
                    "url": "https://huggingface.co/competitions",
                }
            )
    return items


def arxiv_query(config: dict[str, Any]) -> str:
    kw = " OR ".join([f'all:"{k}"' if " " in k else f"all:{k}" for k in config["keywords"]])
    cats = " OR ".join([f"cat:{c}" for c in config["arxiv_categories"]])
    return f"({kw}) AND ({cats})"


def fetch_arxiv(config: dict[str, Any]) -> list[dict[str, Any]]:
    url = "https://export.arxiv.org/api/query"
    params = {
        "search_query": arxiv_query(config),
        "start": 0,
        "max_results": config["max_items_per_source"],
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    feed = feedparser.parse(get_text(url, params))
    items = []
    for entry in feed.entries:
        items.append(
            {
                "source": "arxiv",
                "title": normalize(entry.get("title")),
                "summary": normalize(entry.get("summary")),
                "url": entry.get("link"),
                "published": entry.get("published"),
                "authors": ", ".join(a.get("name", "") for a in entry.get("authors", [])),
            }
        )
    return items


def enrich_with_openalex(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for item in items:
        if item.get("source") != "arxiv":
            continue
        title = item.get("title")
        if not title:
            continue
        try:
            data = get_json("https://api.openalex.org/works", {"search": title, "per-page": 1})
            results = data.get("results", [])
            if results:
                item["cited_by_count"] = results[0].get("cited_by_count", 0)
                item["openalex_url"] = results[0].get("id")
        except Exception:
            pass
    return items


def dedupe(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    result = []
    for item in items:
        key = item_id(item)
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def render_markdown(day: dt.date, items: list[dict[str, Any]]) -> str:
    lines = [
        f"# AI Agent Radar - {day.isoformat()}",
        "",
        "## Today worth checking",
        "",
    ]
    if not items:
        lines.append("No matching items found today.")
        lines.append("")
        return "\n".join(lines)

    for idx, item in enumerate(items, 1):
        lines.append(f"### {idx}. {item['title']}")
        lines.append(f"- Score: {item['score']}/10")
        lines.append(f"- Source: {item['source']}")
        if item.get("authors"):
            lines.append(f"- Authors: {item['authors']}")
        lines.append(f"- Link: {item['url']}")
        if item.get("reasons"):
            lines.append(f"- Why: {', '.join(item['reasons'])}")
        if item.get("summary"):
            summary = item["summary"]
            if len(summary) > 700:
                summary = summary[:700].rsplit(" ", 1)[0] + "..."
            lines.append("")
            lines.append(summary)
        lines.append("")

    lines.extend(
        [
            "## Follow-up",
            "",
            "- [ ] Open the top 3 links",
            "- [ ] Save any strong benchmark or leaderboard to the long-term watch list",
            "- [ ] Mark papers worth reading deeply",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    config = load_config()
    day = today_local()
    state = load_state()
    trends = load_trending_terms()
    seen = set(state.get("seen", []))

    items = []
    for fetcher in [fetch_hf_daily_papers, fetch_hf_spaces, fetch_hf_competitions, fetch_arxiv]:
        try:
            if fetcher.__name__ == "fetch_hf_daily_papers":
                items.extend(fetcher(config, day))
            else:
                items.extend(fetcher(config))
        except Exception as exc:
            items.append(
                {
                    "source": "system",
                    "title": f"{fetcher.__name__} failed",
                    "summary": str(exc),
                    "url": "",
                }
            )

    items = enrich_with_openalex(dedupe(items))
    fresh = [item for item in items if item_id(item) not in seen and item.get("source") != "system"]

    for item in fresh:
        item["score"], item["reasons"] = score_item(item, config, trends)

    fresh.sort(key=lambda x: x.get("score", 0), reverse=True)
    digest_items = fresh[: config["max_digest_items"]]

    output_dir = ROOT / config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{day.isoformat()}.md"

    if not digest_items and output_path.exists():
        print(f"No new items; keeping existing {output_path}")
    else:
        ai_markdown = call_deepseek(day, digest_items, config)
        output_path.write_text(ai_markdown or render_markdown(day, digest_items), encoding="utf-8")

    state["seen"] = sorted((seen | {item_id(i) for i in fresh if i.get("url")}) )[-2000:]
    save_state(state)

    print(f"Wrote {output_path} with {len(digest_items)} items")


if __name__ == "__main__":
    main()
