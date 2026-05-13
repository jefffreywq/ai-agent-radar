from __future__ import annotations

import collections
import datetime as dt
import json
import math
import os
import re
from pathlib import Path
from typing import Any

import feedparser
import requests

from radar import dedupe, fetch_hf_daily_papers, fetch_hf_spaces, get_text, load_config, normalize, today_local


ROOT = Path(__file__).resolve().parent

STOPWORDS = {
    "about",
    "after",
    "agent",
    "agents",
    "based",
    "benchmark",
    "benchmarks",
    "between",
    "from",
    "into",
    "large",
    "language",
    "learning",
    "model",
    "models",
    "multi",
    "paper",
    "reasoning",
    "system",
    "systems",
    "that",
    "their",
    "through",
    "using",
    "with",
}


def parse_arxiv_date(value: str | None) -> dt.date | None:
    if not value:
        return None
    try:
        return dt.datetime.fromisoformat(value.replace("Z", "+00:00")).date()
    except ValueError:
        return None


def weekly_arxiv_query(config: dict[str, Any]) -> str:
    keywords = config.get("keywords", [])
    cats = config.get("arxiv_categories", [])
    kw = " OR ".join([f'all:"{k}"' if " " in k else f"all:{k}" for k in keywords])
    cat_query = " OR ".join([f"cat:{c}" for c in cats])
    return f"({kw}) AND ({cat_query})"


def fetch_recent_arxiv(config: dict[str, Any], start_day: dt.date) -> list[dict[str, Any]]:
    weekly = config.get("weekly", {})
    params = {
        "search_query": weekly_arxiv_query(config),
        "start": 0,
        "max_results": int(weekly.get("max_arxiv_results", 80)),
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    feed = feedparser.parse(get_text("https://export.arxiv.org/api/query", params))
    items = []
    for entry in feed.entries:
        published = parse_arxiv_date(entry.get("published"))
        if published and published < start_day:
            continue
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


def fetch_weekly_hf_papers(config: dict[str, Any], start_day: dt.date, end_day: dt.date) -> list[dict[str, Any]]:
    weekly = config.get("weekly", {})
    original_limit = config.get("max_items_per_source", 12)
    config["max_items_per_source"] = int(weekly.get("max_hf_papers_per_day", 20))
    items = []
    day = start_day
    while day <= end_day:
        try:
            for item in fetch_hf_daily_papers(config, day):
                item["published"] = day.isoformat()
                items.append(item)
        except Exception as exc:
            print(f"HF daily papers failed for {day}: {exc}")
        day += dt.timedelta(days=1)
    config["max_items_per_source"] = original_limit
    return items


def fetch_weekly_hf_spaces(config: dict[str, Any]) -> list[dict[str, Any]]:
    weekly = config.get("weekly", {})
    original_limit = config.get("max_items_per_source", 12)
    config["max_items_per_source"] = int(weekly.get("max_hf_spaces_per_query", 12))
    try:
        return fetch_hf_spaces(config)
    finally:
        config["max_items_per_source"] = original_limit


def term_candidates(text: str, seed_terms: list[str]) -> list[str]:
    text_l = text.lower()
    terms = []
    for term in seed_terms:
        if term.lower() in text_l:
            terms.append(term)

    words = re.findall(r"[a-z][a-z0-9\-]+", text_l)
    filtered = [w for w in words if len(w) > 2 and w not in STOPWORDS]
    for n in (2, 3):
        for idx in range(0, max(0, len(filtered) - n + 1)):
            phrase = " ".join(filtered[idx : idx + n])
            if any(word in STOPWORDS for word in phrase.split()):
                continue
            terms.append(phrase)
    return terms


def extract_trending_terms(items: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    weekly = config.get("weekly", {})
    seed_terms = weekly.get("seed_terms", [])
    counts: collections.Counter[str] = collections.Counter()
    sources_by_term: dict[str, set[str]] = collections.defaultdict(set)

    for item in items:
        text = f"{item.get('title', '')} {item.get('summary', '')}"
        seen_in_item = set(term_candidates(text, seed_terms))
        for term in seen_in_item:
            counts[term] += 1
            sources_by_term[term].add(item.get("source", "unknown"))

    downrank = set(t.lower() for t in config.get("ranking", {}).get("downrank_terms", []))
    scored = []
    for term, count in counts.items():
        if count < 2 and term not in seed_terms:
            continue
        source_diversity = len(sources_by_term[term])
        seed_boost = 1.5 if term in seed_terms else 0.0
        penalty = 2.0 if term.lower() in downrank else 0.0
        score = math.log1p(count) * 3.0 + source_diversity * 1.2 + seed_boost - penalty
        scored.append(
            {
                "term": term,
                "score": round(score, 2),
                "count": count,
                "sources": sorted(sources_by_term[term]),
            }
        )

    scored.sort(key=lambda x: (x["score"], x["count"]), reverse=True)
    return scored[: int(weekly.get("max_terms", 30))]


def select_representative_items(items: list[dict[str, Any]], trending_terms: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    selected = []
    term_list = [t["term"].lower() for t in trending_terms[:20]]
    for item in items:
        text = f"{item.get('title', '')} {item.get('summary', '')}".lower()
        matched = [term for term in term_list if term in text]
        if not matched:
            continue
        summary = normalize(item.get("summary"))
        if len(summary) > 900:
            summary = summary[:900].rsplit(" ", 1)[0] + "..."
        selected.append(
            {
                "title": item.get("title"),
                "source": item.get("source"),
                "url": item.get("url"),
                "published": item.get("published"),
                "authors": item.get("authors"),
                "matched_terms": matched[:5],
                "summary": summary,
            }
        )
        if len(selected) >= limit:
            break
    return selected


def build_weekly_ai_prompt(
    end_day: dt.date,
    start_day: dt.date,
    trending_terms: list[dict[str, Any]],
    representative_items: list[dict[str, Any]],
) -> list[dict[str, str]]:
    system = (
        "You are a trend analyst for AI agents and multi-agent systems. "
        "Write entirely in Simplified Chinese. Be selective and skeptical. "
        "Use only the supplied terms and representative items. Do not invent news, links, metrics, or company claims."
    )
    user = f"""
请生成一份简体中文 Markdown 周报。

时间范围：{start_day.isoformat()} 到 {end_day.isoformat()}

必须使用下面的中文结构：

# Agent Weekly Trend Radar - {end_day.isoformat()}

## 本周结论
用 3-5 句中文说明本周 agent / multi-agent 方向的主要变化。

## 本周升温热词
列出 8-12 个热词。每个热词包含：
- 热度判断：高 / 中 / 低
- 为什么升温：基于输入证据解释
- 应该如何关注：给出一句建议

## 本周值得关注的方向
把热词归并成 3-5 个方向，例如 computer-use、coding agent、evaluation、multi-agent orchestration、observability。

## 噪音和过热信号
指出哪些词可能只是泛化营销、教程、合集或标题党。

## 代表性内容
列出最能代表本周趋势的条目，保留 URL。

## 下周搜索建议
给出下一周应该加入或提高权重的搜索词。

硬性要求：
- 全文必须使用简体中文。
- 不要编造输入中没有的事实。
- 如果证据不足，写“证据不足”。
- 不要声称你已经阅读全文。

热词 JSON：
{json.dumps(trending_terms, ensure_ascii=False, indent=2)}

代表性条目 JSON：
{json.dumps(representative_items, ensure_ascii=False, indent=2)}
""".strip()
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def call_deepseek_weekly(
    end_day: dt.date,
    start_day: dt.date,
    trending_terms: list[dict[str, Any]],
    representative_items: list[dict[str, Any]],
    config: dict[str, Any],
) -> str | None:
    ai_config = config.get("ai", {})
    if not ai_config.get("enabled", False) or ai_config.get("provider") != "deepseek":
        return None
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("DEEPSEEK_API_KEY is not set; using rule-based weekly Markdown.")
        return None

    payload = {
        "model": ai_config.get("model", "deepseek-v4-flash"),
        "messages": build_weekly_ai_prompt(end_day, start_day, trending_terms, representative_items),
        "temperature": 0.2,
        "max_tokens": 3500,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "User-Agent": "ai-agent-radar-weekly/0.1",
    }
    base_url = str(ai_config.get("base_url", "https://api.deepseek.com")).rstrip("/")
    try:
        response = requests.post(f"{base_url}/chat/completions", headers=headers, json=payload, timeout=90)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception as exc:
        print(f"DeepSeek weekly call failed; using rule-based Markdown. Error: {exc}")
        return None


def render_weekly_markdown(
    end_day: dt.date,
    start_day: dt.date,
    trending_terms: list[dict[str, Any]],
    representative_items: list[dict[str, Any]],
) -> str:
    lines = [
        f"# Agent Weekly Trend Radar - {end_day.isoformat()}",
        "",
        f"Range: {start_day.isoformat()} to {end_day.isoformat()}",
        "",
        "## 本周升温热词",
        "",
    ]
    for idx, term in enumerate(trending_terms[:20], 1):
        lines.append(
            f"{idx}. **{term['term']}** - score {term['score']}, count {term['count']}, sources: {', '.join(term['sources'])}"
        )
    lines.extend(["", "## 代表性内容", ""])
    for item in representative_items[:15]:
        lines.append(f"- [{item['title']}]({item['url']})")
        lines.append(f"  - Source: {item.get('source')}; Terms: {', '.join(item.get('matched_terms', []))}")
    lines.extend(["", "## 下周搜索建议", ""])
    for term in trending_terms[:10]:
        lines.append(f"- {term['term']}")
    lines.append("")
    return "\n".join(lines)


def save_trending_state(end_day: dt.date, start_day: dt.date, trending_terms: list[dict[str, Any]], config: dict[str, Any]) -> None:
    weekly = config.get("weekly", {})
    state_dir = ROOT / weekly.get("state_dir", "state")
    state_dir.mkdir(parents=True, exist_ok=True)
    state_path = state_dir / "trending_terms.json"
    data = {
        "generated_at": end_day.isoformat(),
        "range": {"start": start_day.isoformat(), "end": end_day.isoformat()},
        "tier1": [term["term"] for term in trending_terms[:10]],
        "tier2": [term["term"] for term in trending_terms[10:25]],
        "terms": trending_terms,
    }
    state_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    config = load_config()
    weekly = config.get("weekly", {})
    end_day = today_local()
    start_day = end_day - dt.timedelta(days=int(weekly.get("lookback_days", 7)) - 1)

    items = []
    items.extend(fetch_weekly_hf_papers(config, start_day, end_day))
    items.extend(fetch_weekly_hf_spaces(config))
    items.extend(fetch_recent_arxiv(config, start_day))
    items = dedupe(items)

    trending_terms = extract_trending_terms(items, config)
    representative_items = select_representative_items(
        items,
        trending_terms,
        int(weekly.get("ai_max_items", 35)),
    )

    save_trending_state(end_day, start_day, trending_terms, config)

    output_dir = ROOT / weekly.get("output_dir", "weekly")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{end_day.isoformat()}.md"

    ai_markdown = call_deepseek_weekly(end_day, start_day, trending_terms, representative_items, config)
    output_path.write_text(
        ai_markdown or render_weekly_markdown(end_day, start_day, trending_terms, representative_items),
        encoding="utf-8",
    )
    print(f"Wrote {output_path} with {len(trending_terms)} terms from {len(items)} items")


if __name__ == "__main__":
    main()

