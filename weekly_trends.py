from __future__ import annotations

import collections
import datetime as dt
import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any

import feedparser
import requests

from radar import dedupe, fetch_hf_daily_papers, fetch_hf_spaces, get_text, load_config, normalize, today_local


ROOT = Path(__file__).resolve().parent

GENERIC_TERMS = {
    "agent",
    "agents",
    "ai",
    "and the",
    "are increasingly",
    "available https",
    "available https github",
    "existing approaches",
    "existing methods",
    "extensive experiments",
    "github com",
    "has become",
    "https github",
    "https github com",
    "language model",
    "large language",
    "large language model",
    "paradigm for",
    "rather than",
    "the first",
    "the full",
    "the same",
    "these results",
    "this gap",
    "this paper",
    "this propose",
    "this study",
    "this work",
    "while preserving",
}

STOPWORDS = {
    "a",
    "about",
    "after",
    "all",
    "also",
    "an",
    "and",
    "are",
    "as",
    "at",
    "based",
    "be",
    "between",
    "by",
    "can",
    "for",
    "from",
    "has",
    "have",
    "in",
    "into",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "our",
    "than",
    "that",
    "the",
    "their",
    "these",
    "this",
    "through",
    "to",
    "using",
    "via",
    "we",
    "with",
}

DOMAIN_ANCHORS = {
    "agent",
    "agentic",
    "automation",
    "benchmark",
    "browser",
    "coding",
    "computer",
    "desktop",
    "developer",
    "evaluation",
    "function",
    "gaia",
    "glm",
    "gui",
    "kimi",
    "leaderboard",
    "manus",
    "memory",
    "mobile",
    "multi-agent",
    "observability",
    "osworld",
    "planning",
    "qwen",
    "reasoning",
    "research",
    "swe-bench",
    "terminal-bench",
    "tool",
    "trace",
    "tracing",
    "webarena",
    "workflow",
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
    text = None
    for attempt in range(3):
        try:
            text = get_text("https://export.arxiv.org/api/query", params)
            break
        except requests.exceptions.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else None
            if status == 429 and attempt < 2:
                wait_seconds = 20 * (attempt + 1)
                print(f"arXiv rate limited weekly request; retrying in {wait_seconds}s...")
                time.sleep(wait_seconds)
                continue
            print(f"arXiv weekly request failed; continuing without arXiv. Error: {exc}")
            return []
        except Exception as exc:
            print(f"arXiv weekly request failed; continuing without arXiv. Error: {exc}")
            return []

    if text is None:
        return []

    feed = feedparser.parse(text)
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


def clean_term(term: str) -> str:
    term = term.lower().replace("_", "-")
    term = re.sub(r"[^a-z0-9\-\s\.]", " ", term)
    term = re.sub(r"\s+", " ", term).strip()
    return term


def is_domain_term(term: str, seed_terms: set[str]) -> bool:
    term_l = clean_term(term)
    if not term_l or term_l in GENERIC_TERMS:
        return False
    if "http" in term_l or "github com" in term_l or "arxiv" in term_l:
        return False
    parts = term_l.split()
    if len(parts) > 5:
        return False
    if parts[0] in STOPWORDS or parts[-1] in STOPWORDS:
        return False
    if term_l in seed_terms:
        return True
    return any(anchor in term_l for anchor in DOMAIN_ANCHORS)


def extract_candidate_terms(items: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    weekly = config.get("weekly", {})
    seed_terms = {clean_term(t) for t in weekly.get("seed_terms", [])}
    counts: collections.Counter[str] = collections.Counter()
    sources_by_term: dict[str, set[str]] = collections.defaultdict(set)
    examples_by_term: dict[str, list[str]] = collections.defaultdict(list)

    for item in items:
        title = normalize(item.get("title"))
        summary = normalize(item.get("summary"))
        text_l = clean_term(f"{title} {summary}")
        seen_in_item: set[str] = set()

        for seed in seed_terms:
            if seed and seed in text_l:
                seen_in_item.add(seed)

        words = re.findall(r"[a-z][a-z0-9\-\.]+", text_l)
        for n in (2, 3, 4):
            for idx in range(0, max(0, len(words) - n + 1)):
                phrase = clean_term(" ".join(words[idx : idx + n]))
                if is_domain_term(phrase, seed_terms):
                    seen_in_item.add(phrase)

        for term in seen_in_item:
            counts[term] += 1
            sources_by_term[term].add(item.get("source", "unknown"))
            if len(examples_by_term[term]) < 3 and title:
                examples_by_term[term].append(title)

    scored = []
    for term, count in counts.items():
        if count < 2 and term not in seed_terms:
            continue
        source_diversity = len(sources_by_term[term])
        seed_boost = 3.0 if term in seed_terms else 0.0
        score = math.log1p(count) * 2.5 + source_diversity * 1.5 + seed_boost
        scored.append(
            {
                "term": term,
                "score": round(score, 2),
                "count": count,
                "sources": sorted(sources_by_term[term]),
                "examples": examples_by_term[term],
                "seed": term in seed_terms,
            }
        )

    scored.sort(key=lambda x: (x["seed"], x["score"], x["count"]), reverse=True)
    return scored[: int(weekly.get("max_terms", 50))]


def select_representative_items(items: list[dict[str, Any]], candidate_terms: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    selected = []
    term_list = [t["term"] for t in candidate_terms[:30]]
    for item in items:
        text = clean_term(f"{item.get('title', '')} {item.get('summary', '')}")
        matched = [term for term in term_list if term in text]
        if not matched:
            continue
        summary = normalize(item.get("summary"))
        if len(summary) > 500:
            summary = summary[:500].rsplit(" ", 1)[0] + "..."
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


def build_curation_prompt(
    end_day: dt.date,
    start_day: dt.date,
    candidate_terms: list[dict[str, Any]],
    representative_items: list[dict[str, Any]],
) -> list[dict[str, str]]:
    system = (
        "You are a trend curator for AI agents and AI applications. "
        "Write all final content in Simplified Chinese. "
        "Your job is to distinguish real domain trends from generic academic phrases. "
        "Use only the supplied candidate terms and representative items. "
        "Do not invent links, papers, metrics, company claims, or facts."
    )
    user = f"""
请筛选最近一周 AI agent / AI 应用领域的真实热词，并生成短 JSON。

时间范围：{start_day.isoformat()} 到 {end_day.isoformat()}

你需要：
1. 从候选词中选出真正有领域意义的趋势词。
2. 把普通论文套话、URL 片段、泛化表达放入 noise。
3. 优先关注 computer-use、coding agent、agent evaluation、tool use、agent memory、multi-agent orchestration、browser/mobile automation、AI 应用产品化、中国公司生态。
4. 不要把 "rather than"、"this work"、"github com" 这类短语当成热词。

只输出一个 JSON 对象，不要输出 Markdown，不要加解释文本。

JSON schema:
{{
  "tier1": [
    {{
      "term": "computer-use agent",
      "cat": "real-world agent",
      "why": "40字内中文理由"
    }}
  ],
  "tier2": [
    {{
      "term": "agent memory",
      "cat": "agent infra",
      "why": "40字内中文理由"
    }}
  ],
  "downrank": ["prompt collection", "beginner tutorial"],
  "noise": ["rather than", "this work"]
}}

硬性要求：
- tier1 最多 8 个。
- tier2 最多 10 个。
- downrank 最多 8 个。
- noise 最多 10 个。
- why 不超过 40 个中文字符。
- term 必须来自候选词，不要自造新词。
- 只有真正和 AI agent / AI 应用相关的词才能进入 tier1/tier2。
- 如果证据不足，不要放入 tier1。
- 所有 why 必须是简体中文。

候选词 JSON：
{json.dumps(candidate_terms, ensure_ascii=False, indent=2)}

代表性条目 JSON：
{json.dumps(representative_items, ensure_ascii=False, indent=2)}
""".strip()
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_classifier_prompt(
    end_day: dt.date,
    start_day: dt.date,
    candidate_terms: list[dict[str, Any]],
    representative_items: list[dict[str, Any]],
) -> list[dict[str, str]]:
    system = (
        "You are a strict classifier for weekly AI agent and AI application trend terms. "
        "Return valid JSON only. Use only supplied candidate terms. "
        "Do not invent terms, facts, links, papers, metrics, or company claims."
    )
    user = f"""
Classify weekly trend candidate terms for AI agent / AI application monitoring.

Range: {start_day.isoformat()} to {end_day.isoformat()}

Class definitions:
- tier1: Strong weekly trend. Use it to boost Daily Radar. Must be clearly related to AI agents or AI applications and supported by multiple sources, seed status, or strong representative items.
- tier2: Relevant watch term. Related to AI agents or AI applications, but evidence is weaker, narrower, or still emerging.
- downrank: Related but too broad or likely to over-rank low-value items in Daily Radar.
- noise: Generic phrase, URL fragment, academic filler, navigation text, or non-domain phrase.

Few-shot examples:
- "computer use" -> tier1. Note: "真实环境操作方向升温"
- "agentic search" -> tier1. Note: "多步检索型研究代理升温"
- "planning" -> downrank. Note: "相关但过泛，需结合上下文"
- "rather than" -> noise. Note: "普通英语短语"
- "github com" -> noise. Note: "URL 片段"

Output valid JSON only. Do not output Markdown.

JSON schema:
{{
  "tier1": ["computer use", "agentic search"],
  "tier2": ["agent memory"],
  "downrank": ["prompt collection", "beginner tutorial"],
  "noise": ["rather than", "this work"],
  "notes": {{
    "computer use": "真实环境操作方向升温",
    "agentic search": "多步检索型研究代理升温"
  }}
}}

Rules:
- tier1 max 8 terms.
- tier2 max 10 terms.
- downrank max 8 terms.
- noise max 10 terms.
- notes values must be Simplified Chinese, max 30 Chinese characters.
- Every term in tier1/tier2/downrank/noise must exactly match a candidate term.
- If evidence is weak, use tier2 or omit; do not put weak terms in tier1.
- Do not create new terms.

Candidate terms JSON:
{json.dumps(candidate_terms, ensure_ascii=False, indent=2)}

Representative items JSON:
{json.dumps(representative_items, ensure_ascii=False, indent=2)}
""".strip()
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def parse_json_object(text: str) -> dict[str, Any] | None:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def normalize_term_list(value: Any, allowed_terms: set[str], limit: int) -> list[str]:
    if not isinstance(value, list):
        return []
    result = []
    for entry in value:
        term = entry.get("term") if isinstance(entry, dict) else entry
        if not isinstance(term, str):
            continue
        term = clean_term(term)
        if term in allowed_terms and term not in result:
            result.append(term)
        if len(result) >= limit:
            break
    return result


def validate_curated(curated: dict[str, Any] | None, candidate_terms: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not isinstance(curated, dict):
        return None
    allowed_terms = {term["term"] for term in candidate_terms}
    tier1 = normalize_term_list(curated.get("tier1"), allowed_terms, 8)
    tier2 = [term for term in normalize_term_list(curated.get("tier2"), allowed_terms, 10) if term not in tier1]
    downrank = normalize_term_list(curated.get("downrank"), allowed_terms, 8)
    noise = normalize_term_list(curated.get("noise"), allowed_terms, 10)

    raw_notes = curated.get("notes", {})
    notes = {}
    if isinstance(raw_notes, dict):
        for key, value in raw_notes.items():
            term = clean_term(str(key))
            if term in allowed_terms and isinstance(value, str):
                notes[term] = value[:80]

    if not tier1 and not tier2:
        return None
    return {"tier1": tier1, "tier2": tier2, "downrank": downrank, "noise": noise, "notes": notes}


def call_deepseek_json(
    end_day: dt.date,
    start_day: dt.date,
    candidate_terms: list[dict[str, Any]],
    representative_items: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any] | None:
    ai_config = config.get("ai", {})
    if not ai_config.get("enabled", False) or ai_config.get("provider") != "deepseek":
        return None
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("DEEPSEEK_API_KEY is not set; weekly curated state will not be updated.")
        return None

    payload = {
        "model": ai_config.get("model", "deepseek-v4-flash"),
        "messages": build_classifier_prompt(end_day, start_day, candidate_terms, representative_items),
        "temperature": 0.1,
        "max_tokens": int(ai_config.get("max_tokens_weekly", 6000)),
        "response_format": {"type": "json_object"},
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "User-Agent": "ai-agent-radar-weekly/0.2",
    }
    base_url = str(ai_config.get("base_url", "https://api.deepseek.com")).rstrip("/")
    try:
        response = requests.post(f"{base_url}/chat/completions", headers=headers, json=payload, timeout=90)
        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"]["content"].strip()
        parsed = parse_json_object(content)
        if parsed is None:
            print("DeepSeek weekly JSON parse failed. Response head:")
            print(content[:1500])
            print("DeepSeek weekly response tail:")
            print(content[-500:])
        validated = validate_curated(parsed, candidate_terms)
        if validated is None:
            print("DeepSeek weekly JSON validation failed.")
        return validated
    except Exception as exc:
        print(f"DeepSeek weekly JSON curation failed. Error: {exc}")
        return None


def term_names(entries: list[Any]) -> list[str]:
    names = []
    for entry in entries:
        if isinstance(entry, str):
            names.append(entry)
        elif isinstance(entry, dict) and entry.get("term"):
            names.append(str(entry["term"]))
    return names


def save_trending_state(
    end_day: dt.date,
    start_day: dt.date,
    candidate_terms: list[dict[str, Any]],
    curated: dict[str, Any],
    config: dict[str, Any],
) -> None:
    weekly = config.get("weekly", {})
    state_dir = ROOT / weekly.get("state_dir", "state")
    state_dir.mkdir(parents=True, exist_ok=True)
    state_path = state_dir / "trending_terms.json"
    data = {
        "generated_at": end_day.isoformat(),
        "range": {"start": start_day.isoformat(), "end": end_day.isoformat()},
        "curated_by": config.get("ai", {}).get("model", "unknown"),
        "tier1": term_names(curated.get("tier1", [])),
        "tier2": term_names(curated.get("tier2", [])),
        "downrank": term_names(curated.get("downrank", [])),
        "noise": curated.get("noise", []),
        "curated": curated,
        "candidate_terms": candidate_terms,
    }
    state_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def save_empty_trending_state(end_day: dt.date, start_day: dt.date, config: dict[str, Any], reason: str) -> None:
    weekly = config.get("weekly", {})
    state_dir = ROOT / weekly.get("state_dir", "state")
    state_dir.mkdir(parents=True, exist_ok=True)
    state_path = state_dir / "trending_terms.json"
    data = {
        "generated_at": end_day.isoformat(),
        "range": {"start": start_day.isoformat(), "end": end_day.isoformat()},
        "curated_by": None,
        "tier1": [],
        "tier2": [],
        "downrank": [],
        "noise": [],
        "curated": {},
        "error": reason,
    }
    state_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def render_weekly_markdown(
    end_day: dt.date,
    start_day: dt.date,
    candidate_terms: list[dict[str, Any]],
    representative_items: list[dict[str, Any]],
    curated: dict[str, Any] | None,
) -> str:
    lines = [
        f"# Agent Weekly Trend Radar - {end_day.isoformat()}",
        "",
        f"范围：{start_day.isoformat()} 到 {end_day.isoformat()}",
        "",
    ]

    if curated:
        lines.extend(["## 本周结论", ""])
        tier1 = curated.get("tier1", [])
        if tier1:
            lines.append("本周可用于反哺每日雷达的核心热词已经过 AI 精筛。")
        else:
            lines.append("本周没有足够明确的 tier1 热词。")
        lines.extend(["", "## Tier 1 热词", ""])
        for entry in tier1:
            if isinstance(entry, dict):
                lines.append(f"- **{entry.get('term')}**：{entry.get('why', '证据不足')}")
            else:
                lines.append(f"- **{entry}**")
        lines.extend(["", "## Tier 2 热词", ""])
        for entry in curated.get("tier2", []):
            if isinstance(entry, dict):
                lines.append(f"- **{entry.get('term')}**：{entry.get('why', '证据不足')}")
            else:
                lines.append(f"- **{entry}**")
        lines.extend(["", "## 噪音词", ""])
        for entry in curated.get("noise", [])[:15]:
            if isinstance(entry, dict):
                lines.append(f"- {entry.get('term')}：{entry.get('reason', '')}")
            else:
                lines.append(f"- {entry}")
    else:
        lines.extend(
            [
                "## 本周候选热词",
                "",
                "DeepSeek 精筛未成功，因此没有更新 `state/trending_terms.json`。下面仅展示规则粗筛候选，不能直接视为趋势。",
                "",
            ]
        )
        for term in candidate_terms[:20]:
            lines.append(
                f"- **{term['term']}** - score {term['score']}, count {term['count']}, sources: {', '.join(term['sources'])}"
            )

    lines.extend(["", "## 代表性内容", ""])
    for item in representative_items[:15]:
        lines.append(f"- [{item['title']}]({item['url']})")
        lines.append(f"  - 来源：{item.get('source')}；命中词：{', '.join(item.get('matched_terms', []))}")
    lines.append("")
    return "\n".join(lines)


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

    candidate_terms = extract_candidate_terms(items, config)
    representative_items = select_representative_items(
        items,
        candidate_terms,
        int(weekly.get("ai_max_items", 35)),
    )
    curated = call_deepseek_json(end_day, start_day, candidate_terms, representative_items, config)

    if curated:
        save_trending_state(end_day, start_day, candidate_terms, curated, config)
    else:
        print("Writing safe empty state because AI curation did not succeed.")
        save_empty_trending_state(end_day, start_day, config, "AI curation did not succeed")

    output_dir = ROOT / weekly.get("output_dir", "weekly")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{end_day.isoformat()}.md"
    output_path.write_text(
        render_weekly_markdown(end_day, start_day, candidate_terms, representative_items, curated),
        encoding="utf-8",
    )
    print(f"Wrote {output_path} with {len(candidate_terms)} candidate terms from {len(items)} items")


if __name__ == "__main__":
    main()
