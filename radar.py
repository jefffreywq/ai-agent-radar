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
    try:
        from zoneinfo import ZoneInfo

        tz = ZoneInfo(load_config().get("timezone", "UTC"))
        return dt.datetime.now(tz).date()
    except Exception:
        return dt.date.today()


def get_json(url: str, params: dict[str, Any] | None = None) -> Any:
    headers = {"User-Agent": "ai-agent-radar/0.3"}
    resp = requests.get(url, params=params, headers=headers, timeout=30)
    resp.raise_for_status()
    return resp.json()


def get_text(url: str, params: dict[str, Any] | None = None) -> str:
    headers = {"User-Agent": "ai-agent-radar/0.3"}
    resp = requests.get(url, params=params, headers=headers, timeout=30)
    resp.raise_for_status()
    return resp.text


def normalize(text: str | None) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", html.unescape(text)).strip()


def item_id(item: dict[str, Any]) -> str:
    return f"{item.get('source')}::{item.get('url') or item.get('title')}"


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


def apply_weekly_trend_boost(text: str, config: dict[str, Any], trends: dict[str, Any], reasons: list[str]) -> float:
    trend_config = config.get("ranking", {}).get("weekly_trend_boost", {})
    if not trend_config.get("enabled", True):
        return 0.0

    max_total = float(trend_config.get("max_total", 4.0))
    generic_terms = {"agent", "agents", "ai", "model", "models", "benchmark", "framework"}
    boost = 0.0

    for tier_name, default_weight in (("tier1", 2.0), ("tier2", 1.0)):
        weight = float(trend_config.get(tier_name, default_weight))
        for term in trends.get(tier_name, []):
            term_l = str(term).lower()
            if term_l in generic_terms:
                continue
            if term_l in text and boost < max_total:
                add = min(weight, max_total - boost)
                boost += add
                reasons.append(f"weekly-trend {tier_name}: {term}")
    return boost


def apply_weekly_trend_penalty(text: str, trends: dict[str, Any], reasons: list[str]) -> float:
    protected_terms = {
        "agentic search",
        "computer use",
        "computer use agent",
        "coding agent",
        "browser automation",
        "mobile agent",
        "research agent",
        "agent memory",
        "agentic workflow",
        "gui agent",
        "swe-bench",
        "osworld",
        "gaia",
        "manus",
        "deepseek",
        "qwen",
        "kimi",
        "glm",
        "doubao",
        "hunyuan",
    }
    penalty = 0.0
    for term in trends.get("downrank", []):
        term_l = str(term).lower().strip()
        if not term_l or term_l in protected_terms:
            continue
        if term_l in text:
            penalty += 1.0
            reasons.append(f"weekly-downrank: {term}")
    return min(penalty, 3.0)


def score_item(item: dict[str, Any], config: dict[str, Any], trends: dict[str, Any] | None = None) -> tuple[float, list[str]]:
    text = f"{normalize(item.get('title')).lower()} {normalize(item.get('summary')).lower()}"
    score = 0.0
    reasons: list[str] = []

    for kw in config.get("keywords", []):
        if kw.lower() in text:
            score += 1.2
            reasons.append(f"keyword: {kw}")

    for term in config.get("high_value_terms", []):
        if term.lower() in text:
            score += 1.8
            reasons.append(f"high-value: {term}")

    for term in config.get("ranking", {}).get("downrank_terms", []):
        if term.lower() in text:
            score -= 1.5
            reasons.append(f"downrank: {term}")

    score += apply_weekly_trend_boost(text, config, trends or {}, reasons)
    score -= apply_weekly_trend_penalty(text, trends or {}, reasons)

    source_bonus = {
        "hf_daily_papers": (2.5, "appeared on HF Daily Papers"),
        "hf_space": (1.8, "Hugging Face Space"),
        "hf_competition": (2.0, "Hugging Face competition"),
        "arxiv": (1.0, "new arXiv paper"),
    }
    bonus = source_bonus.get(item.get("source", ""))
    if bonus:
        score += bonus[0]
        reasons.append(bonus[1])

    likes = item.get("likes") or 0
    downloads = item.get("downloads") or 0
    cited_by = item.get("cited_by_count") or 0
    if likes:
        score += min(float(likes) / 50.0, 2.0)
        reasons.append(f"HF likes: {likes}")
    if downloads:
        score += min(float(downloads) / 10000.0, 2.0)
        reasons.append(f"HF downloads: {downloads}")
    if cited_by:
        score += min(float(cited_by) / 25.0, 2.0)
        reasons.append(f"OpenAlex citations: {cited_by}")

    return round(max(0.0, min(score, 10.0)), 1), reasons[:6]


def compact_for_ai(item: dict[str, Any], max_summary_chars: int) -> dict[str, Any]:
    summary = normalize(item.get("summary"))
    if len(summary) > max_summary_chars:
        summary = summary[:max_summary_chars].rsplit(" ", 1)[0] + "..."
    return {
        "id": item.get("id"),
        "title": item.get("title"),
        "source": item.get("source"),
        "url": item.get("url"),
        "score": item.get("score"),
        "reasons": item.get("reasons", []),
        "authors": item.get("authors"),
        "published": item.get("published"),
        "summary": summary,
    }


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


def build_daily_classifier_prompt(day: dt.date, items: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, str]]:
    ai_config = config.get("ai", {})
    compact_items = [
        compact_for_ai(item, int(ai_config.get("max_summary_chars_per_item", 1200)))
        for item in items[: int(ai_config.get("max_items", 15))]
    ]
    system = (
        "You are a strict classifier for daily AI agent and multi-agent research radar items. "
        "Return valid JSON only. Use only supplied item ids. "
        "Do not invent papers, links, authors, dates, metrics, claims, or ids."
    )
    user = f"""
Classify today's candidate items for an AI agent / AI application research radar.

Date: {day.isoformat()}

Class definitions:
- must_read: High-value item worth reading today. It should be clearly related to AI agents, AI applications, benchmarks, coding agents, computer-use agents, tool use, memory, evaluation, or multi-agent systems.
- scan: Relevant item worth a quick look, but not urgent or evidence is narrower.
- skip: Low-value, weakly related, generic, marketing-like, duplicated, or evidence-insufficient item.

Few-shot examples:
- ToolCUA / computer-use agent benchmark -> must_read
- New agentic search paper with retrieval + tool use -> must_read
- Generic planning paper without clear agent evaluation -> scan
- Prompt collection or generic tutorial -> skip
- Space/demo with no description or weak evidence -> scan or skip

Output valid JSON only. Do not output Markdown.

JSON schema:
{{
  "must_read": ["item_1", "item_3"],
  "scan": ["item_4"],
  "skip": ["item_7"],
  "notes": {{
    "item_1": "40字内中文理由",
    "item_3": "40字内中文理由"
  }},
  "background": "80字内中文基础知识"
}}

Rules:
- must_read max 5 ids.
- scan max 8 ids.
- skip max 10 ids.
- notes values must be Simplified Chinese, max 40 Chinese characters.
- background must be Simplified Chinese, max 80 Chinese characters.
- Every id must exactly match an input item id.
- If evidence is weak, use scan or skip; do not put weak items in must_read.

Candidate items JSON:
{json.dumps(compact_items, ensure_ascii=False, indent=2)}
""".strip()
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def normalize_id_list(value: Any, allowed_ids: set[str], limit: int) -> list[str]:
    if not isinstance(value, list):
        return []
    result = []
    for entry in value:
        item_id_value = entry.get("id") if isinstance(entry, dict) else entry
        if not isinstance(item_id_value, str):
            continue
        if item_id_value in allowed_ids and item_id_value not in result:
            result.append(item_id_value)
        if len(result) >= limit:
            break
    return result


def validate_daily_classification(curated: dict[str, Any] | None, items: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not isinstance(curated, dict):
        return None
    allowed_ids = {item["id"] for item in items if item.get("id")}
    must_read = normalize_id_list(curated.get("must_read"), allowed_ids, 5)
    scan = [item_id for item_id in normalize_id_list(curated.get("scan"), allowed_ids, 8) if item_id not in must_read]
    skip = [
        item_id
        for item_id in normalize_id_list(curated.get("skip"), allowed_ids, 10)
        if item_id not in must_read and item_id not in scan
    ]
    notes = {}
    raw_notes = curated.get("notes", {})
    if isinstance(raw_notes, dict):
        for key, value in raw_notes.items():
            if key in allowed_ids and isinstance(value, str):
                notes[key] = value[:100]
    background = curated.get("background", "")
    if not isinstance(background, str):
        background = ""
    if not must_read and not scan:
        return None
    return {"must_read": must_read, "scan": scan, "skip": skip, "notes": notes, "background": background[:160]}


def build_deep_read_prompt(day: dt.date, items: list[dict[str, Any]], curated: dict[str, Any], config: dict[str, Any]) -> list[dict[str, str]]:
    ai_config = config.get("ai", {})
    by_id = {item["id"]: item for item in items if item.get("id")}
    must_read_ids = curated.get("must_read", [])[: int(ai_config.get("deep_read_max_items", 5))]
    deep_items = [
        compact_for_ai(by_id[item_id], int(ai_config.get("deep_read_summary_chars_per_item", 1800)))
        for item_id in must_read_ids
        if item_id in by_id
    ]
    system = (
        "You are a careful technical research analyst for AI agents, AI applications, and multi-agent systems. "
        "Return valid JSON only. Use only the supplied item ids and evidence. "
        "Do not invent details, experiments, metrics, authors, dates, links, or claims. "
        "If the abstract/summary is insufficient, say evidence is insufficient in Chinese."
    )
    user = f"""
Create deep-read notes for today's must-read AI agent radar items.

Date: {day.isoformat()}

For each item, explain only what can be supported by its title, summary, source, URL, score, and rule reasons.
Write all values in Simplified Chinese. Keep it concise and useful.

Return exactly this JSON shape:
{{
  "deep_reads": [
    {{
      "id": "item_1",
      "type": "论文 / 工具 / benchmark / 产品 / 公司动态 / 数据集",
      "priority": "高 / 中 / 低",
      "one_liner": "一句话说明它是什么",
      "problem": "它解决什么问题",
      "innovation": ["创新点1", "创新点2", "创新点3"],
      "why_it_matters": "为什么对 AI agent / AI 应用重要",
      "background": "读懂它需要知道的基础知识",
      "follow_up": "后续值得追踪什么"
    }}
  ]
}}

Rules:
- Include only supplied ids.
- deep_reads length must be <= {int(ai_config.get("deep_read_max_items", 5))}.
- one_liner <= 60 Chinese characters.
- problem, why_it_matters, background, follow_up <= 90 Chinese characters each.
- innovation must contain 1-3 short Chinese bullets.
- priority must be 高, 中, or 低.
- If evidence is insufficient, explicitly write "证据不足" in the relevant field.

Must-read items JSON:
{json.dumps(deep_items, ensure_ascii=False, indent=2)}
""".strip()
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def clean_deep_read_text(value: Any, limit: int) -> str:
    if not isinstance(value, str):
        return ""
    return normalize(value)[:limit]


def validate_deep_reads(data: dict[str, Any] | None, items: list[dict[str, Any]], curated: dict[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(data, dict):
        return []
    allowed_ids = set(curated.get("must_read", [])) & {item["id"] for item in items if item.get("id")}
    raw_reads = data.get("deep_reads", [])
    if not isinstance(raw_reads, list):
        return []
    result = []
    seen: set[str] = set()
    for raw in raw_reads:
        if not isinstance(raw, dict):
            continue
        item_id_value = raw.get("id")
        if item_id_value not in allowed_ids or item_id_value in seen:
            continue
        seen.add(item_id_value)
        raw_innovation = raw.get("innovation", [])
        innovation = []
        if isinstance(raw_innovation, list):
            for entry in raw_innovation[:3]:
                text = clean_deep_read_text(entry, 80)
                if text:
                    innovation.append(text)
        priority = clean_deep_read_text(raw.get("priority"), 4)
        if priority not in {"高", "中", "低"}:
            priority = "中"
        result.append(
            {
                "id": item_id_value,
                "type": clean_deep_read_text(raw.get("type"), 20) or "未分类",
                "priority": priority,
                "one_liner": clean_deep_read_text(raw.get("one_liner"), 120),
                "problem": clean_deep_read_text(raw.get("problem"), 180),
                "innovation": innovation,
                "why_it_matters": clean_deep_read_text(raw.get("why_it_matters"), 180),
                "background": clean_deep_read_text(raw.get("background"), 180),
                "follow_up": clean_deep_read_text(raw.get("follow_up"), 180),
            }
        )
    return result


def call_deepseek_deep_read(
    day: dt.date,
    items: list[dict[str, Any]],
    curated: dict[str, Any],
    config: dict[str, Any],
    base_url: str,
    headers: dict[str, str],
) -> list[dict[str, Any]]:
    if not curated.get("must_read"):
        return []
    ai_config = config.get("ai", {})
    payload = {
        "model": ai_config.get("model", "deepseek-v4-flash"),
        "messages": build_deep_read_prompt(day, items, curated, config),
        "temperature": 0.1,
        "max_tokens": int(ai_config.get("max_tokens_deep_read_daily", 4500)),
        "response_format": {"type": "json_object"},
    }
    try:
        response = requests.post(f"{base_url}/chat/completions", headers=headers, json=payload, timeout=90)
        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"]["content"].strip()
        deep_reads = validate_deep_reads(parse_json_object(content), items, curated)
        if not deep_reads:
            print("DeepSeek deep-read JSON was empty or invalid; continuing without deep reads.")
        return deep_reads
    except Exception as exc:
        print(f"DeepSeek deep-read call failed; continuing without deep reads. Error: {exc}")
        return []


def build_ai_prompt(day: dt.date, items: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, str]]:
    ai_config = config.get("ai", {})
    compact_items = [
        compact_for_ai(item, int(ai_config.get("max_summary_chars_per_item", 1200)))
        for item in items[: int(ai_config.get("max_items", 15))]
    ]
    system = (
        "You are a skeptical research assistant for AI agents and multi-agent systems. "
        "Your job is to curate a daily research radar for a technical user. "
        "Do not treat every new item as useful. Be selective, practical, and evidence-based. "
        "Use only the supplied candidate items and URLs. Do not invent links, papers, authors, dates, metrics, claims, or project status. "
        'If the title and summary are not enough to judge an item, explicitly say "证据不足". '
        "The final answer must be written entirely in Simplified Chinese."
    )
    user = f"""
请根据下面的候选条目，生成一份简体中文 Markdown 日报。

日期：{day.isoformat()}

你必须严格使用下面的中文结构和中文标题：

# AI Agent Radar - {day.isoformat()}

## 今日结论
用 3-5 句中文总结今天最值得注意的变化。不要泛泛而谈，要指出哪些方向更值得关注，哪些只是噪音。

## 必看 Top 5
只选择真正值得看的条目，不一定非要凑满 5 条。每条必须包含：

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
列出看起来相关但价值较低的条目，并说明原因，例如：只是关键词相关、缺少 benchmark/代码/实验、摘要信息不足，或更像教程/营销/合集。

## 你需要知道的基础知识
补充一段中文基础知识，帮助我理解今天最重要的内容。选择一个今天候选条目里反复出现、或对理解 Top 5 最关键的概念，解释它是什么、为什么重要、如何判断这个方向的工作是否有价值。

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
    if not ai_config.get("enabled", False) or ai_config.get("provider") != "deepseek" or not items:
        return None

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("DEEPSEEK_API_KEY is not set; using rule-based Markdown.")
        return None

    base_url = str(ai_config.get("base_url", "https://api.deepseek.com")).rstrip("/")
    payload = {
        "model": ai_config.get("model", "deepseek-v4-flash"),
        "messages": build_daily_classifier_prompt(day, items, config),
        "temperature": 0.1,
        "max_tokens": int(ai_config.get("max_tokens_daily", 3500)),
        "response_format": {"type": "json_object"},
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "User-Agent": "ai-agent-radar/0.3",
    }

    try:
        response = requests.post(f"{base_url}/chat/completions", headers=headers, json=payload, timeout=90)
        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"]["content"].strip()
        parsed = parse_json_object(content)
        if parsed is None:
            print("DeepSeek daily JSON parse failed. Response head:")
            print(content[:1200])
            return None
        curated = validate_daily_classification(parsed, items)
        if curated is None:
            print("DeepSeek daily JSON validation failed.")
            return None
        deep_reads = call_deepseek_deep_read(day, items, curated, config, base_url, headers)
        return render_ai_daily_markdown(day, items, curated, deep_reads)
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
    for query in queries:
        try:
            data = get_json(
                "https://huggingface.co/api/spaces",
                {"search": query, "sort": "lastModified", "direction": "-1", "limit": 8, "full": "true"},
            )
        except Exception:
            continue
        for space in data if isinstance(data, list) else []:
            sid = space.get("id")
            if not sid:
                continue
            card_data = space.get("cardData") if isinstance(space.get("cardData"), dict) else {}
            items.append(
                {
                    "source": "hf_space",
                    "title": sid,
                    "summary": normalize(card_data.get("title")),
                    "url": f"https://huggingface.co/spaces/{sid}",
                    "likes": space.get("likes"),
                    "downloads": space.get("downloads"),
                }
            )
    return dedupe(items)


def fetch_hf_competitions(config: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        text = get_text("https://huggingface.co/competitions")
    except Exception:
        return []
    soup = BeautifulSoup(text, "html.parser")
    items: list[dict[str, Any]] = []
    for row in soup.find_all("tr")[1 : config["max_items_per_source"] + 1]:
        cells = [normalize(c.get_text(" ")) for c in row.find_all(["td", "th"])]
        if len(cells) < 2:
            continue
        title = cells[0]
        summary = " | ".join(cells[1:])
        if any(kw.lower() in f"{title} {summary}".lower() for kw in config.get("keywords", [])):
            items.append({"source": "hf_competition", "title": title, "summary": summary, "url": "https://huggingface.co/competitions"})
    return items


def fetch_arxiv(config: dict[str, Any]) -> list[dict[str, Any]]:
    keywords = config.get("keywords", [])
    categories = config.get("arxiv_categories", [])
    kw = " OR ".join([f'all:"{k}"' if " " in k else f"all:{k}" for k in keywords])
    cats = " OR ".join([f"cat:{c}" for c in categories])
    params = {
        "search_query": f"({kw}) AND ({cats})",
        "start": 0,
        "max_results": config["max_items_per_source"],
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    feed = feedparser.parse(get_text("https://export.arxiv.org/api/query", params))
    return [
        {
            "source": "arxiv",
            "title": normalize(entry.get("title")),
            "summary": normalize(entry.get("summary")),
            "url": entry.get("link"),
            "published": entry.get("published"),
            "authors": ", ".join(author.get("name", "") for author in entry.get("authors", [])),
        }
        for entry in feed.entries
    ]


def enrich_with_openalex(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for item in items:
        if item.get("source") != "arxiv" or not item.get("title"):
            continue
        try:
            data = get_json("https://api.openalex.org/works", {"search": item["title"], "per-page": 1})
            results = data.get("results", [])
            if results:
                item["cited_by_count"] = results[0].get("cited_by_count", 0)
                item["openalex_url"] = results[0].get("id")
        except Exception:
            pass
    return items


def render_markdown(day: dt.date, items: list[dict[str, Any]]) -> str:
    lines = [f"# AI Agent Radar - {day.isoformat()}", "", "## Today worth checking", ""]
    if not items:
        return "\n".join(lines + ["No matching items found today.", ""])

    for idx, item in enumerate(items, 1):
        lines.extend(
            [
                f"### {idx}. {item['title']}",
                f"- Score: {item['score']}/10",
                f"- Source: {item['source']}",
                f"- Link: {item['url']}",
            ]
        )
        if item.get("authors"):
            lines.insert(-1, f"- Authors: {item['authors']}")
        if item.get("reasons"):
            lines.append(f"- Why: {', '.join(item['reasons'])}")
        if item.get("summary"):
            summary = item["summary"]
            if len(summary) > 700:
                summary = summary[:700].rsplit(" ", 1)[0] + "..."
            lines.extend(["", summary])
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


def render_item_block(item: dict[str, Any], note: str | None = None) -> list[str]:
    lines = [f"### {item['title']}"]
    lines.append(f"- 来源：{item.get('source')}")
    if item.get("authors"):
        lines.append(f"- 作者：{item.get('authors')}")
    if item.get("published"):
        lines.append(f"- 日期：{item.get('published')}")
    lines.append(f"- 链接：{item.get('url')}")
    lines.append(f"- 规则分：{item.get('score')}/10")
    if item.get("reasons"):
        lines.append(f"- 规则命中：{', '.join(item.get('reasons', []))}")
    if note:
        lines.append(f"- AI 判断：{note}")
    return lines


def render_deep_read_block(item: dict[str, Any], read: dict[str, Any]) -> list[str]:
    lines = [
        f"### {item['title']}",
        f"- 类型：{read.get('type') or '未分类'}",
        f"- 优先级：{read.get('priority') or '中'}",
        f"- 链接：{item.get('url')}",
    ]
    if read.get("one_liner"):
        lines.append(f"- 一句话：{read['one_liner']}")
    if read.get("problem"):
        lines.extend(["", "**它解决什么问题**", "", read["problem"]])
    innovation = read.get("innovation") or []
    if innovation:
        lines.extend(["", "**核心创新**", ""])
        for idx, point in enumerate(innovation, 1):
            lines.append(f"{idx}. {point}")
    if read.get("why_it_matters"):
        lines.extend(["", "**为什么重要**", "", read["why_it_matters"]])
    if read.get("background"):
        lines.extend(["", "**需要知道的基础知识**", "", read["background"]])
    if read.get("follow_up"):
        lines.extend(["", "**后续追踪**", "", read["follow_up"]])
    return lines


def render_ai_daily_markdown(
    day: dt.date,
    items: list[dict[str, Any]],
    curated: dict[str, Any],
    deep_reads: list[dict[str, Any]] | None = None,
) -> str:
    by_id = {item["id"]: item for item in items}
    notes = curated.get("notes", {})
    deep_by_id = {read["id"]: read for read in deep_reads or [] if read.get("id")}
    lines = [
        f"# AI Agent Radar - {day.isoformat()}",
        "",
        "## 今日结论",
        "",
    ]
    must_read = [by_id[item_id] for item_id in curated.get("must_read", []) if item_id in by_id]
    scan = [by_id[item_id] for item_id in curated.get("scan", []) if item_id in by_id]
    skip = [by_id[item_id] for item_id in curated.get("skip", []) if item_id in by_id]

    if must_read:
        lines.append(f"今日有 {len(must_read)} 个高价值 agent / AI 应用相关条目，优先查看必看列表。")
    else:
        lines.append("今天高价值新增内容较少，可快速扫一眼相关条目。")
    lines.append("")

    lines.extend(["## 精读摘要", ""])
    deep_read_items = [
        by_id[item_id]
        for item_id in curated.get("must_read", [])
        if item_id in by_id and item_id in deep_by_id
    ]
    if deep_read_items:
        for item in deep_read_items:
            lines.extend(render_deep_read_block(item, deep_by_id[item["id"]]))
            lines.append("")
    else:
        lines.append("暂无精读摘要。")
        lines.append("")

    lines.extend(["## 必看 Top 5", ""])
    if must_read:
        for item in must_read:
            lines.extend(render_item_block(item, notes.get(item["id"])))
            lines.append("")
    else:
        lines.append("暂无明确必看条目。")
        lines.append("")

    lines.extend(["## 值得扫一眼", ""])
    if scan:
        for item in scan:
            lines.extend(render_item_block(item, notes.get(item["id"])))
            lines.append("")
    else:
        lines.append("暂无。")
        lines.append("")

    lines.extend(["## 低优先级或可跳过", ""])
    if skip:
        for item in skip:
            lines.append(f"- {item['title']}：{notes.get(item['id'], '相关性或证据不足')}")
    else:
        lines.append("暂无。")
    lines.append("")

    lines.extend(["## 你需要知道的基础知识", "", curated.get("background") or "暂无需要额外补充的基础知识。", ""])
    lines.extend(
        [
            "## 后续行动",
            "",
            "- [ ] 打开必看条目的原文链接",
            "- [ ] 判断是否加入长期关注 benchmark / leaderboard 列表",
            "- [ ] 将有复现价值的代码或 Space 单独收藏",
            "",
        ]
    )
    return "\n".join(lines)


def collect_items(config: dict[str, Any], day: dt.date) -> list[dict[str, Any]]:
    items = []
    for fetcher in [fetch_hf_daily_papers, fetch_hf_spaces, fetch_hf_competitions, fetch_arxiv]:
        try:
            if fetcher is fetch_hf_daily_papers:
                items.extend(fetcher(config, day))
            else:
                items.extend(fetcher(config))
        except Exception as exc:
            print(f"{fetcher.__name__} failed: {exc}")
    return enrich_with_openalex(dedupe(items))


def main() -> None:
    config = load_config()
    day = today_local()
    state = load_state()
    trends = load_trending_terms()
    seen = set(state.get("seen", []))

    fresh = [item for item in collect_items(config, day) if item_id(item) not in seen]
    for item in fresh:
        item["score"], item["reasons"] = score_item(item, config, trends)

    fresh.sort(key=lambda x: x.get("score", 0), reverse=True)
    digest_items = fresh[: config["max_digest_items"]]
    for idx, item in enumerate(digest_items, 1):
        item["id"] = f"item_{idx}"

    output_dir = ROOT / config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{day.isoformat()}.md"

    if not digest_items and output_path.exists():
        print(f"No new items; keeping existing {output_path}")
    else:
        ai_markdown = call_deepseek(day, digest_items, config)
        output_path.write_text(ai_markdown or render_markdown(day, digest_items), encoding="utf-8")

    state["seen"] = sorted(seen | {item_id(item) for item in fresh if item.get("url")})[-2000:]
    save_state(state)
    print(f"Wrote {output_path} with {len(digest_items)} items")


if __name__ == "__main__":
    main()
