from __future__ import annotations

import datetime as dt
import json
import os
import re
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

import feedparser
import requests
from bs4 import BeautifulSoup

from radar import get_text, load_config, normalize, today_local


ROOT = Path(__file__).resolve().parent
COMPANY_STATE_PATH = ROOT / "state" / "company_seen.json"

NOISE_TEXT_TERMS = {
    "about us",
    "business",
    "careers",
    "changelog",
    "community",
    "contact",
    "cookie",
    "customer stories",
    "download",
    "download press kit",
    "enterprise",
    "facebook",
    "foundation",
    "linkedin",
    "login",
    "press kit",
    "privacy",
    "products",
    "research",
    "resources",
    "sign in",
    "skip to content",
    "skip to footer",
    "skip to main content",
    "solutions",
    "subscribe",
    "support",
    "terms",
    "try chatgpt",
    "try claude",
    "try meta ai",
    "try studio",
    "use cases",
}

NOISE_URL_PARTS = {
    "#",
    "about",
    "apply",
    "careers",
    "contact",
    "cookie",
    "customer-stories",
    "events",
    "facebook.com",
    "footer",
    "linkedin.com",
    "login",
    "mailto:",
    "mokahr.com",
    "press-kit",
    "privacy",
    "share",
    "signin",
    "signup",
    "support",
    "terms",
    "twitter.com",
    "weibo.com",
}

DYNAMIC_URL_PARTS = {
    "agent",
    "api",
    "blog",
    "changelog",
    "copilot",
    "developer",
    "docs",
    "feature",
    "features",
    "kimi",
    "llm",
    "model",
    "models",
    "news",
    "open-source",
    "product",
    "release",
    "research",
}

DYNAMIC_TITLE_TERMS = {
    "agent",
    "agentic",
    "api",
    "assistant",
    "automation",
    "browser",
    "chat",
    "claude",
    "code",
    "coding",
    "computer",
    "copilot",
    "deepseek",
    "developer",
    "doubao",
    "function calling",
    "glm",
    "hunyuan",
    "kimi",
    "llama",
    "manus",
    "memory",
    "minimax",
    "model",
    "open source",
    "qwen",
    "release",
    "research agent",
    "tool",
    "workflow",
}


def fetch_rss_source(source: dict[str, Any], limit: int) -> list[dict[str, Any]]:
    feed = feedparser.parse(get_text(source["url"]))
    items = []
    for entry in feed.entries[:limit]:
        items.append(
            {
                "company": source["name"],
                "region": source.get("region"),
                "source_type": "rss",
                "title": normalize(entry.get("title")),
                "summary": normalize(entry.get("summary") or entry.get("description")),
                "url": entry.get("link"),
                "published": entry.get("published") or entry.get("updated"),
            }
        )
    return [item for item in items if item["title"] and item["url"]]


def decode_response(response: requests.Response) -> str:
    if not response.encoding or response.encoding.lower() == "iso-8859-1":
        response.encoding = response.apparent_encoding
    return response.text


def is_noise_link(title: str, url: str) -> bool:
    title_l = normalize(title).lower()
    url_l = url.lower()
    if not title_l or len(title_l) < 6 or len(title_l) > 220:
        return True
    if "@" in title_l or url_l.startswith("mailto:"):
        return True
    if "icp" in title_l or "备案" in title_l or "公安" in title_l:
        return True
    if title_l in NOISE_TEXT_TERMS:
        return True
    if any(part in url_l for part in NOISE_URL_PARTS):
        if not any(part in url_l for part in ("blog", "news", "changelog", "release")):
            return True
    return False


def looks_dynamic(title: str, url: str) -> bool:
    title_l = normalize(title).lower()
    url_l = url.lower()
    if any(term in title_l for term in DYNAMIC_TITLE_TERMS):
        return True
    return any(part in url_l for part in DYNAMIC_URL_PARTS)


def fetch_page_source(source: dict[str, Any], limit: int) -> list[dict[str, Any]]:
    # Company homepages are navigation-heavy; keep only links that look like
    # concrete product, model, API, feature, news, or changelog updates.
    headers = {"User-Agent": "ai-agent-radar-company/0.2"}
    response = requests.get(source["url"], headers=headers, timeout=30)
    response.raise_for_status()
    soup = BeautifulSoup(decode_response(response), "html.parser")

    items = []
    seen: set[str] = set()
    for anchor in soup.find_all("a"):
        title = normalize(anchor.get_text(" "))
        href = anchor.get("href")
        if not href:
            continue
        url = urljoin(source["url"], href)
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            continue
        if url in seen or is_noise_link(title, url) or not looks_dynamic(title, url):
            continue
        seen.add(url)
        items.append(
            {
                "company": source["name"],
                "region": source.get("region"),
                "source_type": "page",
                "title": title,
                "summary": "",
                "url": url,
                "published": extract_date_from_text(title) or extract_date_from_text(url),
            }
        )
        if len(items) >= limit:
            break
    return items


def extract_date_from_text(text: str | None) -> str | None:
    if not text:
        return None
    match = re.search(r"(20\d{2})[-/\.](\d{1,2})[-/\.](\d{1,2})", text)
    if not match:
        return None
    year, month, day = match.groups()
    return f"{int(year):04d}-{int(month):02d}-{int(day):02d}"


def company_item_id(item: dict[str, Any]) -> str:
    return str(item.get("url") or f"{item.get('company')}::{item.get('title')}")


def load_company_seen() -> dict[str, str]:
    if not COMPANY_STATE_PATH.exists():
        return {}
    try:
        data = json.loads(COMPANY_STATE_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"Failed to load company seen state; continuing fresh. Error: {exc}")
        return {}
    seen = data.get("seen", data) if isinstance(data, dict) else {}
    return seen if isinstance(seen, dict) else {}


def save_company_seen(seen: dict[str, str]) -> None:
    COMPANY_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    COMPANY_STATE_PATH.write_text(json.dumps({"seen": seen}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def prune_company_seen(seen: dict[str, str], day: dt.date, ttl_days: int) -> dict[str, str]:
    keep: dict[str, str] = {}
    for key, value in seen.items():
        try:
            seen_day = dt.date.fromisoformat(str(value))
        except ValueError:
            continue
        if (day - seen_day).days < ttl_days:
            keep[key] = str(value)
    return keep


def filter_seen_company_items(items: list[dict[str, Any]], seen: dict[str, str]) -> list[dict[str, Any]]:
    return [item for item in items if company_item_id(item) not in seen]


def fetch_company_items(config: dict[str, Any]) -> list[dict[str, Any]]:
    company_config = config.get("company", {})
    limit = int(company_config.get("max_items_per_source", 8))
    items = []
    for region, sources in company_config.get("sources", {}).items():
        for source in sources:
            source = dict(source)
            source["region"] = region
            try:
                if source.get("type") == "rss":
                    items.extend(fetch_rss_source(source, limit))
                else:
                    items.extend(fetch_page_source(source, limit))
            except Exception as exc:
                print(f"Company source failed: {source.get('name')} ({source.get('url')}): {exc}")
    return dedupe_company_items(items)


def dedupe_company_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    result = []
    for item in items:
        key = item.get("url") or f"{item.get('company')}::{item.get('title')}"
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def score_company_item(item: dict[str, Any], config: dict[str, Any]) -> tuple[float, list[str]]:
    company_config = config.get("company", {})
    text = f"{item.get('title', '')} {item.get('summary', '')} {item.get('url', '')}".lower()
    score = 0.0
    reasons: list[str] = []

    for term in company_config.get("focus_terms", []):
        if term.lower() in text:
            score += 1.2
            reasons.append(f"focus: {term}")

    if looks_dynamic(str(item.get("title", "")), str(item.get("url", ""))):
        score += 1.0
        reasons.append("dynamic-looking link")

    company = str(item.get("company", "")).lower()
    strategic_companies = [
        "anthropic",
        "bytedance",
        "cursor",
        "deepmind",
        "deepseek",
        "github",
        "hunyuan",
        "manus",
        "microsoft",
        "minimax",
        "moonshot",
        "openai",
        "qwen",
        "zhipu",
    ]
    if any(name in company for name in strategic_companies):
        score += 0.8
        reasons.append("strategic company")

    if item.get("region") == "china":
        score += 0.6
        reasons.append("china ecosystem")

    return round(min(score, 10.0), 1), reasons[:6]


def compact_items(items: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    company_config = config.get("company", {})
    max_chars = int(company_config.get("max_summary_chars_per_item", 900))
    max_items = int(company_config.get("max_items_for_ai", 40))
    compact = []
    for item in items[:max_items]:
        summary = normalize(item.get("summary"))
        if len(summary) > max_chars:
            summary = summary[:max_chars].rsplit(" ", 1)[0] + "..."
        compact.append(
            {
                "id": item.get("id"),
                "company": item.get("company"),
                "region": item.get("region"),
                "title": item.get("title"),
                "summary": summary,
                "url": item.get("url"),
                "published": item.get("published"),
                "score": item.get("score"),
                "reasons": item.get("reasons", []),
            }
        )
    return compact


def parse_json_object(text: str) -> dict[str, Any] | None:
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        repaired = text.strip()
        if repaired.startswith("{"):
            repaired = repaired.rstrip().rstrip(",")
            repaired += "}" * max(0, repaired.count("{") - repaired.count("}"))
            try:
                data = json.loads(repaired)
                return data if isinstance(data, dict) else None
            except json.JSONDecodeError:
                pass
        match = re.search(r"\{.*\}", text, re.S)
        if not match:
            return None
        try:
            data = json.loads(match.group(0))
            return data if isinstance(data, dict) else None
        except json.JSONDecodeError:
            return None


def build_company_classifier_prompt(day: dt.date, items: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, str]]:
    system = (
        "You are a strict AI company intelligence classifier. Return compact valid JSON only. "
        "Use only supplied ids. Do not invent announcements, dates, features, links, or claims."
    )
    user = f"""
Date: {day.isoformat()}

Classify the supplied company candidates into compact buckets:
- global_important: global company updates worth tracking.
- china_important: China ecosystem updates worth tracking.
- watch: relevant but weaker or uncertain.

Selection rules:
- Prefer concrete announcements, releases, changelogs, benchmark/product capabilities, open-source models, APIs, coding agents, browser/computer/mobile agents, workflow automation, and enterprise agent/application moves.
- Downrank generic pages such as "Try ChatGPT", "Research", "Business", "Products", "Download", "Contact", "Careers", "Publication", ICP/license pages, and social links.
- Manus is a company/product name; classify it only if the item is a concrete capability or announcement, not because the word appears in unrelated text.
- Keep global_important <= 4, china_important <= 4, watch <= 4.
- Every id must exactly match an input id. Do not create new ids.
- Notes are optional. Only include notes for important ids. Each note <= 20 Chinese characters.

Return exactly this JSON shape:
{{
  "global_important": ["item_1"],
  "china_important": ["item_2"],
  "watch": ["item_3"],
  "notes": {{
    "item_1": "中文短理由"
  }}
}}

Few-shot guidance:
- "press@anthropic.com", "Skip to content", "Try Claude", "Careers" => omit.
- "Cursor changelog: agent/code review feature" => global_important or watch.
- "Kimi Agent Swarm", "Qwen agent/API/model release", "DeepSeek model/API release" => china_important when evidence is concrete.
- "OpenAI API generic page" => watch only if it indicates a new/changed capability; otherwise omit.

Company candidates JSON:
{json.dumps(compact_items(items, config), ensure_ascii=False, indent=2)}
""".strip()
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def normalize_id_list(value: Any, allowed: set[str], limit: int) -> list[str]:
    if limit <= 0:
        return []
    if not isinstance(value, list):
        return []
    seen: set[str] = set()
    result: list[str] = []
    for raw in value:
        item_id_value = str(raw).strip()
        if item_id_value in allowed and item_id_value not in seen:
            seen.add(item_id_value)
            result.append(item_id_value)
        if len(result) >= limit:
            break
    return result


def validate_company_classification(data: dict[str, Any] | None, items: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not data:
        return None
    allowed = {str(item.get("id")) for item in items if item.get("id")}
    if not allowed:
        return None
    curated = {
        "global_important": normalize_id_list(data.get("global_important"), allowed, 4),
        "china_important": normalize_id_list(data.get("china_important"), allowed, 4),
        "watch": normalize_id_list(data.get("watch"), allowed, 4),
        "noise": normalize_id_list(data.get("noise"), allowed, 0),
        "notes": {},
    }
    assigned = set(curated["global_important"]) | set(curated["china_important"])
    curated["watch"] = [item_id_value for item_id_value in curated["watch"] if item_id_value not in assigned]
    assigned |= set(curated["watch"])
    curated["noise"] = [item_id_value for item_id_value in curated["noise"] if item_id_value not in assigned]

    notes = data.get("notes", {})
    if isinstance(notes, dict):
        for item_id_value, note in notes.items():
            item_id_value = str(item_id_value).strip()
            if item_id_value in allowed and isinstance(note, str):
                curated["notes"][item_id_value] = normalize(note)[:40]

    if not curated["global_important"] and not curated["china_important"] and not curated["watch"]:
        return None
    return curated


def call_deepseek_company(day: dt.date, items: list[dict[str, Any]], config: dict[str, Any]) -> str | None:
    ai_config = config.get("ai", {})
    if not ai_config.get("enabled", False) or ai_config.get("provider") != "deepseek":
        return None
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("DEEPSEEK_API_KEY is not set; using rule-based company Markdown.")
        return None

    payload = {
        "model": ai_config.get("model", "deepseek-v4-flash"),
        "messages": build_company_classifier_prompt(day, items, config),
        "temperature": 0.1,
        "response_format": {"type": "json_object"},
        "max_tokens": int(ai_config.get("max_tokens_company", 3500)),
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "User-Agent": "ai-agent-radar-company/0.2",
    }
    base_url = str(ai_config.get("base_url", "https://api.deepseek.com")).rstrip("/")
    try:
        response = requests.post(f"{base_url}/chat/completions", headers=headers, json=payload, timeout=90)
        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"]["content"].strip()
        curated = validate_company_classification(parse_json_object(content), items)
        if not curated:
            print("DeepSeek company JSON was empty or invalid; using rule-based Markdown.")
            return None
        return render_ai_company_markdown(day, items, curated)
    except Exception as exc:
        print(f"DeepSeek company call failed; using rule-based Markdown. Error: {exc}")
        return None


def item_by_id(items: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(item.get("id")): item for item in items if item.get("id")}


def render_item_lines(item: dict[str, Any], note: str | None = None) -> list[str]:
    lines = [
        f"### {item.get('company')}: {item.get('title')}",
        f"- 区域：{item.get('region')}",
        f"- 分数：{item.get('score')}/10",
        f"- 链接：{item.get('url')}",
    ]
    if item.get("published"):
        lines.append(f"- 日期：{item.get('published')}")
    if note:
        lines.append(f"- 判断：{note}")
    elif item.get("reasons"):
        lines.append(f"- 规则理由：{', '.join(item.get('reasons', []))}")
    return lines


def render_ai_company_markdown(day: dt.date, items: list[dict[str, Any]], curated: dict[str, Any]) -> str:
    by_id = item_by_id(items)
    notes = curated.get("notes", {})
    global_ids = curated.get("global_important", [])
    china_ids = curated.get("china_important", [])
    watch_ids = curated.get("watch", [])
    noise_ids = curated.get("noise", [])

    lines = [
        f"# AI Company Radar - {day.isoformat()}",
        "",
        "## 今日结论",
        "",
    ]
    if global_ids or china_ids:
        total = len(global_ids) + len(china_ids)
        lines.append(f"DeepSeek 已从候选动态中筛出 {total} 条重点公司动态，并保留低置信度内容供观察。")
    else:
        lines.append("今天没有筛出高置信度的重点公司动态，建议只扫一眼观察项。")
    lines.append("")

    sections = [
        ("## 全球公司重点动态", global_ids, "暂无高价值新增动态。"),
        ("## 中国公司重点动态", china_ids, "暂无高价值新增动态。"),
        ("## 值得观察", watch_ids, "暂无观察项。"),
    ]
    for title, ids, empty_text in sections:
        lines.extend([title, ""])
        if not ids:
            lines.extend([empty_text, ""])
            continue
        for item_id_value in ids:
            item = by_id.get(item_id_value)
            if not item:
                continue
            lines.extend(render_item_lines(item, notes.get(item_id_value)))
            lines.append("")

    lines.extend(["## 噪音和低优先级", ""])
    if noise_ids:
        for item_id_value in noise_ids:
            item = by_id.get(item_id_value)
            if item:
                lines.append(f"- {item.get('company')}: {item.get('title')}")
    else:
        lines.append("未单独标出噪音项。")
    lines.append("")

    lines.extend(
        [
            "## 后续行动",
            "",
            "- [ ] 对重点动态中涉及 agent、API、coding、browser/computer use 的条目做二次确认。",
            "- [ ] 把高价值中国公司动态同步进每日雷达的关注词。",
            "- [ ] 对观察项等待下一次公告或更多证据后再升级。",
        ]
    )
    return "\n".join(lines)


def render_company_markdown(day: dt.date, items: list[dict[str, Any]]) -> str:
    items = items[:12]
    lines = [
        f"# AI Company Radar - {day.isoformat()}",
        "",
        "## 规则候选动态",
        "",
        "DeepSeek 精筛未成功，因此这里只展示规则过滤后的高分候选，不能视为完整公司情报。",
        "",
    ]
    if not items:
        lines.append("暂无公司动态候选条目。")
        return "\n".join(lines)
    for item in items[:10]:
        lines.append(f"### {item['company']}: {item['title']}")
        lines.append(f"- 区域：{item.get('region')}")
        lines.append(f"- 分数：{item.get('score')}/10")
        lines.append(f"- 链接：{item.get('url')}")
        if item.get("published"):
            lines.append(f"- 日期：{item.get('published')}")
        if item.get("reasons"):
            lines.append(f"- 理由：{', '.join(item['reasons'])}")
        lines.append("")
    return "\n".join(lines)


def append_chatgpt_request(markdown: str) -> str:
    if "## 发给 ChatGPT 的公司动态解读请求" in markdown:
        return markdown
    prompt = """

## 发给 ChatGPT 的公司动态解读请求

请基于上面的 AI Company Radar 帮我做二次解读：
1. 哪些公司动态真正值得关注？
2. 它们分别影响 AI agent、AI 应用、开发者工具或模型生态的哪一部分？
3. 标出营销噪音、证据不足或只是普通页面更新的内容。
4. 判断哪些中国公司动态值得后续追踪。
5. 给我今天最应该关注的 3 个行动建议。
""".rstrip()
    return markdown.rstrip() + prompt + "\n"


def main() -> None:
    config = load_config()
    day = today_local()
    company_config = config.get("company", {})
    ttl_days = int(company_config.get("seen_ttl_days", 14))
    seen = prune_company_seen(load_company_seen(), day, ttl_days)

    items = filter_seen_company_items(fetch_company_items(config), seen)
    for item in items:
        item["score"], item["reasons"] = score_company_item(item, config)
    items = [item for item in items if item.get("score", 0) >= 2.0]
    items.sort(key=lambda x: x.get("score", 0), reverse=True)
    for idx, item in enumerate(items, 1):
        item["id"] = f"item_{idx}"

    output_dir = ROOT / config.get("company", {}).get("output_dir", "company")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{day.isoformat()}.md"

    ai_markdown = call_deepseek_company(day, items, config)
    output_path.write_text(append_chatgpt_request(ai_markdown or render_company_markdown(day, items)), encoding="utf-8")

    for item in items[: int(company_config.get("max_items_for_ai", 20))]:
        seen[company_item_id(item)] = day.isoformat()
    save_company_seen(seen)
    print(f"Wrote {output_path} with {len(items)} filtered company items")


if __name__ == "__main__":
    main()

