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


def build_company_prompt(day: dt.date, items: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, str]]:
    system = (
        "You are a company intelligence analyst focused on AI agents, AI applications, and model/product releases. "
        "Write entirely in Simplified Chinese. Be selective and skeptical. "
        "Use only the supplied company items and URLs. Do not invent announcements, dates, features, links, or claims. "
        "Pay special attention to both global AI companies and Chinese AI companies."
    )
    user = f"""
请根据下面的公司动态候选条目，生成一份简体中文 Markdown 公司动态雷达。

日期：{day.isoformat()}

必须使用下面的中文结构：

# AI Company Radar - {day.isoformat()}

## 今日结论
用 3-5 句中文说明今天最值得注意的公司动态。区分真正有价值的产品/模型/agent 动态和普通营销噪音。

## 全球公司重点动态
列出 OpenAI、Anthropic、Google DeepMind、Microsoft、Meta、Mistral、GitHub、Cursor 等相关动态。每条包含：
- 公司：
- 动态：
- 和 AI agent / AI 应用的关系：
- 可能影响：
- 链接：

## 中国公司重点动态
列出 DeepSeek、Qwen、Kimi、GLM、豆包、混元、MiniMax、Manus 等相关动态。每条包含：
- 公司：
- 动态：
- 和 AI agent / AI 应用的关系：
- 可能影响：
- 链接：

## 值得追踪的产品或能力
提取 3-6 个值得后续追踪的产品能力，例如 coding agent、browser automation、mobile automation、agentic workflow、API、open-source model。

## 噪音和低优先级
指出哪些条目可能只是营销、招聘、泛泛新闻、页面导航或证据不足。

## 后续行动
给出 3-5 个中文待办事项，使用 Markdown checkbox。

硬性要求：
- 全文必须使用简体中文。
- 不要输出英文标题。
- 不要编造输入中没有的事实。
- 不要声称你阅读了完整公告。
- 如果证据不足，明确写“证据不足”。
- 保留原始 URL。
- 如果某个区域没有高价值内容，直接写“暂无高价值新增动态”。

公司动态候选条目 JSON：
{json.dumps(compact_items(items, config), ensure_ascii=False, indent=2)}
""".strip()
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


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
        "messages": build_company_prompt(day, items, config),
        "temperature": 0.2,
        "max_tokens": 3500,
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
        return data["choices"][0]["message"]["content"].strip()
    except Exception as exc:
        print(f"DeepSeek company call failed; using rule-based Markdown. Error: {exc}")
        return None


def render_company_markdown(day: dt.date, items: list[dict[str, Any]]) -> str:
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


def main() -> None:
    config = load_config()
    day = today_local()
    items = fetch_company_items(config)
    for item in items:
        item["score"], item["reasons"] = score_company_item(item, config)
    items = [item for item in items if item.get("score", 0) >= 2.0]
    items.sort(key=lambda x: x.get("score", 0), reverse=True)

    output_dir = ROOT / config.get("company", {}).get("output_dir", "company")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{day.isoformat()}.md"

    ai_markdown = call_deepseek_company(day, items, config)
    output_path.write_text(ai_markdown or render_company_markdown(day, items), encoding="utf-8")
    print(f"Wrote {output_path} with {len(items)} filtered company items")


if __name__ == "__main__":
    main()
