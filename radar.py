from __future__ import annotations

import datetime as dt
import html
import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import quote

import feedparser
import requests
import yaml
from bs4 import BeautifulSoup


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config.yaml"
STATE_PATH = ROOT / "state.json"


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


def score_item(item: dict[str, Any], config: dict[str, Any]) -> tuple[float, list[str]]:
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

    return round(min(score, 10.0), 1), reasons[:6]


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
    queries = ["agent leaderboard", "multi-agent benchmark", "web agent", "agent challenge"]
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
        item["score"], item["reasons"] = score_item(item, config)

    fresh.sort(key=lambda x: x.get("score", 0), reverse=True)
    digest_items = fresh[: config["max_digest_items"]]

    output_dir = ROOT / config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{day.isoformat()}.md"
    output_path.write_text(render_markdown(day, digest_items), encoding="utf-8")

    state["seen"] = sorted((seen | {item_id(i) for i in fresh if i.get("url")}) )[-2000:]
    save_state(state)

    print(f"Wrote {output_path} with {len(digest_items)} items")


if __name__ == "__main__":
    main()

