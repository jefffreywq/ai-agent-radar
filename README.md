# AI Agent Radar

Daily GitHub Actions job that collects Hugging Face agent-related spaces, competitions, daily papers, and recent arXiv papers about agents / multi-agent systems, then writes a Markdown digest to `inbox/`.

## Quick Start

1. Create a GitHub repository, for example `ai-agent-radar`.
2. Upload these files to the repository.
3. Open the repository's **Actions** tab and enable workflows if GitHub asks.
4. Run **AI Agent Radar** manually once from **Actions -> AI Agent Radar -> Run workflow**.
5. Check the generated Markdown file under `inbox/`.

The scheduled run is set for 08:30 Australia/Sydney time during AEST, expressed as `22:30 UTC` in GitHub Actions cron.

## Cost

This version can run in two modes:

- Rule-based mode: free public data sources only.
- DeepSeek mode: calls DeepSeek when `DEEPSEEK_API_KEY` is configured.

The public-data collection uses:

- Hugging Face public APIs
- arXiv public API
- OpenAlex public API
- GitHub Actions free quota

DeepSeek usage is pay-as-you-go. The default model is `deepseek-v4-flash`, configured in `config.yaml`.

## Configure

Edit `config.yaml` to change keywords, arXiv categories, item limits, and output folder.

The current DeepSeek digest asks the model to:

- separate useful new items from low-priority noise
- explain each important item's basic information and likely innovation point
- add one short background primer for concepts the user should understand
- keep original URLs and mark weak evidence instead of inventing details

## DeepSeek Setup

Add this repository secret:

```text
Settings -> Secrets and variables -> Actions -> New repository secret
Name: DEEPSEEK_API_KEY
Value: your DeepSeek API key
```

If the secret is missing or the API call fails, the workflow falls back to the rule-based Markdown digest.

## Weekly Trend Radar

The repository also includes a weekly trend radar:

- Script: `weekly_trends.py`
- Workflow: `.github/workflows/weekly-trends.yml`
- Output: `weekly/YYYY-MM-DD.md`
- Dynamic terms state: `state/trending_terms.json`

It scans the last 7 days of Hugging Face Daily Papers, Hugging Face Spaces, and recent arXiv papers, extracts rising terms, and asks DeepSeek to write a Chinese weekly trend digest when `DEEPSEEK_API_KEY` is available.

The scheduled run is set for Sunday 00:15 Australia/Sydney time during AEST, expressed as Saturday `14:15 UTC` in GitHub Actions cron.

For first-time setup, run **Agent Weekly Trend Radar** manually once before the daily radar. This creates `state/trending_terms.json`. The daily radar reads that file and boosts items that match current weekly tier1/tier2 terms. If the file does not exist yet, the daily radar still works and simply skips the weekly trend boost.

## Company Radar

The repository also includes a company dynamics radar:

- Script: `company_radar.py`
- Workflow: `.github/workflows/company-radar.yml`
- Output: `company/YYYY-MM-DD.md`

It scans configured global and Chinese company sources, filters for AI agent / AI application / model and product release signals, and asks DeepSeek to write a Chinese company intelligence digest when `DEEPSEEK_API_KEY` is available.

The scheduled run is set for 00:30 Australia/Sydney time during AEST, expressed as `14:30 UTC` in GitHub Actions cron.
