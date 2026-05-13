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

This starter version does not call paid AI APIs. It uses:

- Hugging Face public APIs
- arXiv public API
- OpenAlex public API
- GitHub Actions free quota

## Configure

Edit `config.yaml` to change keywords, arXiv categories, item limits, and output folder.

