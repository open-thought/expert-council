# Frontier LLM Expert Council

Structured multi-LLM debate tool. Queries multiple frontier models independently, runs anonymous peer review, then synthesizes a final recommendation.

## Protocol

1. **Independent Opinions** — Each model answers independently (parallel API calls)
2. **Anonymous Peer Review** — Each model critiques the others (anonymized as Expert A, B, etc.)
3. **Synthesis** — A randomly selected model integrates all opinions into a final recommendation
4. **Output** — Full report saved as Markdown; TL;DR sent to Discord if configured

## Models

Uses the best available model per provider:

| Variable | Provider | Model |
|---|---|---|
| `ANTHROPIC_API_KEY` | Anthropic | `claude-opus-4-6` |
| `OPENAI_API_KEY` | OpenAI | `gpt-5.4` (reasoning_effort=high) |
| `XAI_API_KEY` | xAI | `grok-4-fast-reasoning` |
| `GEMINI_API_KEY` | Google | `gemini-2.5-pro` _(optional)_ |

At least 2 models required. Models without a key are automatically skipped.

## Setup

```bash
pip install anthropic openai google-generativeai
```

## Usage

```bash
# Direct question
python expert_council.py "What is the best architecture for a robot world model?"

# From file
python expert_council.py --question-file question.txt

# Custom output directory
python expert_council.py "Your question" --output-dir ./reports/
```

Reports are saved to `councils/YYYY-MM-DD_HHMMSS_<slug>.md`.

## Discord Integration

Set `DISCORD_CH` to an OpenClaw Discord channel target to receive a TL;DR after each council. Full report stays in the output directory.

```bash
DISCORD_CH="channel:YOUR_CHANNEL_ID" python expert_council.py "..."
```
