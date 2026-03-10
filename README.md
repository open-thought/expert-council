# Frontier LLM Expert Council

Structured multi-LLM debate tool. Takes a question, queries multiple frontier LLMs independently, runs anonymous peer review, then synthesizes a final recommendation.

## Protocol

1. **Independent Opinions** — Each LLM answers the question independently (max 400 words)
2. **Anonymous Peer Review** — Each LLM reviews the others' opinions (anonymized as Expert A, B, etc.)
3. **Synthesis** — A randomly selected LLM synthesizes all opinions and reviews into a final recommendation
4. **Output** — Structured report saved as markdown + printed to terminal

## Setup

```bash
pip install -r requirements.txt
```

Set API keys in environment (at least 2 required):

| Variable | Service |
|---|---|
| `ANTHROPIC_API_KEY` | Claude (claude-opus-4-6) |
| `OPENAI_API_KEY` | GPT-4o |
| `XAI_API_KEY` | Grok-3 |
| `GEMINI_API_KEY` | Gemini 2.0 Flash |

Optional: `DISCORD_CH` — Discord webhook URL to send the final synthesis.

## Usage

```bash
# Direct question
python expert_council.py "What is the best approach to building a RAG pipeline?"

# Question from file
python expert_council.py --question-file question.txt

# Custom output directory
python expert_council.py "Your question" --output-dir ./my-councils/
```

Output is saved to `councils/YYYY-MM-DD_HHMMSS_<slug>.md`.
