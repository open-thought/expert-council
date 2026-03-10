#!/usr/bin/env python3
"""Frontier LLM Expert Council — structured multi-LLM debate tool."""

import argparse
import os
import re
import sys
import random
import time
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# LLM backend wrappers
# ---------------------------------------------------------------------------

class LLMBackend:
    """Base class for LLM backends."""
    name: str

    def query(self, system_prompt: str, user_prompt: str) -> str:
        raise NotImplementedError


class ClaudeBackend(LLMBackend):
    name = "Claude"

    def __init__(self):
        import anthropic
        self.client = anthropic.Anthropic()

    def query(self, system_prompt: str, user_prompt: str) -> str:
        resp = self.client.messages.create(
            model="claude-opus-4-6",
            max_tokens=1024,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return resp.content[0].text


class GPTBackend(LLMBackend):
    name = "GPT-5.4"

    def __init__(self):
        import openai
        self.client = openai.OpenAI()

    def query(self, system_prompt: str, user_prompt: str) -> str:
        resp = self.client.chat.completions.create(
            model="gpt-5.4",
            reasoning_effort="high",
            max_completion_tokens=16384,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return resp.choices[0].message.content


class GrokBackend(LLMBackend):
    name = "Grok"

    def __init__(self):
        import openai
        self.client = openai.OpenAI(
            base_url="https://api.x.ai/v1",
            api_key=os.environ["XAI_API_KEY"],
        )

    def query(self, system_prompt: str, user_prompt: str) -> str:
        resp = self.client.chat.completions.create(
            model="grok-4-fast-reasoning",
            max_tokens=1024,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return resp.choices[0].message.content


class GeminiBackend(LLMBackend):
    name = "Gemini"

    def __init__(self):
        import requests
        self.api_key = os.environ["GEMINI_API_KEY"]
        self.model = "gemini-2.5-pro"
        # Detect Vertex AI key (not AIza prefix) vs AI Studio key
        if not self.api_key.startswith("AIza"):
            self.endpoint = f"https://aiplatform.googleapis.com/v1/publishers/google/models/{self.model}:generateContent"
        else:
            self.endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"

    def query(self, system_prompt: str, user_prompt: str) -> str:
        import requests
        payload = {
            "contents": [{"role": "user", "parts": [{"text": f"{system_prompt}\n\n{user_prompt}"}]}],
            "generationConfig": {"maxOutputTokens": 1024},
        }
        resp = requests.post(
            f"{self.endpoint}?key={self.api_key}",
            json=payload, timeout=60
        )
        resp.raise_for_status()
        return resp.json()["candidates"][0]["content"]["parts"][0]["text"]


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def discover_backends() -> list[LLMBackend]:
    """Try to instantiate each backend; skip unavailable ones."""
    candidates: list[tuple[str, type, Optional[str]]] = [
        ("Claude", ClaudeBackend, "ANTHROPIC_API_KEY"),
        ("GPT-4o", GPTBackend, "OPENAI_API_KEY"),
        ("Grok", GrokBackend, "XAI_API_KEY"),
        ("Gemini", GeminiBackend, "GEMINI_API_KEY"),
    ]
    backends: list[LLMBackend] = []
    for name, cls, env_key in candidates:
        if env_key and not os.environ.get(env_key):
            print(f"  [skip] {name} — {env_key} not set")
            continue
        try:
            backends.append(cls())
            print(f"  [ok]   {name}")
        except Exception as exc:
            print(f"  [skip] {name} — {exc}")
    return backends


# ---------------------------------------------------------------------------
# Parallel query helper
# ---------------------------------------------------------------------------

def parallel_query(backends: list[LLMBackend], system_prompt: str,
                   user_prompts: dict[str, str]) -> dict[str, str]:
    """Query backends in parallel. user_prompts maps backend.name -> prompt.
    Returns {name: response} for successful calls."""
    results: dict[str, str] = {}

    def _call(backend: LLMBackend) -> tuple[str, str]:
        prompt = user_prompts.get(backend.name, "")
        return backend.name, backend.query(system_prompt, prompt)

    with ThreadPoolExecutor(max_workers=len(backends)) as pool:
        futures = {pool.submit(_call, b): b for b in backends}
        for fut in as_completed(futures):
            b = futures[fut]
            try:
                name, text = fut.result()
                results[name] = text
            except Exception as exc:
                print(f"  [error] {b.name} failed: {exc}")
    return results


# ---------------------------------------------------------------------------
# Council protocol
# ---------------------------------------------------------------------------

LABELS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def run_council(question: str, backends: list[LLMBackend]) -> dict:
    """Execute the 4-phase council protocol. Returns full transcript dict."""
    transcript: dict = {
        "question": question,
        "timestamp": datetime.now().isoformat(),
        "models": [b.name for b in backends],
    }

    # ── Phase 1: Independent Opinions ─────────────────────────────────────
    print("\n▶ Phase 1 — Independent Opinions")
    sys_p1 = (
        "You are an expert analyst participating in a council of experts. "
        "Give your independent analysis of the question below. "
        "Be concrete, cite evidence where possible. Max 400 words."
    )
    prompts_p1 = {b.name: question for b in backends}
    opinions = parallel_query(backends, sys_p1, prompts_p1)
    for name in opinions:
        print(f"  ✓ {name} responded ({len(opinions[name].split())} words)")
    transcript["phase1_opinions"] = opinions

    # Remove backends that failed in phase 1
    active = [b for b in backends if b.name in opinions]
    if len(active) < 2:
        sys.exit("✗ Fewer than 2 LLMs responded. Cannot continue.")

    # ── Anonymous mapping ─────────────────────────────────────────────────
    random.seed(None)
    names = list(opinions.keys())
    random.shuffle(names)
    name_to_label = {n: f"Expert {LABELS[i]}" for i, n in enumerate(names)}
    label_to_name = {v: k for k, v in name_to_label.items()}
    transcript["anon_mapping"] = name_to_label

    def format_opinions_except(exclude_name: str) -> str:
        parts = []
        for n in names:
            if n == exclude_name:
                continue
            parts.append(f"### {name_to_label[n]}\n{opinions[n]}")
        return "\n\n".join(parts)

    # ── Phase 2: Anonymous Peer Review ────────────────────────────────────
    print("\n▶ Phase 2 — Anonymous Peer Review")
    sys_p2 = (
        "You are an expert analyst reviewing other experts' opinions on a question. "
        "Critically review each opinion: what is strong, what is weak, what is missing. "
        "Be specific and constructive. Max 400 words."
    )
    prompts_p2 = {}
    for b in active:
        others_text = format_opinions_except(b.name)
        prompts_p2[b.name] = (
            f"Original question: {question}\n\n"
            f"Other experts' opinions to review:\n\n{others_text}"
        )
    reviews = parallel_query(active, sys_p2, prompts_p2)
    for name in reviews:
        print(f"  ✓ {name} reviewed ({len(reviews[name].split())} words)")
    transcript["phase2_reviews"] = reviews

    # ── Phase 3: Synthesis ────────────────────────────────────────────────
    print("\n▶ Phase 3 — Synthesis")
    random.seed(None)
    synthesizer = random.choice(active)
    print(f"  Synthesizer: {synthesizer.name}")

    all_opinions_text = "\n\n".join(
        f"### {name_to_label[n]}\n{opinions[n]}" for n in names if n in opinions
    )
    all_reviews_text = "\n\n".join(
        f"### Review by {name_to_label[n]}\n{reviews[n]}"
        for n in names if n in reviews
    )

    sys_p3 = (
        "You are the designated Synthesizer for an expert council. "
        "You receive all expert opinions and all peer reviews. "
        "Produce a concrete, actionable final recommendation that integrates "
        "the strongest arguments and addresses the weaknesses identified. "
        "Structure your output as: Summary → Key Points → Recommendation → Caveats."
    )
    prompt_p3 = (
        f"## Original Question\n{question}\n\n"
        f"## Expert Opinions\n{all_opinions_text}\n\n"
        f"## Peer Reviews\n{all_reviews_text}"
    )
    synthesis = synthesizer.query(sys_p3, prompt_p3)
    print(f"  ✓ Synthesis complete ({len(synthesis.split())} words)")
    transcript["phase3_synthesis"] = synthesis
    transcript["synthesizer"] = synthesizer.name

    # ── Phase 4: Actionable Points ───────────────────────────────────────
    print("\n▶ Phase 4 — Actionable Points")
    # Pick the model that did NOT synthesize
    actioneer = next((b for b in active if b.name != synthesizer.name), active[0])
    print(f"  Actioneer: {actioneer.name}")

    sys_p4 = (
        "You are an expert consultant. You receive a synthesis from an expert council. "
        "Extract a prioritized, concrete list of actionable items. "
        "Format as: **Priority (High/Medium/Low)** | **Action** | **Why** | **Effort (1d/1w/1m)**. "
        "Be specific and direct. No preamble. Max 10 items."
    )
    prompt_p4 = (
        f"## Original Question\n{question}\n\n"
        f"## Council Synthesis\n{synthesis}"
    )
    action_items = actioneer.query(sys_p4, prompt_p4)
    print(f"  ✓ Action items extracted ({len(action_items.split())} words)")
    transcript["phase4_action_items"] = action_items
    transcript["actioneer"] = actioneer.name

    # Reveal mapping
    transcript["label_to_name"] = label_to_name
    return transcript


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def slugify(text: str, maxlen: int = 40) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return s[:maxlen].rstrip("-")


def build_markdown(t: dict) -> str:
    lines: list[str] = []
    a = lines.append

    a(f"# Expert Council Report")
    a(f"_Generated {t['timestamp']}_\n")
    a(f"**Models:** {', '.join(t['models'])}\n")

    a(f"## Question\n\n{t['question']}\n")

    # Phase 1
    a("## Phase 1 — Expert Opinions\n")
    mapping = t["anon_mapping"]
    rev_map = t["label_to_name"]
    for label in sorted(rev_map.keys()):
        real = rev_map[label]
        a(f"### {label} _{real}_\n")
        a(t["phase1_opinions"].get(real, "_no response_") + "\n")

    # Phase 2
    a("## Phase 2 — Peer Reviews\n")
    for label in sorted(rev_map.keys()):
        real = rev_map[label]
        a(f"### Review by {label} _{real}_\n")
        a(t["phase2_reviews"].get(real, "_no response_") + "\n")

    # Phase 3
    a("## Phase 3 — Synthesis\n")
    a(f"**Synthesizer:** {t['synthesizer']}\n")
    a(t["phase3_synthesis"] + "\n")

    # Phase 4 — Actionable Points
    a("## Phase 4 — Actionable Points\n")
    a(f"**Extracted by:** {t.get('actioneer', '?')}\n")
    a(t.get("phase4_action_items", "_not available_") + "\n")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Discord notification
# ---------------------------------------------------------------------------

def send_discord(content: str, target: str) -> None:
    """Send a message to Discord via openclaw CLI. Splits at 1900 chars."""
    import subprocess
    chunks = []
    while len(content) > 1900:
        split = content[:1900].rfind("\n")
        if split < 500: split = 1900
        chunks.append(content[:split])
        content = content[split:]
    chunks.append(content)
    for chunk in chunks:
        r = subprocess.run(
            ["openclaw", "message", "send", "--channel", "discord",
             "--target", target, "--message", chunk],
            capture_output=True, text=True, timeout=15
        )
        if r.returncode != 0:
            raise Exception(r.stderr[:100])
def main():
    parser = argparse.ArgumentParser(
        description="Frontier LLM Expert Council — multi-LLM debate tool"
    )
    parser.add_argument("question", nargs="?", help="The question to deliberate on")
    parser.add_argument("--question-file", type=str, help="Read question from file")
    parser.add_argument("--output-dir", type=str, default="councils",
                        help="Directory for markdown output (default: councils/)")
    args = parser.parse_args()

    # Resolve question
    if args.question_file:
        question = Path(args.question_file).read_text().strip()
    elif args.question:
        question = args.question
    else:
        parser.error("Provide a question as argument or via --question-file")

    print("═" * 60)
    print("  FRONTIER LLM EXPERT COUNCIL")
    print("═" * 60)
    print(f"\nQuestion: {question[:120]}{'...' if len(question) > 120 else ''}\n")

    # Discover
    print("Discovering available LLMs...")
    backends = discover_backends()
    if len(backends) < 2:
        sys.exit("✗ Need at least 2 LLMs available. Check API keys.")
    print(f"\n{len(backends)} models active. Starting council.\n")

    # Run
    start = time.time()
    transcript = run_council(question, backends)
    elapsed = time.time() - start

    # Build report
    md = build_markdown(transcript)

    # Save
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    slug = slugify(question)
    out_path = out_dir / f"{ts}_{slug}.md"
    out_path.write_text(md)

    print(f"\n{'═' * 60}")
    print(f"  COUNCIL COMPLETE — {elapsed:.1f}s")
    print(f"  Report: {out_path}")
    print(f"{'═' * 60}\n")

    # Print synthesis to terminal
    print("── Final Recommendation ──\n")
    print(transcript["phase3_synthesis"])
    print()

    # Discord
    discord_url = os.environ.get("DISCORD_CH")
    if discord_url:
        print("Sending TL;DR to Discord...")
        synth = transcript["phase3_synthesis"]
        # First 2-3 sentences as TL;DR
        sentences = synth.replace("\n", " ").split(". ")
        tldr = ". ".join(sentences[:3]).strip()
        if len(tldr) > 500:
            tldr = tldr[:500] + "..."
        actions = transcript.get("phase4_action_items", "")
        top_actions = "\n".join(actions.splitlines()[:4]) if actions else ""
        msg = (
            f"**Expert Council** — _{question[:80]}..._\n\n"
            f"**TL;DR:** {tldr}\n\n"
            f"**Top actions:**\n{top_actions}\n\n"
            f"_Full report → vault: `{out_path.name}`_"
        )
        send_discord(msg, discord_url)


if __name__ == "__main__":
    main()
