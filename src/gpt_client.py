"""Cross-family model client.

The key is read from a file outside the repository so it never enters a prompt,
a log, or version control. Nothing here prints or returns the key, and the
caller sees only completion text.

Expected key file: ~/.config/icaif/openai_key   (chmod 600, one line)
"""
import os
import time
from pathlib import Path

KEY_PATH = Path.home() / ".config" / "icaif" / "openai_key"
DEFAULT_MODEL = os.environ.get("ICAIF_GPT_MODEL", "gpt-4.1")


def _key():
    if not KEY_PATH.exists():
        raise RuntimeError(
            f"No key at {KEY_PATH}. Write it there (chmod 600) and retry.")
    k = KEY_PATH.read_text().strip()
    if not k:
        raise RuntimeError(f"{KEY_PATH} is empty.")
    return k


_client = None


def client():
    global _client
    if _client is None:
        from openai import OpenAI
        _client = OpenAI(api_key=_key())
    return _client


def ask(prompt, model=None, retries=6, max_tokens=400):
    """One-shot completion. Mirrors the claude -p wrapper used elsewhere so the
    experiment scripts can swap families without other changes."""
    model = model or DEFAULT_MODEL
    last = None
    for attempt in range(retries):
        try:
            r = client().chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=max_tokens,
            )
            out = (r.choices[0].message.content or "").strip()
            if out:
                return out
        except Exception as e:  # rate limits, transient 5xx
            last = e
            time.sleep(min(60, 3 * (attempt + 1)))
    raise RuntimeError(f"no completion after {retries} attempts: {last}")


def check():
    """Verify the key works and report the model, without revealing the key."""
    out = ask("Reply with exactly the word: pong", max_tokens=16)
    return f"model={DEFAULT_MODEL} reachable, replied: {out[:40]!r}"


if __name__ == "__main__":
    print(check())
