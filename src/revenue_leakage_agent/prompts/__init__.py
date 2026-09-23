from pathlib import Path


def load_prompt(
    *,
    prompts_dir: Path,
    prompt_name: str,
    fallback: str,
) -> str:
    """Load a Markdown prompt by centralized name, with a safe V1 fallback."""

    path = prompts_dir / f"{prompt_name}.md"
    if not path.exists():
        return fallback.strip()

    content = path.read_text(encoding="utf-8").strip()
    return content or fallback.strip()
