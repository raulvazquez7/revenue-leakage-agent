"""System prompts shipped as package data."""

from importlib.resources import files


def load_prompt(name: str) -> str:
    """Return the packaged ``<name>.md`` prompt.

    Raises ``FileNotFoundError`` when the prompt is missing or empty, so a
    packaging mistake fails loudly instead of running with a degraded prompt.
    """

    resource = files("revenue_leakage_agent.prompts") / f"{name}.md"
    if not resource.is_file():
        raise FileNotFoundError(f"Prompt '{name}' is not packaged.")
    content = resource.read_text(encoding="utf-8").strip()
    if not content:
        raise FileNotFoundError(f"Prompt '{name}' is empty.")
    return content
