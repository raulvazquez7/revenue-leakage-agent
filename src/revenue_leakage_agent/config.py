from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

OpenAIReasoningEffort = Literal["none", "low", "medium", "high", "xhigh"]
OpenAIReasoningSummary = Literal["auto", "concise", "detailed"]


def build_openai_reasoning(
    effort: OpenAIReasoningEffort,
    summary: OpenAIReasoningSummary = "auto",
) -> dict[str, str] | None:
    """Build the ChatOpenAI reasoning config expected by LangChain."""

    if effort == "none":
        return None
    return {"effort": effort, "summary": summary}


def build_chat_openai_kwargs(
    *,
    model: str,
    api_key: Any,
    reasoning_effort: OpenAIReasoningEffort,
    reasoning_summary: OpenAIReasoningSummary = "auto",
    timeout_seconds: int,
) -> dict[str, Any]:
    """Build common ChatOpenAI constructor kwargs without leaking config logic."""

    kwargs: dict[str, Any] = {
        "model": model,
        "api_key": api_key,
        "temperature": 0,
        "timeout": timeout_seconds,
    }
    reasoning = build_openai_reasoning(reasoning_effort, reasoning_summary)
    if reasoning is not None:
        kwargs["reasoning"] = reasoning
    return kwargs


class AppSettings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        frozen=True,
    )

    app_env: str = Field(default="local", description="Runtime environment.")
    log_level: str = Field(default="INFO", description="Application log level.")

    openai_api_key: str = Field(default="", description="OpenAI API key.")

    router_model: str = Field(
        default="gpt-5.4-mini",
        description="Fast model for the router node.",
        validation_alias=AliasChoices("ROUTER_MODEL", "OPENAI_ROUTER_MODEL"),
    )
    router_reasoning_effort: OpenAIReasoningEffort = Field(
        default="none",
        description="Reasoning effort for the router model.",
    )
    router_reasoning_summary: OpenAIReasoningSummary = Field(
        default="auto",
        description="Reasoning summary setting for the router model.",
    )

    agent_model: str = Field(
        default="gpt-5.4-2026-03-05",
        description="Main revenue leakage investigator model.",
        validation_alias=AliasChoices("AGENT_MODEL", "OPENAI_MODEL"),
    )
    agent_reasoning_effort: OpenAIReasoningEffort = Field(
        default="none",
        description="Reasoning effort for the main investigator model.",
    )
    agent_reasoning_summary: OpenAIReasoningSummary = Field(
        default="auto",
        description="Reasoning summary setting for the main investigator model.",
    )

    conversational_model: str = Field(
        default="gpt-5.4-mini",
        description="Fast model for conversational turns.",
    )
    conversational_reasoning_effort: OpenAIReasoningEffort = Field(
        default="low",
        description="Reasoning effort for the conversational model.",
    )
    conversational_reasoning_summary: OpenAIReasoningSummary = Field(
        default="auto",
        description="Reasoning summary setting for the conversational model.",
    )

    llm_timeout_seconds: int = Field(
        default=120,
        ge=5,
        le=300,
        description="Timeout for standard LLM API calls in seconds.",
    )
    agent_timeout_seconds: int = Field(
        default=180,
        ge=30,
        le=300,
        description="Timeout for investigator LLM calls in seconds.",
    )

    data_dir: Path = Field(default=Path("data"), description="Read-only JSON data.")
    sandbox_dir: Path = Field(
        default=Path("sandbox"),
        description="Writable sandbox ledgers.",
    )

    langfuse_secret_key: str | None = Field(default=None)
    langfuse_public_key: str | None = Field(default=None)
    langfuse_base_url: str | None = Field(default=None)


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    return AppSettings()


Settings = AppSettings
