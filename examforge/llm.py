"""LLM provider abstraction + the reliability wrapper every stage calls.

Providers only know how to turn (system, user, schema) into a validated Pydantic object and to classify
failures into three buckets:
  TransientLLMError -> retry with exponential backoff (rate limits, 5xx, timeouts, connection errors)
  OutputError       -> retry with a *structured correction* message (malformed JSON, schema violation,
                       truncated generation)
  LLMError          -> fail fast (auth, bad request, refusal) - retrying cannot help

LLMClient.run() owns retries, prompt rendering/versioning and call logging, so stages stay small.
Secrets are never logged: only stage, model, prompt version, token counts, latency and error class/message.
"""
from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol, TypeVar

from pydantic import BaseModel, ValidationError

from . import prompts
from .config import Settings
from .schemas import now_iso

T = TypeVar("T", bound=BaseModel)


class LLMError(Exception):
    """Non-retryable failure."""


class TransientLLMError(LLMError):
    def __init__(self, msg: str, retry_after: float | None = None):
        super().__init__(msg)
        self.retry_after = retry_after


class OutputError(LLMError):
    """Model answered, but the output is malformed/incomplete. Retry with a correction message."""


@dataclass
class LLMResponse:
    parsed: BaseModel
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""


class Provider(Protocol):
    name: str
    model: str
    is_mock: bool

    def generate(self, system: str, user: str, schema: type[T], variables: dict) -> LLMResponse: ...


# --------------------------------------------------------------------------- Anthropic (Claude)
class AnthropicProvider:
    name = "anthropic"
    is_mock = False

    def __init__(self, model: str, effort: str = "high", max_tokens: int = 16000):
        import os

        import anthropic

        if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
            raise LLMError("ANTHROPIC_API_KEY is not set (add it to .env, or choose another provider)")
        self._anthropic = anthropic
        # SDK retries disabled: LLMClient owns retry policy so every attempt is logged and bounded.
        self.client = anthropic.Anthropic(max_retries=0, timeout=600)
        self.model, self.effort, self.max_tokens = model, effort, max_tokens

    def generate(self, system: str, user: str, schema: type[T], variables: dict) -> LLMResponse:
        a = self._anthropic
        try:
            resp = self.client.beta.messages.parse(
                model=self.model,
                max_tokens=self.max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
                output_format=schema,  # native structured output (JSON schema enforced server-side)
                output_config={"effort": self.effort},
                # Server-side refusal fallback: a policy decline is retried on a fallback model in-call.
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            )
        except ValidationError as e:  # truncated (max_tokens) or schema-violating JSON
            raise OutputError(f"output failed schema validation: {_short(e)}") from e
        except a.RateLimitError as e:
            ra = e.response.headers.get("retry-after") if e.response is not None else None
            raise TransientLLMError(f"rate limited: {e.message}", float(ra) if ra else None) from e
        except (a.APIConnectionError, a.APITimeoutError) as e:
            raise TransientLLMError(f"connection/timeout: {e}") from e
        except a.APIStatusError as e:
            if e.status_code >= 500:
                raise TransientLLMError(f"server error {e.status_code}") from e
            raise LLMError(f"API error {e.status_code}: {e.message}") from e

        if resp.stop_reason == "refusal":
            raise LLMError("model refused the request (stop_reason=refusal)")
        if resp.stop_reason == "max_tokens":
            raise OutputError("generation truncated at max_tokens (incomplete output)")
        parsed = resp.parsed_output
        if parsed is None:
            raise OutputError("no structured output in response")
        return LLMResponse(parsed, resp.usage.input_tokens, resp.usage.output_tokens, resp.model)


# --------------------------------------------------------------------------- OpenAI-compatible
class OpenAICompatProvider:
    """OpenAI or any OpenAI-compatible endpoint (Gemini, Groq, Ollama...). JSON mode + Pydantic validation."""

    name = "openai"
    is_mock = False

    def __init__(self, model: str, api_key: str, base_url: str | None = None, max_tokens: int = 16000):
        import openai

        self._openai = openai
        self.client = openai.OpenAI(api_key=api_key or None, base_url=base_url or None, max_retries=0, timeout=600)
        self.model, self.max_tokens = model, max_tokens

    def generate(self, system: str, user: str, schema: type[T], variables: dict) -> LLMResponse:
        o = self._openai
        sys = f"{system}\n\nReturn ONLY a JSON object that validates against this JSON schema:\n{json.dumps(schema.model_json_schema())}"
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"},
                messages=[{"role": "system", "content": sys}, {"role": "user", "content": user}],
            )
        except o.RateLimitError as e:
            raise TransientLLMError(f"rate limited: {e}") from e
        except (o.APIConnectionError, o.APITimeoutError, o.InternalServerError) as e:
            raise TransientLLMError(f"transient: {e}") from e
        except o.APIStatusError as e:
            raise LLMError(f"API error {e.status_code}") from e
        choice = resp.choices[0]
        if choice.finish_reason == "length":
            raise OutputError("generation truncated (finish_reason=length)")
        try:
            parsed = schema.model_validate_json(choice.message.content or "")
        except ValidationError as e:
            raise OutputError(f"output failed schema validation: {_short(e)}") from e
        u = resp.usage
        return LLMResponse(parsed, u.prompt_tokens if u else 0, u.completion_tokens if u else 0, resp.model)


# --------------------------------------------------------------------------- AWS Bedrock (open models)
class BedrockProvider:
    """Bedrock Converse API (GLM, Qwen, ...). Auth = standard AWS chain (AWS_PROFILE / SSO), no key in .env."""

    name = "bedrock"
    is_mock = False
    TRANSIENT = {"ThrottlingException", "ServiceUnavailableException", "ModelTimeoutException",
                 "InternalServerException", "ModelNotReadyException"}

    def __init__(self, model: str, max_tokens: int = 8000, client: Any = None):
        if client is None:
            import boto3
            from botocore.config import Config

            # botocore retries off: LLMClient owns retries.
            client = boto3.client("bedrock-runtime", config=Config(retries={"max_attempts": 1}, read_timeout=600))
        self.client, self.model, self.max_tokens = client, model, max_tokens

    def generate(self, system: str, user: str, schema: type[T], variables: dict) -> LLMResponse:
        from botocore.exceptions import BotoCoreError, ClientError

        sys = f"{system}\n\nReturn ONLY a JSON object that validates against this JSON schema:\n{json.dumps(schema.model_json_schema())}"
        try:
            resp = self.client.converse(
                modelId=self.model,
                system=[{"text": sys}],
                messages=[{"role": "user", "content": [{"text": user}]}],
                inferenceConfig={"maxTokens": self.max_tokens, "temperature": 0.3},
            )
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "")
            if code in self.TRANSIENT:
                raise TransientLLMError(f"{code}") from e
            raise LLMError(f"Bedrock {code}: {e.response.get('Error', {}).get('Message', '')}") from e
        except BotoCoreError as e:
            kind = type(e).__name__
            if "Token" in kind or "Credential" in kind:  # expired SSO session: retrying cannot help
                raise LLMError(f"AWS credentials unavailable ({kind}): run `aws sso login --profile <profile>`") from e
            raise TransientLLMError(f"connection: {kind}") from e
        if resp.get("stopReason") == "max_tokens":
            raise OutputError("generation truncated at max_tokens (incomplete output)")
        text = "".join(b.get("text", "") for b in resp["output"]["message"]["content"])  # skips reasoning blocks
        try:
            parsed = schema.model_validate_json(_json_text(text))
        except ValidationError as e:
            raise OutputError(f"output failed schema validation: {_short(e)}") from e
        u = resp.get("usage", {})
        return LLMResponse(parsed, u.get("inputTokens", 0), u.get("outputTokens", 0), self.model)


def _json_text(text: str) -> str:
    """Open models often wrap JSON in ``` fences or <think> blocks; keep the outermost {...}."""
    text = text.split("</think>")[-1]
    start, end = text.find("{"), text.rfind("}")
    return text[start : end + 1] if start != -1 and end > start else text


def make_provider(settings: Settings, reviewer: bool = False) -> Provider:
    import os

    name = (settings.reviewer_provider if reviewer and settings.reviewer_provider else settings.llm_provider).lower()
    model = settings.reviewer_model if reviewer and settings.reviewer_model else settings.llm_model
    if name == "anthropic":
        return AnthropicProvider(model, settings.llm_effort, settings.llm_max_tokens)
    if name == "openai":
        return OpenAICompatProvider(model, os.environ.get("LLM_API_KEY", ""), settings.llm_base_url, settings.llm_max_tokens)
    if name == "bedrock":
        return BedrockProvider(model, settings.llm_max_tokens)
    if name == "mock":
        from .mock_llm import MockProvider

        return MockProvider()
    raise ValueError(f"unknown LLM_PROVIDER {name!r} (expected anthropic | openai | bedrock | mock)")


# --------------------------------------------------------------------------- reliability wrapper
@dataclass
class CallRecord:
    ts: str
    stage: str
    prompt_version: str
    provider: str
    model: str
    attempts: int
    ok: bool
    latency_s: float
    input_tokens: int = 0
    output_tokens: int = 0
    errors: list[str] = field(default_factory=list)


@dataclass
class LLMClient:
    provider: Provider
    reviewer: Provider | None = None  # optional independent model for reviewer stages
    max_retries: int = 3
    prompt_versions: dict[str, str] = field(default_factory=dict)
    log_path: Path | None = None
    sleep: Callable[[float], None] = time.sleep
    calls: list[CallRecord] = field(default_factory=list)

    REVIEW_STAGES = frozenset({"critical_examiner", "evidence_verifier", "difficulty_reviewer", "judge"})

    def provider_for(self, stage: str) -> Provider:
        return self.reviewer if (self.reviewer and stage in self.REVIEW_STAGES) else self.provider

    def run(self, stage: str, schema: type[T], variables: dict[str, Any]) -> T:
        version = self.prompt_versions.get(stage, "v1")
        system, user = prompts.render(stage, version, variables)
        provider = self.provider_for(stage)
        rec = CallRecord(now_iso(), stage, version, provider.name, provider.model, 0, False, 0.0)
        correction = ""
        t0 = time.perf_counter()
        try:
            for attempt in range(1, self.max_retries + 1):
                rec.attempts = attempt
                try:
                    resp = provider.generate(system, user + correction, schema, variables)
                except TransientLLMError as e:
                    rec.errors.append(f"transient: {e}")
                    if attempt < self.max_retries:
                        self.sleep(e.retry_after or min(30.0, 2 ** attempt + random.random()))
                    continue
                except OutputError as e:
                    rec.errors.append(f"output: {e}")
                    correction = (
                        "\n\n## CORRECTION REQUIRED\nYour previous answer was rejected: "
                        f"{e}. Return a complete, valid JSON object matching the schema. Do not truncate."
                    )
                    continue
                rec.ok = True
                rec.input_tokens, rec.output_tokens = resp.input_tokens, resp.output_tokens
                if resp.model:
                    rec.model = resp.model
                return resp.parsed  # type: ignore[return-value]
            raise LLMError(f"{stage}: failed after {self.max_retries} attempts: {rec.errors[-1] if rec.errors else '?'}")
        except LLMError as e:
            rec.errors.append(f"fatal: {e}")
            raise
        finally:
            rec.latency_s = round(time.perf_counter() - t0, 3)
            self.calls.append(rec)
            if self.log_path:
                self.log_path.parent.mkdir(parents=True, exist_ok=True)
                with self.log_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(asdict(rec)) + "\n")

    def stats(self, since: int = 0) -> dict:
        cs = self.calls[since:]
        return {
            "llm_calls": len(cs),
            "retries": sum(c.attempts - 1 for c in cs),
            "latency_s": round(sum(c.latency_s for c in cs), 2),
            "input_tokens": sum(c.input_tokens for c in cs),
            "output_tokens": sum(c.output_tokens for c in cs),
            "failed_calls": sum(not c.ok for c in cs),
        }


def _short(e: ValidationError) -> str:
    return "; ".join(f"{'.'.join(map(str, err['loc']))}: {err['msg']}" for err in e.errors()[:5])
