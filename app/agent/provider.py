"""
Provider abstraction for one model turn. Real providers (OpenAIProvider today;
Anthropic/Gemini would be additional classes with the same interface) wrap a
live API. FakeProvider is a scripted, deterministic stand-in used by tests and
by pipeline-validation runs when no live API key is configured - it never
calls a network.
"""
import asyncio
import json
from dataclasses import dataclass, field


class ProviderError(Exception):
    """Base class. Carries whether a retry is sensible."""
    retryable = False


class TransientProviderError(ProviderError):
    """Timeout, rate limit, or an explicitly transient failure - retry once."""
    retryable = True


class PermanentProviderError(ProviderError):
    """Malformed output or an authentication failure - never retry."""
    retryable = False


@dataclass(frozen=True)
class ToolCallRequest:
    call_id: str
    name: str
    arguments: dict


@dataclass(frozen=True)
class ModelTurnResult:
    tool_calls: tuple  # of ToolCallRequest, empty if this turn produced a final answer
    final_content: str | None  # raw text (expected to be JSON) if no tool call
    usage: dict = field(default_factory=dict)  # {"input_tokens":.., "output_tokens":..} if known


class LLMProvider:
    async def complete_turn(self, messages: list[dict], tools: list[dict], timeout_s: float) -> ModelTurnResult:
        raise NotImplementedError


class OpenAIProvider(LLMProvider):
    def __init__(self, model: str, api_key: str):
        from openai import AsyncOpenAI
        self._client = AsyncOpenAI(api_key=api_key)
        self._model = model

    async def complete_turn(self, messages: list[dict], tools: list[dict], timeout_s: float) -> ModelTurnResult:
        from openai import APITimeoutError, AuthenticationError, RateLimitError, APIError

        try:
            response = await asyncio.wait_for(
                self._client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    tools=tools,
                    tool_choice="auto",
                ),
                timeout=timeout_s,
            )
        except (asyncio.TimeoutError, APITimeoutError, RateLimitError) as e:
            raise TransientProviderError(str(e)) from e
        except AuthenticationError as e:
            raise PermanentProviderError(str(e)) from e
        except APIError as e:
            raise PermanentProviderError(str(e)) from e

        choice = response.choices[0]
        usage = {}
        if response.usage:
            usage = {
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
            }

        if choice.message.tool_calls:
            calls = tuple(
                ToolCallRequest(
                    call_id=tc.id,
                    name=tc.function.name,
                    arguments=json.loads(tc.function.arguments),
                )
                for tc in choice.message.tool_calls
            )
            return ModelTurnResult(tool_calls=calls, final_content=None, usage=usage)

        return ModelTurnResult(tool_calls=(), final_content=choice.message.content, usage=usage)


class FakeProvider(LLMProvider):
    """Returns a pre-scripted sequence of ModelTurnResult (or raises a
    pre-scripted exception) on successive calls - one entry consumed per
    complete_turn call, in order. Used by tests, and by the no-key
    pipeline-validation run."""

    def __init__(self, script: list):
        self._script = list(script)
        self._calls = 0

    async def complete_turn(self, messages: list[dict], tools: list[dict], timeout_s: float) -> ModelTurnResult:
        if self._calls >= len(self._script):
            raise PermanentProviderError("FakeProvider script exhausted")
        item = self._script[self._calls]
        self._calls += 1
        if isinstance(item, Exception):
            raise item
        return item
