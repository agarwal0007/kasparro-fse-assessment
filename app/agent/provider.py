"""
Provider abstraction for one model turn. Real providers (OpenAIProvider today;
Anthropic/Gemini would be additional classes with the same interface) wrap a
live API. FakeProvider is a scripted, deterministic stand-in used by tests and
by pipeline-validation runs when no live API key is configured - it never
calls a network.
"""
import asyncio
import json
import uuid
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


class GeminiProvider(LLMProvider):
    """Wraps Gemini's function-calling protocol behind the same complete_turn
    interface. Our internal message list is OpenAI-shaped (system/user/
    assistant-with-tool_calls/tool); this converts it to Gemini's
    system_instruction + Content(role="user"|"model") turns on every call,
    since the SDK is stateless per-request and we already keep the full
    message history ourselves in runner.py. Only one tool exists in this
    project (get_brand_facts), so tool-result messages (which carry only a
    tool_call_id in our OpenAI-shaped history, not a name) are converted back
    to a function_response under that one known name."""

    TOOL_NAME = "get_brand_facts"

    def __init__(self, model: str, api_key: str):
        from google import genai
        self._client = genai.Client(api_key=api_key)
        self._model = model
        # Gemini 3.x requires echoing a "thought_signature" back on a function-call
        # Part in later turns. One provider instance is shared across all answers
        # in a run (and they execute concurrently), so call IDs must be globally
        # unique - a uuid, not a per-answer counter - or two answers' caches could
        # collide. Keyed by that unique call_id, never cleared (bounded by one run).
        self._thought_signatures: dict[str, bytes] = {}

    @staticmethod
    def _strip_unsupported_schema_keys(schema):
        """Gemini's function-parameters Schema is a Google proto type, not a
        full JSON Schema - it rejects keys like 'additionalProperties' that
        OpenAI's tool-schema format includes. Strip recursively."""
        if isinstance(schema, dict):
            return {
                k: GeminiProvider._strip_unsupported_schema_keys(v)
                for k, v in schema.items()
                if k not in ("additionalProperties",)
            }
        if isinstance(schema, list):
            return [GeminiProvider._strip_unsupported_schema_keys(v) for v in schema]
        return schema

    def _to_gemini_tool(self, tools: list[dict]):
        from google.genai import types

        declarations = []
        for t in tools:
            fn = t["function"]
            declarations.append(types.FunctionDeclaration(
                name=fn["name"], description=fn.get("description", ""),
                parameters=self._strip_unsupported_schema_keys(fn.get("parameters")),
            ))
        return [types.Tool(function_declarations=declarations)]

    def _to_gemini_contents(self, messages: list[dict]):
        from google.genai import types

        system_instruction = None
        contents = []
        for m in messages:
            role = m["role"]
            if role == "system":
                system_instruction = m["content"]
            elif role == "user":
                contents.append(types.Content(role="user", parts=[types.Part(text=m["content"])]))
            elif role == "assistant":
                if m.get("tool_calls"):
                    parts = [
                        types.Part(
                            function_call=types.FunctionCall(
                                name=tc["function"]["name"],
                                args=json.loads(tc["function"]["arguments"]),
                            ),
                            thought_signature=self._thought_signatures.get(tc["id"]),
                        )
                        for tc in m["tool_calls"]
                    ]
                    contents.append(types.Content(role="model", parts=parts))
                else:
                    contents.append(types.Content(role="model", parts=[types.Part(text=m.get("content") or "")]))
            elif role == "tool":
                response_dict = json.loads(m["content"])
                contents.append(types.Content(role="user", parts=[
                    types.Part(function_response=types.FunctionResponse(
                        name=self.TOOL_NAME, response=response_dict,
                    ))
                ]))
        return system_instruction, contents

    async def complete_turn(self, messages: list[dict], tools: list[dict], timeout_s: float) -> ModelTurnResult:
        from google.genai import errors, types

        system_instruction, contents = self._to_gemini_contents(messages)
        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            tools=self._to_gemini_tool(tools),
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        )

        try:
            response = await asyncio.wait_for(
                self._client.aio.models.generate_content(model=self._model, contents=contents, config=config),
                timeout=timeout_s,
            )
        except asyncio.TimeoutError as e:
            raise TransientProviderError(str(e)) from e
        except errors.ServerError as e:
            raise TransientProviderError(str(e)) from e
        except errors.ClientError as e:
            if getattr(e, "code", None) == 429:
                raise TransientProviderError(str(e)) from e
            raise PermanentProviderError(str(e)) from e
        except errors.APIError as e:
            raise PermanentProviderError(str(e)) from e

        usage = {}
        if response.usage_metadata:
            usage = {
                "input_tokens": response.usage_metadata.prompt_token_count,
                "output_tokens": response.usage_metadata.candidates_token_count,
            }

        if not response.candidates:
            raise PermanentProviderError("Gemini returned no candidates.")
        parts = response.candidates[0].content.parts or []

        call_parts = [p for p in parts if p.function_call is not None]
        if call_parts:
            calls = []
            for p in call_parts:
                call_id = f"gemini-{uuid.uuid4().hex}"
                if p.thought_signature:
                    self._thought_signatures[call_id] = p.thought_signature
                calls.append(ToolCallRequest(call_id=call_id, name=p.function_call.name,
                                              arguments=dict(p.function_call.args or {})))
            return ModelTurnResult(tool_calls=tuple(calls), final_content=None, usage=usage)

        text = "".join(p.text or "" for p in parts)
        return ModelTurnResult(tool_calls=(), final_content=text, usage=usage)


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
