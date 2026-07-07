"""Feature extraction agent with CLI tool loop via LangGraph.

The agent receives only three paths (card, diff, repo). It uses CLI tools to
read the card and diff, then explores the repo freely before calling
submit_features() with the final EnhancedPRFeatures JSON.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.prebuilt import create_react_agent
from pydantic import ValidationError

from .env import load_project_env
from .prompts import CLI_FEATURE_EXTRACTION_SYSTEM_PROMPT
from .schemas import EnhancedPRFeatures

load_project_env()

MAX_TOOL_OUTPUT_CHARS = 8_000


# ---------------------------------------------------------------------------
# LLM factory
# ---------------------------------------------------------------------------

def _build_llm() -> Any:
    backend = os.getenv("LLM_BACKEND", "stub").lower()
    if backend in {"openai", "openai_compatible", "local", "vllm"}:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=os.getenv("LOCAL_LLM_MODEL") or os.getenv("OPENAI_MODEL", "local-model"),
            base_url=os.getenv("LOCAL_LLM_BASE_URL") or os.getenv("OPENAI_BASE_URL", "http://localhost:8000/v1"),
            api_key=os.getenv("LOCAL_LLM_API_KEY") or os.getenv("OPENAI_API_KEY", "not-set"),
            temperature=0,
        )
    if backend == "vertex_mistral":
        return _build_vertex_mistral_llm()
    if backend == "vertex_gemma":
        return _build_vertex_gemma_llm()
    raise RuntimeError(
        f"LLM_BACKEND={backend!r} not supported. "
        "Set LLM_BACKEND=vertex_gemma, vertex_mistral, or openai_compatible."
    )


def _build_vertex_mistral_llm() -> Any:
    """Custom BaseChatModel for Vertex AI Mistral rawPredict (urllib, same as setup script)."""
    import json as _json
    import ssl as _ssl
    import time as _time
    import urllib.request
    import urllib.error
    import google.auth
    import google.auth.transport.requests
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
    from langchain_core.outputs import ChatGeneration, ChatResult

    project = os.getenv("VERTEX_PROJECT") or os.getenv("GOOGLE_CLOUD_PROJECT")
    location = os.getenv("VERTEX_LOCATION", "us-central1")
    model_id = os.getenv("VERTEX_MISTRAL_MODEL", "mistral-medium-3")

    if not project:
        raise RuntimeError("Set VERTEX_PROJECT for vertex_mistral backend.")

    rawpredict_url = (
        f"https://{location}-aiplatform.googleapis.com/v1"
        f"/projects/{project}/locations/{location}"
        f"/publishers/mistralai/models/{model_id}:rawPredict"
    )

    creds, _ = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    creds.refresh(google.auth.transport.requests.Request())

    def _get_token() -> str:
        if not creds.valid:
            creds.refresh(google.auth.transport.requests.Request())
        return creds.token

    def _to_messages(messages) -> list:
        result = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                result.append({"role": "system", "content": str(msg.content)})
            elif isinstance(msg, HumanMessage):
                result.append({"role": "user", "content": str(msg.content)})
            elif isinstance(msg, AIMessage):
                m: dict = {"role": "assistant", "content": str(msg.content or "")}
                if msg.tool_calls:
                    m["tool_calls"] = [
                        {"id": tc["id"], "type": "function",
                         "function": {"name": tc["name"], "arguments": _json.dumps(tc["args"])}}
                        for tc in msg.tool_calls
                    ]
                result.append(m)
            elif isinstance(msg, ToolMessage):
                result.append({"role": "tool", "tool_call_id": msg.tool_call_id or "",
                                "content": str(msg.content)})
        return result

    def _to_tools(tools) -> list:
        result = []
        for t in tools:
            schema = {}
            if hasattr(t, "args_schema") and t.args_schema:
                schema = t.args_schema.model_json_schema()
            result.append({"type": "function", "function": {
                "name": t.name, "description": t.description or "", "parameters": schema,
            }})
        return result

    def _call(payload: dict) -> dict:
        body = _json.dumps(payload).encode("utf-8")
        try:
            import certifi as _certifi
            ctx = _ssl.create_default_context(cafile=_certifi.where())
        except Exception:
            ctx = _ssl.create_default_context()
        req = urllib.request.Request(
            rawpredict_url, data=body,
            headers={"Authorization": f"Bearer {_get_token()}", "Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=120, context=ctx) as resp:
                return _json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"HTTP {e.code}: {e.read().decode('utf-8', errors='replace')[:500]}") from e

    class _ChatVertexMistral(BaseChatModel):
        _bound_tools: list = []

        def bind_tools(self, tools, **kwargs) -> "_ChatVertexMistral":
            new = _ChatVertexMistral()
            new._bound_tools = list(tools)
            return new

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            payload: dict = {
                "model": model_id,
                "messages": _to_messages(messages),
                "temperature": 0,
                "max_tokens": 4096,
            }
            if self._bound_tools:
                payload["tools"] = _to_tools(self._bound_tools)
                payload["tool_choice"] = "auto"

            started = _time.perf_counter()
            data = _call(payload)
            elapsed = _time.perf_counter() - started

            _write_usage_log({
                "backend": "vertex_mistral",
                "model": model_id,
                "elapsed_seconds": round(elapsed, 3),
                "usage": data.get("usage") or {},
            })

            msg = data["choices"][0]["message"]
            tool_calls = []
            for tc in msg.get("tool_calls") or []:
                args = tc["function"].get("arguments", "{}")
                try:
                    args = _json.loads(args)
                except Exception:
                    args = {}
                tool_calls.append({
                    "name": tc["function"]["name"],
                    "args": args,
                    "id": tc.get("id", f"call_{len(tool_calls)}"),
                    "type": "tool_call",
                })

            return ChatResult(generations=[ChatGeneration(message=AIMessage(
                content=msg.get("content") or "",
                tool_calls=tool_calls,
            ))])

        @property
        def _llm_type(self) -> str:
            return "vertex-mistral"

    return _ChatVertexMistral()


def _build_vertex_gemma_llm() -> Any:
    """Custom LangChain LLM wrapping genai.Client for Vertex AI Gemma MaaS."""
    from google import genai as _genai
    from google.genai import types as _types
    from pydantic import PrivateAttr
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
    from langchain_core.outputs import ChatGeneration, ChatResult

    project = os.getenv("VERTEX_PROJECT") or os.getenv("GOOGLE_CLOUD_PROJECT")
    location = os.getenv("VERTEX_GEMMA_LOCATION", os.getenv("VERTEX_LOCATION", "global"))
    model_id = os.getenv("VERTEX_GEMMA_MODEL", "google/gemma-4-26b-a4b-it-maas")

    if not project:
        raise RuntimeError("Set VERTEX_PROJECT for vertex_gemma backend.")

    _client = _genai.Client(vertexai=True, project=project, location=location)

    def _to_genai_contents(messages):
        contents = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                continue
            if isinstance(msg, HumanMessage):
                contents.append(_types.Content(
                    role="user", parts=[_types.Part(text=str(msg.content))],
                ))
            elif isinstance(msg, AIMessage):
                if msg.tool_calls:
                    parts = [
                        _types.Part(function_call=_types.FunctionCall(
                            name=tc["name"], args=tc["args"]
                        ))
                        for tc in msg.tool_calls
                    ]
                else:
                    parts = [_types.Part(text=str(msg.content or ""))]
                contents.append(_types.Content(role="model", parts=parts))
            elif isinstance(msg, ToolMessage):
                contents.append(_types.Content(
                    role="user",
                    parts=[_types.Part(function_response=_types.FunctionResponse(
                        name=msg.name or "",
                        response={"output": str(msg.content)},
                    ))],
                ))
        return contents

    def _to_genai_tools(tools):
        declarations = []
        for tool in tools:
            schema = {}
            if hasattr(tool, "args_schema") and tool.args_schema:
                schema = tool.args_schema.model_json_schema()
            declarations.append(_types.FunctionDeclaration(
                name=tool.name,
                description=tool.description or "",
                parameters=schema,
            ))
        return [_types.Tool(function_declarations=declarations)]

    class _ChatVertexGemma(BaseChatModel):
        _bound_tools: list = PrivateAttr(default_factory=list)

        def bind_tools(self, tools, **kwargs) -> "_ChatVertexGemma":
            new = _ChatVertexGemma()
            object.__setattr__(new, "_bound_tools", list(tools))
            return new

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            import time as _time
            system_instruction = next(
                (m.content for m in messages if isinstance(m, SystemMessage)), None
            )
            config_kwargs: dict = {"temperature": 0}
            if system_instruction:
                config_kwargs["system_instruction"] = system_instruction
            if self._bound_tools:
                config_kwargs["tools"] = _to_genai_tools(self._bound_tools)

            started = _time.perf_counter()
            response = _client.models.generate_content(
                model=model_id,
                contents=_to_genai_contents(messages),
                config=_types.GenerateContentConfig(**config_kwargs),
            )
            elapsed = _time.perf_counter() - started

            _write_usage_log({
                "backend": "vertex_gemma",
                "model": model_id,
                "elapsed_seconds": round(elapsed, 3),
                "usage_metadata": _dump_usage(getattr(response, "usage_metadata", None)),
            })

            parts = response.candidates[0].content.parts
            tool_calls, text_parts = [], []
            for part in parts:
                fc = getattr(part, "function_call", None)
                if fc and getattr(fc, "name", None):
                    tool_calls.append({
                        "name": fc.name,
                        "args": dict(fc.args or {}),
                        "id": f"call_{fc.name}_{len(tool_calls)}",
                        "type": "tool_call",
                    })
                elif getattr(part, "text", None):
                    text_parts.append(part.text)

            return ChatResult(generations=[ChatGeneration(message=AIMessage(
                content="".join(text_parts),
                tool_calls=tool_calls,
            ))])

        @property
        def _llm_type(self) -> str:
            return "vertex-gemma"

    return _ChatVertexGemma()


# ---------------------------------------------------------------------------
# Usage logging helpers (shared by Gemma and Mistral wrappers)
# ---------------------------------------------------------------------------

def _write_usage_log(payload: dict) -> None:
    log_path = os.getenv("LLM_USAGE_LOG_PATH")
    if not log_path:
        return
    from pathlib import Path as _Path
    import json as _json
    p = _Path(log_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(_json.dumps(payload, ensure_ascii=False) + "\n")


def _dump_usage(obj: Any) -> dict:
    if obj is None:
        return {}
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    if hasattr(obj, "__dict__"):
        return dict(obj.__dict__)
    return {}


# ---------------------------------------------------------------------------
# Tool builder (closures so repo_dir is captured)
# ---------------------------------------------------------------------------

def _build_tools(repo_dir: Path) -> list:

    def read_file(path: str) -> str:
        """Read the full content of a file. Accepts absolute paths or paths relative to REPO_DIR."""
        p = _resolve(path, repo_dir)
        if not p.is_file():
            return f"File not found: {path}"
        return _truncate(p.read_text(encoding="utf-8", errors="replace"), MAX_TOOL_OUTPUT_CHARS)

    def list_dir(path: str) -> str:
        """List the contents of a directory. Accepts absolute paths or paths relative to REPO_DIR."""
        p = _resolve(path, repo_dir)
        if not p.is_dir():
            return f"Directory not found: {path}"
        entries = sorted(p.iterdir(), key=lambda e: (e.is_file(), e.name))
        return "\n".join(f"{'  ' if e.is_file() else 'd '}{e.name}" for e in entries[:300])

    def head_file(path: str, lines: int = 50) -> str:
        """Read the first N lines of a file."""
        p = _resolve(path, repo_dir)
        if not p.is_file():
            return f"File not found: {path}"
        return "\n".join(p.read_text(encoding="utf-8", errors="replace").splitlines()[:lines])

    def search(pattern: str, path: str = ".", include: str = "") -> str:
        """Search for a regex pattern in files. path is absolute or relative to REPO_DIR."""
        p = _resolve(path, repo_dir)
        cmd = ["grep", "-r", "-n", "--max-count=5", pattern, str(p)]
        if include:
            cmd += ["--include", include]
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True,
                encoding="utf-8", errors="replace", timeout=15,
            )
            output = result.stdout or result.stderr or "(no matches)"
        except subprocess.TimeoutExpired:
            output = "(search timed out)"
        return _truncate(output, MAX_TOOL_OUTPUT_CHARS)

    def submit_features(features_json: str) -> str:
        """Submit the complete EnhancedPRFeatures JSON. Returns validation errors if invalid so you can fix and retry."""
        try:
            EnhancedPRFeatures.model_validate(json.loads(features_json))
            return features_json  # valid — extraction loop captures this
        except json.JSONDecodeError as e:
            return f"JSON parse error — fix and resubmit: {e}"
        except ValidationError as e:
            return f"Validation failed — fix these fields and call submit_features again:\n{e}"

    return [
        StructuredTool.from_function(read_file,      name="read_file"),
        StructuredTool.from_function(list_dir,        name="list_dir"),
        StructuredTool.from_function(head_file,       name="head_file"),
        StructuredTool.from_function(search,          name="search"),
        StructuredTool.from_function(submit_features, name="submit_features"),
    ]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def extract_features(
    diff_path: Path,
    card_path: Path,
    repo_dir: Path,
    *,
    pr_url: str | None = None,
    max_iterations: int = 30,
) -> EnhancedPRFeatures:
    """Run the LangGraph CLI tool loop and return validated EnhancedPRFeatures."""
    diff_path = diff_path.resolve()
    card_path = card_path.resolve()
    repo_dir = repo_dir.resolve()

    llm = _build_llm()
    tools = _build_tools(repo_dir)
    agent = create_react_agent(llm, tools, prompt=CLI_FEATURE_EXTRACTION_SYSTEM_PROMPT)

    initial_message = (
        f"PR URL:    {pr_url or 'unknown'}\n"
        f"CARD_PATH: {card_path}\n"
        f"DIFF_PATH: {diff_path}\n"
        f"REPO_DIR:  {repo_dir}\n\n"
        "Start by reading the card (read_file CARD_PATH), then the diff "
        "(read_file DIFF_PATH), then explore the repo as needed. "
        "When done, call submit_features with the complete JSON."
    )

    result = agent.invoke(
        {"messages": [HumanMessage(content=initial_message)]},
        config={"recursion_limit": max_iterations},
    )

    return _extract_features_from_result(result)


# ---------------------------------------------------------------------------
# Result extraction
# ---------------------------------------------------------------------------

def _extract_features_from_result(result: dict) -> EnhancedPRFeatures:
    messages = result.get("messages", [])
    debug = os.getenv("AGENT_DEBUG", "0") != "0"

    if debug:
        print(f"\n[DEBUG] Total messages: {len(messages)}")
        for i, msg in enumerate(messages):
            name = getattr(msg, "name", "")
            content_preview = str(getattr(msg, "content", ""))[:120].replace("\n", " ")
            tool_calls = getattr(msg, "tool_calls", [])
            print(f"  [{i}] {msg.type} name={name!r} tool_calls={len(tool_calls)} | {content_preview}")

    # Primary: find the ToolMessage produced by submit_features
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage) and msg.name == "submit_features":
            if debug:
                print(f"\n[DEBUG] submit_features called. Content preview:\n{msg.content[:500]}")
            try:
                return EnhancedPRFeatures.model_validate(json.loads(msg.content))
            except json.JSONDecodeError as e:
                if debug:
                    print(f"[DEBUG] JSON parse error: {e}")
            except ValidationError as e:
                if debug:
                    print(f"[DEBUG] Pydantic validation error:\n{e}")

    if debug:
        print("\n[DEBUG] submit_features never called or all attempts failed. Trying fallback...")

    # Fallback: parse the last AI message that contains a JSON object
    for msg in reversed(messages):
        if msg.type == "ai" and isinstance(msg.content, str):
            raw = _extract_json_block(msg.content)
            if raw:
                try:
                    return EnhancedPRFeatures.model_validate(json.loads(raw))
                except (json.JSONDecodeError, ValidationError) as e:
                    if debug:
                        print(f"[DEBUG] Fallback parse failed: {e}")

    raise RuntimeError("Agent finished without producing valid EnhancedPRFeatures.")


def _extract_json_block(text: str) -> str | None:
    value = text.strip()
    if value.startswith("```"):
        value = re.sub(r"^```(?:json)?\s*", "", value)
        value = re.sub(r"\s*```$", "", value)
    match = re.search(r"\{[\s\S]*\}", value)
    return match.group() if match else None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve(raw: str, repo_dir: Path) -> Path:
    p = Path(raw)
    return p if p.is_absolute() else repo_dir / p


def _truncate(text: str, max_chars: int) -> str:
    return text if len(text) <= max_chars else text[:max_chars] + "\n...[truncated]..."
