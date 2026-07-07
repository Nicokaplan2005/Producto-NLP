"""LLM adapter boundary.

Replace this file when the backend decision is made. The rest of the pipeline
should call these functions instead of importing a provider SDK directly.
"""

from __future__ import annotations

import json
import os
import platform
import re
import ssl
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import TypeVar
from urllib.error import URLError
from urllib.request import Request, urlopen

from pydantic import BaseModel, ValidationError

from .env import load_project_env

try:
    import certifi
except ImportError:  # pragma: no cover - optional runtime hardening.
    certifi = None

load_project_env()

ModelT = TypeVar("ModelT", bound=BaseModel)


class LLMNotConfiguredError(RuntimeError):
    """Raised when a real LLM backend has not been wired yet."""


class LLMResponseError(RuntimeError):
    """Raised when an LLM backend returns unusable output."""


def llm_backend() -> str:
    return os.getenv("LLM_BACKEND", "stub").lower()


def llm_is_configured() -> bool:
    return llm_backend() != "stub"


def predict_text(
    prompt: str,
    *,
    system_prompt: str | None = None,
    response_mime_type: str = "text/plain",
    max_output_tokens: int | None = None,
    context: str | None = None,
) -> str:
    backend = os.getenv("LLM_BACKEND", "stub").lower()
    if backend == "stub":
        raise LLMNotConfiguredError(
            "LLM_BACKEND=stub: no real LLM backend configured yet."
        )
    if backend in {"openai", "openai_compatible", "local", "vllm"}:
        return _predict_text_openai_compatible(
            prompt,
            system_prompt=system_prompt,
            response_mime_type=response_mime_type,
            max_output_tokens=max_output_tokens,
            context=context,
        )
    if backend == "vertex_mistral":
        return _predict_text_vertex_mistral(
            prompt,
            system_prompt=system_prompt,
            max_output_tokens=max_output_tokens,
            context=context,
        )
    if backend == "vertex_gemma":
        return _predict_text_vertex_gemma(
            prompt,
            system_prompt=system_prompt,
            max_output_tokens=max_output_tokens,
            context=context,
        )
    raise LLMNotConfiguredError(f"Unsupported LLM_BACKEND={backend!r}.")


def predict_model(
    model: type[ModelT],
    prompt: str,
    *,
    system_prompt: str | None = None,
    context: str | None = None,
    max_output_tokens: int | None = None,
) -> ModelT:
    resolved_tokens = max_output_tokens or int(os.getenv("GEMINI_JSON_MAX_OUTPUT_TOKENS", "8192"))
    text = predict_text(
        prompt,
        system_prompt=system_prompt,
        response_mime_type="application/json",
        max_output_tokens=resolved_tokens,
        context=context,
    )
    _write_debug_text("raw", text)
    try:
        return model.model_validate(json.loads(_extract_json_text(text)))
    except (json.JSONDecodeError, ValidationError) as exc:
        repaired = _repair_json_with_llm(model, text, original_error=str(exc))
        _write_debug_text("repaired", repaired)
        return model.model_validate(json.loads(_extract_json_text(repaired)))


def _repair_json_with_llm(
    model: type[ModelT], broken_text: str, *, original_error: str
) -> str:
    if llm_backend() == "stub":
        raise LLMNotConfiguredError("Cannot repair JSON with LLM_BACKEND=stub.")
    schema_hint = json.dumps(model.model_json_schema(), ensure_ascii=False)[:20000]
    repair_prompt = "\n\n".join(
        [
            "Repair the following model output into one valid JSON object.",
            "Return only JSON. No markdown. No comments. No trailing commas. No ellipses.",
            f"Pydantic model name: {model.__name__}",
            f"Original parse/validation error: {original_error}",
            "Schema hint:",
            schema_hint,
            "Broken output:",
            _extract_json_text(broken_text)[:60000],
        ]
    )
    return predict_text(
        repair_prompt,
        system_prompt="You repair invalid JSON into valid schema-compatible JSON.",
        response_mime_type="application/json",
        max_output_tokens=int(os.getenv("GEMINI_JSON_MAX_OUTPUT_TOKENS", "8192")),
        context=f"json_repair:{model.__name__}",
    )


def _predict_text_vertex_gemma(
    prompt: str,
    *,
    system_prompt: str | None,
    max_output_tokens: int | None,
    context: str | None,
) -> str:
    _configure_tls()
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise LLMNotConfiguredError("Install google-genai to use LLM_BACKEND=vertex_gemma.") from exc

    project = os.getenv("VERTEX_PROJECT") or os.getenv("GOOGLE_CLOUD_PROJECT")
    location = os.getenv("VERTEX_GEMMA_LOCATION", os.getenv("VERTEX_LOCATION", "global"))
    model = os.getenv("VERTEX_GEMMA_MODEL", "google/gemma-4-26b-a4b-it-maas")

    if not project:
        raise LLMNotConfiguredError("Set VERTEX_PROJECT for vertex_gemma backend.")

    client = genai.Client(vertexai=True, project=project, location=location)
    config_kwargs: dict = {
        "temperature": 0.0,
        "max_output_tokens": max_output_tokens or 8192,
    }
    if system_prompt:
        config_kwargs["system_instruction"] = system_prompt

    started = time.perf_counter()
    try:
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(**config_kwargs),
        )
    except Exception as exc:
        raise LLMResponseError(f"Vertex Gemma request failed: {exc}") from exc
    elapsed_seconds = time.perf_counter() - started

    _write_usage_log({
        "backend": "vertex_gemma",
        "model": model,
        "context": context or "",
        "elapsed_seconds": round(elapsed_seconds, 3),
        "prompt_chars": len(prompt),
        "system_prompt_chars": len(system_prompt or ""),
        "usage_metadata": _model_dump(getattr(response, "usage_metadata", None)),
    })

    if isinstance(response.text, str) and response.text:
        return response.text
    raise LLMResponseError(f"Empty Vertex Gemma response: {_model_json(response)[:500]}")


def _predict_text_openai_compatible(
    prompt: str,
    *,
    system_prompt: str | None,
    response_mime_type: str,
    max_output_tokens: int | None,
    context: str | None,
) -> str:
    api_key = (
        os.getenv("LOCAL_LLM_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or "api-key-not-set"
    )
    model = (
        os.getenv("LOCAL_LLM_MODEL")
        or os.getenv("OPENAI_MODEL")
        or os.getenv("PYDANTIC_AI_MODEL")
        or "local-model"
    )
    base_url = (
        os.getenv("LOCAL_LLM_BASE_URL")
        or os.getenv("OPENAI_BASE_URL")
        or "http://localhost:8000/v1"
    )
    url = base_url.rstrip("/") + "/chat/completions"
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    payload: dict[str, object] = {
        "model": model,
        "messages": messages,
        "temperature": float(os.getenv("LOCAL_LLM_TEMPERATURE", "0")),
        "max_tokens": max_output_tokens
        if max_output_tokens is not None
        else int(os.getenv("LOCAL_LLM_MAX_OUTPUT_TOKENS", "8192")),
    }
    if (
        response_mime_type == "application/json"
        and os.getenv("LOCAL_LLM_USE_RESPONSE_FORMAT", "1") != "0"
    ):
        payload["response_format"] = {"type": "json_object"}
    started = time.perf_counter()
    response = _post_json_with_headers(
        url,
        payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        secret=api_key,
    )
    elapsed_seconds = time.perf_counter() - started
    _write_usage_log(
        {
            "backend": "openai_compatible",
            "model": model,
            "base_url": base_url,
            "context": context or "",
            "response_mime_type": response_mime_type,
            "elapsed_seconds": round(elapsed_seconds, 3),
            "prompt_chars": len(prompt),
            "system_prompt_chars": len(system_prompt or ""),
            "usage": response.get("usage") or {},
        }
    )
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMResponseError(
            "Unexpected OpenAI-compatible response shape: "
            f"{_redact_secret(json.dumps(response)[:1000], api_key)}"
        ) from exc
    if not isinstance(content, str):
        raise LLMResponseError("OpenAI-compatible response content was not text.")
    return content


def _post_json(
    url: str, payload: dict[str, object], *, api_key: str
) -> dict[str, object]:
    retries = int(os.getenv("GEMINI_RETRIES", "3"))
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            return _post_json_once(url, payload, api_key=api_key)
        except LLMResponseError as exc:
            last_error = exc
            if attempt >= retries or not _is_retryable_error(str(exc)):
                raise
            base_sleep = float(os.getenv("GEMINI_RETRY_BASE_SLEEP_SECONDS", "2"))
            max_sleep = float(os.getenv("GEMINI_RETRY_MAX_SLEEP_SECONDS", "60"))
            time.sleep(min(base_sleep * (2**attempt), max_sleep))
    raise LLMResponseError(str(last_error))


def _post_json_once(
    url: str, payload: dict[str, object], *, api_key: str
) -> dict[str, object]:
    body = json.dumps(payload).encode("utf-8")
    request = Request(url, data=body, headers={"Content-Type": "application/json"})
    try:
        context = _ssl_context()
        with urlopen(request, timeout=120, context=context) as response:
            return json.loads(response.read().decode("utf-8"))
    except (ssl.SSLError, URLError) as exc:
        if platform.system().lower() == "windows":
            return _post_json_with_powershell(url, payload, api_key=api_key)
        raise LLMResponseError(
            _redact_secret(f"Gemini request failed: {exc}", api_key)
        ) from exc


def _ssl_context() -> ssl.SSLContext | None:
    if certifi is None:
        return None
    return ssl.create_default_context(cafile=certifi.where())


def _configure_tls() -> None:
    try:
        import truststore

        truststore.inject_into_ssl()
    except Exception:
        if certifi is None:
            return
        ca_bundle = certifi.where()
        os.environ.setdefault("SSL_CERT_FILE", ca_bundle)
        os.environ.setdefault("REQUESTS_CA_BUNDLE", ca_bundle)


def _post_json_with_powershell(
    url: str, payload: dict[str, object], *, api_key: str
) -> dict[str, object]:
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", delete=False, suffix=".json"
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False)
        body_path = handle.name
    ps_script = "\n".join(
        [
            "$ProgressPreference = 'SilentlyContinue'",
            f"$body = Get-Content -LiteralPath '{body_path}' -Raw -Encoding UTF8",
            "$response = Invoke-WebRequest "
            f"-UseBasicParsing -Uri '{url}' "
            "-Method POST -ContentType 'application/json' -Body $body -TimeoutSec 120",
            "$response.Content",
        ]
    )
    try:
        import subprocess

        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps_script],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    finally:
        try:
            os.remove(body_path)
        except OSError:
            pass
    if result.returncode != 0:
        message = _redact_secret(
            result.stderr.strip() or result.stdout.strip(), api_key
        )
        raise LLMResponseError(f"Gemini PowerShell request failed: {message}")
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        stdout = _redact_secret(result.stdout[:1000], api_key)
        stderr = _redact_secret(result.stderr[:1000], api_key)
        raise LLMResponseError(
            f"Gemini PowerShell returned non-JSON response. stdout={stdout!r} stderr={stderr!r}"
        ) from exc


def _predict_text_vertex_mistral(
    prompt: str,
    *,
    system_prompt: str | None,
    max_output_tokens: int | None,
    context: str | None,
) -> str:
    _configure_tls()
    try:
        import google.auth
        import google.auth.transport.requests
    except ImportError as exc:
        raise LLMNotConfiguredError("Install google-auth to use LLM_BACKEND=vertex_mistral.") from exc

    project = os.getenv("VERTEX_PROJECT") or os.getenv("GOOGLE_CLOUD_PROJECT")
    location = os.getenv("VERTEX_LOCATION", "us-central1")
    model_id = os.getenv("VERTEX_MISTRAL_MODEL", "mistral-medium-3")

    if not project:
        raise LLMNotConfiguredError("Set VERTEX_PROJECT or GOOGLE_CLOUD_PROJECT for vertex_mistral.")

    url = (
        f"https://{location}-aiplatform.googleapis.com/v1"
        f"/projects/{project}/locations/{location}"
        f"/publishers/mistralai/models/{model_id}:rawPredict"
    )

    creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    creds.refresh(google.auth.transport.requests.Request())

    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    payload: dict[str, object] = {
        "model": model_id,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": max_output_tokens if max_output_tokens is not None else 8192,
    }

    started = time.perf_counter()
    response = _post_json_with_headers(
        url,
        payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {creds.token}",
        },
        secret=creds.token,
    )
    elapsed_seconds = time.perf_counter() - started
    _write_usage_log({
        "backend": "vertex_mistral",
        "model": model_id,
        "context": context or "",
        "elapsed_seconds": round(elapsed_seconds, 3),
        "prompt_chars": len(prompt),
        "system_prompt_chars": len(system_prompt or ""),
        "usage": response.get("usage") or {},
    })

    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMResponseError(f"Unexpected Vertex Mistral response: {str(response)[:500]}") from exc
    if not isinstance(content, str):
        raise LLMResponseError("Vertex Mistral response content was not text.")
    return content


def _post_json_with_headers(
    url: str,
    payload: dict[str, object],
    *,
    headers: dict[str, str],
    secret: str,
) -> dict[str, object]:
    body = json.dumps(payload).encode("utf-8")
    request = Request(url, data=body, headers=headers)
    try:
        context = _ssl_context() if url.lower().startswith("https://") else None
        with urlopen(request, timeout=300, context=context) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        raise LLMResponseError(
            _redact_secret(f"OpenAI-compatible request failed: {exc}", secret)
        ) from exc


def _extract_json_text(text: str) -> str:
    value = text.strip()
    if value.startswith("```"):
        value = re.sub(r"^```(?:json)?\s*", "", value)
        value = re.sub(r"\s*```$", "", value)
    return value


def _redact_secret(text: str, secret: str) -> str:
    return text.replace(secret, "[REDACTED]")


def _model_dump(value: object) -> object:
    if value is None:
        return {}
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")  # type: ignore[no-any-return]
    if hasattr(value, "__dict__"):
        return dict(value.__dict__)
    return str(value)


def _model_json(value: object) -> str:
    if hasattr(value, "model_dump_json"):
        return value.model_dump_json(exclude_none=True)
    return str(value)


def _is_retryable_error(message: str) -> bool:
    lowered = message.lower()
    retryable_markers = ("(503)", " 503", "(429)", " 429", "timeout", "temporar")
    return any(marker in lowered for marker in retryable_markers)


def _write_debug_text(prefix: str, text: str) -> None:
    debug_dir = os.getenv("LLM_DEBUG_DIR")
    if not debug_dir:
        return
    path = Path(debug_dir)
    path.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    (path / f"{prefix}_{stamp}.txt").write_text(text, encoding="utf-8")


def _write_usage_log(payload: dict[str, object]) -> None:
    log_path = os.getenv("LLM_USAGE_LOG_PATH")
    if not log_path:
        return
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
