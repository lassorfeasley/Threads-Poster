"""Claude calls for the clip engine: streamed, schema-constrained JSON.

``llm._create`` serves the app's short calls. Engine calls differ enough to
need their own path:

- They are long (a 150-frame perception chunk, a ranking request with ~150
  images, 32k-token drafts), so they stream: the SDK refuses a non-streaming
  request whose ``max_tokens`` implies more than ~10 minutes.
- Claude 5.x always thinks (adaptive) and rejects ``temperature``; depth is set
  with ``output_config.effort`` and nothing else is sent.
- The answer is constrained to a JSON schema (``output_config.format``), so a
  parse failure means the answer was cut off or declined. Those surface as
  ``ClaudeTruncated`` / ``ClaudeRefused``. Spend is recorded from the response
  usage BEFORE raising and the error carries ``.usage``, so a failed call still
  counts against the engine's cap.
- Strict schemas reject numeric/string/array constraints and need every object
  closed (``additionalProperties: false``, every key required);
  ``strict_schema`` adapts the Gemini-flavoured schemas the engine shares.

Server-side fallback (beta ``server-side-fallback-2026-07-01`` with
``fallbacks: "default"``): Opus 5.5's safety classifiers occasionally decline
benign news footage. With the beta the API re-runs a declined request on
Anthropic's recommended fallback model inside the same call. The answer then
follows a ``fallback`` content block (the declined model's partial text comes
before it), and ``usage.iterations`` itemises every attempt with the model that
ran it — spend is recorded per attempt from that. A 400 that names the
fallback/beta retries once without both (and one that names ``effort``, once
without it, so a role pointed at a model without effort control still runs).

The request uses the non-beta ``messages.stream`` with the header and body
field added by hand. Its stream accumulator (SDK 0.117) drops
``usage.iterations`` from ``message_delta`` and keeps the requested model's
name after a mid-answer fallback, so both are read off the raw events while the
stream is consumed.
"""
from __future__ import annotations

import base64
import copy
import importlib
import logging
import time
from typing import Any

import anthropic

from .. import llm, spend
from .gemini import parse_json_object


def _transport_errors() -> tuple[type[BaseException], ...]:
    """Errors from the SDK's HTTP layer, which a stream that breaks off midway
    raises unwrapped. The SDK talks through httpx up to 0.x and httpx2 (the
    same API, separate classes) from 2.x; both may be installed, so catch each."""
    errors: list[type[BaseException]] = [OSError]
    for name in ("httpx", "httpx2"):
        try:
            errors.append(importlib.import_module(name).HTTPError)
        except ImportError:
            continue
    return tuple(errors)


_TRANSPORT_ERRORS = _transport_errors()

log = logging.getLogger("clip_engine.claude")

FALLBACK_BETA = "server-side-fallback-2026-07-01"

# JSON Schema keywords Claude's structured outputs reject (constraints are
# validated by our own cleaning code instead) plus Gemini's propertyOrdering.
_DROP_KEYS = frozenset({
    "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf",
    "minLength", "maxLength", "minItems", "maxItems", "uniqueItems", "pattern",
    "propertyOrdering",
})
# When a type list splits into anyOf branches, these follow their type.
_BRANCH_KEYS = {
    "object": ("properties", "required", "additionalProperties"),
    "array": ("items", "prefixItems"),
    "string": ("format",),
}
_SHARED_KEYS = ("description", "title")


class ClaudeError(RuntimeError):
    """A Claude call failed. ``usage`` is set when the failed response was
    billed (it has already been recorded in the spend ledger); ``status`` is
    the HTTP status when there was one."""

    usage: dict | None = None

    def __init__(self, message: str = "", *, usage: dict | None = None,
                 status: int | None = None) -> None:
        super().__init__(message)
        self.usage = usage
        self.status = status


class ClaudeRefused(ClaudeError):
    """Every model in the fallback chain declined (``stop_reason: refusal``)."""


class ClaudeTruncated(ClaudeError):
    """The answer hit ``max_tokens`` (thinking counts toward it)."""


def is_configured() -> bool:
    from ..config import env

    return bool(env("ANTHROPIC_API_KEY", "").strip())


# --- schemas ------------------------------------------------------------------------------

def strict_schema(schema: dict) -> dict:
    """A copy of ``schema`` that Claude's strict structured outputs accept.

    Drops numeric/string/array constraints and ``propertyOrdering``; closes
    every object (``additionalProperties: false``, ``required`` = all of its
    properties); turns a type list such as ``["number", "null"]`` into
    ``anyOf`` branches (``description`` stays on the outside, an ``enum`` is
    split across the branches whose type can hold its values); ``nullable``
    and ``oneOf`` become the same. The input is never modified."""
    return _strict(copy.deepcopy(schema))


def _strict(node: Any) -> Any:
    if isinstance(node, list):
        return [_strict(x) for x in node]
    if not isinstance(node, dict):
        return node
    node = {k: v for k, v in node.items() if k not in _DROP_KEYS}
    if node.pop("nullable", False) is True and isinstance(node.get("type"), str):
        node["type"] = [node["type"], "null"]
    if "oneOf" in node and "anyOf" not in node:
        node["anyOf"] = node.pop("oneOf")
    if isinstance(node.get("type"), list):
        return _split_types(node)
    for key in ("properties", "$defs", "definitions"):
        if isinstance(node.get(key), dict):
            node[key] = {name: _strict(sub) for name, sub in node[key].items()}
    for key in ("items", "prefixItems", "anyOf", "allOf"):
        if key in node:
            node[key] = _strict(node[key])
    if node.get("type") == "object" or isinstance(node.get("properties"), dict):
        props = node.get("properties") if isinstance(node.get("properties"), dict) else {}
        node["properties"] = props
        node["required"] = list(props)
        node["additionalProperties"] = False
    return node


def _holds(value: Any, type_name: str) -> bool:
    """Whether a JSON value is an instance of a JSON Schema type."""
    if type_name == "null":
        return value is None
    if type_name == "boolean":
        return isinstance(value, bool)
    if type_name == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if type_name == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if type_name == "string":
        return isinstance(value, str)
    if type_name == "array":
        return isinstance(value, list)
    if type_name == "object":
        return isinstance(value, dict)
    return True


def _split_types(node: dict) -> dict:
    names = list(dict.fromkeys(str(t) for t in node["type"]))
    if len(names) == 1:
        return _strict({**node, "type": names[0]})
    enum = node.get("enum") if isinstance(node.get("enum"), list) else None
    branches = []
    for name in names:
        branch: dict[str, Any] = {"type": name}
        branch.update({k: node[k] for k in _BRANCH_KEYS.get(name, ()) if k in node})
        if enum is not None:
            allowed = [v for v in enum if _holds(v, name)]
            if not allowed:  # this type can never satisfy the enum
                continue
            if name != "null":
                branch["enum"] = allowed
        branches.append(_strict(branch))
    out: dict[str, Any] = {k: node[k] for k in _SHARED_KEYS if k in node}
    if branches:
        out["anyOf"] = branches
    else:  # an enum none of the listed types can hold: keep it as written
        out["enum"] = enum
    return out


# --- content blocks ---------------------------------------------------------------------------

def text_block(text: str) -> dict:
    return {"type": "text", "text": str(text)}


def image_block(jpeg: bytes) -> dict:
    return {"type": "image", "source": {
        "type": "base64", "media_type": "image/jpeg",
        "data": base64.standard_b64encode(bytes(jpeg)).decode("ascii")}}


def frames_blocks(frames: list[tuple[float, bytes]], *, clock: str = "video") -> list[dict]:
    """Each frame as a time label then the image, in order: "t=12.0s" on the
    source clock (``clock="video"``) or "clip 12.0s" from the start of a
    rendered clip (``clock="clip"``). Separate labelled images rather than a
    contact sheet, so the model can say exactly when something happens.
    Frames without bytes are skipped."""
    if clock not in ("video", "clip"):
        raise ValueError(f"clock must be 'video' or 'clip', not {clock!r}")
    blocks: list[dict] = []
    for t, jpeg in frames:
        if not jpeg:
            log.debug("skipping an empty frame at %s", t)
            continue
        label = f"t={float(t):.1f}s" if clock == "video" else f"clip {float(t):.1f}s"
        blocks.append(text_block(label))
        blocks.append(image_block(jpeg))
    return blocks


# --- the call ---------------------------------------------------------------------------------

def _field(obj: Any, name: str) -> Any:
    """``obj.name`` or ``obj[name]``: the non-beta SDK keeps fields it doesn't
    model (fallback blocks, usage.iterations) as plain dicts."""
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _int(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


class _Seen:
    """What the raw stream events said that the non-beta accumulator drops."""

    def __init__(self) -> None:
        self.iterations: list | None = None
        self.served_by: str | None = None
        self.fallbacks = 0

    def note(self, event: Any) -> None:
        kind = _field(event, "type")
        if kind == "message_delta":
            iterations = _field(_field(event, "usage"), "iterations")
            if iterations:
                self.iterations = list(iterations)
        elif kind == "content_block_start":
            block = _field(event, "content_block")
            if _field(block, "type") == "fallback":
                self.fallbacks += 1
                served = _field(_field(block, "to"), "model")
                if served:
                    self.served_by = str(served)


def _request(model: str, *, system: str, content: Any, schema: dict, effort: str | None,
             max_tokens: int) -> dict:
    output_config: dict[str, Any] = {}
    effort = str(effort or "").strip().lower()
    if effort:
        output_config["effort"] = effort
    output_config["format"] = {"type": "json_schema", "schema": strict_schema(schema)}
    request: dict[str, Any] = {
        "model": model,
        "max_tokens": int(max_tokens),
        "messages": [{"role": "user", "content": content}],
        "output_config": output_config,
    }
    if system:
        request["system"] = system
    return request


def _stream(client: Any, request: dict, *, fallbacks: bool) -> tuple[Any, _Seen]:
    kwargs = dict(request)
    if fallbacks:
        kwargs["extra_headers"] = {"anthropic-beta": FALLBACK_BETA}
        kwargs["extra_body"] = {"fallbacks": "default"}
    seen = _Seen()
    with client.messages.stream(**kwargs) as stream:
        for event in stream:
            seen.note(event)
        message = stream.get_final_message()
    return message, seen


def _api_message(exc: BaseException) -> str:
    body = getattr(exc, "body", None)
    error = body.get("error") if isinstance(body, dict) else None
    message = error.get("message") if isinstance(error, dict) else None
    return str(message or exc)[:300]


def _api_error(exc: BaseException, label: str) -> ClaudeError:
    status = getattr(exc, "status_code", None)
    if isinstance(exc, anthropic.AuthenticationError):
        message = f"{label}: Claude rejected the API key (check ANTHROPIC_API_KEY)"
    elif isinstance(exc, anthropic.APIStatusError) and (status or 0) < 400:
        # An ``error`` event inside a 200 stream (e.g. overloaded mid-answer):
        # the HTTP status says nothing, the error type does.
        body = getattr(exc, "body", None)
        error = body.get("error") if isinstance(body, dict) else None
        kind = error.get("type") if isinstance(error, dict) else None
        message = f"{label}: Claude stopped mid-answer ({kind or 'error'}): {_api_message(exc)}"
        status = None
    elif isinstance(exc, anthropic.APIStatusError):
        message = f"{label}: Claude API error {status}: {_api_message(exc)}"
    elif isinstance(exc, anthropic.APITimeoutError):
        message = f"{label}: Claude timed out"
    elif isinstance(exc, anthropic.APIConnectionError):
        message = f"{label}: could not reach Claude: {exc}"
    elif isinstance(exc, _TRANSPORT_ERRORS):
        message = f"{label}: the Claude stream broke off: {exc}"
    else:
        message = f"{label}: Claude call failed: {exc}"
    return ClaudeError(message, status=status)


def _call(request: dict, label: str) -> tuple[Any, _Seen]:
    """Stream the request with server-side fallbacks.

    One-shot downgrades on a 400: one naming the fallback beta retries without
    the beta header and ``fallbacks``; one naming ``effort`` (a model without
    effort control, e.g. an operator-chosen Haiku 4.5) retries at the model's
    default effort. SDK and transport errors become ClaudeError."""
    client = llm.client()
    fallbacks = True
    while True:
        try:
            return _stream(client, request, fallbacks=fallbacks)
        except anthropic.BadRequestError as exc:
            text = f"{exc} {getattr(exc, 'body', '')}".lower()
            config = request.get("output_config") or {}
            if fallbacks and ("fallback" in text or "beta" in text):
                fallbacks = False
                log.warning("%s: Claude rejected server-side fallback (%s); retrying without it",
                            label, _api_message(exc))
            elif "effort" in config and "effort" in text:
                request = {**request, "output_config": {k: v for k, v in config.items()
                                                        if k != "effort"}}
                log.warning("%s: Claude rejected effort %r (%s); retrying at the model's default",
                            label, config["effort"], _api_message(exc))
            else:
                raise _api_error(exc, label) from exc
        except (anthropic.APIError, *_TRANSPORT_ERRORS) as exc:
            raise _api_error(exc, label) from exc


def _usage_tokens(usage: Any) -> tuple[int, int]:
    """(input incl. cache writes and reads, output) for a usage object/entry."""
    tokens_in = (_int(_field(usage, "input_tokens")) + _int(_field(usage, "cache_creation_input_tokens"))
                 + _int(_field(usage, "cache_read_input_tokens")))
    return tokens_in, _int(_field(usage, "output_tokens"))


def _record_spend(message: Any, seen: _Seen, *, model: str, bucket: str | None,
                  label: str) -> dict:
    """Record what the call cost and return the usage dict.

    With ``usage.iterations`` (one entry per attempt when a fallback ran) each
    attempt is priced at the model that ran it; top-level usage covers only
    the attempt that answered, so it is used only when there are no
    iterations."""
    bucket = bucket or spend.current_bucket() or "engine"
    served = seen.served_by or _field(message, "model") or model
    usage = _field(message, "usage")
    iterations = seen.iterations or _field(usage, "iterations") or []
    entries = [e for e in iterations
               if _field(e, "input_tokens") is not None or _field(e, "output_tokens") is not None]
    tokens_in = tokens_out = 0
    cost = 0.0
    if entries:
        for entry in entries:
            entry_in, entry_out = _usage_tokens(entry)
            cost += spend.record(str(_field(entry, "model") or served), entry_in, entry_out,
                                 bucket=bucket)
            tokens_in += entry_in
            tokens_out += entry_out
    else:
        if usage is None:
            log.warning("%s: the response carried no usage; recording 0 tokens", label)
        tokens_in, tokens_out = _usage_tokens(usage)
        cost = spend.record(str(served), tokens_in, tokens_out, bucket=bucket)
    return {"model": str(served), "input_tokens": tokens_in, "output_tokens": tokens_out,
            "cost_usd": round(cost, 6)}


def _raise_for_stop(message: Any, usage: dict, label: str) -> None:
    reason = _field(message, "stop_reason")
    if reason == "refusal":
        details = _field(message, "stop_details")
        category = _field(details, "category") or "no category given"
        explanation = _field(details, "explanation")
        raise ClaudeRefused(f"{label}: Claude declined ({category})"
                            + (f": {explanation}" if explanation else ""), usage=usage)
    if reason == "max_tokens":
        raise ClaudeTruncated(f"{label}: the answer hit max_tokens ({usage['output_tokens']} "
                              "output tokens incl. thinking)", usage=usage)


def _answer_texts(message: Any) -> list[str]:
    """Text blocks after the last ``fallback`` block — anything before it is
    the declined model's abandoned partial answer."""
    blocks = list(_field(message, "content") or [])
    start = 0
    for i, block in enumerate(blocks):
        if _field(block, "type") == "fallback":
            start = i + 1
    texts = []
    for block in blocks[start:]:
        text = _field(block, "text")
        if _field(block, "type") == "text" and isinstance(text, str):
            texts.append(text)
    return texts


def json_call(model: str, *, system: str, content, schema: dict, effort: str = "medium",
              max_tokens: int = 32000, label: str = "",
              bucket: str | None = "engine") -> tuple[dict, dict]:
    """One streamed, schema-constrained Claude call; returns ``(data, usage)``.

    ``content`` is the user turn (a string or content blocks). No temperature,
    no ``thinking`` (5.x thinks adaptively); ``effort`` sets the depth. Usage
    ``{"model", "input_tokens", "output_tokens", "cost_usd"}`` is recorded
    under ``bucket`` (else the process default, else "engine") before any
    check. Raises ``ClaudeRefused`` / ``ClaudeTruncated`` (with ``.usage``),
    or ``ClaudeError`` for an API failure or an answer that isn't a JSON
    object (with ``.usage`` when billed)."""
    model = str(model or "").strip()
    label = label or model or "claude"
    if not model:
        raise ClaudeError(f"{label}: no Claude model configured")
    request = _request(model, system=system, content=content, schema=schema, effort=effort,
                       max_tokens=max_tokens)
    started = time.monotonic()
    message, seen = _call(request, label)
    usage = _record_spend(message, seen, model=model, bucket=bucket, label=label)
    _raise_for_stop(message, usage, label)
    texts = _answer_texts(message)
    data = parse_json_object(texts[-1]) if texts else None
    if data is None and len(texts) > 1:
        data = parse_json_object("".join(texts))
    if data is None:
        shown = texts[-1] if texts else ""
        raise ClaudeError(f"{label}: Claude did not return a JSON object: {shown[:200]!r}",
                          usage=usage)
    log.info("%s: %s in=%d out=%d $%.4f %.1fs%s", label, usage["model"], usage["input_tokens"],
             usage["output_tokens"], usage["cost_usd"], time.monotonic() - started,
             f" (after {seen.fallbacks} fallback)" if seen.fallbacks else "")
    return data, usage
