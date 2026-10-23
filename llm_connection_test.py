#!/usr/bin/env python3
"""Benchmark OpenAI-compatible Chat Completions and tool-calling agents.

Run from this directory with:
    .venv/bin/python llm_connection_test.py

Use --json for structured output.
"""

from __future__ import annotations

import argparse
import html
import http.client
import json
import logging
import math
import os
import re
import socket
import ssl
import sys
import tempfile
import threading
import time
import tomllib
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Iterator, TextIO
from urllib.parse import SplitResult, urlsplit, urlunsplit

import tiktoken
from azure.core.exceptions import ClientAuthenticationError, ServiceRequestError
from azure.identity import ClientSecretCredential, CredentialUnavailableError, ManagedIdentityCredential, get_bearer_token_provider


class ConfigurationError(ValueError):
    """Raised when config.toml cannot describe a valid benchmark."""


class BenchmarkError(RuntimeError):
    """Raised when a benchmark request cannot complete."""


class CallFailure(BenchmarkError):
    def __init__(self, record: dict[str, Any]):
        super().__init__(record["error"])
        self.record = record


@dataclass(frozen=True)
class BenchmarkConfig:
    endpoint: str
    models: tuple[str, ...]
    api_key: str
    auth_header: str
    auth_scheme: str
    timeout_seconds: float
    rounds: int
    continue_on_error: bool
    system_prompt: str
    user_prompt: str
    max_tokens: int | None
    max_completion_tokens: int | None
    temperature: float | None
    reasoning_effort: str | None
    include_usage: bool
    extra_headers: dict[str, str]
    mode: str = "streaming"
    scenarios: tuple[str, ...] = ("baseline",)
    connection_reuse: bool = True
    warmup_rounds: int = 0
    tokenizer_model: str = ""
    short_requests: int = 20
    short_token_budget: int = 1024
    long_token_budget: int = 8192
    prefill_short_words: int = 128
    prefill_long_words: int = 4096
    agent_max_steps: int = 4
    agent_max_calls: int = 12
    agent_token_budget: int = 1024
    agent_total_token_budget: int = 12288
    task_timeout_seconds: float = 600
    auth_type: str = "api_key"
    azure_scope: str = "https://cognitiveservices.azure.com/.default"
    azure_tenant_id: str = ""
    azure_client_id: str = ""
    azure_client_secret_env: str = "AZURE_CLIENT_SECRET"
    azure_managed_identity_client_id: str | None = ""
    azure_client_secret: str = field(default="", repr=False)
    protocol: str = "chat_completions"


@dataclass
class Timing:
    started_at: float
    dns_seconds: float | None = None
    tcp_seconds: float | None = None
    tls_seconds: float | None = None
    request_sent_at: float | None = None
    response_headers_at: float | None = None
    first_body_at: float | None = None
    first_event_at: float | None = None
    first_token_at: float | None = None
    last_token_at: float | None = None
    completed_at: float | None = None
    deadline: float | None = None
    tcp_attempts: int = 0


@dataclass
class StreamResult:
    text: str
    usage: dict[str, Any] | None
    event_count: int
    first_chunk: str = ""
    text_chunk_count: int = 0
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    finish_reason: str | None = None
    server_timing: dict[str, Any] = field(default_factory=dict)
    response_output: list[dict[str, Any]] = field(default_factory=list)


_ENV_REFERENCE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
_METRIC_NAMES = (
    "connection_setup_ms",
    "response_headers_ms",
    "first_sse_event_ms",
    "time_to_first_token_ms",
    "first_token_after_request_ms",
    "end_to_end_ms",
    "generation_duration_ms",
    "output_tokens_per_second",
    "ttft_ms",
    "ttlt_ms",
    "prefill_proxy_ms",
    "tps",
    "end_to_end_output_tokens_per_second",
    "authentication_ms",
)
_REASONING_EFFORTS = frozenset(
    {"none", "minimal", "low", "medium", "high", "xhigh", "max"}
)
_MODES = frozenset({"streaming", "non_streaming", "both"})
_PROTOCOL_PATHS = {"chat_completions": "/chat/completions", "responses": "/responses"}
_PROTOCOLS = frozenset({*_PROTOCOL_PATHS, "both"})
_AUTH_CHAIN = ("azure_managed_identity", "azure_app_registration", "api_key")
_AUTH_TYPES = frozenset({"auto", *_AUTH_CHAIN})
_SCENARIOS = (
    "baseline", "prefill_short", "prefill_long", "long_output",
    "short_burst", "short_conversation", "agent", "agent_serial", "agent_parallel",
)


def expand_environment(value: str, field_name: str) -> str:
    """Expand ${NAME}, failing explicitly when an environment variable is absent."""

    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        environment_value = os.environ.get(name)
        if environment_value is None:
            raise ConfigurationError(
                f"{field_name} references environment variable {name!r}, "
                "but it is not set."
            )
        return environment_value

    return _ENV_REFERENCE.sub(replace, value)


def get_table(document: dict[str, Any], name: str) -> dict[str, Any]:
    value = document.get(name, {})
    if not isinstance(value, dict):
        raise ConfigurationError(f"[{name}] must be a TOML table.")
    return value


def get_string(
    table: dict[str, Any],
    name: str,
    *,
    default: str | None = None,
    required: bool = False,
) -> str:
    value = table.get(name, default)
    if value is None:
        if required:
            raise ConfigurationError(f"Missing required setting: {name}.")
        return ""
    if not isinstance(value, str):
        raise ConfigurationError(f"{name} must be a string.")
    return value


def get_boolean(table: dict[str, Any], name: str, default: bool) -> bool:
    value = table.get(name, default)
    if not isinstance(value, bool):
        raise ConfigurationError(f"{name} must be true or false.")
    return value


def get_number(table: dict[str, Any], name: str, default: float) -> float:
    value = table.get(name, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigurationError(f"{name} must be a number.")
    result = float(value)
    if not math.isfinite(result):
        raise ConfigurationError(f"{name} must be finite.")
    return result


def get_optional_number(table: dict[str, Any], name: str) -> float | None:
    if name not in table:
        return None
    return get_number(table, name, 0)


def get_optional_positive_integer(table: dict[str, Any], name: str) -> int | None:
    if name not in table:
        return None
    value = table[name]
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigurationError(f"{name} must be a positive integer.")
    return value


def get_optional_choice(
    table: dict[str, Any], name: str, allowed_values: frozenset[str]
) -> str | None:
    if name not in table:
        return None
    value = table[name]
    if not isinstance(value, str) or value not in allowed_values:
        choices = ", ".join(sorted(allowed_values))
        raise ConfigurationError(f"{name} must be one of: {choices}.")
    return value


def get_positive_number(
    table: dict[str, Any], name: str, default: float
) -> float:
    value = get_number(table, name, default)
    if value <= 0:
        raise ConfigurationError(f"{name} must be greater than zero.")
    return value


def get_positive_integer(table: dict[str, Any], name: str, default: int) -> int:
    value = table.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigurationError(f"{name} must be a positive integer.")
    return value


def get_models(table: dict[str, Any]) -> tuple[str, ...]:
    values = table.get("models")
    if not isinstance(values, list) or not values:
        raise ConfigurationError("llm.models must be a non-empty TOML string list.")

    models: list[str] = []
    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise ConfigurationError(
                "Every llm.models item must be a non-empty string."
            )
        models.append(value.strip())
    return tuple(models)


def load_config(config_path: Path, *, require_credentials: bool = True) -> BenchmarkConfig:
    try:
        with config_path.open("rb") as config_file:
            document = tomllib.load(config_file)
    except FileNotFoundError as error:
        raise ConfigurationError(f"Configuration file not found: {config_path}") from error
    except tomllib.TOMLDecodeError as error:
        raise ConfigurationError(f"Invalid TOML in {config_path}: {error}") from error

    llm = get_table(document, "llm")
    benchmark = get_table(document, "benchmark")
    request = get_table(document, "request")
    headers = get_table(document, "headers")
    scenarios = get_table(document, "scenarios")
    agents = get_table(document, "agents")
    azure_auth = get_table(document, "azure_auth")

    endpoint = get_string(llm, "endpoint", required=True).strip()
    if not endpoint:
        raise ConfigurationError("llm.endpoint cannot be empty.")

    auth_type = get_optional_choice(llm, "auth_type", _AUTH_TYPES) or "api_key"
    uses_api_key = auth_type in {"api_key", "auto"}
    api_key = get_string(llm, "api_key", default="")
    api_key_env = get_string(llm, "api_key_env", default="").strip()
    if not uses_api_key and api_key:
        raise ConfigurationError("Remove llm.api_key when using Azure identity authentication.")
    if uses_api_key and api_key and api_key_env:
        raise ConfigurationError(
            "Use either llm.api_key or llm.api_key_env, not both."
        )
    if uses_api_key and api_key_env:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", api_key_env):
            raise ConfigurationError("llm.api_key_env is not a valid variable name.")
        api_key = (os.environ.get(api_key_env, "") if auth_type == "auto"
                   else expand_environment("${" + api_key_env + "}", "llm.api_key_env")) if require_credentials else ""
    elif api_key and require_credentials:
        if auth_type == "auto" and any(not os.environ.get(match.group(1)) for match in _ENV_REFERENCE.finditer(api_key)):
            api_key = ""
        else:
            api_key = expand_environment(api_key, "llm.api_key")

    auth_header = get_string(llm, "auth_header", default="Authorization").strip()
    auth_scheme = get_string(llm, "auth_scheme", default="Bearer").strip()
    if auth_type == "api_key" and require_credentials and auth_header and not api_key:
        raise ConfigurationError(
            "llm.auth_header is set, but no API key was supplied. Set "
            "llm.api_key_env, llm.api_key, or leave llm.auth_header empty."
        )

    system_prompt = get_string(request, "system_prompt", default="")
    user_prompt = get_string(request, "user_prompt", required=True)
    if not user_prompt.strip():
        raise ConfigurationError("request.user_prompt cannot be empty.")

    max_tokens = get_optional_positive_integer(request, "max_tokens")
    max_completion_tokens = get_optional_positive_integer(
        request, "max_completion_tokens"
    )
    if max_tokens is not None and max_completion_tokens is not None:
        raise ConfigurationError(
            "Set only one of request.max_tokens or request.max_completion_tokens."
        )
    if max_tokens is None and max_completion_tokens is None:
        max_tokens = 256

    extra_headers: dict[str, str] = {}
    for header_name, value in headers.items():
        if not isinstance(header_name, str) or not isinstance(value, str):
            raise ConfigurationError("All [headers] names and values must be strings.")
        extra_headers[header_name] = expand_environment(
            value, f"headers.{header_name}"
        ) if require_credentials else value

    azure_scope = get_string(azure_auth, "scope", default="https://cognitiveservices.azure.com/.default").strip()
    app_registration = auth_type == "azure_app_registration"
    uses_app_registration = app_registration or auth_type == "auto"
    azure_tenant_id = get_string(azure_auth, "tenant_id", default="${AZURE_TENANT_ID}" if uses_app_registration else "").strip()
    azure_client_id = get_string(azure_auth, "client_id", default="${AZURE_CLIENT_ID}" if uses_app_registration else "").strip()
    azure_client_secret = get_string(azure_auth, "client_secret", default="")
    azure_client_secret_env = get_string(azure_auth, "client_secret_env",
        default="" if "client_secret" in azure_auth else "AZURE_CLIENT_SECRET").strip()
    managed_identity_client_id: str | None = get_string(azure_auth, "managed_identity_client_id", default="").strip()
    if auth_type != "api_key":
        if parse_endpoint(endpoint).scheme != "https":
            raise ConfigurationError("Azure identity authentication requires an HTTPS endpoint.")
        if not azure_scope.endswith("/.default") or any(character.isspace() for character in azure_scope):
            raise ConfigurationError("azure_auth.scope must be a single resource scope ending in /.default.")
        if any(name.lower() in {"authorization", "api-key"} for name in headers):
            raise ConfigurationError("Do not set Authorization or api-key headers when using Azure identity authentication.")
        if uses_app_registration and azure_client_secret and azure_client_secret_env:
            raise ConfigurationError("Use either azure_auth.client_secret or client_secret_env, not both.")
        if require_credentials:
            def resolve_identity_setting(value: str, name: str) -> str | None:
                if auth_type == "auto" and any(not os.environ.get(match.group(1)) for match in _ENV_REFERENCE.finditer(value)):
                    return None
                return expand_environment(value, f"azure_auth.{name}").strip()

            azure_tenant_id = resolve_identity_setting(azure_tenant_id, "tenant_id") or ""
            azure_client_id = resolve_identity_setting(azure_client_id, "client_id") or ""
            managed_identity_client_id = resolve_identity_setting(managed_identity_client_id, "managed_identity_client_id")
        if auth_type == "azure_managed_identity" and managed_identity_client_id:
            azure_client_id = managed_identity_client_id
        if uses_app_registration and azure_client_secret_env and not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", azure_client_secret_env):
            raise ConfigurationError("azure_auth.client_secret_env must be an environment variable name.")
        if app_registration:
            if not azure_tenant_id or not azure_client_id:
                raise ConfigurationError("App Registration requires azure_auth.tenant_id and client_id.")
            if not azure_client_secret and not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", azure_client_secret_env):
                raise ConfigurationError("azure_auth.client_secret_env must be an environment variable name.")
            if require_credentials and not azure_client_secret and not os.environ.get(azure_client_secret_env):
                raise ConfigurationError("Set azure_auth.client_secret or the configured client secret environment variable.")

    selected = benchmark.get("scenarios", ["baseline"])
    if not isinstance(selected, list) or not selected or any(
        not isinstance(name, str) or name not in _SCENARIOS for name in selected
    ):
        raise ConfigurationError(f"benchmark.scenarios must select from {', '.join(_SCENARIOS)}.")
    if len(set(selected)) != len(selected):
        raise ConfigurationError("benchmark.scenarios must not contain duplicates.")
    warmups = benchmark.get("warmup_rounds", 0)
    if isinstance(warmups, bool) or not isinstance(warmups, int) or warmups < 0:
        raise ConfigurationError("warmup_rounds must be a non-negative integer.")
    parse_endpoint(endpoint)

    return BenchmarkConfig(
        endpoint=endpoint,
        models=get_models(llm),
        api_key=api_key,
        auth_header=auth_header,
        auth_scheme=auth_scheme,
        timeout_seconds=get_positive_number(llm, "timeout_seconds", 120),
        rounds=get_positive_integer(benchmark, "rounds", 3),
        continue_on_error=get_boolean(benchmark, "continue_on_error", True),
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_tokens=max_tokens,
        max_completion_tokens=max_completion_tokens,
        temperature=get_optional_number(request, "temperature"),
        reasoning_effort=get_optional_choice(
            request, "reasoning_effort", _REASONING_EFFORTS
        ),
        include_usage=get_boolean(request, "include_usage", True),
        extra_headers=extra_headers,
        mode=get_optional_choice(benchmark, "mode", _MODES) or "streaming",
        scenarios=tuple(selected),
        connection_reuse=get_boolean(benchmark, "connection_reuse", True),
        warmup_rounds=warmups,
        tokenizer_model=get_string(benchmark, "tokenizer_model", default=""),
        short_requests=get_positive_integer(scenarios, "short_requests", 20),
        short_token_budget=get_positive_integer(scenarios, "short_token_budget", 1024),
        long_token_budget=get_positive_integer(scenarios, "long_token_budget", 8192),
        prefill_short_words=get_positive_integer(scenarios, "prefill_short_words", 128),
        prefill_long_words=get_positive_integer(scenarios, "prefill_long_words", 4096),
        agent_max_steps=get_positive_integer(agents, "max_steps", 4),
        agent_max_calls=get_positive_integer(agents, "max_calls", 12),
        agent_token_budget=get_positive_integer(agents, "token_budget", 1024),
        agent_total_token_budget=get_positive_integer(agents, "total_token_budget", 12288),
        task_timeout_seconds=get_positive_number(benchmark, "task_timeout_seconds", 600),
        auth_type=auth_type,
        azure_scope=azure_scope,
        azure_tenant_id=azure_tenant_id,
        azure_client_id=azure_client_id,
        azure_client_secret_env=azure_client_secret_env,
        azure_managed_identity_client_id=managed_identity_client_id,
        azure_client_secret=azure_client_secret,
        protocol=get_optional_choice(llm, "protocol", _PROTOCOLS) or "chat_completions",
    )


def parse_endpoint(endpoint: str) -> SplitResult:
    try:
        parsed = urlsplit(endpoint)
        port = parsed.port
    except ValueError:
        raise ConfigurationError("llm.endpoint contains an invalid host or port.") from None
    if parsed.scheme not in {"http", "https"}:
        raise ConfigurationError("llm.endpoint must start with http:// or https://.")
    if not parsed.hostname:
        raise ConfigurationError("llm.endpoint must include a hostname.")
    if parsed.username or parsed.password:
        raise ConfigurationError("Do not embed credentials in llm.endpoint.")
    if parsed.fragment:
        raise ConfigurationError("llm.endpoint must not include a URL fragment.")
    if port == 0 or parsed.netloc.endswith(":"):
        raise ConfigurationError("llm.endpoint port must be an integer between 1 and 65535.")
    return parsed


def display_endpoint(parsed: SplitResult) -> str:
    """Avoid showing endpoint query strings, which may contain credentials."""

    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))


def remaining_seconds(deadline: float) -> float:
    remaining = deadline - time.perf_counter()
    if remaining <= 0:
        raise BenchmarkError("Request/task deadline exceeded.")
    return remaining


def open_timed_tcp_socket(
    host: str, port: int, timeout_seconds: float, timing: Timing,
    *, deadline: float | None = None,
) -> socket.socket:
    dns_started_at = time.perf_counter()
    deadline = min(deadline if deadline is not None else math.inf, dns_started_at + timeout_seconds)
    remaining_seconds(deadline)
    try:
        addresses = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except socket.gaierror as error:
        raise BenchmarkError(f"DNS lookup failed for {host!r}: {error}") from error
    finally:
        timing.dns_seconds = time.perf_counter() - dns_started_at

    last_error: OSError | None = None
    for family, socket_type, protocol, _, address in addresses:
        remaining = remaining_seconds(deadline)
        candidate = socket.socket(family, socket_type, protocol)
        candidate.settimeout(remaining)
        tcp_started_at = time.perf_counter()
        timing.tcp_attempts += 1
        try:
            candidate.connect(address)
            candidate.settimeout(remaining_seconds(deadline))
        except OSError as error:
            candidate.close()
            last_error = error
            continue
        except Exception:
            candidate.close()
            raise
        finally:
            timing.tcp_seconds = (timing.tcp_seconds or 0) + time.perf_counter() - tcp_started_at

        return candidate

    message = f"TCP connection failed for {host}:{port}"
    if last_error is not None:
        message += f": {last_error}"
    raise BenchmarkError(message)


class TimedHTTPConnection(http.client.HTTPConnection):
    def __init__(
        self, host: str, port: int, timeout_seconds: float, timing: Timing
    ) -> None:
        super().__init__(host, port=port, timeout=timeout_seconds)
        self._timing = timing
        self._deadline = timing.deadline if timing.deadline is not None else time.perf_counter() + timeout_seconds

    def connect(self) -> None:
        self.sock = open_timed_tcp_socket(
            self.host, self.port, self.timeout, self._timing, deadline=self._deadline
        )
        if self._tunnel_host:
            self._tunnel()

    def send(self, data: Any) -> None:
        if self.sock is None and self.auto_open:
            self.connect()
        if self.sock is not None:
            self.sock.settimeout(remaining_seconds(self._deadline))
        super().send(data)


class TimedHTTPSConnection(http.client.HTTPSConnection):
    def __init__(
        self, host: str, port: int, timeout_seconds: float, timing: Timing
    ) -> None:
        super().__init__(host, port=port, timeout=timeout_seconds)
        self._timing = timing
        self._deadline = timing.deadline if timing.deadline is not None else time.perf_counter() + timeout_seconds

    def connect(self) -> None:
        raw_socket = open_timed_tcp_socket(
            self.host, self.port, self.timeout, self._timing, deadline=self._deadline
        )
        self.sock = raw_socket
        if self._tunnel_host:
            self._tunnel()

        tls_started_at = time.perf_counter()
        try:
            raw_socket.settimeout(remaining_seconds(self._deadline))
            self.sock = self._context.wrap_socket(
                raw_socket, server_hostname=self.host
            )
            self.sock.settimeout(remaining_seconds(self._deadline))
        except Exception:
            raw_socket.close()
            self.close()
            raise
        finally:
            self._timing.tls_seconds = time.perf_counter() - tls_started_at

    def send(self, data: Any) -> None:
        if self.sock is None and self.auto_open:
            self.connect()
        if self.sock is not None:
            self.sock.settimeout(remaining_seconds(self._deadline))
        super().send(data)


def create_connection(
    endpoint: SplitResult, timeout_seconds: float, timing: Timing
) -> http.client.HTTPConnection:
    host = endpoint.hostname
    if host is None:
        raise ConfigurationError("llm.endpoint must include a hostname.")
    if endpoint.scheme == "https":
        return TimedHTTPSConnection(
            host, endpoint.port or 443, timeout_seconds, timing
        )
    return TimedHTTPConnection(host, endpoint.port or 80, timeout_seconds, timing)


def resolve_protocol_endpoint(endpoint: SplitResult, protocol: str) -> SplitResult:
    if protocol not in _PROTOCOL_PATHS:
        raise ConfigurationError("A request must select chat_completions or responses, not both.")
    path = endpoint.path.rstrip("/")
    for suffix in _PROTOCOL_PATHS.values():
        if path.endswith(suffix):
            path = path[:-len(suffix)]
            break
    return endpoint._replace(path=path + _PROTOCOL_PATHS[protocol])


def responses_input(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    items = []
    for message in messages:
        if message.get("responses_output"):
            items.extend(message["responses_output"])
        elif message["role"] == "tool":
            items.append({"type": "function_call_output", "call_id": message["tool_call_id"],
                          "output": message["content"]})
        else:
            if message.get("content") is not None:
                items.append({"role": message["role"], "content": message["content"]})
            for call in message.get("tool_calls", []):
                items.append({"type": "function_call", "call_id": call["id"],
                              "name": call["function"]["name"], "arguments": call["function"]["arguments"]})
    return items


def build_request(
    config: BenchmarkConfig, model: str, *, streaming: bool = True,
    messages: list[dict[str, Any]] | None = None,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: Any = None,
) -> tuple[bytes, dict[str, str]]:
    if messages is None:
        messages = []
        if config.system_prompt:
            messages.append({"role": "system", "content": config.system_prompt})
        messages.append({"role": "user", "content": config.user_prompt})

    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": streaming,
    }
    if config.max_tokens is not None and config.max_completion_tokens is not None:
        raise ConfigurationError(
            "Configure exactly one of request.max_tokens or "
            "request.max_completion_tokens."
        )
    if config.max_tokens is not None:
        payload["max_tokens"] = config.max_tokens
    elif config.max_completion_tokens is not None:
        payload["max_completion_tokens"] = config.max_completion_tokens
    else:
        raise ConfigurationError(
            "Configure exactly one of request.max_tokens or "
            "request.max_completion_tokens."
        )
    if config.temperature is not None:
        payload["temperature"] = config.temperature
    if config.reasoning_effort is not None:
        payload["reasoning_effort"] = config.reasoning_effort
    if streaming and config.include_usage:
        payload["stream_options"] = {"include_usage": True}
    if tools:
        payload["tools"] = tools
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice

    if config.protocol == "responses":
        payload["input"] = responses_input(payload.pop("messages"))
        budget_name = "max_completion_tokens" if "max_completion_tokens" in payload else "max_tokens"
        payload["max_output_tokens"] = payload.pop(budget_name)
        payload["store"] = False
        payload["include"] = ["reasoning.encrypted_content"]
        payload.pop("stream_options", None)
        if config.reasoning_effort is not None:
            payload["reasoning"] = {"effort": payload.pop("reasoning_effort")}
        if tools:
            payload["tools"] = [{"type": "function", **tool["function"]} for tool in tools]
            if isinstance(tool_choice, dict):
                payload["tool_choice"] = {"type": "function", "name": tool_choice["function"]["name"]}
    elif config.protocol != "chat_completions":
        raise ConfigurationError("A request must select chat_completions or responses, not both.")

    headers = {
        "Accept": "text/event-stream" if streaming else "application/json",
        "Content-Type": "application/json",
        **config.extra_headers,
    }
    if config.auth_type == "api_key" and config.auth_header:
        headers[config.auth_header] = (
            f"{config.auth_scheme} {config.api_key}".strip()
        )
    return json.dumps(payload, separators=(",", ":")).encode("utf-8"), headers


def content_from_delta(delta: dict[str, Any]) -> str:
    content = delta.get("content")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""

    text_parts: list[str] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        text = part.get("text")
        if isinstance(text, str):
            text_parts.append(text)
    return "".join(text_parts)


def parse_response_document(document: Any) -> StreamResult:
    if not isinstance(document, dict) or document.get("object") != "response":
        raise BenchmarkError("Expected a Responses API response object.")
    if document.get("error") or document.get("status") in {"failed", "cancelled"}:
        raise BenchmarkError(f"Responses endpoint returned an error: {document.get('error') or document.get('status')}")
    status = document.get("status")
    if status not in {"completed", "incomplete"}:
        raise BenchmarkError("Incomplete Responses response: missing terminal status.")
    output = document.get("output")
    if not isinstance(output, list) or any(not isinstance(item, dict) for item in output):
        raise BenchmarkError("Responses output must be an item list.")
    text_parts = []
    tool_calls = []
    for item in output:
        if item.get("type") == "message":
            content = item.get("content")
            if item.get("role") != "assistant" or not isinstance(content, list):
                raise BenchmarkError("Invalid Responses assistant message.")
            for part in content:
                if not isinstance(part, dict):
                    raise BenchmarkError("Invalid Responses message content.")
                if part.get("type") == "output_text":
                    if not isinstance(part.get("text"), str):
                        raise BenchmarkError("Responses output text must be a string.")
                    text_parts.append(part["text"])
        elif item.get("type") == "function_call":
            if any(not isinstance(item.get(name), str) for name in ("call_id", "name", "arguments")):
                raise BenchmarkError("Invalid Responses function call.")
            tool_calls.append({"id": item["call_id"], "type": "function",
                               "function": {"name": item["name"], "arguments": item["arguments"]}})
    reason = "tool_calls" if tool_calls else "stop"
    if status == "incomplete":
        detail = document.get("incomplete_details") or {}
        reason = "length" if detail.get("reason") == "max_output_tokens" else "incomplete"
    return StreamResult("".join(text_parts), document.get("usage") if isinstance(document.get("usage"), dict) else None,
        0, tool_calls=tool_calls, finish_reason=reason, response_output=output,
        server_timing={name: document[name] for name in ("timings", "timing") if isinstance(document.get(name), dict)})


def consume_sse(response: Any, timing: Timing, result: StreamResult | None = None,
                *, protocol: str = "chat_completions",
                on_first_text: Callable[[], None] | None = None) -> StreamResult:
    """Consume OpenAI-style SSE events and record first-event/TTFT timestamps."""

    text_parts: list[str] = []
    usage: dict[str, Any] | None = None
    event_count = 0
    data_lines: list[str] = []
    result = result if result is not None else StreamResult("", None, 0)
    tool_calls: dict[int, dict[str, Any]] = {}
    output_items: dict[int, dict[str, Any]] = {}
    done = False

    def observe_text(text: str) -> None:
        if text:
            observed_at = time.perf_counter()
            if timing.first_token_at is None:
                timing.first_token_at = observed_at
                result.first_chunk = text
                if on_first_text is not None:
                    on_first_text()
            timing.last_token_at = observed_at
            result.text_chunk_count += 1
            text_parts.append(text)

    def consume_response_event(event: dict[str, Any]) -> None:
        nonlocal done, usage
        event_type = event.get("type")
        if not isinstance(event_type, str):
            raise BenchmarkError("Responses SSE event has no type.")
        if event_type == "error":
            raise BenchmarkError(f"Responses stream returned an error: {event.get('message', 'unknown')}")
        if event_type in {"response.completed", "response.incomplete", "response.failed"}:
            document = event.get("response")
            if not isinstance(document, dict) or document.get("status") != event_type.removeprefix("response."):
                raise BenchmarkError("Responses terminal event/status mismatch.")
            if isinstance(document.get("usage"), dict):
                usage = document["usage"]
            parsed = parse_response_document(document)
            if parsed.text != "".join(text_parts):
                raise BenchmarkError("Responses final text does not match received text deltas.")
            result.finish_reason = parsed.finish_reason
            result.server_timing.update(parsed.server_timing)
            for index, item in enumerate(parsed.response_output):
                output_items[index] = {**output_items.get(index, {}), **item}
            tool_calls.clear()
            tool_calls.update(enumerate(parsed.tool_calls))
            done = True
        elif event_type == "response.output_text.delta":
            if not isinstance(event.get("delta"), str):
                raise BenchmarkError("Responses text delta must be a string.")
            observe_text(event["delta"])
        elif event_type in {"response.output_item.added", "response.output_item.done",
                            "response.function_call_arguments.delta", "response.function_call_arguments.done"}:
            index = event.get("output_index")
            if type(index) is not int or index < 0:
                raise BenchmarkError("Responses output index must be a nonnegative integer.")
            if event_type.startswith("response.output_item."):
                item = event.get("item")
                if not isinstance(item, dict):
                    raise BenchmarkError("Invalid Responses output item event.")
                output_items[index] = item.copy()
                if item.get("type") == "function_call":
                    tool_calls[index] = {"id": item.get("call_id", ""), "type": "function",
                        "function": {"name": item.get("name", ""), "arguments": item.get("arguments", "")}}
            else:
                fragment = event.get("delta" if event_type.endswith(".delta") else "arguments")
                if not isinstance(fragment, str) or index not in tool_calls:
                    raise BenchmarkError("Responses argument fragment has no function call or string value.")
                function = tool_calls[index]["function"]
                function["arguments"] = function["arguments"] + fragment if event_type.endswith(".delta") else fragment
                output_items[index]["arguments"] = function["arguments"]

    def consume_event(serialized_event: str) -> None:
        nonlocal event_count, usage, done
        if serialized_event == "[DONE]":
            if protocol == "responses" and not done:
                raise BenchmarkError("Incomplete Responses stream: missing terminal response event.")
            done = True
            return
        if done:
            raise BenchmarkError("Received SSE data after terminal event or [DONE].")

        try:
            event = json.loads(serialized_event)
        except json.JSONDecodeError as error:
            raise BenchmarkError(
                "Received a non-JSON SSE data event. Ensure llm.endpoint is an "
                "OpenAI-compatible streaming Chat Completions endpoint."
            ) from error
        if not isinstance(event, dict):
            raise BenchmarkError("Received an SSE event with an unexpected shape.")

        if timing.first_event_at is None:
            timing.first_event_at = time.perf_counter()
        event_count += 1

        if event.get("error"):
            raise BenchmarkError(f"LLM stream returned an error: {event['error']}")

        possible_usage = event.get("usage")
        if isinstance(possible_usage, dict):
            usage = possible_usage
        for name in ("timings", "timing"):
            if isinstance(event.get(name), dict):
                result.server_timing[name] = event[name]

        if protocol == "responses":
            consume_response_event(event)
            return

        choices = event.get("choices")
        if not isinstance(choices, list):
            return
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            if choice.get("index", 0) != 0:
                raise BenchmarkError("Multiple completion choices are not supported.")
            if choice.get("finish_reason") is not None:
                result.finish_reason = choice["finish_reason"]
            delta = choice.get("delta")
            if not isinstance(delta, dict):
                continue
            text = content_from_delta(delta)
            observe_text(text)
            fragments = delta.get("tool_calls") or []
            if not isinstance(fragments, list):
                raise BenchmarkError("Invalid tool-call fragment list.")
            for fragment in fragments:
                if not isinstance(fragment, dict):
                    raise BenchmarkError("Invalid tool-call fragment.")
                index = fragment.get("index")
                if not isinstance(index, int) or isinstance(index, bool):
                    raise BenchmarkError("Tool-call fragment has no integer index.")
                target = tool_calls.setdefault(index, {
                    "id": "", "type": "function",
                    "function": {"name": "", "arguments": ""},
                })
                if fragment.get("id"):
                    if not isinstance(fragment["id"], str):
                        raise BenchmarkError("Tool-call id must be a string.")
                    target["id"] += fragment["id"]
                function = fragment.get("function") or {}
                if not isinstance(function, dict):
                    raise BenchmarkError("Invalid tool-call function fragment.")
                for name in ("name", "arguments"):
                    if function.get(name):
                        if not isinstance(function[name], str):
                            raise BenchmarkError("Tool function name/arguments must be strings.")
                        target["function"][name] += function[name]

    try:
        while True:
            raw_line = response.readline()
            if not raw_line:
                break
            if timing.first_body_at is None:
                timing.first_body_at = time.perf_counter()

            try:
                line = raw_line.decode("utf-8").rstrip("\r\n")
            except UnicodeDecodeError as error:
                raise BenchmarkError("The streaming response is not valid UTF-8.") from error

            if not line:
                if data_lines:
                    consume_event("\n".join(data_lines))
                    data_lines.clear()
                continue
            if line.startswith(":"):
                continue
            if line.startswith("data:"):
                data = line[5:]
                if data.startswith(" "):
                    data = data[1:]
                data_lines.append(data)

        if data_lines:
            consume_event("\n".join(data_lines))
    finally:
        result.text = "".join(text_parts)
        result.usage = usage
        result.event_count = event_count
        result.tool_calls = [tool_calls[index] for index in sorted(tool_calls)]
        if protocol == "responses":
            result.response_output = [output_items[index] for index in sorted(output_items)]
    if event_count == 0:
        raise BenchmarkError(
            "The endpoint returned no OpenAI-compatible SSE events. Confirm that "
            "streaming is enabled for this endpoint."
        )

    if not done or result.finish_reason is None:
        if protocol == "responses":
            raise BenchmarkError("Incomplete Responses SSE stream: missing terminal response event.")
        raise BenchmarkError("Incomplete SSE stream: missing [DONE] or finish_reason.")
    return result


def consume_json(response: Any, *, protocol: str = "chat_completions") -> StreamResult:
    try:
        document = json.loads(response.read())
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BenchmarkError("Response is not valid JSON.") from error
    if protocol == "responses":
        return parse_response_document(document)
    if not isinstance(document, dict) or document.get("error"):
        raise BenchmarkError("Endpoint returned an invalid or error JSON response.")
    choices = document.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise BenchmarkError("Expected exactly one completion choice.")
    choice = choices[0]
    if not isinstance(choice, dict) or not isinstance(choice.get("message"), dict):
        raise BenchmarkError("Completion has no assistant message.")
    if not choice.get("finish_reason"):
        raise BenchmarkError("Incomplete completion: missing finish_reason.")
    message = choice["message"]
    tool_calls = message.get("tool_calls") or []
    if not isinstance(tool_calls, list):
        raise BenchmarkError("Invalid completion tool-call list.")
    return StreamResult(
        content_from_delta(message),
        document.get("usage") if isinstance(document.get("usage"), dict) else None,
        0,
        tool_calls=tool_calls,
        finish_reason=choice["finish_reason"],
        server_timing={name: document[name] for name in ("timings", "timing")
                       if isinstance(document.get(name), dict)},
    )


def resolve_tokenizer(config: BenchmarkConfig, model: str) -> Any:
    try:
        return tiktoken.encoding_for_model(config.tokenizer_model or model)
    except KeyError:
        if config.tokenizer_model:
            raise ConfigurationError(f"Unknown tokenizer_model: {config.tokenizer_model}.") from None
        return None


def count_visible_tokens(text: str, tokenizer: Any) -> tuple[int, str]:
    if tokenizer is None:
        return estimate_token_count(text), "estimated"
    return len(tokenizer.encode(text, disallowed_special=())), f"tiktoken:{tokenizer.name}"


def estimate_token_count(text: str) -> int:
    """Use a labelled estimate only when the endpoint omits usage metadata."""

    if not text:
        return 0
    segments = re.findall(r"[A-Za-z0-9_]+|[^\sA-Za-z0-9_]", text)
    return sum(
        max(1, math.ceil(len(segment) / 4))
        if segment[0].isascii() and segment[0].isalnum()
        else 1
        for segment in segments
    )


def usage_value(usage: dict[str, Any] | None, *names: str) -> int | None:
    if not isinstance(usage, dict):
        return None
    for name in names:
        value = usage.get(name)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value
    return None


def milliseconds(start: float | None, end: float | None) -> float | None:
    if start is None or end is None:
        return None
    return round((end - start) * 1_000, 2)


class DeadlineResponse:
    def __init__(self, response: http.client.HTTPResponse, active_socket: Any, deadline: float):
        self.response = response
        self.active_socket = active_socket
        self.deadline = deadline
        self.buffer = bytearray()

    def remaining(self) -> None:
        remaining = self.deadline - time.perf_counter()
        if remaining <= 0:
            raise BenchmarkError("Request/task deadline exceeded.")
        if self.active_socket is not None and self.active_socket.fileno() >= 0:
            self.active_socket.settimeout(remaining)

    def read_chunk(self) -> bytes:
        self.remaining()
        chunk = self.response.read1(65536)
        remaining_length = self.response.length
        if not chunk and remaining_length is not None and remaining_length > 0:
            raise http.client.IncompleteRead(b"", remaining_length)
        return chunk

    def readline(self) -> bytes:
        while b"\n" not in self.buffer:
            chunk = self.read_chunk()
            if not chunk:
                line = bytes(self.buffer)
                self.buffer.clear()
                return line
            self.buffer.extend(chunk)
            if len(self.buffer) > 8 * 1024 * 1024:
                raise BenchmarkError("SSE line exceeds the 8 MiB safety limit.")
        index = self.buffer.index(b"\n") + 1
        line = bytes(self.buffer[:index])
        del self.buffer[:index]
        return line

    def read(self, limit: int | None = None) -> bytes:
        chunks = []
        received = 0
        while limit is None or received < limit:
            chunk = self.read_chunk()
            if not chunk:
                break
            if limit is not None:
                chunk = chunk[:limit - received]
            chunks.append(chunk)
            received += len(chunk)
        return b"".join(chunks)


def endpoint_error_details(response: http.client.HTTPResponse, reader: DeadlineResponse,
                           redact: Callable[[Exception], str]) -> dict[str, Any]:
    details: dict[str, Any] = {"http_status": response.status}
    request_ids = {}
    for name in ("apim-request-id", "x-request-id", "x-ms-request-id"):
        value = response.getheader(name)
        if value:
            request_ids[name] = redact(BenchmarkError(value))[:512]
    if request_ids:
        details["request_ids"] = request_ids
    try:
        body = reader.read(limit=16384)
        document = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError):
        details["body_status"] = "empty_or_non_json_or_over_16_kib"
        return details
    except (BenchmarkError, http.client.HTTPException, OSError):
        details["body_status"] = "unavailable"
        return details
    error = document.get("error", document) if isinstance(document, dict) else None
    if isinstance(error, dict):
        for name in ("code", "message"):
            value = error.get(name)
            if isinstance(value, str):
                details[name] = redact(BenchmarkError(value))[:1500]
    return details


_IDENTITY_ERROR_HINTS = {
    "AADSTS7000215": "Invalid client secret: use the secret value, not its Secret ID, and verify the application.",
    "AADSTS7000222": "The client secret has expired; create a new secret and update the local credential.",
    "AADSTS700016": "Application not found in the requested tenant; verify tenant_id and client_id.",
    "AADSTS90002": "Tenant not found; verify tenant_id and the Azure cloud.",
    "AADSTS500011": "Token resource not found in the tenant; verify scope and tenant_id.",
    "AADSTS70011": "Invalid token scope; use the resource's supported /.default scope.",
}
_IDENTITY_LOG_CONTEXT = ContextVar("identity_log_context", default=False)
_IDENTITY_LOG_LOCK = threading.Lock()
_identity_log_factory_installed = False


@contextmanager
def protected_identity_logging() -> Iterator[None]:
    global _identity_log_factory_installed
    with _IDENTITY_LOG_LOCK:
        if not _identity_log_factory_installed:
            previous_factory = logging.getLogRecordFactory()

            def safe_factory(*args: Any, **kwargs: Any) -> logging.LogRecord:
                record = previous_factory(*args, **kwargs)
                if _IDENTITY_LOG_CONTEXT.get() and any(
                    record.name == name or record.name.startswith(name + ".")
                    for name in ("azure", "msal", "msal_extensions")
                ):
                    record.msg = "Azure identity diagnostic suppressed; consult the sanitized benchmark error."
                    record.args = ()
                    record.exc_info = None
                    record.exc_text = None
                    record.stack_info = None
                return record

            logging.setLogRecordFactory(safe_factory)
            _identity_log_factory_installed = True
    context_token = _IDENTITY_LOG_CONTEXT.set(True)
    try:
        yield
    finally:
        _IDENTITY_LOG_CONTEXT.reset(context_token)


def client_secret_value(config: BenchmarkConfig) -> str:
    if config.azure_client_secret:
        return config.azure_client_secret
    return os.environ.get(config.azure_client_secret_env, "") if config.azure_client_secret_env else ""


class Authentication:
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.lock = threading.Lock()
        self.credential: Any = None
        self.provider: Any = None
        self.tokens: set[str] = set()
        self.candidates: dict[str, Authentication] = {}
        self.chain_index = 0
        self.selected_type: str | None = None if config.auth_type == "auto" else config.auth_type
        self.attempts: list[dict[str, str]] = []
        self.request_state = threading.local()

    def apply(self, headers: dict[str, str]) -> float:
        if self.config.auth_type == "auto":
            return self.apply_chain(headers)
        if self.config.auth_type == "api_key":
            return 0.0
        started_at = time.perf_counter()
        with self.lock, protected_identity_logging():
            try:
                if self.credential is None:
                    options = {"connection_timeout": self.config.timeout_seconds,
                               "read_timeout": self.config.timeout_seconds,
                               "retry_total": 0, "logging_enable": False}
                    if self.config.auth_type == "azure_app_registration":
                        secret = client_secret_value(self.config)
                        if not secret:
                            raise ValueError("Client secret is not configured.")
                        self.credential = ClientSecretCredential(
                            tenant_id=self.config.azure_tenant_id,
                            client_id=self.config.azure_client_id,
                            client_secret=secret, **options,
                        )
                    elif self.config.auth_type == "azure_managed_identity":
                        self.credential = ManagedIdentityCredential(
                            client_id=self.config.azure_client_id or None, **options,
                        )
                    else:
                        raise ValueError("Unsupported authentication type.")
                if self.provider is None:
                    self.provider = get_bearer_token_provider(self.credential, self.config.azure_scope)
                token = self.provider()
                if not isinstance(token, str) or not token:
                    raise ValueError("Azure identity returned an empty access token.")
                self.tokens.add(token)
                headers["Authorization"] = f"Bearer {token}"
            except Exception as error:
                category = "identity_error"
                if isinstance(error, CredentialUnavailableError):
                    category = "credential_unavailable"
                elif isinstance(error, ClientAuthenticationError):
                    category = "client_authentication_error"
                elif isinstance(error, (ServiceRequestError, OSError)):
                    category = "identity_network_error"
                status = getattr(error, "status_code", None)
                status_detail = f", HTTP {status}" if type(status) is int and 100 <= status <= 599 else ""
                diagnostic = safe_error(error, self.config)
                for token in self.tokens:
                    diagnostic = diagnostic.replace(token, "[REDACTED]")
                codes = sorted(set(re.findall(r"\bAADSTS[0-9]{5,9}\b", diagnostic)))[:3]
                code_detail = ", " + ", ".join(codes) if codes else ""
                guidance = "".join(_IDENTITY_ERROR_HINTS[code] + " "
                                   for code in codes if code in _IDENTITY_ERROR_HINTS)
                raise BenchmarkError(
                    f"Azure {self.config.auth_type} authentication failed ({category}{status_detail}{code_detail}). "
                    f"{guidance}Check identity credentials, "
                    "scope and identity endpoint availability; credential details are suppressed."
                ) from None
        return round((time.perf_counter() - started_at) * 1000, 2)

    def apply_chain(self, headers: dict[str, str]) -> float:
        started_at = time.perf_counter()
        with self.lock:
            while self.chain_index < len(_AUTH_CHAIN):
                auth_type = _AUTH_CHAIN[self.chain_index]
                config = self.config
                candidate_headers: dict[str, str] = {}
                try:
                    if auth_type == "api_key":
                        if not config.api_key or not config.auth_header:
                            raise BenchmarkError("LLM API key or authentication header is not configured.")
                        candidate_headers[config.auth_header] = (
                            f"{config.auth_scheme} {config.api_key}" if config.auth_scheme else config.api_key)
                    else:
                        if auth_type == "azure_managed_identity" and config.azure_managed_identity_client_id is None:
                            raise BenchmarkError("Managed Identity client ID environment variable is not configured.")
                        if auth_type == "azure_app_registration" and not (
                            config.azure_tenant_id and config.azure_client_id and client_secret_value(config)
                        ):
                            raise BenchmarkError("App Registration tenant/client ID or client secret is not configured.")
                        if auth_type not in self.candidates:
                            candidate_config = replace(config, auth_type=auth_type,
                                azure_client_id=(config.azure_managed_identity_client_id or "")
                                if auth_type == "azure_managed_identity" else config.azure_client_id)
                            self.candidates[auth_type] = Authentication(candidate_config)
                        self.candidates[auth_type].apply(candidate_headers)
                except BenchmarkError as error:
                    self.attempts.append({"type": auth_type, "status": "failed", "error": str(error)})
                    self.chain_index += 1
                    continue
                headers.update(candidate_headers)
                self.selected_type = auth_type
                if not self.attempts or self.attempts[-1].get("status") != "selected":
                    self.attempts.append({"type": auth_type, "status": "selected"})
                self.request_state.details = {"type": auth_type, "configured_type": "auto",
                                              "attempts": [attempt.copy() for attempt in self.attempts]}
                return round((time.perf_counter() - started_at) * 1000, 2)
            self.request_state.details = {"type": None, "configured_type": "auto",
                                          "attempts": [attempt.copy() for attempt in self.attempts]}
            detail = "; ".join(f"{attempt['type']}: {attempt['error']}"
                               for attempt in self.attempts if attempt["status"] == "failed")
            raise BenchmarkError(f"Authentication chain exhausted (Managed Identity -> App Registration -> LLM API key). {detail}")

    def details(self) -> dict[str, Any]:
        return getattr(self.request_state, "details", {"type": self.config.auth_type})

    def redact(self, error: Exception) -> str:
        message = safe_error(error, self.config)
        with self.lock:
            for token in self.tokens:
                message = message.replace(token, "[REDACTED]")
            for candidate in self.candidates.values():
                message = candidate.redact(BenchmarkError(message))
        return message

    def close(self) -> None:
        with self.lock, protected_identity_logging():
            try:
                for candidate in self.candidates.values():
                    candidate.close()
                if self.credential is not None:
                    self.credential.close()
            finally:
                self.credential = None
                self.provider = None
                self.tokens.clear()
                self.candidates.clear()


_PROGRESS = ContextVar("benchmark_progress", default=None)


class ProgressReporter:
    def __init__(self, *, enabled: bool = True, interval_seconds: float = 5.0, stream: TextIO | None = None):
        self.enabled = enabled
        self.interval_seconds = interval_seconds
        self.stream = stream if stream is not None else sys.stderr
        self.lock = threading.Lock()
        self.output_lock = threading.RLock()
        self.stop = threading.Event()
        self.wake = threading.Event()
        self.pending: list[str] = []
        self.thread: threading.Thread | None = None
        self.active: dict[int, dict[str, Any]] = {}
        self.sequence = 0

    def __enter__(self) -> ProgressReporter:
        self.context_token = _PROGRESS.set(self if self.enabled else None)
        if self.enabled:
            self.thread = threading.Thread(target=self._run, name="benchmark-progress", daemon=True)
            self.thread.start()
        return self

    def __exit__(self, *args: Any) -> None:
        self.stop.set()
        self.wake.set()
        if self.thread is not None:
            self.thread.join()
        self.flush_updates()
        _PROGRESS.reset(self.context_token)

    def write(self, text: str) -> None:
        if not self.enabled:
            return
        with self.output_lock:
            try:
                print(re.sub(r"[\x00-\x1f\x7f]", " ", text), file=self.stream, flush=True)
            except (OSError, ValueError):
                self.stop.set()

    def begin(self, label: str) -> int | None:
        if not self.enabled:
            return None
        with self.lock:
            self.sequence += 1
            identifier = self.sequence
            self.active[identifier] = {"label": label, "stage": "running", "started_at": time.monotonic()}
        self.write(f"[START #{identifier}] {label}")
        return identifier

    def update(self, identifier: int | None, stage: str) -> None:
        with self.lock:
            if identifier in self.active:
                entry = self.active[identifier]
                entry["stage"] = stage
                self.pending.append(f"[STATUS #{identifier}] {entry['label']} - {stage}")
                self.wake.set()

    def flush_updates(self) -> None:
        with self.output_lock:
            with self.lock:
                pending, self.pending = self.pending, []
            for message in pending:
                self.write(message)

    def finish(self, identifier: int | None, outcome: str | None = None) -> None:
        with self.output_lock:
            self.flush_updates()
            with self.lock:
                entry = self.active.pop(identifier, None)
            if entry is not None:
                stage = entry["stage"]
                status = outcome or (stage if stage in {"PASS", "FAILED", "UNVERIFIED"} else "DONE")
                elapsed = time.monotonic() - entry["started_at"]
                self.write(f"[{status} #{identifier}] {entry['label']} - {stage} - elapsed {elapsed:.1f}s")

    def heartbeat(self) -> None:
        with self.output_lock:
            with self.lock:
                entries = [(identifier, entry.copy()) for identifier, entry in self.active.items()]
            now = time.monotonic()
            for identifier, entry in entries:
                self.write(f"[WAIT #{identifier}] {entry['label']} - {entry['stage']} - elapsed {now - entry['started_at']:.1f}s")

    def _run(self) -> None:
        next_heartbeat = time.monotonic() + self.interval_seconds
        while not self.stop.is_set():
            self.wake.wait(max(0, next_heartbeat - time.monotonic()))
            self.wake.clear()
            self.flush_updates()
            if time.monotonic() >= next_heartbeat:
                self.heartbeat()
                next_heartbeat = time.monotonic() + self.interval_seconds


@contextmanager
def progress_scope(label: str, reporter: ProgressReporter | None = None) -> Iterator[Callable[[str], None]]:
    reporter = reporter if reporter is not None else _PROGRESS.get()
    if reporter is None:
        yield lambda stage: None
        return
    identifier = reporter.begin(label)
    try:
        yield lambda stage: reporter.update(identifier, stage)
    except BaseException as error:
        reporter.finish(identifier, "CANCELLED" if isinstance(error, KeyboardInterrupt) else "FAILED")
        raise
    else:
        reporter.finish(identifier)


def progress_label(protocol: str, model: str, scenario: str, mode: str) -> str:
    name = "Responses" if protocol == "responses" else "Chat Completions"
    scenario_name = {"long_output": "long output", "short_burst": "short output (independent)",
                     "short_conversation": "short output (conversation)"}.get(scenario, scenario)
    return f"{name} endpoint testing - model={model} - {scenario_name} - {mode}"


class RequestSession:
    def __init__(self, config: BenchmarkConfig, endpoint: SplitResult, model: str,
                 authentication: Authentication | None = None, *, measure_tokens: bool = True):
        self.config = config
        self.endpoint = resolve_protocol_endpoint(endpoint, config.protocol)
        self.model = model
        self.progress = _PROGRESS.get()
        self.progress_context = ""
        self.call_detail = "request"
        self.authentication = authentication or Authentication(config)
        self.owns_authentication = authentication is None
        self.connection: Any = None
        self.tokenizer = resolve_tokenizer(config, model) if measure_tokens else None
        if self.tokenizer is not None:
            self.tokenizer.encode("", disallowed_special=())

    def close(self) -> None:
        self.close_connection()
        if self.owns_authentication:
            self.authentication.close()

    def close_connection(self) -> None:
        if self.connection is not None:
            self.connection.close()
            self.connection = None

    def call(self, round_number: int, *, streaming: bool = True,
             config: BenchmarkConfig | None = None,
             messages: list[dict[str, Any]] | None = None,
             tools: list[dict[str, Any]] | None = None, tool_choice: Any = None,
             deadline: float | None = None, diagnostics: bool = False) -> dict[str, Any]:
        effective = config or self.config
        label = self.progress_context or progress_label(effective.protocol, self.model, "request",
                                                       "streaming" if streaming else "non_streaming")
        with progress_scope(f"{label} - {self.call_detail}", self.progress) as update:
            record = self._call(round_number, streaming=streaming, config=config, messages=messages,
                                tools=tools, tool_choice=tool_choice, deadline=deadline,
                                diagnostics=diagnostics, on_stage=update)
            update(f"response complete - HTTP {record['http_status']}; checking output")
            return record

    def _call(self, round_number: int, *, streaming: bool = True,
              config: BenchmarkConfig | None = None,
              messages: list[dict[str, Any]] | None = None,
              tools: list[dict[str, Any]] | None = None, tool_choice: Any = None,
              deadline: float | None = None, diagnostics: bool = False,
              on_stage: Callable[[str], None]) -> dict[str, Any]:
        config = config or self.config
        body, headers = build_request(config, self.model, streaming=streaming,
                                      messages=messages, tools=tools, tool_choice=tool_choice)
        if deadline is not None and time.perf_counter() >= deadline:
            raise BenchmarkError("Task deadline exceeded before authentication.")
        on_stage(f"authenticating ({config.auth_type})")
        try:
            authentication_ms = self.authentication.apply(headers)
        except BenchmarkError as error:
            on_stage("authentication failed")
            raise CallFailure({"model": self.model, "round": round_number, "success": False,
                "request_success": False, "mode": "streaming" if streaming else "non_streaming",
                "error": self.authentication.redact(error), "authentication": self.authentication.details(),
                "response_complete": False, "interrupted": False, "protocol": config.protocol,
                "endpoint": display_endpoint(self.endpoint)}) from None
        authentication_details = self.authentication.details()
        on_stage(f"authentication ready ({authentication_details['type']}); connecting/sending request")
        path = self.endpoint.path or "/"
        if self.endpoint.query:
            path += f"?{self.endpoint.query}"
        timing = Timing(started_at=time.perf_counter())
        deadline = min(deadline or math.inf, timing.started_at + config.timeout_seconds)
        timing.deadline = deadline
        remaining = deadline - timing.started_at
        if remaining <= 0:
            raise BenchmarkError("Task deadline exceeded.")
        reused = self.connection is not None and self.connection.sock is not None
        if self.connection is None:
            self.connection = create_connection(self.endpoint, remaining, timing)
        else:
            self.connection._timing = timing
            self.connection._deadline = deadline
            self.connection.timeout = remaining
            if self.connection.sock is not None:
                self.connection.sock.settimeout(remaining)
        stream_result = StreamResult("", None, 0)
        response = None
        reading_body = False
        diagnostic_details = None
        try:
            self.connection.request("POST", path, body=body, headers=headers)
            timing.request_sent_at = time.perf_counter()
            active_socket = self.connection.sock
            remaining = deadline - timing.request_sent_at
            if remaining <= 0:
                raise BenchmarkError("Request deadline exceeded while sending.")
            if active_socket is not None:
                active_socket.settimeout(remaining)
            on_stage("request sent; waiting for response headers")
            response = self.connection.getresponse()
            timing.response_headers_at = time.perf_counter()
            if not 200 <= response.status < 300:
                if diagnostics:
                    diagnostic_details = endpoint_error_details(response,
                        DeadlineResponse(response, active_socket, deadline), self.authentication.redact)
                raise BenchmarkError(f"Endpoint returned HTTP {response.status} {response.reason}")
            expected = "text/event-stream" if streaming else "application/json"
            if expected not in response.getheader("Content-Type", "").lower():
                raise BenchmarkError(f"Expected {expected} response.")
            reader = DeadlineResponse(response, active_socket, deadline)
            server_timing_header = response.getheader("Server-Timing")
            reading_body = True
            if streaming:
                on_stage("headers received; waiting for first visible text or tool call")
                consume_sse(reader, timing, stream_result, protocol=config.protocol,
                            on_first_text=lambda: on_stage("first visible text received; generating"))
            else:
                on_stage("headers received; waiting for complete JSON response")
                stream_result = consume_json(reader, protocol=config.protocol)
            if server_timing_header:
                stream_result.server_timing["http_server_timing"] = server_timing_header
            timing.completed_at = time.perf_counter()
            if timing.completed_at > deadline:
                raise BenchmarkError("Request/task deadline exceeded.")
            record = build_call_result(self.model, round_number, streaming, timing,
                                       stream_result, self.tokenizer, response.status)
            record["connection_reused"] = reused
            record["protocol"] = config.protocol
            record["endpoint"] = display_endpoint(self.endpoint)
            record["authentication"] = authentication_details
            record["metrics"]["authentication_ms"] = authentication_ms
            record["request_parameters"] = {
                "max_tokens": config.max_tokens, "max_completion_tokens": config.max_completion_tokens,
                "temperature": config.temperature, "reasoning_effort": config.reasoning_effort,
                "max_output_tokens": (config.max_completion_tokens or config.max_tokens) if config.protocol == "responses" else None,
            }
            return record
        except Exception as error:
            on_stage(f"request failed - HTTP {response.status}" if response is not None else "connection/request failed")
            self.close_connection()
            message = self.authentication.redact(error)
            record = {"model": self.model, "round": round_number,
                      "mode": "streaming" if streaming else "non_streaming"}
            if streaming and reading_body and response is not None:
                server_timing_header = response.getheader("Server-Timing")
                if server_timing_header:
                    stream_result.server_timing["http_server_timing"] = server_timing_header
                record = build_call_result(self.model, round_number, streaming, timing,
                                           stream_result, self.tokenizer, response.status)
                record["tokens"]["visible_output_tokens_complete"] = False
                record["stream"]["last_observed_text_ms"] = milliseconds(timing.started_at, timing.last_token_at)
                for name in ("ttlt_ms", "tps", "output_tokens_per_second", "generation_duration_ms",
                             "end_to_end_ms", "end_to_end_output_tokens_per_second"):
                    record["metrics"][name] = None
                record["metrics"]["authentication_ms"] = authentication_ms
            record.update({"success": False, "request_success": False, "error": message,
                           "interrupted": reading_body, "response_complete": False,
                           "http_status": response.status if response is not None else None,
                           "protocol": config.protocol, "endpoint": display_endpoint(self.endpoint),
                           "auth_type": authentication_details["type"], "authentication": authentication_details,
                           "connection_reused": reused,
                           "failure_elapsed_ms": milliseconds(timing.started_at, time.perf_counter()),
                           "validation": {"status": "failed", "reason": message}})
            if diagnostic_details is not None:
                record["endpoint_error"] = diagnostic_details
            raise CallFailure(record) from None
        finally:
            if not config.connection_reuse:
                self.close_connection()


def run_single_benchmark(
    config: BenchmarkConfig, endpoint: SplitResult, model: str, round_number: int,
    *, streaming: bool = True,
) -> dict[str, Any]:
    session = RequestSession(config, endpoint, model)
    try:
        return session.call(round_number, streaming=streaming)
    finally:
        session.close()


def build_call_result(
    model: str, round_number: int, streaming: bool, timing: Timing,
    stream_result: StreamResult, tokenizer: Any, http_status: int,
) -> dict[str, Any]:
    output_tokens, output_token_count_source = count_visible_tokens(stream_result.text, tokenizer)
    first_chunk_tokens, _ = count_visible_tokens(stream_result.first_chunk, tokenizer)
    generation_seconds = (
        timing.last_token_at - timing.first_token_at
        if streaming and timing.last_token_at is not None and timing.first_token_at is not None
        else None
    )
    tokens_per_second = (
        round(max(0, output_tokens - first_chunk_tokens) / generation_seconds, 2)
        if generation_seconds is not None and generation_seconds > 0
        and stream_result.text_chunk_count > 1
        else None
    )
    end_to_end_seconds = (
        timing.completed_at - timing.started_at if timing.completed_at is not None else 0
    )
    connection_setup_ms = sum(
        value * 1_000
        for value in (timing.dns_seconds, timing.tcp_seconds, timing.tls_seconds)
        if value is not None
    )

    return {
        "model": model,
        "round": round_number,
        "mode": "streaming" if streaming else "non_streaming",
        "success": True,
        "response_complete": True,
        "interrupted": False,
        "http_status": http_status,
        "metrics": {
            "dns_lookup_ms": round(timing.dns_seconds * 1_000, 2)
            if timing.dns_seconds is not None
            else None,
            "tcp_connect_ms": round(timing.tcp_seconds * 1_000, 2)
            if timing.tcp_seconds is not None
            else None,
            "tcp_connect_attempts": timing.tcp_attempts,
            "tls_handshake_ms": round(timing.tls_seconds * 1_000, 2)
            if timing.tls_seconds is not None
            else None,
            "connection_setup_ms": round(connection_setup_ms, 2),
            "response_headers_ms": milliseconds(
                timing.request_sent_at, timing.response_headers_at
            ),
            "first_sse_event_ms": milliseconds(
                timing.request_sent_at, timing.first_event_at
            ),
            "time_to_first_token_ms": milliseconds(
                timing.started_at, timing.first_token_at
            ),
            "first_token_after_request_ms": milliseconds(
                timing.request_sent_at, timing.first_token_at
            ),
            "end_to_end_ms": milliseconds(timing.started_at, timing.completed_at),
            "generation_duration_ms": round(generation_seconds * 1_000, 2)
            if generation_seconds is not None
            else None,
            "output_tokens_per_second": tokens_per_second,
            "ttft_ms": milliseconds(timing.started_at, timing.first_token_at),
            "ttlt_ms": milliseconds(timing.started_at, timing.last_token_at),
            "prefill_proxy_ms": milliseconds(timing.request_sent_at, timing.first_token_at),
            "tps": tokens_per_second,
            "end_to_end_output_tokens_per_second": round(output_tokens / end_to_end_seconds, 2)
            if end_to_end_seconds > 0 else None,
            "authentication_ms": 0.0,
        },
        "tokens": {
            "input_tokens": usage_value(
                stream_result.usage, "prompt_tokens", "input_tokens"
            ),
            "output_tokens": output_tokens,
            "total_tokens": usage_value(stream_result.usage, "total_tokens"),
            "output_token_count_source": output_token_count_source,
            "visible_output_tokens": output_tokens,
            "visible_output_tokens_complete": True,
            "api_completion_tokens": usage_value(stream_result.usage, "completion_tokens", "output_tokens"),
            "reasoning_tokens": usage_value((stream_result.usage or {}).get("completion_tokens_details")
                or (stream_result.usage or {}).get("output_tokens_details"), "reasoning_tokens"),
            "cached_input_tokens": usage_value((stream_result.usage or {}).get("prompt_tokens_details")
                or (stream_result.usage or {}).get("input_tokens_details"), "cached_tokens"),
        },
        "stream": {
            "event_count": stream_result.event_count,
            "response_characters": len(stream_result.text),
            "response_preview": stream_result.text[:300],
            "text_chunk_count": stream_result.text_chunk_count,
        },
        "finish_reason": stream_result.finish_reason,
        "api_usage": stream_result.usage,
        "server_timing": {"source": "provider_metadata", "raw": stream_result.server_timing}
        if stream_result.server_timing else None,
        "assistant_message": {"role": "assistant", "content": stream_result.text,
                              **({"tool_calls": stream_result.tool_calls} if stream_result.tool_calls else {}),
                              **({"responses_output": stream_result.response_output} if stream_result.response_output else {})},
    }


def percentile(values: list[float], percentage: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentage / 100
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def summarize_metric(records: list[dict[str, Any]], name: str) -> dict[str, float] | None:
    values = [
        float(record["metrics"][name])
        for record in records
        if not record.get("interrupted", False) and record["metrics"].get(name) is not None
    ]
    if not values:
        return None
    return {
        "sample_count": len(values),
        "average": round(sum(values) / len(values), 2),
        "minimum": round(min(values), 2),
        "p50": round(percentile(values, 50), 2),
        "p95": round(percentile(values, 95), 2),
        "maximum": round(max(values), 2),
    }


def summarize_model(
    model: str, records: list[dict[str, Any]], total_rounds: int
) -> dict[str, Any]:
    successes = [record for record in records if record["success"]]
    measured = [record for record in records if "metrics" in record and not record.get("interrupted", False)]
    failures = [record for record in records if not record["success"]]
    metrics = {
        name: summarize_metric(measured, name) for name in _METRIC_NAMES
    }
    return {
        "model": model,
        "successful_rounds": len(successes),
        "failed_rounds": len(failures),
        "total_rounds": total_rounds,
        "completed_rounds": len(records),
        "success_rate": round(len(successes) / len(records), 4) if records else None,
        "unverified_rounds": sum(record.get("validation", {}).get("status") == "unverified" for record in records),
        "percentile_note": "P95 is descriptive only with fewer than 20 samples." if len(measured) < 20 else None,
        "metrics": metrics,
    }


def with_token_budget(config: BenchmarkConfig, budget: int) -> BenchmarkConfig:
    if config.max_completion_tokens is not None:
        return replace(config, max_completion_tokens=budget)
    return replace(config, max_tokens=budget)


def scenario_config(config: BenchmarkConfig, scenario: str) -> BenchmarkConfig:
    if scenario == "long_output":
        return replace(with_token_budget(config, config.long_token_budget), user_prompt=(
            "Write a detailed technical article of 1500 to 1800 English words about network "
            "latency, measurement, routing, congestion, and performance tuning. Develop each "
            "section with concrete examples. Do not end early or include a summary instead."
        ))
    if scenario in {"short_burst", "short_conversation"}:
        return replace(with_token_budget(config, config.short_token_budget), user_prompt=(
            "Explain one practical network latency measurement technique in 40 to 60 English "
            "words. Keep the entire answer below 150 tokens."
        ))
    if scenario in {"prefill_short", "prefill_long"}:
        words = config.prefill_short_words if scenario == "prefill_short" else config.prefill_long_words
        vocabulary = ("latency", "routing", "packet", "queue", "network", "measurement", "response", "time")
        context = " ".join(vocabulary[index % len(vocabulary)] for index in range(words))
        return replace(config, user_prompt=f"Read this network context:\n{context}\nReply with a single sentence about latency.")
    return config


def safe_error(error: Exception, config: BenchmarkConfig) -> str:
    message = str(error)
    azure_secret = client_secret_value(config) if config.auth_type in {"azure_app_registration", "auto"} else ""
    for secret in (config.api_key, config.azure_client_secret, azure_secret, *config.extra_headers.values()):
        if secret:
            message = message.replace(secret, "[REDACTED]")
    return message


def validate_output(record: dict[str, Any], scenario: str) -> None:
    tokens = record["tokens"]["visible_output_tokens"]
    limited = scenario in {"long_output", "short_burst", "short_conversation"}
    status, reason = "passed", "Complete visible response."
    if not record["assistant_message"]["content"].strip():
        status, reason = "failed", "Response contains no visible text."
    elif record["finish_reason"] != "stop":
        status, reason = "failed", f"Response did not finish normally: {record['finish_reason']}."
    elif limited and record["tokens"]["output_token_count_source"] == "estimated":
        status, reason = "unverified", "Visible-output length requires a known tokenizer or explicit tokenizer_model."
    elif scenario == "long_output" and tokens <= 1000:
        status, reason = "failed", f"Long output requires >1000 visible tokens; received {tokens}."
    elif scenario in {"short_burst", "short_conversation"} and tokens >= 200:
        status, reason = "failed", f"Short output requires <200 visible tokens; received {tokens}."
    record["request_success"] = True
    record["validation"] = {"status": status, "reason": reason}
    record["success"] = status == "passed"
    if not record["success"]:
        record["error"] = reason


def aggregate_calls(model: str, round_number: int, mode: str,
                    calls: list[dict[str, Any]], started_at: float,
                    error: str | None = None) -> dict[str, Any]:
    elapsed = time.perf_counter() - started_at
    measured = [call for call in calls if "tokens" in call]
    token_names = ("input_tokens", "output_tokens", "visible_output_tokens", "total_tokens",
                   "api_completion_tokens", "reasoning_tokens", "cached_input_tokens")
    totals = {}
    for name in token_names:
        values = [call.get("tokens", {}).get(name) for call in calls]
        totals[name] = sum(values) if values and all(value is not None for value in values) else None
    visible = sum(call["tokens"]["visible_output_tokens"] for call in measured)
    totals["output_tokens"] = totals["visible_output_tokens"] = visible
    sources = sorted({call["tokens"]["output_token_count_source"] for call in measured})
    totals["output_token_count_source"] = ",".join(sources) or "unavailable"
    metrics = dict.fromkeys(_METRIC_NAMES)
    metrics["end_to_end_ms"] = round(elapsed * 1000, 2)
    metrics["end_to_end_output_tokens_per_second"] = round(visible / elapsed, 2) if elapsed > 0 else None
    metrics["connection_setup_ms"] = round(sum(call["metrics"]["connection_setup_ms"] for call in measured), 2)
    metrics["authentication_ms"] = round(sum(call["metrics"]["authentication_ms"] for call in measured), 2)
    unverified = any(call.get("validation", {}).get("status") == "unverified" for call in calls)
    interrupted = any(call.get("interrupted", False) for call in calls)
    success = bool(calls) and all(call["success"] for call in calls) and error is None
    reason = error or next((call.get("error", "Call failed.") for call in calls if not call["success"]), "All calls passed.")
    return {
        "model": model, "round": round_number, "mode": mode, "success": success,
        "metrics": metrics, "tokens": totals, "calls": calls, "call_count": len(calls),
        "interrupted": interrupted,
        "visible_token_totals_complete": len(measured) == len(calls) and not interrupted,
        "token_totals_complete": bool(calls) and not interrupted and all(totals[name] is not None for name in token_names),
        "validation": {"status": "passed" if success else "unverified" if unverified and not error else "failed", "reason": reason},
        "call_metrics": {name: summarize_metric(measured, name) for name in _METRIC_NAMES},
        **({"error": reason} if not success else {}),
    }


def run_text_scenario(config: BenchmarkConfig, session: RequestSession,
                      scenario: str, round_number: int, streaming: bool) -> dict[str, Any]:
    effective = scenario_config(config, scenario)
    started_at = time.perf_counter()
    deadline = started_at + config.task_timeout_seconds
    count = config.short_requests if scenario in {"short_burst", "short_conversation"} else 1
    calls = []
    history: list[dict[str, Any]] = []
    if effective.system_prompt:
        history.append({"role": "system", "content": effective.system_prompt})
    for sequence in range(1, count + 1):
        session.call_detail = f"request {sequence}/{count}"
        messages = None
        if scenario == "short_conversation":
            history.append({"role": "user", "content": effective.user_prompt})
            messages = history
        try:
            record = session.call(round_number, streaming=streaming, config=effective,
                                  messages=messages, deadline=deadline)
            validate_output(record, scenario)
            if messages is not None:
                history.append(record["assistant_message"])
        except (BenchmarkError, http.client.HTTPException, OSError) as error:
            record = error.record if isinstance(error, CallFailure) else {
                      "model": session.model, "round": round_number, "success": False,
                      "mode": "streaming" if streaming else "non_streaming",
                      "error": safe_error(error, config)}
            history = history[:1] if effective.system_prompt else []
        record["sequence"] = sequence
        calls.append(record)
        if not record["success"] and not config.continue_on_error:
            break
    if count == 1 and scenario not in {"short_burst", "short_conversation"}:
        if scenario in {"prefill_short", "prefill_long"}:
            calls[0]["input_context_words"] = config.prefill_short_words if scenario == "prefill_short" else config.prefill_long_words
        return calls[0]
    result = aggregate_calls(session.model, round_number, "streaming" if streaming else "non_streaming", calls, started_at)
    result["configured_requests"] = count
    result["history_policy"] = "growing" if scenario == "short_conversation" else "independent"
    return result


_FACTS = {
    "latency": "Latency is elapsed time for an operation; report median and tail percentiles with units.",
    "throughput": "Throughput is completed work per unit time; report output tokens per second separately from latency.",
}


def agent_tool(delegation: bool) -> dict[str, Any]:
    if delegation:
        name, description = "delegate", "Ask two sub-agents to research latency and throughput."
        properties = {"topics": {"type": "array", "items": {"type": "string", "enum": list(_FACTS)},
                                  "minItems": 2, "maxItems": 2}}
    else:
        name, description = "lookup_fact", "Read a deterministic local measurement fact."
        properties = {"topic": {"type": "string", "enum": list(_FACTS)}}
    return {"type": "function", "function": {"name": name, "description": description,
            "parameters": {"type": "object", "properties": properties,
                           "required": list(properties), "additionalProperties": False}}}


class AgentBudget:
    def __init__(self, config: BenchmarkConfig, started_at: float):
        self.config = config
        self.started_at = started_at
        self.deadline = started_at + config.task_timeout_seconds
        self.lock = threading.Lock()
        self.call_count = 0
        self.reserved_tokens = 0
        self.calls: list[dict[str, Any]] = []
        self.tools: list[dict[str, Any]] = []

    def reserve(self) -> tuple[int, int]:
        with self.lock:
            if time.perf_counter() >= self.deadline:
                raise BenchmarkError("Agent task deadline exceeded.")
            if self.call_count >= self.config.agent_max_calls:
                raise BenchmarkError("Agent maximum LLM call count exceeded.")
            remaining = self.config.agent_total_token_budget - self.reserved_tokens
            if remaining <= 0:
                raise BenchmarkError("Agent total completion-token budget exhausted.")
            tokens = min(self.config.agent_token_budget, remaining)
            self.call_count += 1
            self.reserved_tokens += tokens
            return self.call_count, tokens

    def append_call(self, call: dict[str, Any]) -> None:
        with self.lock:
            self.calls.append(call)

    def append_tool(self, tool: dict[str, Any]) -> None:
        with self.lock:
            self.tools.append(tool)


def parse_tool_call(tool_call: Any, expected_name: str) -> tuple[str, dict[str, Any]]:
    if not isinstance(tool_call, dict) or not isinstance(tool_call.get("function"), dict):
        raise BenchmarkError("Invalid tool-call shape.")
    identifier = tool_call.get("id")
    function = tool_call["function"]
    if not isinstance(identifier, str) or not identifier or function.get("name") != expected_name:
        raise BenchmarkError("Unexpected tool name or missing tool-call id.")
    try:
        arguments = json.loads(function.get("arguments", ""))
    except (TypeError, json.JSONDecodeError) as error:
        raise BenchmarkError("Tool arguments must be a JSON object.") from error
    if not isinstance(arguments, dict):
        raise BenchmarkError("Tool arguments must be a JSON object.")
    return identifier, arguments


def run_agent_loop(config: BenchmarkConfig, session: RequestSession, budget: AgentBudget,
                   round_number: int, streaming: bool, role: str, topic: str,
                   delegation: bool, parallel: bool) -> str:
    tool = agent_tool(delegation)
    tool_name = tool["function"]["name"]
    prompt = (
        "Delegate both topics, latency and throughput, exactly once using delegate, then summarize "
        "the two returned findings in under 100 words."
        if delegation else f"Topic: {topic}. Use lookup_fact for this topic exactly once, then explain the finding in under 60 words."
    )
    messages: list[dict[str, Any]] = []
    if config.system_prompt:
        messages.append({"role": "system", "content": config.system_prompt})
    messages.append({"role": "user", "content": prompt})
    tool_executed = False
    for step in range(1, config.agent_max_steps + 1):
        call_number, tokens = budget.reserve()
        session.call_detail = f"agent={role} - step {step}/{config.agent_max_steps} - call {call_number}/{config.agent_max_calls}"
        choice: Any = "none" if tool_executed else {"type": "function", "function": {"name": tool_name}}
        offset_ms = round((time.perf_counter() - budget.started_at) * 1000, 2)
        try:
            record = session.call(round_number, streaming=streaming,
                                  config=with_token_budget(config, tokens), messages=messages,
                                  tools=[tool], tool_choice=choice, deadline=budget.deadline)
        except (BenchmarkError, http.client.HTTPException, OSError) as error:
            failed_call = error.record if isinstance(error, CallFailure) else {
                "success": False, "error": safe_error(error, config)}
            failed_call.update({"model": session.model, "round": round_number,
                "mode": "streaming" if streaming else "non_streaming", "agent": role,
                "step": step, "call_number": call_number, "task_offset_ms": offset_ms})
            budget.append_call(failed_call)
            raise
        record.update({"agent": role, "step": step, "call_number": call_number,
                       "task_offset_ms": offset_ms, "reserved_completion_tokens": tokens})
        budget.append_call(record)
        message = record["assistant_message"]
        tool_calls = message.get("tool_calls") or []
        if not tool_calls:
            if not tool_executed:
                raise BenchmarkError("Required tool call not returned; tool support is unavailable or the request was ignored.")
            validate_output(record, "baseline")
            if not record["success"]:
                raise BenchmarkError(record["error"])
            return message["content"]
        if tool_executed or len(tool_calls) != 1 or record["finish_reason"] != "tool_calls":
            raise BenchmarkError("Expected exactly one tool invocation before the final answer.")
        identifier, arguments = parse_tool_call(tool_calls[0], tool_name)
        messages.append(message)
        tool_started_at = time.perf_counter()
        if delegation:
            topics = arguments.get("topics")
            if set(arguments) != {"topics"} or not isinstance(topics, list) or sorted(topics, key=str) != sorted(_FACTS):
                raise BenchmarkError("Delegate must request latency and throughput exactly once each.")

            def run_child(child_topic: str) -> str:
                child = RequestSession(config, session.endpoint, session.model, session.authentication)
                child.progress = session.progress
                child.progress_context = session.progress_context
                try:
                    return run_agent_loop(config, child, budget, round_number, streaming,
                                          f"subagent:{child_topic}", child_topic, False, False)
                finally:
                    child.close()

            if parallel:
                if session.progress is not None:
                    session.progress.write(f"[TOOL] {session.progress_context} - agent={role} - delegating parallel subagents")
                with ThreadPoolExecutor(max_workers=2) as executor:
                    futures = [executor.submit(run_child, child_topic) for child_topic in topics]
                    answers = [future.result() for future in futures]
            else:
                if session.progress is not None:
                    session.progress.write(f"[TOOL] {session.progress_context} - agent={role} - delegating serial subagents")
                answers = [run_child(child_topic) for child_topic in topics]
            output = dict(zip(topics, answers))
        else:
            if set(arguments) != {"topic"} or arguments.get("topic") != topic:
                raise BenchmarkError("lookup_fact must request the assigned topic.")
            if session.progress is not None:
                session.progress.write(f"[TOOL] {session.progress_context} - agent={role} - lookup_fact")
            output = {"topic": topic, "fact": _FACTS[topic]}
        budget.append_tool({"agent": role, "name": tool_name, "arguments": arguments,
            "kind": "delegation" if delegation else "local", "success": True,
            "task_offset_ms": round((tool_started_at - budget.started_at) * 1000, 2),
            "duration_ms": round((time.perf_counter() - tool_started_at) * 1000, 2)})
        messages.append({"role": "tool", "tool_call_id": identifier,
                         "content": json.dumps(output, separators=(",", ":"))})
        tool_executed = True
    raise BenchmarkError("Agent maximum step count reached without a final answer.")


def run_agent_scenario(config: BenchmarkConfig, session: RequestSession, scenario: str,
                       round_number: int, streaming: bool) -> dict[str, Any]:
    started_at = time.perf_counter()
    budget = AgentBudget(config, started_at)
    error = None
    answer = None
    try:
        answer = run_agent_loop(config, session, budget, round_number, streaming,
                                "main", "latency", scenario != "agent", scenario == "agent_parallel")
    except (BenchmarkError, http.client.HTTPException, OSError) as exception:
        error = safe_error(exception, config)
    calls = sorted(budget.calls, key=lambda call: call["call_number"])
    result = aggregate_calls(session.model, round_number, "streaming" if streaming else "non_streaming", calls, started_at, error)
    result.update({
        "final_answer": answer, "tools": budget.tools,
        "final_answer_visible_tokens": count_visible_tokens(answer, session.tokenizer)[0] if answer is not None else None,
        "tool_support": "observed" if any(call.get("assistant_message", {}).get("tool_calls") for call in calls) else "not_observed",
        "local_tool_duration_ms": round(sum(tool["duration_ms"] for tool in budget.tools if tool["kind"] == "local"), 2),
        "delegation_duration_ms": round(sum(tool["duration_ms"] for tool in budget.tools if tool["kind"] == "delegation"), 2),
        "budget": {"max_calls": config.agent_max_calls, "reserved_completion_tokens": budget.reserved_tokens,
                   "total_completion_token_limit": config.agent_total_token_budget,
                   "reservation_policy": "Reserve each call's maximum completion tokens, including reasoning; do not refund unused tokens."},
    })
    return result


def run_validation(config: BenchmarkConfig, *,
                   on_record: Callable[[dict[str, Any]], None] | None = None) -> dict[str, Any]:
    if config.auth_type == "auto":
        return validate_authentication_methods(config, on_record=on_record)
    endpoint = parse_endpoint(config.endpoint)
    model = config.models[0]
    budget = min(config.max_completion_tokens or config.max_tokens or 64, 64)
    probe_config = with_token_budget(replace(config, system_prompt="", user_prompt="Reply with OK.",
        connection_reuse=False, temperature=None), budget)
    probes = []
    authentication = Authentication(config)
    try:
        for protocol in selected_protocols(config):
            session = RequestSession(replace(probe_config, protocol=protocol), endpoint, model,
                                     authentication, measure_tokens=False)
            session.progress_context = progress_label(protocol, model, "access validation", "non_streaming") + f" - auth={config.auth_type}"
            try:
                with progress_scope(progress_label(protocol, model, "access validation", "non_streaming")
                                    + f" - auth={config.auth_type}") as update:
                    try:
                        call = session.call(1, streaming=False, diagnostics=True)
                    except CallFailure as error:
                        call = error.record
                    except (BenchmarkError, http.client.HTTPException, OSError) as error:
                        call = {"success": False, "error": authentication.redact(error), "http_status": None}
                    update("PASS" if call["success"] else "FAILED")
            finally:
                session.close()
            status = call.get("http_status")
            stage = "complete" if call["success"] else (
                "authentication" if "http_status" not in call else
                "network" if status is None else "response" if 200 <= status < 300 else "endpoint")
            probe = {"type": "access_validation", "protocol": protocol, "model": model,
                "endpoint": display_endpoint(session.endpoint), "mode": "non_streaming", "success": call["success"],
                "stage": stage, "http_status": status, "authentication_success": "http_status" in call,
                "authentication": call.get("authentication", authentication.details()),
                "finish_reason": call.get("finish_reason"), "api_usage": call.get("api_usage")}
            if not call["success"]:
                probe["error"] = call["error"]
            if "endpoint_error" in call:
                probe["endpoint_error"] = call["endpoint_error"]
            if status in {401, 403} and config.auth_type == "api_key":
                probe["guidance"] = (
                    "The endpoint rejected the API key. Check that the key belongs to this resource, key/local "
                    "authentication is enabled, and auth_header/auth_scheme match the endpoint. Azure-style keys "
                    "normally use auth_header=api-key and an empty auth_scheme.")
            elif status in {401, 403}:
                probe["guidance"] = (
                    "Credential acquisition succeeded but the endpoint rejected access. Check the app service principal's "
                    "Cognitive Services OpenAI User/Contributor role on the target resource, tenant, token audience/scope, "
                    "and resource endpoint. Your own user permissions do not grant the app access.")
            elif status == 404:
                probe["guidance"] = "Check the resource endpoint, selected protocol and actual model deployment name."
            elif status == 400:
                probe["guidance"] = "The endpoint rejected request parameters; inspect its message and selected model/protocol compatibility."
            if call["success"] and call.get("finish_reason") != "stop":
                probe["warning"] = "Access accepted; the small validation budget did not produce a normal final answer."
            probes.append(probe)
            if on_record is not None:
                on_record(probe)
            if stage == "authentication":
                break
    finally:
        authentication.close()
    checked = {probe["protocol"] for probe in probes}
    return {"schema_version": 2, "type": "access_validation",
        "success": len(probes) == len(selected_protocols(config)) and all(probe["success"] for probe in probes),
        "endpoint": display_endpoint(endpoint), "model": model, "auth_type": config.auth_type,
        "scope": config.azure_scope if config.auth_type != "api_key" else None,
        "token_budget_per_request": budget, "maximum_model_requests": len(selected_protocols(config)),
        "skipped_protocols": [protocol for protocol in selected_protocols(config) if protocol not in checked],
        "probes": probes, "note": "Online access check, not a benchmark. First configured model only; one non-streaming request per protocol. No warmups, retries, scenarios or length validation."}


def validate_authentication_methods(config: BenchmarkConfig, *,
                                    on_record: Callable[[dict[str, Any]], None] | None = None) -> dict[str, Any]:
    protocols = selected_protocols(config)
    endpoint = parse_endpoint(config.endpoint)
    probes: list[dict[str, Any]] = []
    auth_results = []
    for auth_type in _AUTH_CHAIN:
        reporter = _PROGRESS.get()
        if reporter is not None:
            reporter.write(f"[AUTH] checking {auth_type}")
        candidate = replace(config, auth_type=auth_type,
            azure_client_id=(config.azure_managed_identity_client_id or "")
            if auth_type == "azure_managed_identity" else config.azure_client_id)
        error = None
        if auth_type == "azure_managed_identity" and config.azure_managed_identity_client_id is None:
            error = "Managed Identity client ID environment variable is not configured."
        elif auth_type == "azure_app_registration" and not (
            config.azure_tenant_id and config.azure_client_id and client_secret_value(config)
        ):
            error = "App Registration tenant/client ID or client secret is not configured."
        elif auth_type == "api_key" and not (config.api_key and config.auth_header):
            error = "LLM API key or authentication header is not configured."
        if error:
            probe = {"type": "access_validation", "protocol": protocols[0], "model": config.models[0],
                "endpoint": display_endpoint(resolve_protocol_endpoint(endpoint, protocols[0])),
                "mode": "non_streaming", "success": False, "stage": "authentication", "http_status": None,
                "authentication_success": False, "authentication": {"type": auth_type}, "error": error}
            method_result = {"success": False, "probes": [probe], "skipped_protocols": list(protocols[1:])}
            if on_record is not None:
                on_record(probe)
        else:
            method_result = run_validation(candidate, on_record=on_record)
        probes.extend(method_result["probes"])
        auth_results.append({"auth_type": auth_type, "success": method_result["success"],
            "status": "success" if method_result["success"] else "failed",
            "authentication_success": any(probe["authentication_success"] for probe in method_result["probes"]),
            "probes": method_result["probes"], "skipped_protocols": method_result["skipped_protocols"]})
        if reporter is not None:
            reporter.write(f"[AUTH] {auth_type} - {'Success' if method_result['success'] else 'Failed'}")
    usable = [method["auth_type"] for method in auth_results if method["success"]]
    checked = {probe["protocol"] for probe in probes}
    return {"schema_version": 2, "type": "access_validation", "success": bool(usable),
        "endpoint": display_endpoint(endpoint), "model": config.models[0], "auth_type": "auto",
        "scope": config.azure_scope, "token_budget_per_request": min(config.max_completion_tokens or config.max_tokens or 64, 64),
        "maximum_model_requests": len(_AUTH_CHAIN) * len(protocols),
        "authentication_order": list(_AUTH_CHAIN), "auth_results": auth_results, "usable_auth_types": usable,
        "skipped_protocols": [protocol for protocol in protocols if protocol not in checked], "probes": probes,
        "note": "Auto validation independently checks Managed Identity, App Registration, then LLM API key, including methods after a success. Overall success requires one method to pass all selected protocols. Benchmark credential fallback is unchanged."}


def print_validation_result(result: dict[str, Any]) -> None:
    print(f"Endpoint access validation: {'PASS' if result['success'] else 'FAIL'}")
    print(f"Endpoint: {result['endpoint']}")
    print(f"Authentication: {result['auth_type']}")
    if result["scope"]:
        print(f"Scope: {result['scope']}")
    print(f"Model: {result['model']} (first configured model only)")
    print(f"Maximum model requests: {result['maximum_model_requests']}; completion budget: {result['token_budget_per_request']} per request")
    for probe in result["probes"]:
        auth_type = probe["authentication"].get("type") or "unavailable"
        print(f"  {probe['protocol']} / auth {auth_type}: {'PASS' if probe['success'] else 'FAIL'}"
              f" | stage {probe['stage']} | HTTP {format_value(probe['http_status'])}")
        print(f"    Credential acquisition: {'PASS' if probe['authentication_success'] else 'FAIL'}")
        if probe.get("error"):
            print(f"    Error: {probe['error']}")
        details = probe.get("endpoint_error", {})
        for name in ("code", "message", "body_status"):
            if details.get(name):
                print(f"    {name}: {details[name]}")
        for name, value in details.get("request_ids", {}).items():
            print(f"    {name}: {value}")
        for name in ("guidance", "warning"):
            if probe.get(name):
                print(f"    {probe[name]}")
    if result["skipped_protocols"]:
        print(f"Skipped after credential acquisition failed: {', '.join(result['skipped_protocols'])}")
    if "auth_results" in result:
        names = {"azure_managed_identity": "Managed Identity", "azure_app_registration": "App Registration", "api_key": "LLM API Key"}
        print("Authentication checks (configured order):")
        for method in result["auth_results"]:
            print(f"  {names[method['auth_type']]} - {'Success' if method['success'] else 'Failed'}")
        print("Overall PASS means at least one method can access all selected protocols.")
    print("Access validation only; no performance conclusion or full-model coverage.")


def selected_modes(config: BenchmarkConfig) -> tuple[str, ...]:
    return ("streaming", "non_streaming") if config.mode == "both" else (config.mode,)


def selected_protocols(config: BenchmarkConfig) -> tuple[str, ...]:
    return tuple(_PROTOCOL_PATHS) if config.protocol == "both" else (config.protocol,)


def run_benchmarks(config: BenchmarkConfig, *,
                   on_record: Callable[[dict[str, Any]], None] | None = None) -> dict[str, Any]:
    authentication = Authentication(config)
    try:
        return run_authenticated_benchmarks(config, authentication, on_record=on_record)
    finally:
        authentication.close()


def run_authenticated_benchmarks(config: BenchmarkConfig, authentication: Authentication, *,
                                 on_record: Callable[[dict[str, Any]], None] | None = None) -> dict[str, Any]:
    endpoint = parse_endpoint(config.endpoint)
    records: list[dict[str, Any]] = []
    warmups: list[dict[str, Any]] = []
    for model in config.models:
        for scenario in config.scenarios:
            with progress_scope(f"Initializing sessions/tokenizers - model={model} - scenario={scenario}"):
                sessions = {(protocol, mode): RequestSession(replace(config, protocol=protocol), endpoint, model, authentication)
                            for protocol in selected_protocols(config) for mode in selected_modes(config)}
            try:
                for warmup in (True, False):
                    rounds = config.warmup_rounds if warmup else config.rounds
                    for round_number in range(1, rounds + 1):
                        combinations = tuple(sessions)
                        if round_number % 2 == 0:
                            combinations = tuple(reversed(combinations))
                        for order, (protocol, mode) in enumerate(combinations, 1):
                            session = sessions[protocol, mode]
                            protocol_config = session.config
                            label = progress_label(protocol, model, scenario, mode)
                            label += f" - {'warmup' if warmup else 'measured'} round {round_number}/{rounds}"
                            session.progress_context = label
                            with progress_scope(label) as update:
                                if scenario.startswith("agent"):
                                    record = run_agent_scenario(protocol_config, session, scenario, round_number, mode == "streaming")
                                else:
                                    record = run_text_scenario(protocol_config, session, scenario, round_number, mode == "streaming")
                                update("PASS" if record["success"] else "FAILED")
                            record.update({"scenario": scenario, "mode": mode, "execution_order": order, "warmup": warmup,
                                           "protocol": protocol, "endpoint": display_endpoint(session.endpoint)})
                            for call in record.get("calls", []):
                                call.update({"protocol": protocol, "endpoint": display_endpoint(session.endpoint)})
                            (warmups if warmup else records).append(record)
                            if on_record is not None:
                                on_record(record)
                            if not record["success"] and not config.continue_on_error:
                                return build_result(config, endpoint, records, warmups)
            finally:
                for session in sessions.values():
                    session.close()
    return build_result(config, endpoint, records, warmups)


def build_comparisons(config: BenchmarkConfig, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if config.mode != "both":
        return []
    if config.protocol == "both":
        return [comparison for protocol in selected_protocols(config)
                for comparison in build_comparisons(replace(config, protocol=protocol), records)]
    comparisons = []
    for model in config.models:
        for scenario in config.scenarios:
            pairs = []
            for round_number in range(1, config.rounds + 1):
                matched = {record["mode"]: record for record in records if record["model"] == model
                           and record["scenario"] == scenario and record["round"] == round_number and record["success"]
                           and record.get("protocol", "chat_completions") == config.protocol}
                if "streaming" not in matched or "non_streaming" not in matched:
                    continue
                pair = {"round": round_number, "metrics": {}}
                for name in ("end_to_end_ms", "end_to_end_output_tokens_per_second", "visible_output_tokens"):
                    group = "tokens" if name == "visible_output_tokens" else "metrics"
                    stream_value = matched["streaming"][group][name]
                    non_stream_value = matched["non_streaming"][group][name]
                    pair["metrics"][name] = {
                        "streaming": stream_value, "non_streaming": non_stream_value,
                        "delta": round(non_stream_value - stream_value, 4),
                        "delta_percent": round((non_stream_value - stream_value) / stream_value * 100, 2) if stream_value else None,
                    }
                pairs.append(pair)
            comparisons.append({
                "model": model, "scenario": scenario, "protocol": config.protocol, "paired_rounds": len(pairs),
                "skipped_rounds": config.rounds - len(pairs), "pairs": pairs,
                "delta_definition": "non_streaming minus streaming; prompts and budgets match, generated text may differ",
                "deltas": {name: {
                    statistic: summarize_metric([{"metrics": {name: pair["metrics"][name][statistic]}} for pair in pairs], name)
                    for statistic in ("delta", "delta_percent")
                } for name in ("end_to_end_ms", "end_to_end_output_tokens_per_second", "visible_output_tokens")},
            })
    return comparisons


def build_result(
    config: BenchmarkConfig, endpoint: SplitResult, records: list[dict[str, Any]],
    warmups: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    summaries = [
        {**summarize_model(
            model,
            [record for record in records if record["model"] == model
             and record["scenario"] == scenario and record["mode"] == mode
             and record.get("protocol", "chat_completions") == protocol],
            config.rounds,
        ), "scenario": scenario, "mode": mode, "protocol": protocol}
        for model in config.models
        for scenario in config.scenarios
        for protocol in selected_protocols(config)
        for mode in selected_modes(config)
    ]
    failed_runs = sum(not record["success"] for record in records)
    return {
        "schema_version": 2,
        "success": bool(records) and failed_runs == 0 and not any(not record["success"] for record in warmups or []),
        "endpoint": display_endpoint(endpoint),
        "endpoints": {protocol: display_endpoint(resolve_protocol_endpoint(endpoint, protocol))
                  for protocol in selected_protocols(config)},
        "models": list(config.models),
        "configured_rounds": config.rounds,
        "completed_runs": len(records),
        "failed_runs": failed_runs,
        "runs": records,
        "summaries": summaries,
        "comparisons": build_comparisons(config, records),
        "warmup_runs": warmups or [],
        "failed_warmups": sum(not record["success"] for record in warmups or []),
        "settings": {"mode": config.mode, "scenarios": list(config.scenarios), "protocol": config.protocol,
                     "auth_type": config.auth_type,
                     "auth_order": list(_AUTH_CHAIN) if config.auth_type == "auto" else [config.auth_type],
                     "connection_reuse": config.connection_reuse, "warmup_rounds": config.warmup_rounds,
                     "tokenizer_model": config.tokenizer_model or "automatic", "automatic_retries": False},
        "metric_notes": {
            "protocol": "Chat Completions and Responses use separate sessions and summaries. Mode comparisons pair only matching protocols. Responses is stateless (store=false); input history includes returned reasoning/tool items.",
            "tcp_connect_ms": "Total TCP attempt time, including failed addresses; all attempts and TLS share the request deadline. Synchronous OS DNS resolution cannot be forcibly cancelled.",
            "ttft_ms": "Request start to first visible text chunk; null for non-streaming.",
            "ttlt_ms": "Request start to last visible text chunk; not stream-close time; null for non-streaming.",
            "prefill_proxy_ms": "Request sent to first visible text; includes network/queue/reasoning, not pure server prefill.",
            "tps": "Visible tokens excluding first text chunk / first-to-last text chunk seconds; null for single-chunk/non-streaming.",
            "end_to_end_output_tokens_per_second": "Visible output tokens / wall time; comparable across response modes.",
            "task_metrics": "Batch/agent top-level latency and output rate cover the whole task; call_metrics contain individual LLM statistics.",
            "token_counts": "output_tokens now counts visible text only; api_completion_tokens preserves provider totals separately.",
            "server_timing": "Raw provider timing retained with source; no provider-independent prefill semantics assumed.",
            "interrupted": "Partial text/tokens and observed TTFT are retained. TTLT and full-response rates are null; interrupted runs are excluded from metric summaries and comparisons.",
            "authentication_ms": "Token provider time, including cache/refresh and lock waiting; excluded from individual model request timing. Task wall time includes authentication; task authentication_ms sums call values.",
        },
    }


def format_value(value: Any, unit: str = "") -> str:
    return "n/a" if value is None else f"{value}{unit}"


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    def cell(value: Any) -> str:
        text = html.escape(format_value(value), quote=False)
        for character in ("\\", "`", "*", "_", "[", "]"):
            text = text.replace(character, "\\" + character)
        return text.replace("|", "&#124;").replace("\r\n", "\n").replace("\r", "\n").replace("\n", "<br>")

    lines = ["| " + " | ".join(cell(header) for header in headers) + " |",
             "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(cell(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def markdown_statistics(statistics: dict[str, Any] | None) -> str:
    if not statistics:
        return "n/a"
    return (f"{statistics['average']} / {statistics['p50']} / {statistics['p95']} "
            f"(n={statistics['sample_count']})")


def render_markdown(result: dict[str, Any]) -> str:
    validation = result.get("type") == "access_validation"
    settings = result.get("settings", {})
    sections = ["# LLM Access Validation" if validation else "# LLM Benchmark Report"]
    overview = [["Status", "PASS" if result["success"] else "FAIL" if validation else "PARTIAL FAILURE"],
                ["Endpoint", result.get("endpoint")],
                ["Authentication", result.get("auth_type", settings.get("auth_type"))]]
    if validation:
        overview.extend([["Model (first configured only)", result["model"]], ["Scope", result.get("scope")],
            ["Maximum model requests", result["maximum_model_requests"]],
            ["Completion budget per request", result["token_budget_per_request"]]])
    else:
        overview.extend([["Models", ", ".join(result.get("models", []))], ["Protocol", settings.get("protocol")],
            ["Response mode", settings.get("mode")], ["Configured rounds per combination", result.get("configured_rounds")],
            ["Recorded measured rounds", result.get("completed_runs")], ["Failed measured rounds", result.get("failed_runs")],
            ["Recorded warmups", len(result.get("warmup_runs", []))], ["Failed warmups", result.get("failed_warmups")],
            ["Connection reuse", settings.get("connection_reuse")], ["Tokenizer selection", settings.get("tokenizer_model")]])
    sections.extend(["## Overview", markdown_table(["Setting", "Value"], overview)])
    if validation:
        methods = result.get("auth_results")
        if methods:
            sections.extend(["## Authentication Checks", markdown_table(["Method", "Credentials Observed", "Access"],
                [[method["auth_type"], method["authentication_success"], "Success" if method["success"] else "Failed"]
                 for method in methods])])
        rows = []
        for probe in result["probes"]:
            details = probe.get("endpoint_error", {})
            messages = [probe.get("error"), details.get("message"), details.get("body_status"),
                        probe.get("guidance"), probe.get("warning")]
            rows.append([probe.get("authentication", {}).get("type"), probe["protocol"], probe["endpoint"],
                "PASS" if probe["success"] else "FAIL", probe["stage"], probe.get("http_status"),
                probe["authentication_success"], details.get("code"), "\n".join(message for message in messages if message),
                "\n".join(f"{name}: {value}" for name, value in details.get("request_ids", {}).items())])
        sections.extend(["## Access Probes", markdown_table(
            ["Method", "Protocol", "Endpoint", "Result", "Stage", "HTTP", "Credentials", "Error Code", "Details", "Request IDs"], rows),
            "## Notes", markdown_table(["Topic", "Explanation"], [["Scope", result.get("note")],
                ["Skipped protocols", ", ".join(result.get("skipped_protocols", [])) or "None"],
                ["Access versus credentials", "A token alone does not prove model access. Auto PASS requires one method to pass all selected protocols."],
                ["Limits", "Online access checks may incur cost. Small-budget truncation can pass access validation with a warning; this is not a performance or full-model test."]])])
        return "\n\n".join(sections) + "\n"
    endpoints = result.get("endpoints", {})
    if endpoints:
        sections.extend(["## Resolved Endpoints", markdown_table(["Protocol", "Endpoint"],
            [[protocol, endpoint] for protocol, endpoint in endpoints.items()])])
    for title, columns in (
        ("Latency Summary (ms)", [("ttft_ms", "TTFT"), ("ttlt_ms", "TTLT"),
                                  ("prefill_proxy_ms", "Prefill Proxy"), ("end_to_end_ms", "End To End")]),
        ("Throughput And Authentication", [("tps", "Decode TPS (tokens/s)"),
            ("end_to_end_output_tokens_per_second", "End-To-End Rate (tokens/s)"),
            ("authentication_ms", "Authentication (ms)")]),
    ):
        rows = []
        for summary in result.get("summaries", []):
            rows.append([summary["model"], summary["scenario"], summary.get("protocol", "chat_completions"), summary["mode"],
                f"{summary['successful_rounds']}/{summary['completed_rounds']}/{summary['total_rounds']}",
                *[markdown_statistics(summary["metrics"].get(name)) for name, _ in columns]])
        sections.extend(["## " + title, "Metric cells: average / P50 / P95 (n = measured samples).",
            markdown_table(["Model", "Scenario", "Protocol", "Mode", "Passed / Recorded / Configured",
                            *[label for _, label in columns]], rows)])
    comparison_rows = []
    for comparison in result.get("comparisons", []):
        for name, statistics in comparison["deltas"].items():
            comparison_rows.append([comparison["model"], comparison["scenario"], comparison.get("protocol", "chat_completions"),
                comparison["paired_rounds"], comparison["skipped_rounds"], name,
                markdown_statistics(statistics["delta"]), markdown_statistics(statistics["delta_percent"])])
    sections.extend(["## Paired Mode Comparisons",
        "Delta = non-streaming minus streaming, within the same protocol. Only successful paired rounds participate. "
        "A negative latency delta means non-streaming was faster in those samples; output length can differ.",
        markdown_table(["Model", "Scenario", "Protocol", "Pairs", "Skipped", "Metric", "Delta Avg / P50 / P95", "Delta % Avg / P50 / P95"], comparison_rows)])
    scope_headers = ["Phase", "Model", "Scenario", "Protocol", "Mode", "Round"]
    rounds = []
    entries = []
    calls = []
    tasks = []
    for phase, records in (("measured", result.get("runs", [])), ("warmup", result.get("warmup_runs", []))):
        for record in records:
            scope = [phase, record["model"], record["scenario"], record.get("protocol", "chat_completions"),
                     record["mode"], record["round"]]
            rounds.append((scope, record))
            entries.append((scope, "task" if "calls" in record else "request", record))
            if "calls" in record:
                tasks.append([*scope, record.get("call_count"), record.get("local_tool_duration_ms"),
                              record.get("delegation_duration_ms"), record.get("token_totals_complete")])
            for index, call in enumerate(record.get("calls", []), 1):
                label = f"{call.get('call_number', index)}: {call.get('agent', 'request')}"
                entries.append((scope, label, call))
                calls.append((scope, label, call))

    def state(record: dict[str, Any]) -> str:
        status = "UNVERIFIED" if record.get("validation", {}).get("status") == "unverified" else "PASS" if record["success"] else "FAIL"
        return status + (" / interrupted" if record.get("interrupted") else "")

    def timings(record: dict[str, Any]) -> list[Any]:
        metrics = record.get("metrics", {})
        return [metrics.get(name) for name in ("ttft_ms", "ttlt_ms", "tps", "end_to_end_ms")]

    sections.extend(["## Round Results",
        markdown_table([*scope_headers, "Result", "TTFT (ms)", "TTLT (ms)", "Decode TPS", "End To End (ms)", "Visible Tokens"],
            [[*scope, state(record), *timings(record), record.get("tokens", {}).get("visible_output_tokens")]
             for scope, record in rounds])])
    if calls:
        sections.extend(["## Individual LLM Calls",
            "Task-level TTFT/TTLT/decode TPS are unavailable; these rows contain individual model observations, including failed calls.",
            markdown_table([*scope_headers, "Call / Role", "Result", "TTFT (ms)", "TTLT (ms)", "Decode TPS", "End To End (ms)"],
                [[*scope, label, state(call), *timings(call)] for scope, label, call in calls])])
    if tasks:
        sections.extend(["## Task Details",
            "Round end-to-end time covers the entire task. Delegation includes child model calls; do not add it to task time again.",
            markdown_table([*scope_headers, "Calls", "Local Tools (ms)", "Delegation (ms)", "Token Totals Complete"], tasks)])
    token_rows = []
    failure_rows = []
    server_rows = []
    for scope, label, record in entries:
        tokens = record.get("tokens", {})
        token_rows.append([*scope, label, tokens.get("visible_output_tokens"), tokens.get("api_completion_tokens"),
            tokens.get("reasoning_tokens"), tokens.get("cached_input_tokens"), tokens.get("total_tokens"),
            tokens.get("output_token_count_source"), record.get("visible_token_totals_complete", tokens.get("visible_output_tokens_complete"))])
        if not record["success"]:
            failure_rows.append([*scope, label, state(record), record.get("http_status"), record.get("finish_reason"),
                record.get("validation", {}).get("status"), record.get("error", record.get("validation", {}).get("reason"))])
        if record.get("server_timing"):
            timing = record["server_timing"]
            server_rows.append([*scope, label, timing.get("source"), json.dumps(timing.get("raw"), ensure_ascii=False)])
    sections.extend(["## Token Accounting",
        "Visible text is separate from API output/reasoning/tool tokens. Task totals include children/intermediate outputs; "
        "do not sum task rows with their call rows. Incomplete visible totals are observed portions, not full output or billing totals.",
        markdown_table([*scope_headers, "Scope / Call", "Visible", "API Completion", "Reasoning", "Cached Input", "API Total", "Tokenizer / Source", "Visible Complete"], token_rows),
        "## Failures And Length Validation",
        markdown_table([*scope_headers, "Scope / Call", "Result", "HTTP", "Finish Reason", "Validation", "Reason"], failure_rows)
        if failure_rows else "No failed or unverified records."])
    if server_rows:
        sections.extend(["## Provider Timing Metadata",
            "Raw provider metadata is retained with its source; it is not normalized or asserted to be measured server prefill.",
            markdown_table([*scope_headers, "Scope / Call", "Source", "Raw Metadata"], server_rows)])
    notes = {"Unavailable values": "n/a means unavailable, not zero. Non-streaming TTFT, TTLT, decode TPS and prefill proxy are unobservable.",
        "Percentiles": "P95 with fewer than 20 samples is descriptive only; inspect each metric's n before drawing conclusions.",
        "Sample inclusion": "Warmups are excluded from measured summaries. Completed length-invalid samples retain metrics, but are excluded from successful pairs. Interrupted samples are excluded from metric summaries.",
        **result.get("metric_notes", {})}
    sections.extend(["## Metric Definitions And Limitations", markdown_table(["Topic", "Explanation"],
        [[name, note] for name, note in notes.items()])])
    return "\n\n".join(sections) + "\n"


def print_human_result(result: dict[str, Any]) -> None:
    status = "PASS" if result["success"] else "PARTIAL FAILURE"
    print(f"LLM connection benchmark: {status}")
    print(f"Endpoint: {result['endpoint']}")
    print(f"Protocol: {result['settings'].get('protocol', 'chat_completions')}")
    print(f"Authentication: {result['settings']['auth_type']}")
    if result["settings"]["auth_type"] == "auto":
        print("Authentication order: Managed Identity -> App Registration -> LLM API key")
    print(
        f"Configured rounds: {result['configured_rounds']} per model; "
        f"completed: {result['completed_runs']}; failed: {result['failed_runs']}; "
        f"failed warmups: {result['failed_warmups']}"
    )
    print()
    print("Round results")
    for record in result["runs"]:
        prefix = f"  {record['model']} / {record['scenario']} / {record.get('protocol', 'chat_completions')} / {record['mode']} / round {record['round']}"
        if "authentication" in record:
            prefix += f" / auth {record['authentication']['type'] or 'unavailable'}"
        if not record["success"]:
            print(f"{prefix}: FAIL - {record['error']}")
            continue
        metrics = record["metrics"]
        tokens = record["tokens"]
        print(
            f"{prefix}: PASS | connection {format_value(metrics['connection_setup_ms'], ' ms')}"
            f" | TTFT {format_value(metrics['time_to_first_token_ms'], ' ms')}"
            f" | TTLT {format_value(metrics['ttlt_ms'], ' ms')}"
            f" | prefill proxy {format_value(metrics['prefill_proxy_ms'], ' ms')}"
            f" | total {format_value(metrics['end_to_end_ms'], ' ms')}"
            f" | throughput {format_value(metrics['output_tokens_per_second'], ' tokens/s')}"
            f" | output {tokens['output_tokens']} tokens"
            f" ({tokens['output_token_count_source']})"
        )
        for call in record.get("calls", []):
            label = call.get("agent", f"request {call.get('sequence', '?')}")
            if "authentication" in call:
                label += f" / auth {call['authentication']['type'] or 'unavailable'}"
            if "metrics" not in call:
                print(f"    {label}: FAIL - {call['error']}")
                continue
            print(f"    {label}: TTFT {format_value(call['metrics']['ttft_ms'], ' ms')}"
                  f" | TTLT {format_value(call['metrics']['ttlt_ms'], ' ms')}"
                  f" | TPS {format_value(call['metrics']['tps'])}")
    print()
    print("Per-model/scenario/mode summary (average / p50 / p95)")
    for summary in result["summaries"]:
        print(
            f"  {summary['model']} / {summary['scenario']} / {summary.get('protocol', 'chat_completions')} / {summary['mode']}: {summary['successful_rounds']}/"
            f"{summary['total_rounds']} successful"
        )
        for name, label, unit in (
            ("connection_setup_ms", "Connection setup", " ms"),
            ("authentication_ms", "Authentication (outside model timing)", " ms"),
            ("time_to_first_token_ms", "TTFT", " ms"),
            ("ttlt_ms", "TTLT", " ms"),
            ("prefill_proxy_ms", "Prefill proxy", " ms"),
            ("first_token_after_request_ms", "First token after request", " ms"),
            ("response_headers_ms", "Response headers", " ms"),
            ("end_to_end_ms", "End to end", " ms"),
            ("output_tokens_per_second", "Output throughput", " tokens/s"),
            ("end_to_end_output_tokens_per_second", "End-to-end output rate", " tokens/s"),
        ):
            values = summary["metrics"][name]
            if values is None:
                print(f"    {label}: n/a")
                continue
            print(
                f"    {label}: {values['average']}{unit} / "
                f"{values['p50']}{unit} / {values['p95']}{unit}"
            )
        if summary["percentile_note"]:
            print(f"    Note: {summary['percentile_note']}")
    for comparison in result["comparisons"]:
        print(f"  Comparison {comparison['model']} / {comparison['scenario']} / {comparison.get('protocol', 'chat_completions')}: {comparison['paired_rounds']} valid pairs")
        for name, values in comparison["deltas"].items():
            delta = values["delta"]
            percentage = values["delta_percent"]
            print(f"    {name}: non-streaming minus streaming = "
                  f"{format_value(delta['average'] if delta else None)} "
                  f"({format_value(percentage['average'] if percentage else None, '%')})")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure latency and throughput for OpenAI-compatible LLMs."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.toml"),
        help="TOML configuration path (default: config.toml).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Write the complete result as JSON.",
    )
    parser.add_argument("--mode", choices=sorted(_MODES), help="Override benchmark response mode.")
    parser.add_argument("--protocol", choices=sorted(_PROTOCOLS), help="Select Chat Completions, Responses, or both protocols.")
    parser.add_argument("--no-progress", action="store_true", help="Disable live status and 5-second waiting messages on stderr.")
    parser.add_argument("--scenario", nargs="+", choices=_SCENARIOS, help="Select one or more scenarios.")
    parser.add_argument("--output", type=Path, help="Preflight, then atomically replace this file with complete JSON results.")
    parser.add_argument("--output-markdown", type=Path,
                        help="Preflight, then atomically save Markdown comparison tables and metric explanations.")
    parser.add_argument("--output-jsonl", type=Path,
                        help="Create a new JSONL file; flush and sync each completed round, including warmups and failures.")
    validation = parser.add_mutually_exclusive_group()
    validation.add_argument("--check-config", action="store_true", help="Validate config without credentials or network requests.")
    validation.add_argument("--validate", action="store_true",
                            help="ONLINE access check: first model, up to 64 completion tokens per request. auto checks all 3 methods in order (up to 6 requests with both protocols); single auth checks only that method. May incur cost.")
    return parser.parse_args()


def validate_output_paths(config_path: Path, *paths: Path | None) -> None:
    used = [config_path]
    for path in paths:
        if path is None:
            continue
        for other in used:
            if path.resolve() == other.resolve() or (path.exists() and other.exists() and path.samefile(other)):
                raise ConfigurationError("Output files must differ from the configuration file and from each other.")
        if path.exists() and not path.is_file():
            raise ConfigurationError(f"Output path must be a regular file: {path}")
        used.append(path)


@contextmanager
def prepared_json_output(path: Path | None) -> Iterator[TextIO | None]:
    if path is None:
        yield None
        return
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w+", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", suffix=".tmp", delete=False) as output:
            temporary_path = Path(output.name)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
            output.seek(0)
            output.truncate()
            yield output
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


@contextmanager
def prepared_jsonl_output(path: Path | None) -> Iterator[TextIO | None]:
    if path is None:
        yield None
        return
    with path.open("x+", encoding="utf-8") as output:
        try:
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
            output.seek(0)
            output.truncate()
            output.flush()
            os.fsync(output.fileno())
        except OSError:
            output.close()
            path.unlink(missing_ok=True)
            raise
        yield output


def main() -> int:
    arguments = parse_arguments()
    try:
        config = load_config(arguments.config, require_credentials=not arguments.check_config)
        if arguments.mode:
            config = replace(config, mode=arguments.mode)
        if arguments.protocol:
            config = replace(config, protocol=arguments.protocol)
        if arguments.scenario:
            config = replace(config, scenarios=tuple(dict.fromkeys(arguments.scenario)))
        if arguments.check_config:
            print(json.dumps({"success": True, "mode": config.mode, "scenarios": config.scenarios,
                              "protocol": config.protocol,
                              "endpoints": {protocol: display_endpoint(resolve_protocol_endpoint(parse_endpoint(config.endpoint), protocol))
                                            for protocol in selected_protocols(config)},
                              "models": config.models, "auth_type": config.auth_type,
                              "auth_order": list(_AUTH_CHAIN) if config.auth_type == "auto" else [config.auth_type],
                              "credentials_checked": False}))
            return 0
        validate_output_paths(arguments.config, arguments.output, arguments.output_jsonl, arguments.output_markdown)
        with (ProgressReporter(enabled=not arguments.no_progress) as progress,
              prepared_json_output(arguments.output) as output,
              prepared_json_output(arguments.output_markdown) as markdown,
              prepared_jsonl_output(arguments.output_jsonl) as jsonl):
            def save_round(record: dict[str, Any]) -> None:
                if jsonl is not None:
                    jsonl.write(json.dumps({"schema_version": 2,
                        "endpoint": record.get("endpoint", display_endpoint(parse_endpoint(config.endpoint))), "record": record},
                        ensure_ascii=False) + "\n")
                    jsonl.flush()
                    os.fsync(jsonl.fileno())

            runner = run_validation if arguments.validate else run_benchmarks
            progress.write("[STATUS] Starting access validation" if arguments.validate else "[STATUS] Starting benchmark")
            result = runner(config, on_record=save_round) if jsonl is not None else runner(config)
            progress.write("[STATUS] Saving reports" if any((output, markdown, jsonl)) else "[STATUS] Preparing final report")
            if output is not None:
                output.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
            if markdown is not None:
                markdown.write(render_markdown(result))
    except (ConfigurationError, OSError, ssl.SSLError, KeyboardInterrupt) as error:
        interrupted = isinstance(error, KeyboardInterrupt)
        message = "Benchmark interrupted; any completed JSONL rounds are retained." if interrupted else str(error)
        failure = {"success": False, "error": message}
        if arguments.json:
            print(json.dumps(failure, ensure_ascii=False))
        else:
            print(f"LLM connection benchmark: FAIL\nReason: {message}", file=sys.stderr)
        return 130 if interrupted else 1

    if arguments.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif arguments.validate:
        print_validation_result(result)
    else:
        print_human_result(result)
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
