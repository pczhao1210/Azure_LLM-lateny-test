# Repository Instructions

## Scope

- Benchmark OpenAI-compatible Chat Completions endpoints using Python 3.11+.
- Keep the standard-library HTTP transport and DNS/TCP/TLS measurements.
- Preserve the existing CLI and result fields where practical; document changes
  to metric semantics. Avoid heavyweight agent frameworks.
- Declare dependencies in pyproject.toml and pin resolved versions in
  requirements.lock. Use stable releases only.

## Measurement Contract

- Support streaming, non-streaming, and alternating paired comparisons using
  identical prompts, token budgets, connection policies, and warmup settings.
- Streaming TTFT/TTLT measure request start to first/last visible text chunk.
  Record request-sent-relative TTFT separately.
- Decode TPS excludes the first text chunk and uses first-to-last text time,
  not stream-close time. A single text chunk has no measurable decode TPS.
- Non-streaming TTFT, TTLT, decode TPS, and client prefill proxy are null.
- A client prefill proxy includes network, queuing, and potentially reasoning;
  never label it as measured server prefill. Preserve server timing metadata
  separately when available, with its source.
- Separate visible text tokens from API completion/reasoning/tool-call tokens.
  Label tokenizer and heuristic counts; unsupported tokenizers cannot establish
  that a strict visible-output length requirement passed.
- Long output must exceed 1000 visible tokens. Short output must stay below
  200 visible tokens. Token limits alone do not prove either requirement.
- Continuous short requests run sequentially without artificial delays;
  independent requests and growing conversation history are separate scenarios.
- Agents must execute real tool calls with bounded steps and deterministic
  local tools. Sub-agents make separate model calls, serially or concurrently.
  Report per-call metrics and whole-task timing/token totals separately.
- Report failures, interrupted streams, length violations, cache usage,
  sample counts, and small-sample percentile limitations explicitly.

## Safety And Validation

- Never commit API keys or print authorization headers. Configure credentials
  through environment variables and omit endpoint query strings from reports.
- Do not call a real model endpoint without explicit approval: it incurs cost.
- Use local unittest fixtures and a mock HTTP server for validation. Do not
  add sleeps or depend on live API responses for deterministic timing tests.
- Run `.venv/bin/python -m unittest discover -s tests -v` after behavioral edits.
- Run `.venv/bin/python llm_connection_test.py --help` for CLI changes.
- Keep documentation and example configuration aligned with implemented options.