# LLM Latency Test

[中文使用说明](Readme_CN.md)

Benchmark OpenAI-compatible Chat Completions and Responses endpoints with streaming and
non-streaming responses, visible-output checks, and tool-calling agents.
Python 3.11 or newer is required; the development environment uses Python 3.12.

## Setup

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock
cp -n config.example.EN.toml config.toml
export LLM_API_KEY='<your rotated API key>'
.venv/bin/python llm_connection_test.py --check-config
.venv/bin/python llm_connection_test.py --json --output benchmark-results.json --output-jsonl benchmark-results.jsonl
```

Copy [config.example.EN.toml](config.example.EN.toml) to [config.toml](config.toml)
and configure the endpoint and model names. The copy command preserves an existing
local configuration; config.toml is ignored by Git. Prefer environment
credentials outside disposable tests; never commit credentials. An inline
App Registration client_secret is supported for local tests. The previously
configured plaintext credential must be rotated by its
owner. `--check-config` validates settings without resolving credentials,
loading tokenizers, or making network requests.

The standard-library model transport keeps DNS, TCP, and TLS timing visible.
Direct runtime dependencies are tiktoken and azure-identity. requirements.lock pins the resolved
stable dependencies and pip version. To refresh them deliberately, upgrade the
packages, run the local tests, and update both dependency files.

tiktoken may download its public encoding data on first use. Initialization is
outside individual request timing. To cache encodings ahead of a benchmark:

```bash
.venv/bin/python -c 'import tiktoken; tiktoken.encoding_for_model("gpt-4o")'
```

For model aliases, set `benchmark.tokenizer_model` only when the underlying
tokenizer is known. An unrecognized model uses a labelled heuristic estimate;
strict output-length validation then reports `unverified`, not `passed`.
Tokenizer counts are local measurements, not provider-certified token counts.

## Live Progress

Normal runs and `--validate` print live, flushed status lines to **stderr** by
default. Labels identify the endpoint protocol, model, scenario, response mode,
warmup/measured round, short-request sequence, and agent/sub-agent step. For example:

```text
[START #3] Responses endpoint testing - model=YOUR-MODEL - short output (independent) - streaming - measured round 1/3 - request 1/20
[STATUS #3] Responses endpoint testing - model=YOUR-MODEL - short output (independent) - streaming - measured round 1/3 - request 1/20 - authenticating (azure_app_registration)
[WAIT #3] Responses endpoint testing - model=YOUR-MODEL - short output (independent) - streaming - measured round 1/3 - request 1/20 - request sent; waiting for response headers - elapsed 5.0s
[STATUS #3] Responses endpoint testing - model=YOUR-MODEL - short output (independent) - streaming - measured round 1/3 - request 1/20 - first visible text received; generating
```

Background heartbeats show the current stage and elapsed time every five seconds
while an operation is active, including tokenizer initialization and authentication.
They do not delay requests, add model calls, or extend request/task deadlines.
`PASS`/`FAILED` on round lines reflect the scenario result; request completion alone
does not prove a strict output-length requirement passed. Non-streaming requests
wait for complete JSON and do not report a first-text event.

Progress excludes credentials, headers, endpoint query strings, prompts, and
generated text. It does not enter stdout JSON or saved JSON/JSONL/Markdown reports.

```bash
.venv/bin/python llm_connection_test.py --json 2>progress.log
.venv/bin/python llm_connection_test.py --no-progress
```

In-request status changes are queued for background output, not printed for each
text chunk. Progress still has some scheduling/logging overhead; use
`--no-progress` consistently for measurements that need minimal observer overhead.
`--help` and `--check-config` do not start progress reporting or online operations.

## Protocol And Base URL

```toml
[llm]
endpoint = "https://YOUR-RESOURCE.openai.azure.com/openai/v1"
protocol = "chat_completions" # chat_completions, responses, or both
```

The base URL expands to `/openai/v1/chat/completions` or `/openai/v1/responses`.
Trailing slashes are accepted. Existing full endpoint URLs work too: their
known `/chat/completions` or `/responses` suffix is replaced for the selected
protocol. Query parameters are preserved in requests but excluded from reports.
For proxies use their actual prefix, for example `https://proxy.example/v1`;
the program does not guess or insert `/openai/v1` into other base paths.

```bash
.venv/bin/python llm_connection_test.py --protocol responses --mode both
.venv/bin/python llm_connection_test.py --protocol both --mode both --scenario baseline \
	--output benchmark-results.json --output-jsonl benchmark-protocols.jsonl
.venv/bin/python llm_connection_test.py --protocol both --check-config
```

`--protocol` overrides `llm.protocol`; omitted protocol defaults to
`chat_completions`. Protocol is independent of `--mode`: `protocol=both` and
`mode=both` test all four combinations. They have separate connection sessions
and warmups, reverse execution order on alternating rounds, and report separate
summaries. Records and JSONL include `protocol` and the resolved `endpoint`;
top-level `endpoints` lists both paths. Streaming/non-streaming comparisons never
pair across protocols; compare protocol-specific summaries for protocol effects.
Two protocols double request count and potential cost; no protocol fallback or
automatic retry is performed if the server/model rejects one.

Responses supports the same text, length, conversation and agent scenarios. The
configured completion budget maps to `max_output_tokens`, and reasoning_effort
maps to `reasoning.effort`. Chat-only `stream_options.include_usage` is not sent;
Responses usage comes from its response object/terminal event. Raw `api_usage`
is retained, while input/cache/reasoning/completion counts use shared report
fields. Only `response.output_text.delta` updates visible-text timing; lifecycle,
reasoning and tool-argument events do not. A terminal response event is required,
not Chat Completions' `[DONE]`. Incomplete generation fails output validation;
missing terminal events retain partial observations as interrupted failures.

Responses requests use `store=false`, explicitly resend history, and request
encrypted reasoning items for stateless continuation. Completed output items,
including function call IDs and encrypted reasoning, are returned in subsequent
agent/conversation input. Neither `previous_response_id` nor server-managed
conversation state is used. Reports may therefore contain encrypted reasoning
items as well as generated text. This benchmark covers text and deterministic
local function tools, not hosted tools, multimodal inputs, background jobs or
WebSocket transport. The target endpoint/model must support the selected API.

## Online Access Validation

`--check-config` checks configuration offline; it cannot establish whether an
App Registration can acquire a token or invoke the model. Use `--validate` for
a small **online** access check with the configured authentication method:

```bash
.venv/bin/python llm_connection_test.py --validate --protocol both
.venv/bin/python llm_connection_test.py --validate --protocol chat_completions --json \
	--output validation.json --output-jsonl validation.jsonl
```

This checks only the **first configured model**, once per selected protocol,
using non-streaming and a short `Reply with OK.` prompt. The completion budget
is capped at 64 tokens (or your configured budget if smaller). Both protocols
therefore make at most two model requests with a single auth method. In `auto`,
validation independently checks **Managed Identity -> App Registration -> LLM
API key**, even after an earlier success: at most six requests with both
protocols. Missing credentials fail that method without a model request; token
acquisition failure skips its remaining protocols but continues to the next
method. It may incur identity-network traffic and model charges. It
does not load tokenizers, run warmups/rounds/agents, or test streaming, output
length, other deployments, or performance. Mode/scenario selections and
benchmark continue_on_error do not change this bounded probe.

`azure_app_registration` uses only that application; it never substitutes MI,
your CLI login or a key. Only auto validation checks all three methods; ordinary
auto benchmarks still stop credential selection at the first usable credential
and do not switch identities on model HTTP errors. Output
separates credential acquisition from network, endpoint, and response failures.
On HTTP errors it reads at most 16 KiB under the request deadline and reports
only sanitized error code/message and whitelisted request IDs. Other body data
and authorization headers are not dumped. Empty, malformed or oversized error
bodies can leave only status/IDs; a failed body read does not hide the HTTP error.

If credential acquisition passes but HTTP 401/403 follows, the app obtained a
token but the resource rejected access. Check the app's **service principal**
role on the exact target Azure OpenAI resource (for example Cognitive Services
OpenAI User), resource tenant, token audience/scope and endpoint; granting your
personal account a role does not authorize the app. The server's diagnostic
message and request ID help distinguish these cases. Validation does not change
roles/scopes or retry model calls with another identity.

`--json`/`--output` save an `access_validation` report with `probes`, not benchmark
`runs`/summaries. JSONL writes one probe per line. Auto validation additionally
returns ordered `auth_results` and `usable_auth_types`, and prints each method
as `Managed Identity - Failed`, `App Registration - Success`, etc. Here Success
means endpoint access passed, not merely token acquisition. Auto exits with 0
when at least one method passes **all** selected protocols; failed alternatives
remain visible. Single auth exits with 0 only if all its protocol probes pass.
Otherwise the exit code is 1 (130 for Ctrl+C). A terminal truncated
response can pass the access check with a warning: accepted access does not
prove a normal completed answer under this small budget. `--validate` and
`--check-config` are mutually exclusive; output path preflight still precedes
all online work.

## Output Persistence

`--output-markdown PATH` saves a rendered-table-friendly Markdown report without
changing terminal output. It works for benchmarks and `--validate`, alone or
alongside JSON/JSONL:

```bash
.venv/bin/python llm_connection_test.py --protocol both --mode both \
	--output-markdown benchmark-results.md
.venv/bin/python llm_connection_test.py --validate --protocol both \
	--output-markdown validation.md
```

Benchmark tables cover protocol/mode latency and throughput summaries, paired
mode differences, recorded rounds/warmups, individual agent calls, task timing,
token accounting/completeness, failure categories and failed-call usage, length validation, and provider timing
metadata with its source. Metric explanations and limitations are included.
Summary cells show average / P50 / P95 and their own measured sample count;
each summary has both **All completed** and **Successful only** rows.
`n/a` means unavailable, never zero. Task totals must not be summed with their
child-call rows. Small-sample P95 and client-prefill-proxy caveats remain explicit.
Heuristic token counts are flagged in Measurement Warnings; their rates are
approximate and cannot be compared strictly with known-tokenizer rates.
Access reports include ordered authentication results, probes, HTTP errors and
request IDs. Markdown focuses on measurements/diagnostics, not full generated
answers or encrypted reasoning histories; use JSON for the complete raw report.

Markdown paths are preflighted before online work and atomically replaced after
the complete report is generated, including failure reports with a nonzero exit
code. Existing Markdown survives interruption or a failed write of that report.
Each final file is replaced independently, not as a multi-file transaction.
Markdown is not incremental; use JSONL as well to retain completed work on
interruption. Open the result in VS Code's Markdown preview to compare tables.
Exporting a report itself makes no additional model requests.

`--output PATH` preflights the destination by writing and syncing a temporary
file in its parent directory before any benchmark requests. On completion,
including a report with failed samples, it atomically replaces the final JSON
file. Interruption or a failed save leaves any previous report intact.

`--output-jsonl PATH` creates a **new** file, checks writing and syncing before
requests, then writes one JSON object per completed model/scenario/protocol/mode round
in execution order, including warmups and failures. It can be used alone or
with `--output`; each line has `schema_version`, a query-free `endpoint`, and
`record` (the same record found in `runs` or `warmup_runs`, with a `warmup` flag).
There is no summary/footer line. Existing JSONL files are rejected, not appended
or overwritten; use a new filename for each execution.

All output parent directories must already exist. Outputs cannot alias the config
file or each other, including symbolic/hard links. `--check-config` remains
config-only: it neither preflights nor creates output files.

Every JSONL line is flushed and synced after the round's timing is complete,
before starting the next round. An agent or short batch is one round: its nested
`calls` are saved only when the whole round finishes, not after each call/chunk.
Ctrl+C exits with status 130 and retains completed JSONL records; the current
unfinished round is not saved. Write/sync errors abort subsequent requests even
with `continue_on_error=true`. Preflight does not reserve disk space or guarantee
later writes; disk failure or forced termination can leave an incomplete trailing
line, which readers should discard. Reports contain model output and should be
handled as application data.

## Azure Authentication

The example config uses `llm.auth_type = "auto"`: Managed Identity, then App
Registration, then LLM API key. All three are enabled; missing credentials,
identity network errors and token acquisition failures advance to the next one.
The selected method is reused across calls; if token refresh later fails, the
chain moves forward, not back to a previously failed method. Failed methods are
not probed again until a new benchmark execution. If all fail, no model request
is sent. There is no Azure CLI or interactive-login fallback.

```toml
[llm]
auth_type = "auto"
api_key_env = "LLM_API_KEY"
auth_header = "Authorization"
auth_scheme = "Bearer"

[azure_auth]
scope = "https://cognitiveservices.azure.com/.default"
managed_identity_client_id = "" # Empty selects system-assigned identity.
tenant_id = ""
client_id = ""
client_secret = ""
```

In auto mode, `managed_identity_client_id` is independent of the App Registration
`client_id`. Fill the latter three blank fields to enable a usable App Registration
credential (IDs or environment references; client_secret is the literal secret value).
Leave them blank to skip it. Export `LLM_API_KEY` for the final fallback. An
unresolved user-assigned identity ID skips Managed Identity rather than silently
using system-assigned identity. Auto mode requires HTTPS. For Azure-style keys,
set `llm.auth_header="api-key"` and `llm.auth_scheme=""`; do not put auth headers
in `[headers]` when using auto.

Fallback only acquires credentials: a model HTTP 401/403 or other HTTP failure
does **not** retry with another identity/key. A token does not prove resource
authorization. Per-call `authentication.type` reports the actual method;
`authentication.attempts` contains sanitized chain history, including failures.
Both JSON and JSONL retain it. `authentication_ms` includes fallback attempts
but excludes them from individual model timing. Each identity operation uses
the configured SDK timeouts; discovering an unavailable identity is not always
instantaneous.

For a single method, `llm.auth_type` still accepts `api_key`,
`azure_app_registration`, or `azure_managed_identity`, with no fallback. Omitting
auth_type keeps the backward-compatible `api_key` behavior. Single Azure modes do not read
llm.api_key_env and always set `Authorization: Bearer <token>`, independently
of llm.auth_header / auth_scheme. Remove any plaintext llm.api_key and any
Authorization or api-key entries from [headers]. Azure modes require HTTPS.
Merge the following fragments into the existing config rather than duplicating
tables; retain the benchmark and request settings.

### App Registration / Service Principal

This mode uses ClientSecretCredential, not interactive login or a credential
chain. Supply the application client ID, tenant ID, and client secret **value**
(not the secret ID) through environment variables:

```bash
export AZURE_TENANT_ID='<tenant-id>'
export AZURE_CLIENT_ID='<application-client-id>'
read -rsp 'Azure client secret: ' AZURE_CLIENT_SECRET
export AZURE_CLIENT_SECRET
printf '\n'
```

```toml
[llm]
endpoint = "https://YOUR-RESOURCE.openai.azure.com/openai/v1/chat/completions"
models = ["YOUR-DEPLOYMENT-NAME"]
auth_type = "azure_app_registration"

[azure_auth]
scope = "https://cognitiveservices.azure.com/.default"
tenant_id = "${AZURE_TENANT_ID}"
client_id = "${AZURE_CLIENT_ID}"
client_secret_env = "AZURE_CLIENT_SECRET"
```

The three Azure environment names above are defaults when the corresponding
fields are omitted. The example config contains explicit blanks: replace them
with the values/references shown above before selecting App Registration alone.
tenant_id and client_id may alternatively be literal IDs. For disposable tests,
replace client_secret_env with `client_secret = "<temporary-test-secret>"` to
read the value directly without any environment lookup or `${...}` expansion.
Do not configure both nonempty sources. The inline value is hidden from config
repr and redacted from errors/SDK logs, but remains plaintext on disk; do not
commit the config with credentials. Certificate and federated
workload credentials are not exposed by this mode.

### Managed Identity

Run on an Azure host with Managed Identity enabled and access to its identity
endpoint; an ordinary local terminal cannot use this mode simply by running
`az login`. No client secret or tenant ID is required.

```toml
[llm]
endpoint = "https://YOUR-RESOURCE.openai.azure.com/openai/v1/chat/completions"
models = ["YOUR-DEPLOYMENT-NAME"]
auth_type = "azure_managed_identity"

[azure_auth]
scope = "https://cognitiveservices.azure.com/.default"
# Leave managed_identity_client_id and the legacy client_id empty for system-assigned identity.
# For user-assigned identity, use its client ID, not its principal/object ID:
# managed_identity_client_id = "USER-ASSIGNED-IDENTITY-CLIENT-ID"
```

Both modes require authorization at the destination. For Azure OpenAI, assign
the identity an appropriate data-plane role such as Cognitive Services OpenAI
User on the target resource. For a custom Entra-protected proxy, configure its
scope, for example `api://TARGET-API-APP-ID/.default`, and grant the application's
required permissions/app roles and consent. The proxy must accept the token's
audience and issuer. A valid token alone does not establish access. HTTP 401
usually warrants checking audience, issuer, token and endpoint; HTTP 403 also
warrants checking roles and resource policy.

The Azure SDK caches and refreshes tokens. One thread-safe authentication
selection is shared across response modes, models and sub-agents. Only auto mode
falls back between the three configured methods; single modes do not. Neither
uses Azure CLI or interactive authentication. Credential details are suppressed on
failure, and tokens and client secrets are not written to reports.
During synchronous credential operations, Azure/MSAL log messages and exception
stacks are replaced before log handlers receive them; logging in unrelated
threads or outside those operations is unchanged. Sanitized failures retain a
category (`credential_unavailable`, `client_authentication_error`,
`identity_network_error`, or `identity_error`) and an HTTP status when available.
After credential redaction, up to three distinct, format-checked `AADSTS` codes
are retained, with fixed guidance for common failures. Raw SDK messages remain hidden:

| Code | What to check |
| --- | --- |
| `AADSTS7000215` | Use the client secret value, not its Secret ID; verify the application. |
| `AADSTS7000222` | The secret expired; create a new secret and update the local credential. |
| `AADSTS700016` / `AADSTS90002` | Verify the application Client ID and Tenant ID. |
| `AADSTS500011` / `AADSTS70011` | Verify the target resource scope and tenant. |

`Credential acquisition: FAIL` means model access has not been tested. Fix the
identity error before investigating model-endpoint HTTP 401/403 or resource roles.
When no safe code is available, the generic sanitized category is still reported.

`authentication_ms` measures token-provider time, including cache checks,
refresh and lock waiting, outside individual LLM latency/throughput timing.
DNS/TCP/TLS fields describe the model endpoint only, not Entra or identity
endpoints. Agent/batch wall time includes authentication; their top-level
authentication_ms sums per-call values and can include concurrent lock waits.
With no warmups, the first task may include cold token acquisition; inspect this
field before comparing whole-task timings. Identity network operations use the
configured connection/read timeouts. Task deadlines are checked before and after
authentication but do not forcibly cancel an in-progress SDK network operation.

`--check-config` remains fully offline for both modes, even with no Azure
credentials present, and rejects malformed hosts and invalid ports before any
request. Local tests mock Azure credentials; they do not contact
Microsoft Entra, Managed Identity endpoints or real model services.

## Modes And Scenarios

```bash
.venv/bin/python llm_connection_test.py --mode streaming
.venv/bin/python llm_connection_test.py --mode non_streaming
.venv/bin/python llm_connection_test.py --mode both --scenario long_output short_burst
.venv/bin/python llm_connection_test.py --mode both --scenario prefill_short prefill_long
.venv/bin/python llm_connection_test.py --mode both --scenario agent agent_serial agent_parallel
.venv/bin/python llm_connection_test.py --scenario short_conversation --json
```

| Scenario | Workload |
| --- | --- |
| baseline | Configured system/user prompts and completion budget |
| prefill_short / prefill_long | Deterministic 128 / 4096-word input context by default |
| long_output | Requests a long article; passes only with more than 1000 visible tokens |
| short_burst | 20 sequential independent requests by default; each must produce fewer than 200 visible tokens |
| short_conversation | Same short requests, retaining history within each batch |
| agent | Model calls a deterministic local lookup tool, then writes an answer |
| agent_serial | Main agent delegates two topics to separate sub-agents sequentially, then summarizes |
| agent_parallel | Identical two topics, executed by two concurrent sub-agents, then summarized |

The example config runs only baseline to limit accidental cost. With two models,
three measured rounds, one warmup round, and both modes, baseline makes 16 API
calls. Short scenarios multiply requests by short_requests; a successful agent
case normally uses two LLM calls and a delegated case uses six. Warmup rounds
execute the entire selected workload and incur the same cost as measured rounds.
The 16-call example above assumes one protocol; both protocols make 32 calls
with those same two-model settings.

### Generation Budget And Reasoning

The example baseline budget is `request.max_completion_tokens = 2048`.
If neither token-limit option is supplied, the legacy `max_tokens` default is
also 2048; explicit limits are never increased automatically. Responses maps
either option to `max_output_tokens`. The cap covers hidden reasoning as well
as visible output, so 256 tokens can be exhausted before any answer appears.
HTTP 200 with `finish_reason=length` remains a failed benchmark, classified as
`budget_exhausted`, rather than an HTTP or connection failure.

2048 is a starting point, not a guarantee. A higher cap can increase cost.
First compare budgets with prompts, reasoning effort, rounds, and connection
policy unchanged. Only after verifying support on every selected deployment,
try `request.reasoning_effort = "low"` in a separate experiment. Use prompts to
request concise answers instead of truncating them with a tiny generation cap;
actual visible length is still validated for strict short/long scenarios.
Do not set an arbitrary tokenizer to make length checks pass. Stabilize the
workload before increasing sample counts; small-sample P95 remains descriptive.
There are no automatic retries, budget escalation, or extra model probes.

## Measurement Contract

| Metric | Streaming | Non-streaming |
| --- | --- | --- |
| ttft_ms | Request start to first visible text chunk | null |
| first_token_after_request_ms | Request send complete to first visible text chunk | null |
| ttlt_ms | Request start to last visible text chunk | null |
| prefill_proxy_ms | Same client interval as first_token_after_request_ms | null |
| tps | Visible tokens after the first chunk divided by first-to-last chunk time | null |
| end_to_end_ms | Request start to complete HTTP response | Same definition |
| end_to_end_output_tokens_per_second | Visible text tokens divided by end-to-end seconds | Same definition |
| authentication_ms | Token-provider time, outside individual model request timing | Same definition |

Text arrives in chunks, not individually observable tokens. TTFT and TTLT use
those chunk arrival times. Decode TPS is chunk-observed and has tokenizer-boundary
limitations; a single text chunk has no measurable decode TPS. Stream-close
waiting and trailing usage events do not inflate generation duration.

The prefill proxy includes network, server queuing, scheduling, and potentially
hidden reasoning. It is **not pure server prefill time or prefill throughput**.
Raw provider `timing` / `timings` metadata and the `Server-Timing` header are
retained under server_timing with their source. They are not assigned
provider-independent semantics. Cached input tokens are reported when available;
repeated prompts may hit provider caches, which the client cannot reliably reset.

API completion tokens, reasoning tokens, total tokens, and visible text token
counts are separate. Tool-call arguments are not visible text. A normal final
response must contain text and finish with `stop`; truncation, interruption,
empty responses, and length violations are failures with retained measurements.
Missing usage remains null rather than becoming a fabricated API count.

An interrupted response is unsuccessful even if its received text looks complete:
an unfinished `Content-Length`, invalid chunked framing, or missing SSE completion
markers fails validation. Streaming failures retain fully decoded text events,
observed TTFT, received usage/tool fragments, and server timing metadata.
`interrupted=true`, `response_complete=false`, and
`tokens.visible_output_tokens_complete=false` distinguish these partial records.
TTLT, decode TPS, generation duration, request end-to-end time and output rate
are null; `failure_elapsed_ms` records time until failure, and
`stream.last_observed_text_ms` records request start to the last text seen, not
a completed response's TTLT. Undecodable fragments are not counted as text.

For compatibility, time_to_first_token_ms aliases ttft_ms and
output_tokens_per_second aliases tps. In schema version 2, output_tokens counts
visible text instead of the API completion total; generation_duration_ms ends
at the last text chunk instead of stream close.

## Comparison And Connection Policy

`both` alternates execution order each measured round. Each mode has its own
connection session; prompts, parameters, warmups, and budgets match. Reuse is
enabled by default and can be disabled with benchmark.connection_reuse. If a
server closes a connection, a later request reconnects and reports fresh setup
timing. There are no automatic request retries or artificial inter-request delays.

DNS, successive TCP address attempts, TLS and model HTTP I/O share the remaining
request/task budget. `tcp_connect_ms` includes failed attempts, and
`tcp_connect_attempts` reports the number tried. Budget checks happen at phase/read
boundaries, not through process-level cancellation: synchronous OS DNS resolution
cannot be forcibly cancelled, but an overrun prevents subsequent connection work.

Comparisons pair successful measured rounds and show non-streaming minus
streaming differences, including percentages, for end-to-end latency, end-to-end
output rate, and visible output length. Different generated lengths are reported;
latency differences alone do not establish which mode generates tokens faster.
Warmups are recorded separately and excluded from measured percentiles.

Summaries group by model, scenario, protocol, and mode, with sample count, mean, min/max,
P50, P95, and success rate. P95 with fewer than 20 samples is descriptive only.
Length-invalid but completed responses remain in measurement summaries; valid
paired comparisons exclude them. Interrupted responses (and task runs containing
them) are excluded from metric summaries and successful comparisons; observed
partial timings remain in the individual records. Failures without observations
have no fabricated timings.

The existing `summaries[].metrics` inclusion policy is unchanged.
`summaries[].successful_metrics` adds the same statistics restricted to successful
non-interrupted rounds; `successful_call_metrics` does the same for task calls.
Token count sources and heuristic warnings are available as `token_count_sources`
and `token_count_note` in each summary. Request budgets and configured reasoning
effort are retained in `settings` (null effort means the model default).

Each failed record has a `failure_category`: `budget_exhausted`, `empty_output`,
`length_violation`, `length_unverified`, `abnormal_finish`, `authentication_error`,
`http_error`, `transport_error`, `response_error`, `interrupted_response`, or
`task_error`. Tasks inherit the first failed call's category, or `task_error`
when only local task execution fails; inspect all call rows for other causes.
Legacy records without a category are reported as `unclassified`, not guessed
from error text.

`failure_summary` separates measured and warmup failures; each grouped summary
also has its own `failure_summary`. `categories` counts failed/unverified rounds,
while `failed_calls` and `tokens` count failed individual calls only, without
double-counting tasks or charging successful intermediate calls to failed-call
usage. For each token field, `observed_total` sums available counts,
`reported_calls` counts calls with that field, and `complete` is false if counts
are missing or a response was interrupted. No available counts means null,
not zero; zero is valid when there were no failed calls. These are observed
counts, not monetary costs or complete billing totals.

## Agent And Batch Results

Agent tests use actual protocol-native function/tool calls, not role descriptions alone.
Tool requests are forced for a reproducible workload; unsupported or ignored
tool requests fail explicitly and never fall back to plain chat. Both children
use the currently benchmarked model. The local tool data and child topics are
fixed to avoid external-service variability.

Each agent or short batch reports whole-task end-to-end timing and cumulative
visible tokens, plus per-call records and call_metrics. Task-level TTFT, TTLT,
and decode TPS are null: inspect individual calls instead. Cumulative tokens
include child/intermediate answers, not just the main agent's final answer.
Local-tool time and delegation time are separate; delegation time includes
child LLM calls and must not be added to task time again.

Agent max_steps applies to each agent, while max_calls, total_token_budget, and
task_timeout_seconds bound the whole task. Each request reserves its maximum
completion budget, including hidden reasoning. Reservations are conservative
and are not refunded when usage is missing or smaller than the limit. Failed
tasks retain completed and failed call records. Request timeout_seconds also
applies independently to every HTTP call.

Failed calls without token data do not abort aggregation. Visible totals sum the
observed portions; `visible_token_totals_complete=false` means these are not a
complete output count. Missing API usage makes the corresponding total null;
`token_totals_complete=false` also covers interruption. Do not use partial totals
as complete billing figures. Successful calls in a failed task remain available
in `calls` and `call_metrics`.

## Local Validation

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python llm_connection_test.py --help
.venv/bin/python llm_connection_test.py --check-config
.venv/bin/python -m pip check
```

The suite uses controlled timestamps, a test tokenizer, and a local HTTP server.
It does not use credentials or call a real model endpoint. CLI exit status is
zero only when all measured runs and warmups pass. Real API validation incurs
cost and should be approved separately.