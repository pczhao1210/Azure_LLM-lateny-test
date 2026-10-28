# LLM 延迟与吞吐测试

[English](README.md) | 中文使用说明

本工具用于测试兼容 OpenAI Chat Completions 和 Responses 的接口，支持流式与非流式对比、
长短输出、连续对话，以及真实工具调用的 Agent 和 Sub-Agent 场景。
结果既可以直接在终端查看，也可以保存为 JSON。

## 1. 快速开始

### 环境与安装

需要 Python 3.11 或更新版本，当前项目开发环境为 Python 3.12。
在仓库根目录执行以下命令；已有 `.venv` 时无需重新创建。

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock
```

以下示例直接使用虚拟环境中的 Python，不需要先执行 `activate`。
依赖声明见 [pyproject.toml](pyproject.toml)，精确版本见 [requirements.lock](requirements.lock)。

### 配置接口与凭据

从 [config.example.CN.toml](config.example.CN.toml) 创建本地配置；已有配置不会被覆盖：

```bash
cp -n config.example.CN.toml config.toml
```

本地 [config.toml](config.toml) 已加入 Git 忽略规则，不应提交真实凭据。

打开 [config.toml](config.toml)，将 `[llm]` 中的 `endpoint` 和 `models`
修改为实际接口和模型。`endpoint` 可以简写为 API 基础地址，
例如 `https://xxxxx.openai.azure.com/openai/v1`；程序按 `llm.protocol` 自动补齐路径。
旧的完整 `/chat/completions` 或 `/responses` 地址也支持。

示例配置默认 `llm.auth_type = "auto"`，依次尝试 **Managed Identity → App Registration → LLM Key**。
前一项缺少配置或取 Token 失败时进入下一项，不需要手动切换。只使用 Key 时可设置：

```bash
export LLM_API_KEY='<your-api-key>'
```

前面的 Azure 身份成功时无需设置 `LLM_API_KEY`。模板从 `AZURE_TENANT_ID`、
`AZURE_CLIENT_ID` 和 `AZURE_CLIENT_SECRET` 环境变量读取 App Registration 凭据，
未设置时自动跳过；测试环境也可按下文通过 `client_secret` 直接填写临时 Secret。

非临时测试建议通过环境变量配置凭据；含密钥的配置不要提交到版本库。泄露的凭据应在服务端轮换。

### 校验并运行

```bash
# 只检查配置格式和参数，不读取凭据、不加载分词器、不调用接口。
.venv/bin/python llm_connection_test.py --check-config

# 按配置运行，直接查看文本报告。
.venv/bin/python llm_connection_test.py

# 输出完整 JSON，同时逐轮保存 JSONL。每次执行使用新的 JSONL 文件名。
.venv/bin/python llm_connection_test.py --json --output benchmark-results.json --output-jsonl benchmark-results.jsonl
```

**除 `--help` 和 `--check-config` 外，正常运行会调用配置中的模型接口，可能产生费用。**
配置检查通过不代表密钥有效、模型存在或接口支持所选功能。

示例配置默认只运行 `baseline`、一种协议、两种响应模式，每种组合预热 1 轮、正式测试 3 轮。
请求数量为“模型数 × 协议数 × 模式数 ×（预热 + 正式轮数）”；四个模型时单协议为 32 次，
启用两种协议为 64 次。请以当前配置中的模型和轮数为准。

## 2. 命令行参数

```bash
.venv/bin/python llm_connection_test.py --help
```

| 参数 | 用法 |
| --- | --- |
| `--config PATH` | 指定 TOML 配置；默认读取当前目录的 `config.toml` |
| `--mode streaming` | 只测试流式响应 |
| `--mode non_streaming` | 只测试非流式响应 |
| `--mode both` | 使用相同场景参数交替测试两种模式，并生成配对对比 |
| `--protocol chat_completions` | 使用 Chat Completions 协议，兼容原有默认行为 |
| `--protocol responses` | 使用 Responses 协议 |
| `--protocol both` | 测试两种协议，独立统计；与 `--mode both` 组合时共四种组合 |
| `--scenario NAME ...` | 选择一个或多个场景，用空格分隔；覆盖配置中的场景列表 |
| `--json` | 将终端输出切换为完整 JSON |
| `--no-progress` | 关闭默认写到 stderr 的实时状态和每 5 秒一次的等待提示 |
| `--output PATH` | 运行前预检，结束后原子替换完整 JSON；不改变终端输出格式 |
| `--output-jsonl PATH` | 新建 JSONL，每轮完成后写入并同步磁盘，包含预热和失败轮次 |
| `--output-markdown PATH` | 保存可渲染的 Markdown 表格报告与指标说明，支持基准和访问验证 |
| `--check-config` | 离线校验配置，不执行测试 |
| `--validate` | 在线访问探测：首个模型、小预算；auto 依次独立检查三种鉴权，可能产生费用 |

命令行指定的协议、模式和场景优先于配置。轮数、预热、请求数量等其他设置通过 TOML 修改。
所有输出路径的父目录必须已经存在，不能指向配置文件或互相指向同一文件（包括符号链接和硬链接）。
`--output` 和 `--output-markdown` 会在完整报告生成后替换已有文件；
`--output-jsonl` 不覆盖、不追加已有文件，每次运行需换一个新文件名。
真正运行前会在目标目录试写并同步磁盘；`--check-config` 只检查配置，不检查或创建输出文件。

### 实时运行状态

普通测试和 `--validate` 默认实时刷新状态，显示协议、模型、场景、流式/非流式模式、
预热/正式轮次；连续短输出还会显示请求序号，Agent 会显示角色、步骤和调用编号。例如：

```text
[START #3] Responses endpoint testing - model=YOUR-MODEL - short output (independent) - streaming - measured round 1/3 - request 1/20
[STATUS #3] Responses endpoint testing - model=YOUR-MODEL - short output (independent) - streaming - measured round 1/3 - request 1/20 - authenticating (azure_app_registration)
[WAIT #3] Responses endpoint testing - model=YOUR-MODEL - short output (independent) - streaming - measured round 1/3 - request 1/20 - request sent; waiting for response headers - elapsed 5.0s
[STATUS #3] Responses endpoint testing - model=YOUR-MODEL - short output (independent) - streaming - measured round 1/3 - request 1/20 - first visible text received; generating
```

执行中的阶段每 5 秒提示一次当前状态和已等待时间，覆盖 tokenizer 初始化、鉴权、
连接/发送、等待响应头、等待首段可见文本和生成过程。该后台心跳不会人为延迟请求、
增加模型调用或延长原有请求/任务超时。非流式请求显示等待完整 JSON，不报告首段文本。
轮次结束显示 `PASS` 或 `FAILED`；单次请求收到完整响应并不代表严格输出长度校验通过。

状态仅写入 **stderr**，不会混入 `--json` 的 stdout，也不会进入 JSON、JSONL 或 Markdown 报告。
不会输出凭据、请求头、URL 查询参数、提示词或生成文本。可单独保存状态或关闭它：

```bash
.venv/bin/python llm_connection_test.py --json 2>progress.log
.venv/bin/python llm_connection_test.py --no-progress
```

请求内的阶段变化交给后台线程输出，不逐个文本块打印，但仍有少量调度和日志开销。
对观察开销敏感的测量，所有对比组统一使用 `--no-progress`。
`--help` 和 `--check-config` 不启动实时状态或在线操作。

### 协议与基础 URL

```toml
[llm]
endpoint = "https://xxxxx.openai.azure.com/openai/v1"
protocol = "both" # chat_completions、responses 或 both
```

上面的基础地址分别补全为 `/openai/v1/chat/completions` 和 `/openai/v1/responses`。
末尾带 `/` 也可以；已有完整地址的已知协议后缀会被替换，不会重复拼接。
代理可使用其真实基础路径，如 `https://proxy.example/v1`；程序不会擅自增加 `/openai/v1`。
URL 查询参数在请求中保留，在报告中省略。配置省略 `protocol` 时仍默认 `chat_completions`。

```bash
# 只测 Responses，包含流式和非流式。
.venv/bin/python llm_connection_test.py --protocol responses --mode both

# 同时测试两种协议、两种响应模式，并逐轮保存。
.venv/bin/python llm_connection_test.py --protocol both --mode both --scenario baseline \
  --output benchmark-results.json --output-jsonl benchmark-protocols.jsonl

# 离线查看补全后的两个端点，不调用接口。
.venv/bin/python llm_connection_test.py --protocol both --check-config
```

协议与流式模式是独立维度。四种组合分别维护连接和预热，偶数轮反转执行顺序；
正式结果、预热、逐次调用和 JSONL 都带 `protocol` 和实际 `endpoint`，汇总也按协议分开。
流式/非流式配对只在同一协议内进行，不会将 Chat 与 Responses 样本混配。
启用两种协议会使请求量和潜在费用翻倍；接口拒绝某种协议时不会自动换协议重试。

Responses 沿用全部文本、长短输出、对话和 Agent 场景。现有 token 预算映射为 `max_output_tokens`，
`reasoning_effort` 映射为 `reasoning.effort`；不发送 Chat 专用的 `stream_options.include_usage`。
Responses 的用量从响应对象/终止事件读取，`api_usage` 保存原始值，归一化字段仍区分可见输出、
API 输出、推理和缓存 token。只有 `response.output_text.delta` 更新可见文本计时，生命周期、
推理和工具参数事件不计入文本块。流式完成需要终止响应事件，不要求 Chat 的 `[DONE]`。

Responses 固定 `store=false`，显式回传完整历史，并请求加密推理项以支持无状态续接。
Agent/对话后续输入会保留已完成的输出项、函数 `call_id` 和加密推理上下文，
不使用 `previous_response_id` 或服务端会话。报告中因此也可能包含加密推理项，应按业务数据保管。
当前范围是文本和确定性本地函数工具，不含托管工具、多模态输入、后台任务或 WebSocket。
目标服务和模型必须支持所选协议及相关参数。

### 在线访问验证

`--check-config` 只能检查配置，不能证明 App Registration 能获取 Token 或访问模型。
手动指定 `auth_type="azure_app_registration"` 后可先运行：

```bash
# 单独鉴权时最多两次；auto 依次检查三种方式，最多六次。
.venv/bin/python llm_connection_test.py --validate --protocol both

# 只探测 Chat，并保存详细诊断；JSONL 文件名必须是新的。
.venv/bin/python llm_connection_test.py --validate --protocol chat_completions --json \
  --output validation.json --output-jsonl validation.jsonl
```

验证仅使用**配置里的第一个模型**，固定非流式和简短的 `Reply with OK.` 提示词，
生成预算最多 64 token；如果原预算更小则沿用更小值。不会执行预热、正式循环、Agent 或分词器加载，
也不验证其他模型、流式、输出长度或性能；`--mode`、`--scenario` 和 `continue_on_error` 不会扩大探测。
这是**真实在线请求**，可能产生身份端点网络流量和模型调用费用。

单独 App Registration 模式只用该应用，不会切换成 MI、个人 Azure CLI 登录或 Key；
auto 验证按 **Managed Identity → App Registration → LLM Key** 依次独立检查全部三种方式，
前一项成功也继续检查下一项；两种协议下最多发起 6 次小预算模型请求。
缺少凭据时该方式标记失败、不发送模型请求；取 Token 失败时跳过该方式的剩余协议，继续下一种方式。
正式基准的 auto 仍在取得第一种可用凭据后停止选取，不因模型 HTTP 401/403 而切换身份。
输出区分 `authentication`（凭据）、`network`（网络）、`endpoint`（HTTP 拒绝）和 `response`（响应解析）阶段。

HTTP 错误只在请求时限内读取最多 16 KiB 的错误体，提取脱敏后的 `code`、`message`，
以及白名单 request ID，不输出原始响应体或鉴权头。非 JSON、超大或读取失败的错误体可能只显示状态和请求 ID。
若显示 `Credential acquisition: PASS`，随后 HTTP 401/403，表示应用已取得 Token，但资源拒绝访问。
优先检查该应用的 **Service Principal** 在实际目标资源上的数据面角色，例如 `Cognitive Services OpenAI User`，
再核对资源租户、Token audience/scope 和端点；给个人账号授权不等于给应用授权。
程序不会修改角色或 scope，也不会因模型 HTTP 拒绝而换身份重发。

`--json` / `--output` 输出 `type="access_validation"` 和 `probes`，不是正式基准的 `runs` / 汇总；
JSONL 每行保存一条探测。auto 的完整 JSON 额外包含按顺序排列的 `auth_results`、`usable_auth_types`，
终端最后也会显示类似：

```text
Authentication checks (configured order):
  Managed Identity - Failed
  App Registration - Success
  LLM API Key - Failed
```

`Success` 表示真正通过模型端点访问，不只是取 Token 成功。
auto 至少有一种鉴权通过**全部所选协议**时退出码为 `0`，其他方式的失败仍保留；
单独鉴权则要求该方式的全部协议通过。整体失败为 `1`，Ctrl+C 为 `130`。
小预算导致的正常终止截断可以显示“访问通过但回答未正常完成”的警告，不会被当成性能结论。
`--validate` 与 `--check-config` 不能同时使用；已有输出路径预检仍在在线操作前执行。

## 3. 场景用法

| 场景名 | 测试内容 | 验收或特点 |
| --- | --- | --- |
| `baseline` | 使用配置中的系统提示词、用户提示词和生成预算 | 基础连接、延迟和吞吐测试 |
| `prefill_short` | 使用较短的固定输入上下文 | 默认 128 个英文词，不是 128 token |
| `prefill_long` | 使用较长的固定输入上下文 | 默认 4096 个英文词，用于观察长输入对首段等待的影响 |
| `long_output` | 要求生成长篇英文技术文章 | 实际可见输出必须严格大于 1000 token |
| `short_burst` | 无人工间隔地连续发送独立短请求 | 默认每批 20 次，每次实际可见输出严格小于 200 token |
| `short_conversation` | 连续短请求，并在当前批次中保留对话历史 | 每次输出小于 200 token，输入上下文逐步增长 |
| `agent` | 模型调用固定本地查询工具，再生成最终回答 | 正常完成通常需要 2 次模型请求 |
| `agent_serial` | 主 Agent 委派两个 Sub-Agent，串行执行后汇总 | 正常完成通常需要 6 次模型请求 |
| `agent_parallel` | 与串行场景相同的两个子任务，并行执行后汇总 | 正常完成通常需要 6 次模型请求 |

除 `baseline` 外，场景会使用内置的专用用户提示词；系统提示词仍沿用配置。
长输出、短输出和 Agent 场景各有独立生成预算，不直接沿用基线的 token 上限。

### 流式与非流式对比

```bash
.venv/bin/python llm_connection_test.py --mode both --scenario baseline
```

也可以单独执行一种模式：

```bash
.venv/bin/python llm_connection_test.py --mode streaming --scenario baseline
.venv/bin/python llm_connection_test.py --mode non_streaming --scenario baseline
```

### 长输出与连续短输出

```bash
.venv/bin/python llm_connection_test.py \
  --mode both --scenario long_output short_burst \
  --json --output benchmark-results.json
```

增加 `max_tokens` 或 `max_completion_tokens` 只会提高生成预算，不能保证模型一定输出足够长。
长输出不足、短输出超限或响应被截断时，场景不会通过，已取得的测量结果仍会保留。

独立短请求不累计历史；要观察连续对话时上下文增长的影响，单独选择：

```bash
.venv/bin/python llm_connection_test.py --mode both --scenario short_conversation
```

### 长短输入与 Prefill 代理值

```bash
.venv/bin/python llm_connection_test.py \
  --mode both --scenario prefill_short prefill_long
```

建议结合 `prefill_proxy_ms`、输入 token 数和缓存命中 token 数观察。
该测试不能从客户端分离出纯服务端 prefill 时间，也不能自动清空服务端缓存。

### Agent 与 Sub-Agent

```bash
.venv/bin/python llm_connection_test.py \
  --mode both --scenario agent agent_serial agent_parallel \
  --json --output benchmark-results.json
```

这些场景使用真实的工具调用协议，不是在提示词中假装存在子 Agent。
本地工具返回固定数据；委派场景的两个主题固定为延迟和吞吐，便于串行、并行对照。
主 Agent 和子 Agent 都使用当前正在测试的模型。
接口必须支持工具调用及 `tool_choice`；拒绝或忽略工具请求会明确报错，不会自动降级为普通聊天。

## 4. 配置说明

完整示例见 [config.example.CN.toml](config.example.CN.toml)。以下片段应修改或合并到对应表中，
不要在同一个 TOML 文件里重复声明同名表。

### 接口与请求参数

| 配置项 | 说明 |
| --- | --- |
| `llm.endpoint` | API 基础 URL 或完整 Chat Completions / Responses 地址 |
| `llm.protocol` | `chat_completions`（默认）、`responses` 或 `both` |
| `llm.models` | 非空模型名称数组，按列表顺序测试 |
| `llm.auth_type` | 示例默认 `auto`，按 MI → App Registration → Key 回退；也支持三个单独模式 |
| `llm.api_key_env` | API Key 单独模式或 auto 最后回退使用的密钥环境变量名，示例为 `LLM_API_KEY` |
| `llm.auth_header` | 仅 API Key 模式使用的鉴权请求头；空字符串表示不添加此鉴权头 |
| `llm.auth_scheme` | 仅 API Key 模式使用的鉴权前缀，默认示例为 `Bearer` |
| `llm.timeout_seconds` | 单次 HTTP 请求时限，示例值为 120 秒 |
| `request.system_prompt` | 所有场景沿用的系统提示词 |
| `request.user_prompt` | 基线场景使用的用户提示词 |
| `request.max_tokens` | 传统 Chat 模型的生成预算，与 `max_completion_tokens` 二选一；Responses 映射为 `max_output_tokens` |
| `request.max_completion_tokens` | 可能包括隐藏推理的生成预算；Responses 映射为 `max_output_tokens` |
| `request.temperature` | 可选采样参数，仅在接口和模型支持时设置 |
| `request.reasoning_effort` | 可选推理强度，仅在模型支持时设置 |
| `request.include_usage` | Chat 流式是否发送 `stream_options.include_usage`；Responses 从响应对象读取用量，不发送此选项 |
| `headers` | 自定义请求头，值支持 `${ENVIRONMENT_VARIABLE}` 引用 |

示例基线使用 `max_completion_tokens = 2048`。两种 token 上限都未配置时，
兼容旧模型的 `max_tokens` 默认值也为 2048；显式设置的上限不会被自动提高。
推理和可见回答共享预算，256 可能在模型开始回答前就耗尽。
HTTP 200 但 `finish_reason=length` 仍判为失败，分类为 `budget_exhausted`，不是网络或 HTTP 错误。

2048 只是试验起点，不保证成功，提高上限可能增加费用。先仅调整预算，保持提示词、
推理强度、轮数和连接策略不变；确认所有部署支持后，再单独测试 `reasoning_effort = "low"`。
通过提示词要求简短回答，不要用过小预算强行截断；严格长短场景仍校验实际可见输出。
不要为通过校验随意指定 tokenizer。先稳定负载，再增加样本量；少量样本的 P95 仍仅供描述。
程序不会自动重试、自动提高预算或额外探测真实模型。

不要同时设置 `llm.api_key` 和 `llm.api_key_env`；推荐只使用环境变量配置。

对于 Azure 风格的 `api-key` 请求头，可采用下面的配置片段，并在环境中设置
`AZURE_OPENAI_API_KEY`。同时删除原有的 `llm.api_key_env` 和 `llm.api_key`，
避免仍被要求读取默认密钥。

```toml
[llm]
# 在这里保留实际的 endpoint、models 和其他必要设置。
auth_type = "api_key"
auth_header = ""

[headers]
"api-key" = "${AZURE_OPENAI_API_KEY}"
```

连接免鉴权的本地服务时，也应删除 `llm.api_key_env`、`llm.api_key` 以及不需要的鉴权头，
并保留 `llm.auth_type = "api_key"`，将 `llm.auth_header` 设置为 `""`。

### Azure 身份鉴权

支持通过 Azure 官方 `azure-identity` 库获取 Microsoft Entra ID Access Token。
已有虚拟环境也应重新执行安装命令，以安装新增依赖：

```bash
.venv/bin/python -m pip install -r requirements.lock
```

Azure 单独模式及 auto 必须使用 HTTPS。Azure Token 自动添加 `Authorization: Bearer <token>`，
单独 Azure 模式
不使用 `llm.auth_header`、`llm.auth_scheme` 或 `llm.api_key_env`。
因此现有示例中的 `api_key_env = "LLM_API_KEY"` 可以保留，但应删除明文 `llm.api_key`，
以及 `[headers]` 中的 `Authorization`、`api-key`，避免混用鉴权。

#### 默认自动验证顺序

```toml
[llm]
auth_type = "auto"
api_key_env = "LLM_API_KEY"
auth_header = "Authorization"
auth_scheme = "Bearer"

[azure_auth]
scope = "https://cognitiveservices.azure.com/.default"
managed_identity_client_id = ""
tenant_id = ""
client_id = ""
client_secret = ""
```

1. **Managed Identity**：先尝试；`managed_identity_client_id` 留空代表系统分配身份，用户分配身份填该身份的 Client ID。
2. **App Registration**：上一项取 Token 失败后尝试；填写 `tenant_id`、`client_id`、`client_secret`，缺少任一凭据时跳过；也兼容 `client_secret_env`。
3. **LLM Key**：前两项均不可用时，使用 `api_key_env` 指定环境变量中的 Key；三项均失败则不发送模型请求。

MI 的 `managed_identity_client_id` 与 App Registration 的 `client_id` 独立，不能混用。
ID 支持 `${ENVIRONMENT_VARIABLE}`；`client_secret` 是 Secret 原值，不读取或展开环境变量。
选择环境变量方式时，改用 `client_secret_env` 填变量名，两种非空来源不能同时配置。
显式指定用户分配身份的环境变量但变量未设置时，会跳过 MI，不会改用系统身份。
为 Azure 风格的 Key 配置 auto 时，设置 `llm.auth_header="api-key"`、`llm.auth_scheme=""`，
不要在 `[headers]` 中设置 `Authorization` 或 `api-key`。

成功选择后会在模型、响应模式和子 Agent 之间复用，不会每个请求都重新探测失败的方式。
若 Token 刷新后来失败，就继续往后回退；只有重新执行测试才会从第一项重新尝试。
**回退只发生在获取凭据阶段**：模型接口返回 401/403 或其他 HTTP 错误不会换身份重发，
以免改变基准请求次数和测量口径。取到 Token 不代表该身份一定有模型资源权限。
逐次结果的 `authentication.type` 是实际使用的方式，`authentication.attempts` 保留脱敏后的尝试历史，
JSON 和 JSONL 均可查看。自动验证包含网络等待，会使用 SDK 配置的连接/读取超时，不一定瞬间失败。

需要禁用回退时，仍可将 `auth_type` 改为 `azure_managed_identity`、`azure_app_registration` 或 `api_key`。
旧配置未写 `auth_type` 时仍按 `api_key` 处理，保持兼容。

#### App Registration：应用注册 / Service Principal

使用应用的 Tenant ID、Application Client ID 和 Client Secret，
通过 `ClientSecretCredential` 执行应用身份认证，不弹出浏览器。
下面的 Bash 命令在终端静默读取 Secret，避免将其写进命令历史：

```bash
export AZURE_TENANT_ID='<tenant-id>'
export AZURE_CLIENT_ID='<application-client-id>'
read -rsp 'Azure client secret: ' AZURE_CLIENT_SECRET
export AZURE_CLIENT_SECRET
printf '\n'
```

Client Secret 应填写创建密钥时得到的**值**，不是 Secret ID；不要将真实 Secret 发到聊天或提交到版本库。

将下面片段合并进现有配置，保留其他测试和请求设置：

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

上述 Azure 环境变量名是字段省略时的默认值；当前示例显式留了空项，
需将空项填写为上述值/环境引用，再选择 App Registration 单独模式。
`tenant_id` 和 `client_id` 也可以填写实际 ID。临时测试不使用环境变量时，将上面的
`client_secret_env` 替换为：

```toml
client_secret = "<临时测试Secret的值>"
```

`client_secret` 原样传给 SDK，不查询环境变量，也不展开其中的 `${...}`。
它不会出现在配置对象的 `repr` 中，错误及 SDK 日志仍脱敏；但文件里是明文，应避免提交或共享。
不要同时配置非空的 `client_secret` 和 `client_secret_env`。
此模式当前不提供证书或工作负载联合身份配置。

#### Managed Identity：托管身份

适用于已启用 Managed Identity 的 Azure 主机或服务，运行环境必须能访问其托管身份端点。
普通本机执行 `az login` 不会获得云主机的 Managed Identity；本机测试可改用上述应用注册模式。
Managed Identity 不需要 Client Secret 或 Tenant ID。

```toml
[llm]
endpoint = "https://YOUR-RESOURCE.openai.azure.com/openai/v1/chat/completions"
models = ["YOUR-DEPLOYMENT-NAME"]
auth_type = "azure_managed_identity"

[azure_auth]
scope = "https://cognitiveservices.azure.com/.default"
# 系统分配身份：managed_identity_client_id 和旧版 client_id 均留空。
# 用户分配身份：填写已绑定到运行环境的身份 Client ID，不是 Object/Principal ID。
# managed_identity_client_id = "USER-ASSIGNED-IDENTITY-CLIENT-ID"
```

用户分配身份的 `managed_identity_client_id` 也支持 `${ENVIRONMENT_VARIABLE}`。
单独 MI 模式兼容旧版 `client_id`，auto 模式只用独立的 `managed_identity_client_id`。
从 App Registration 切换为系统分配身份时，应删除原有的 `tenant_id`、`client_id` 配置，
避免残留环境变量引用或错误地选择其他身份。

#### Scope 与资源授权

默认 `https://cognitiveservices.azure.com/.default` 适用于公有云 Azure OpenAI 的资源 audience。
示例 v1 地址中的 `models` 应填写实际部署名称。
身份必须在目标资源上拥有相应数据面权限，例如 `Cognitive Services OpenAI User`；
仅创建 App Registration 或启用 Managed Identity 并不等于已经有调用权限。

如果连接的是使用自定义 Entra 应用保护的代理，而不是直接访问 Azure OpenAI，
应按代理配置填写 scope，例如 `api://TARGET-API-APP-ID/.default`。
同时配置目标应用要求的应用权限或 App Role，并按租户策略完成管理员同意。
代理需要接受该 Token 的 audience、issuer 和权限声明，不能假定 Azure OpenAI 的 Token 能用于任意代理。

`401` 时应检查 audience、issuer、Token 和实际接口地址；`403` 时还应检查角色分配、资源策略等授权设置。
Token 获取成功并不能单独证明模型调用已经获得授权。

#### Token 缓存与计时

SDK 会缓存并在需要时刷新 Token；同次运行中的流式、非流式、不同模型和子 Agent
共享线程安全的鉴权选择。auto 按指定三项顺序回退，三个单独模式不回退，
任何模式均不会回退到 Azure CLI 或交互式登录。
Token 和 Client Secret 不写入报告，鉴权失败时不直接输出 SDK 异常中的凭据详情。
同步凭据操作期间，Azure/MSAL 日志中的原始消息和异常堆栈会在进入日志处理器前替换；
其他线程或操作之外的日志不受影响。脱敏错误保留安全分类：`credential_unavailable`（身份不可用）、
`client_authentication_error`（认证失败）、`identity_network_error`（身份网络故障）或
`identity_error`（其他错误），以及可取得的 HTTP 状态码。
先对已知凭据脱敏，再保留最多三个去重且格式受限的 `AADSTS` 错误码，
并为常见错误提供固定建议；不会放开 SDK 原始异常或日志：

| 错误码 | 排查方向 |
| --- | --- |
| `AADSTS7000215` | Secret 无效；填写 Secret 的值而非 Secret ID，并确认属于该应用。 |
| `AADSTS7000222` | Secret 已过期；创建新 Secret 并更新本地凭据。 |
| `AADSTS700016` / `AADSTS90002` | 核对应用 Client ID 和 Tenant ID，检查应用与租户是否匹配。 |
| `AADSTS500011` / `AADSTS70011` | 核对目标资源 scope 和租户。 |

`Credential acquisition: FAIL` 表示尚未验证模型访问，应先解决取 Token 失败，
再排查模型接口 HTTP 401/403 或资源角色。没有可提取的安全错误码时，仍显示通用脱敏分类。

`authentication_ms` 单独记录 Token 获取、缓存检查、刷新和锁等待耗时，
auto 模式也包含此前失败方法的尝试时间；
不计入单次模型请求的 TTFT、TTLT、端到端耗时和 TPS；DNS/TCP/TLS 仍只描述模型接口连接。
批次或 Agent 的整任务耗时会包含鉴权，顶层 `authentication_ms` 是各次调用的合计，
并行时可能累计多个线程的锁等待，不能将它直接当作可从任务墙钟时间扣除的值。
未预热时首次任务可能包含冷启动取 Token 开销，对比整任务耗时应同时查看此字段。

SDK 身份网络操作使用配置中的连接和读取超时；任务时限会在鉴权前后检查，
但不会强制取消正在执行的 SDK 网络操作。
两种 Azure 模式的 `--check-config` 仍完全离线，即使未设置 Azure 环境变量也可以检查配置。
非法主机格式和端口也会在配置阶段拒绝，不会等到发送请求才报错。
自动化测试使用模拟凭据，不连接真实 Entra、托管身份或模型端点。

### 测试组织与连接策略

```toml
[benchmark]
rounds = 3
continue_on_error = true
mode = "both"
scenarios = ["baseline"]
connection_reuse = true
warmup_rounds = 1
task_timeout_seconds = 600
# 只有确定底层模型使用该分词器时才显式指定。
# tokenizer_model = "gpt-4o"
```

- `rounds` 是每个“模型 × 场景 × 模式”的正式测试轮数，不包含预热。
- `warmup_rounds` 可设为 `0`；每一轮预热都会执行完整场景，同样消耗接口额度。
- `continue_on_error = true` 会保留失败并继续其他测试；`false` 遇到失败即停止。
- `connection_reuse = true` 允许顺序请求复用连接；设为 `false` 可观察每次重新建连的影响。
- 两种模式分别维护连接会话；服务端关闭连接后，下次请求会重新建连并记录连接开销。
- `task_timeout_seconds` 限制一轮场景的任务时间；批量短请求和 Agent 的整轮任务共用此时限。
- 不会自动重试模型请求，也不会在连续短请求之间人为等待。

DNS、多个 TCP 地址尝试、TLS 和模型 HTTP 请求共用剩余的请求/任务预算。
`tcp_connect_ms` 包含失败地址尝试的耗时，`tcp_connect_attempts` 记录尝试次数。
预算在阶段切换和读取边界检查，不是进程级强制取消；系统同步 DNS 查询无法被强制中断，
但查询返回后若已超时，就不会继续尝试 TCP 连接。

### 场景与 Agent 预算

```toml
[scenarios]
short_requests = 20
short_token_budget = 1024
long_token_budget = 8192
prefill_short_words = 128
prefill_long_words = 4096

[agents]
max_steps = 4
max_calls = 12
token_budget = 1024
total_token_budget = 12288
```

`short_requests` 控制每批短输出请求数量；两个 `*_token_budget` 分别控制短输出和长输出的请求预算。
`prefill_*_words` 控制输入上下文的英文词数，实际输入 token 数以接口 usage 为准。

Agent 的 `max_steps` 对每个 Agent 分别生效，`max_calls` 则限制整个任务的模型调用总次数。
`token_budget` 是每次请求的 completion 预算，`total_token_budget` 是整轮任务共享的预算。
每次请求按其最大预算保守预留，包括可能的隐藏推理；即使实际用量较少，也不会退回预留额度。

### 控制测试费用

完整运行时，请求数量约为“模型数 × 协议数 × 模式数 ×（预热轮数 + 正式轮数）× 每轮调用数”。
基线、长输出和每个 prefill 场景每轮调用 1 次，短请求场景每轮调用 `short_requests` 次，
普通 Agent 通常调用 2 次，委派场景通常调用 6 次。失败或预算限制可能提前结束任务。

初次验证建议只配置一个模型，设置 `rounds = 1`、`warmup_rounds = 0`，从 `baseline` 开始。
确认兼容性后再增加场景、请求数量和样本轮数。

## 5. 指标怎么读

| 指标字段 | 含义 | 非流式是否可测 |
| --- | --- | --- |
| `dns_lookup_ms` / `tcp_connect_ms` / `tls_handshake_ms` | DNS、TCP、TLS 分段耗时 | 可以；复用连接时没有新的分段测量 |
| `connection_setup_ms` | 本次连接建立总耗时 | 可以；复用连接时为 0 |
| `response_headers_ms` | 请求发送完成到收到响应头的时间 | 可以，但不能代替 TTFT |
| `ttft_ms` | 请求开始到首个可见文本块到达 | 不可以，返回 `null` |
| `first_token_after_request_ms` | 请求发送完成到首个可见文本块到达 | 不可以，返回 `null` |
| `ttlt_ms` | 请求开始到最后可见文本块到达 | 不可以，返回 `null` |
| `prefill_proxy_ms` | 与 `first_token_after_request_ms` 相同的客户端等待区间 | 不可以，返回 `null` |
| `tps` | 扣除首块 token 后的可见输出量，除以首末文本块到达间隔 | 不可以，返回 `null` |
| `end_to_end_ms` | 请求开始到完整 HTTP 响应接收结束 | 可以直接对比 |
| `end_to_end_output_tokens_per_second` | 可见输出 token 数除以端到端秒数 | 可以按相同口径对比 |
| `authentication_ms` | 模型计时之外的 Token 提供器耗时；API Key 模式为 0 | 可以 |

**Prefill 代理值不是纯 prefill 耗时。** 它还包含网络、排队、调度以及可能的隐藏推理。
服务端返回的 `timing`、`timings` 或 `Server-Timing` 会连同来源保存在 `server_timing`，
但不会擅自解释为统一的服务端 prefill 指标。

流式文本以块到达，客户端无法逐个观测 token 生成时刻，因此 TTFT、TTLT 和 TPS 都受分块方式影响。
单块输出没有可测的解码 TPS。末尾 usage 事件和等待流关闭的时间不计入首末文本块的生成间隔。
非流式只返回完整响应，不能把整包接收耗时冒充 TTFT 或 TTLT。

可见文本 token 使用 `tiktoken` 计数，并标明来源；API completion token、隐藏推理 token 和工具参数
不直接作为可见输出量。无法识别分词器时会使用估算，并将严格长度验收标记为 `unverified`。
显式设置 `tokenizer_model` 时必须确认底层分词器；本地计数不等同于服务商认证的 token 用量。

分词器首次使用时可能下载公开编码数据；初始化不计入单次请求耗时。可按实际模型提前缓存：

```bash
.venv/bin/python -c 'import tiktoken; tiktoken.encoding_for_model("gpt-4o")'
```

为兼容旧字段，`time_to_first_token_ms` 对应 `ttft_ms`，`output_tokens_per_second` 对应 `tps`。
当前结果的 `schema_version` 为 `2`：`output_tokens` 表示可见文本 token，
API completion 总量单独存入 `api_completion_tokens`；`generation_duration_ms` 截止到最后文本块。

## 6. 报告与对比结果

终端文本报告适合快速查看通过情况、延迟和吞吐。需要分析完整调用、失败原因或导入其他工具时，
使用 `--json` 或 `--output`。长任务建议同时启用 `--output-jsonl` 保留逐轮记录。

### Markdown 对比报告

```bash
# 基准结果、对比表和指标说明。
.venv/bin/python llm_connection_test.py --protocol both --mode both \
  --output-markdown benchmark-results.md

# 可同时保留完整 JSON 和增量 JSONL。
.venv/bin/python llm_connection_test.py --protocol both --mode both \
  --output benchmark-results.json --output-jsonl benchmark-results.jsonl \
  --output-markdown benchmark-results.md

# 访问验证也可生成表格；auto 时包含三种鉴权的检查汇总。
.venv/bin/python llm_connection_test.py --validate --protocol both \
  --output-markdown validation.md
```

生成后在 VS Code 中打开 Markdown 预览即可渲染表格。该选项不改变终端文本/JSON 输出格式，
导出本身也不会额外调用模型，但上面的正常运行和在线验证仍有原本的调用费用。

报告包含：协议/模式延迟和吞吐汇总、成功配对差异、正式轮次与预热、Agent 逐次调用和任务计时、
可见/API/推理/缓存 token、计数来源及完整性、失败分类与失败调用用量、长度验证原因、
带来源的服务端原始计时，以及指标说明。
访问验证报告则包含鉴权结果、逐项探测、脱敏错误和 request ID。
Markdown 面向测量和诊断对比，不嵌入完整生成回答或加密推理历史；需要原始完整结果时同时保存 JSON。

汇总单元格为“均值 / P50 / P95 (n=有效测量样本数)”。`n/a` 表示不可观测或缺失，不是零。
每个组合同时展示 **All completed**（含已完成但验证失败的样本）和 **Successful only**（仅成功样本）。
Measurement Warnings 会标明启发式 token 估算；其吞吐率不能与已知分词器的结果严格比较。
预热不混入正式汇总；少于 20 个样本的 P95 仅供描述。任务累计 token 包含子调用，
不能将任务行与子调用行重复相加。不完整计数只代表已观测部分，不能当作完整计费用量。
Prefill proxy 仍是客户端代理指标，不是实测服务端 prefill。

Markdown 在在线操作前试写预检，整份报告完成后原子替换，基准失败、退出码非零时也会保存失败报告。
Ctrl+C 或该报告写盘失败时保留旧 Markdown。多个最终输出文件分别替换，不构成跨文件事务。
Markdown 不是逐轮保存；需要保留长任务中断前的数据时，请同时启用 JSONL。

### 增量保存与中断

```bash
.venv/bin/python llm_connection_test.py --mode both --scenario baseline \
  --output benchmark-results.json --output-jsonl benchmark-results.jsonl
```

JSONL 每行是独立 JSON 对象，结构为 `schema_version`、已去除查询字符串的 `endpoint` 和 `record`。
`record` 与完整报告中的 `runs` / `warmup_runs` 记录一致，用 `warmup` 区分是否预热；
记录按实际执行顺序写入，包括失败轮次，文件没有额外的汇总或结束行。
可以只使用 `--output-jsonl`，不必同时生成最终 JSON 文件。

每完成一个“模型 × 场景 × 协议 × 模式 × 轮次”就写入一行并执行 `flush` / `fsync`，之后才开始下一轮；
写盘开销在该轮测量结束后发生，不计入该轮指标。Agent 或连续短请求的**整个批次算一轮**，
其中的 `calls` 要等这一轮结束才一起保存，不是每次模型调用或每个文本块就写入。

- 按 Ctrl+C 后返回退出码 `130`，已完成的 JSONL 记录保留；当前未完成轮次不会保存。
- 最终 JSON 先写临时文件，再原子替换；中断或保存失败时，已有的最终报告不会被半份新内容覆盖。
- 写入或同步失败会立即停止后续模型请求，即使 `continue_on_error=true` 也不继续消耗额度。
- 预检不能预留磁盘容量或保证之后的存储状态；磁盘错误或强制终止可能留下不完整的尾行，读取时应忽略该尾行。
- 重跑时为 JSONL 使用新文件名，例如 `benchmark-results-2.jsonl`，否则会在请求前报错。

### 完整报告字段

| JSON 字段 | 查看内容 |
| --- | --- |
| `success` | 全部正式测试及预热是否通过 |
| `runs` | 每轮正式测试，包括指标、token 数、输出内容和验证结果 |
| `warmup_runs` | 预热记录，不参与正式样本分位数统计 |
| `summaries` | 按模型、场景、协议和模式分组的均值、最小值、最大值、P50、P95、样本数与成功率 |
| `summaries[].successful_metrics` | 与原 `metrics` 并存的仅成功、未中断样本统计，原口径不变 |
| `summaries[].token_count_sources` / `token_count_note` | 可见 token 计数来源及启发式估算警告 |
| `failure_summary` | 正式/预热分别统计失败分类与失败调用用量；各分组也有独立汇总 |
| `settings.max_tokens` / `max_completion_tokens` / `reasoning_effort` | 基线预算及推理强度配置；推理强度为 `null` 表示采用模型默认值 |
| `endpoints` | 各协议自动补全后的端点，不含查询字符串 |
| `comparisons` | `both` 模式下，成功轮次的配对差异 |
| `metric_notes` | 指标口径及兼容性说明 |

`both` 模式会逐轮交替两种模式的执行顺序，保持提示词和预算一致。
配对差值为“非流式值减流式值”，百分比以流式值为基准：延迟差为负表示非流式在该样本中更快，
输出速率差为正表示非流式的端到端输出速率更高。两种响应的生成长度可能不同，报告也会展示长度差；
不能仅凭总耗时差认定某种模式的解码速度更快。

完成请求但长度不合格的样本会保留测量值并参与测量汇总，但不会参与成功配对。
断流样本及包含断流的任务不参与指标汇总或成功配对，已观测的数据仍保留在逐次记录中。
没有观测值的失败请求不会虚构指标。少于 20 个样本时，P95 仅供描述，不宜作为稳定性能结论。
重复输入可能命中服务端缓存，应同时查看 `cached_input_tokens`，不能假定每次都是冷缓存。

失败记录的 `failure_category` 区分 `budget_exhausted`（生成预算耗尽）、`empty_output`（空回答）、
`length_violation`（可见长度不合格）、`length_unverified`（分词器未知，无法严格验证）、
`abnormal_finish`（其他非正常结束）、`authentication_error`（取凭据失败）、`http_error`、
`transport_error`、`response_error`（响应格式错误）、`interrupted_response`（响应读取/解析未完成）
和 `task_error`（任务执行失败）。任务采用首个失败调用的分类；仅本地任务失败时为 `task_error`。
其他原因仍保留在逐次调用中；缺少分类的旧记录显示 `unclassified`，不通过错误文本猜测。

`failure_summary.categories` 按失败/未验证轮次计数，`failed_calls` 和 `tokens` 只统计失败的独立
模型调用，不重复加上任务累计值，也不将失败任务中已成功的中间调用算作失败调用用量。
各 token 字段包含 `observed_total`（已有计数之和）、`reported_calls`（有该计数的调用数）
和 `complete`（缺少计数或响应中断时为 `false`）。完全缺少计数时为 `null`，不能当成零；
没有失败调用时才可直接记零。这些值不是金额，也不代表完整账单用量。

即使已收到的内容看起来完整，缺少 `Content-Length` 声明的字节、损坏的 chunked 边界或缺失 SSE
结束标记都会被判定失败。流式失败保留已完整解析的文本事件、可见 token、TTFT、已收到的 API
用量/工具片段和服务端计时，无法解码的片段不算可见文本。
此类记录标记 `interrupted=true`、`response_complete=false`、`tokens.visible_output_tokens_complete=false`；
TTLT、解码 TPS、生成时长、单次请求端到端耗时及输出速率为 `null`。
`failure_elapsed_ms` 表示到失败为止的耗时，`stream.last_observed_text_ms` 表示请求开始到最后一次
观测到文本的时间，不能当成完整响应的 TTLT。

Agent 和短请求批次的顶层耗时、输出速率描述**整个任务**；顶层 TTFT、TTLT、解码 TPS 为 `null`。
逐次模型调用保存在 `calls` 中，调用统计见 `call_metrics`；
新增 `successful_call_metrics` 仅统计其中成功且未中断的调用，原统计口径不变。
累计可见 token 包括子 Agent 和中间回答，不只是主 Agent 最终回答。
Agent 最终回答见 `final_answer`，工具记录见 `tools`；本地工具时间与委派时间分别记录，
委派时间已经包含子模型调用，不能再重复加到任务耗时上。

失败调用即使没有 token 数据也不会中断汇总。可见 token 累计值是已观测部分；
`visible_token_totals_complete=false` 时不能当成完整输出量。
某次调用缺失 API 用量时，对应累计 API 字段为 `null`；断流也会使 `token_totals_complete=false`。
这些部分数据不能当作完整计费用量。失败任务中已经完成的调用仍保留在 `calls` 和 `call_metrics` 中。

输出文件含模型生成内容，应按业务数据要求保管，不要默认将报告提交到版本库。

## 7. 常见问题与本地验证

| 现象 | 处理方式 |
| --- | --- |
| 提示环境变量未设置 | 在当前运行终端设置 `llm.api_key_env` 指定的变量；`export` 只影响当前 shell 及其子进程 |
| HTTP 400 或参数不兼容 | 检查端点、模型、`max_tokens` / `max_completion_tokens`、采样与推理参数；工具场景还需支持工具协议 |
| 流式不支持 usage 选项 | 将 `request.include_usage` 设为 `false`；接口用量缺失时对应 API 计数字段为 `null` |
| 长输出不足或没有可见文本 | 检查 `finish_reason`；增加长输出预算或调整受支持的推理强度，并确认请求及任务时限足够 |
| 长短输出显示 `unverified` | 只有确定模型底层分词器时才设置 `benchmark.tokenizer_model`，不要为通过验收随意指定 |
| 非流式 TTFT、TTLT、TPS 为 `null` | 正常限制；改看端到端耗时及端到端输出速率 |
| 流式响应被标记为不完整 | Chat 检查 `finish_reason` 和 `[DONE]`；Responses 检查终止响应事件、最终文本与增量是否一致 |
| 配置检查通过但实际请求失败 | 配置检查不会验证鉴权、网络、模型能力或服务端参数兼容性 |
| Azure 身份鉴权失败 | auto 查看 `authentication.attempts` 中的回退原因；单独模式不回退，检查环境变量、scope 和身份端点 |
| Azure 返回 401 / 403 | 检查 Token audience、issuer、应用权限、目标资源角色与访问策略 |

无需密钥、不会调用真实模型的本地验证：

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python llm_connection_test.py --help
.venv/bin/python llm_connection_test.py --check-config
.venv/bin/python -m pip check
```

测试使用本地模拟 HTTP 服务、可控时间戳和测试分词器。
实际测试运行的退出码仅在全部正式轮次和预热均通过时为 `0`；失败或长度未验证时为非零。