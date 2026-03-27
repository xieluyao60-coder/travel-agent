# 旅游规划助手 MVP（企业微信 + QQ）

基于 `FastAPI + Redis + 第三方API` 的旅游规划助手，提供以下能力：

- 实时天气：和风天气
- 两地通勤：高德地图路径规划（公交/驾车/步行）
- 聊天入口：企业微信（URL 回调 / 长连接）+ QQ（官方机器人 WebSocket）
- 联网搜索：SerpAPI 聚合搜索
- 兜底对话：OpenAI 兼容 LLM

## 1. 项目结构

```text
app/
  main.py                 # FastAPI 入口
  container.py            # 依赖装配
  config.py               # 配置管理
  schemas.py              # 业务模型
  errors.py               # 异常定义
  providers/              # 第三方 API 适配层（天气/路线/搜索/QQ）
  services/               # 解析Agent、回复Agent、编排、会话、企业微信/QQ 协议
tests/                    # 单元与集成测试
Dockerfile
docker-compose.yml
```

## 2. 环境变量

复制示例配置：

```powershell
Copy-Item .env.example .env
```

核心变量：

- 企业微信
- `WECOM_CONNECTION_MODE`：`webhook` 或 `long_connection`
- `WECOM_TOKEN` `WECOM_ENCODING_AES_KEY` `WECOM_CORP_ID`
- `WECOM_BOT_ID` `WECOM_BOT_SECRET`
- `WECOM_WS_URL` `WECOM_WS_MAX_AUTH_FAILURE_ATTEMPTS`
- QQ
- `QQ_ENABLED`：`true/false`
- `QQ_BOT_APP_ID` `QQ_BOT_CLIENT_SECRET`
- `QQ_API_BASE_URL`（默认 `https://api.sgroup.qq.com`）
- `QQ_AUTH_BASE_URL`（默认 `https://bots.qq.com`）
- `QQ_WS_INTENTS`（默认 `1107296256`）
- `QQ_WS_MAX_AUTH_FAILURE_ATTEMPTS`
- `QQ_WS_MAX_MISSED_HEARTBEAT`
- `QQ_WS_RECONNECT_BASE_DELAY_SECONDS` `QQ_WS_RECONNECT_MAX_DELAY_SECONDS`
- `QQ_EVENT_DEDUP_TTL_SECONDS` `QQ_EVENT_DEDUP_MAX_SIZE`
- 业务能力
- `QWEATHER_API_KEY` `QWEATHER_API_HOST`
- `AMAP_API_KEY`
- `SERPAPI_API_KEY`
- `LLM_BASE_URL` `LLM_API_KEY` `LLM_MODEL`

## 3. 本地运行（Docker）

```powershell
docker compose up --build
```

启动后：

- 健康检查：`GET http://localhost/healthz`
- 企业微信 webhook 模式：`GET/POST http://localhost/webhook/wecom`
- 企业微信长连接模式：服务启动后自动建立长连接
- QQ 模式：`QQ_ENABLED=true` 时自动建立 QQ 官方网关长连接

## 4. 企业微信接入模式

- 明文模式：`signature + timestamp + nonce` 验签，消息体直接 XML
- 安全模式：`msg_signature + timestamp + nonce` 验签 + `Encrypt` AES 解密
- 长连接模式：`WECOM_CONNECTION_MODE=long_connection` + `WECOM_BOT_ID` / `WECOM_BOT_SECRET`
- 当使用企业微信长连接模式时，`/webhook/wecom` 会返回 `409`

## 5. QQ 接入（官方平台）

- 通道：QQ 官方机器人开放平台 API v2
- 方式：WebSocket 网关接收事件 + OpenAPI 回发消息
- 事件：`GROUP_AT_MESSAGE_CREATE`（QQ群@）和 `C2C_MESSAGE_CREATE`（单聊）
- 会话键：
- 群聊：`qq-group:{group_openid}:{user_id}`
- 单聊：`qq-c2c:{user_id}`
- 群里回复：直接文本回复（不强制 @ 提问者）

开通步骤（简版）：

1. 在 QQ 机器人平台创建机器人并开通群聊/单聊消息权限。
2. 将机器人加入目标 QQ 群并允许被 @。
3. 在 `.env` 配置 `QQ_ENABLED=true`、`QQ_BOT_APP_ID`、`QQ_BOT_CLIENT_SECRET`。
4. 启动服务并检查 `GET /healthz` 中 `qq_longconn.authenticated=true`。

## 6. 接口行为

### `GET /healthz`

返回服务状态，示例：

```json
{
  "status": "ok",
  "env": "dev",
  "service": "travel-planner-assistant",
  "wecom_mode": "long_connection",
  "qq_enabled": true,
  "wecom_longconn": {
    "connected": true,
    "authenticated": true,
    "reconnect_attempt": 0,
    "last_error": ""
  },
  "qq_longconn": {
    "connected": true,
    "authenticated": true,
    "reconnect_attempt": 0,
    "last_error": ""
  }
}
```

### `POST /webhook/wecom`

企业微信回调入口（明文或安全模式）。

### `GET /webhook/wecom`

企业微信 URL 验证入口（回调模式专用）。

## 7. 双 Agent 编排（当前实现）

- 解析 Agent（Planner）先运行：抽取意图、工具、参数，必要时追问补槽
- 工具层执行：天气、路线、联网搜索
- 回复 Agent 再运行：结合用户原问题和工具结果生成自然回答
- 失败兜底：统一降级文案，不暴露堆栈

## 8. 测试

```powershell
pytest -q
```

覆盖：

- 意图路由单元测试
- Provider 适配器测试（Mock HTTP）
- 企业微信 webhook 测试（签名 + XML + 回复链路）
- 企业微信长连接测试（认证、文本消息、事件）
- QQ 长连接测试（网关帧、群/单聊路由、去重）

## 9. 已知限制

- 天气超过和风可预报范围时会降级联网检索，并提示“超过7天，天气数据来自于网络，不一定准确”
- 会话仅短期存储（默认 TTL 24h）
- QQ 通道当前仅支持文本输入输出（不处理图片/卡片/富媒体）
