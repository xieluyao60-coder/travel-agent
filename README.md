# Travel Agent

一个面向中文场景的旅游助手后端，支持企业微信与 QQ 两个通道，共用同一套编排核心。

## 核心能力
- 天气查询：实时天气 + 7 天预报（和风天气）。
- 路线规划：驾车/步行/公交路线（高德地图）。
- 周边搜索：基于位置 + 关键词查找附近 POI（高德地图）。
- 联网搜索：检索公开网页信息（SerpAPI）。
- 记忆能力：记住用户旅行城市、酒店位置、常用偏好（Redis，TTL 可配置）。

## 架构说明
- `planner agent`：解析用户意图、决定调用工具、提取参数。
- `tool layer`：执行天气/路线/周边/搜索等外部 API 调用。
- `reply agent`：结合原始问题与工具结果生成自然回复。
- `orchestrator`：串联全链路，做会话状态与降级处理。

## 技术栈
- Python 3.12+
- FastAPI
- Redis
- httpx / pydantic
- Docker Compose

## 目录结构
```text
app/
  main.py
  config.py
  providers/
  services/
docker-compose.yml
docker-compose.prod.yml
```

## 快速启动
1. 复制配置：
```powershell
Copy-Item .env.example .env
```

2. 填写 `.env` 中关键参数（示例）：
- `QWEATHER_API_KEY`
- `QWEATHER_API_HOST`
- `AMAP_API_KEY`
- `SERPAPI_API_KEY`
- `LLM_BASE_URL`
- `LLM_API_KEY`
- `LLM_MODEL`
- `WECOM_*` / `QQ_*`

3. 启动服务：
```powershell
docker compose up --build
```

4. 健康检查：
```text
GET http://localhost/healthz
```

## 通道接入
- 企业微信：支持 URL 回调与长连接模式。
- QQ 机器人：支持官方群聊 @消息与单聊（WebSocket 网关）。

## 测试
```powershell
pytest -q
```

## 说明
- 本项目默认只处理文本输入输出。
- 天气超过官方可预报范围时会触发降级策略并提示可靠性说明。
- 请勿在仓库中提交真实密钥与个人隐私配置，统一使用 `.env` 本地注入。
