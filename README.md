# RAG 智能知识库系统（shopkeeper-brain）

基于 **FastAPI + LangGraph** 构建的本地知识库系统，支持文档导入、向量检索与问答生成。

## 主要功能

- **文档导入**：上传 PDF / Markdown → 解析/切分 → 商品名识别 → BGE-M3 向量生成 → 写入 Milvus。
- **知识检索**：识别商品名 → 向量检索 + HYDE + Web 搜索 → RRF 融合 → BGE 重排序 → LLM 生成答案，支持流式 SSE 输出。
- **前端页面**：内置 `import.html`（导入页）和 `chat.html`（问答页）。

## 技术栈

- Python 3.x
- FastAPI / Uvicorn
- LangGraph / LangChain-OpenAI
- Milvus（向量库）
- MongoDB（会话历史）
- MinIO（对象存储）
- BGE-M3 / bge-reranker-large（Embedding & Rerank）
- MinerU（PDF 解析）

## 项目结构

```
shopkeeper-brain-BJ0108/
├── main.py                          # PyCharm 占位脚本（非业务入口）
├── knowledge/                       # 主包
│   ├── api/
│   │   ├── import_api.py            # 导入服务（端口 8000）
│   │   └── query_api.py             # 查询服务（端口 8011）
│   ├── core/                        # 依赖与路径常量
│   ├── processor/
│   │   ├── import_processor/        # LangGraph 导入流程
│   │   └── query_processor/         # LangGraph 查询流程
│   ├── service/                     # 文件处理与查询调度
│   ├── schema/                      # Pydantic 模型
│   ├── prompts/                     # LLM 提示词
│   ├── utils/                       # 客户端与工具函数
│   ├── front/                       # 静态前端页面
│   ├── docs/                        # 流程说明文档
│   ├── test/                        # 组件测试脚本
│   ├── requirements.txt             # Python 依赖
│   └── .env.example                 # 环境变量模板
└── README.md
```

## 快速开始

1. **克隆仓库**

```bash
git clone https://github.com/fengmeili1228-ux/ZK.git
cd ZK
```

2. **安装依赖**

```bash
pip install -r knowledge/requirements.txt
```

3. **配置环境变量**

复制模板文件并填入真实配置：

```bash
cp knowledge/.env.example knowledge/.env
```

然后编辑 `knowledge/.env`，补全 API 密钥、数据库地址、模型路径等信息。

4. **启动服务**

导入服务：

```bash
python -m knowledge.api.import_api
```

查询服务：

```bash
python -m knowledge.api.query_api
```

5. **访问前端**

- 导入页面：`http://localhost:8000/front/import.html`
- 问答页面：`http://localhost:8011/front/chat.html`

## 环境变量说明

真实配置存放在 `knowledge/.env`，该文件已被 `.gitignore` 排除，**不会提交到 GitHub**。仓库中只保留 `knowledge/.env.example` 作为模板，方便其他开发者复用。

主要配置项：

| 类别 | 关键变量 |
|------|----------|
| LLM/VLM | `OPEN_API_KEY`, `OPEN_API_BASE`, `LLM_DEFAULT_MODEL`, `VL_MODEL` |
| Embedding | `BGE_M3_PATH`, `BGE_RERANKER_LARGE`, `BGE_DEVICE` |
| 向量库 | `MILVUS_URL`, `CHUNKS_COLLECTION`, `ITEM_NAME_COLLECTION` |
| 文档库 | `MONGO_URL`, `MONGO_DB_NAME` |
| 对象存储 | `MINIO_ENDPOINT`, `MINIO_ACCESS_KEY`, `MINIO_SECRET_KEY`, `MINIO_BUCKET_NAME` |
| PDF 解析 | `MINERU_API_TOKEN`, `MINERU_BASE_URL` |

## 注意事项

- 运行前请确保 Milvus、MongoDB、MinIO 等外部服务已正常启动。
- `knowledge/temp_data/` 与 `knowledge/processor/import_processor/` 下的输入/输出文件属于运行时产物，不会被提交。
- `main.py` 为 PyCharm 默认生成的占位脚本，实际业务入口为 `knowledge/api/` 下的两个 API 文件。

## License

MIT
