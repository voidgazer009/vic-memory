<div align="center">

# 🧠 Vic Memory

**一套无限趋近人脑的 AI 记忆体系 —— 适配任意 Agent、任意大模型**
Brain-inspired, model-agnostic long-term memory for any AI agent.

`Python ≥3.9` · `零依赖` · `一个 SQLite 文件` · `内置 MCP Server` · `中英双语检索` · `MIT`

[快速开始](#快速开始) · [人脑机制](#我们实现了哪些人脑机制) · [接入任意-agent](#接入任意-agent) · [基准](#基准) · [架构](docs/ARCHITECTURE.md) · [路线图](ROADMAP.md)

姊妹项目 → [**vic-awesome-memory**](https://github.com/voidgazer009/vic-awesome-memory)：全网记忆系统精选与人脑映射

</div>

---

## 为什么

大模型的"记忆"只是一张越来越贵的草稿纸：窗口一满就截断，事实变了新旧混在一起，不知道的时候还会编。
人脑不是这样工作的。人脑会**过滤、暂存、巩固、关联、遗忘、更新、在睡眠中整理**。
Vic Memory 把这些机制做成一个可以挂在任何 Agent 上的记忆层。

## 我们实现了哪些人脑机制

| 人脑机制 | 在 Vic Memory 里 | 代码 |
|---|---|---|
| 感觉记忆过滤 | `observe()` 先算显著性 + 新颖度，废话不进长期记忆 | `core.observe` |
| 工作记忆（容量有限） | K 个槽位（默认 7），溢出时显著的晋升、其余遗忘 | `core.observe` |
| 情景 / 语义 / 程序 / 前瞻 四类长期记忆 | `kind=episodic/semantic/procedural/prospective` | `core.remember` |
| 艾宾浩斯遗忘曲线 + 间隔重复 | ACT-R 基础激活方程，每次回忆加强 | `dynamics.py` |
| 情绪 / 重要性调节 | 显著性越高衰减越慢、检索权重越高 | `dynamics.py` |
| 海马体索引 + 扩散激活 | 写入时自动建立关联网络，回忆时沿边扩散 | `core.recall` |
| 再巩固 | 被回忆的记忆被强化；新值让旧值失效而非删除 | `core.remember` |
| 双时序记忆 | `as_of=` 问"当时是什么情况"，`history(key)` 看全时间线 | `core.recall` |
| 睡眠巩固 | `sleep()`：衰减归档、合并重复、把情景簇抽象成知识 | `core.sleep` |
| 前瞻记忆 | 带时间或关键词触发的"记得要去做" | `core.due` |
| 元记忆 | `confident=false` → 知道自己不知道，不编造 | `core.recall` |
| 可解释 | 一键导出可读 Markdown，每条可见、可删 | `export_markdown` |

## 快速开始

```bash
git clone https://github.com/voidgazer009/vic-memory.git && cd vic-memory
pip install -e .          # 零第三方依赖
python examples/quickstart.py
```

```python
from vic_memory import Memory

m = Memory("vic.db", scope="vic")

m.remember("我叫 Vic，主营 LED 工矿灯出口", "semantic", pinned=True)       # 核心记忆，永远在上下文
m.remember("仓库在深圳", "semantic", key="warehouse.city")
m.remember("仓库搬到宁波了", "semantic", key="warehouse.city")             # 自动让旧值失效，历史保留
m.remember("德国客户一律报 CIF 汉堡", "procedural")                        # 程序性记忆
m.remember("给 Hans 发修改后的报价", "prospective", trigger={"keywords": ["Hans"]})

m.observe("好的")                                   # 废话：留在工作记忆，溢出即遗忘
m.observe("记住：Hans 讨厌长邮件，控制在 5 行以内")   # 显著：会被晋升到长期记忆

m.recall("仓库在哪").items[0].content     # -> 仓库搬到宁波了
m.recall("Vic 的血型").confident          # -> False（不知道就说不知道）
print(m.context("给 Hans 写邮件"))         # -> 直接塞进 system prompt 的记忆块
m.sleep()                                 # 离线巩固（建议每天或每次会话结束）
```

## 接入任意 Agent

**1. MCP（Claude Desktop / Claude Code / Cursor / Codex / Cline / Windsurf …）**

```json
{
  "mcpServers": {
    "vic-memory": {
      "command": "python",
      "args": ["-m", "vic_memory.mcp_server"],
      "env": { "VIC_MEMORY_DB": "~/.vic_memory.db" }
    }
  }
}
```
提供 8 个工具：`memory_remember` `memory_observe` `memory_recall` `memory_context` `memory_history` `memory_forget` `memory_sleep` `memory_stats`。

**2. Python 库**：任何框架（LangGraph、CrewAI、AutoGen、OpenAI Agents SDK、自研）里三行接入：
`observe(用户输入)` → `context(当前问题)` 拼进 system prompt → 会话结束 `sleep()`。完整示例：[examples/agent_loop.py](examples/agent_loop.py)

**3. CLI**：`vic-memory remember "..." --kind semantic --key user.city` · `vic-memory recall "..."` · `vic-memory export`

**可选：接任意大模型增强**（OpenAI 兼容接口：OpenAI / DeepSeek / Qwen / Kimi / Ollama）
```bash
export VIC_LLM_BASE_URL=https://api.deepseek.com/v1
export VIC_LLM_API_KEY=sk-...
export VIC_LLM_MODEL=deepseek-chat              # 睡眠巩固时用 LLM 抽象知识
export VIC_EMBED_MODEL=text-embedding-3-small   # 可选：向量语义检索
```
不配置也能完整运行（纯本地词法 + 认知动力学）。

## 基准

`python bench/run.py` —— VicMemBench-lite，2000 轮对话（约 14 天），确定性、无需 API：

| 系统 | 久远回忆 | 知识更新 | 时间回溯 | 拒答 | 总分 | 注入 token |
|---|---|---|---|---|---|---|
| 滑动窗口 40 轮（普通聊天） | 0/6 | 0/2 | 0/3 | 0/5 | 0% | 192 |
| 全量存储 + BM25（普通 RAG） | 6/6 | 0/2 | 1/3 | 1/5 | 50% | 50 |
| **Vic Memory** | **6/6** | **2/2** | **3/3** | **4/5** | **94%** | 130 |

> ⚠️ 诚实说明：这是我们**自建的合成基准**，只测"检索到了什么"，用来验证机制是否生效，不是与 Mem0 / Zep 等的公开对比。LoCoMo、LongMemEval 的公开复现在 [路线图 v1.0](ROADMAP.md)。

## 设计原则

1. **模型无关**：记忆属于你，不属于任何模型厂商
2. **零依赖、单文件**：`pip install` 后离线可用
3. **可解释、可删除**：随时导出 Markdown；软删除 / 硬删除（隐私合规）
4. **遗忘是功能**：衰减与归档让上下文干净、便宜
5. **永不覆盖历史**：事实会变，但"以前是怎样"也是记忆

## 测试

```bash
python -m unittest discover -s tests -v
```

## License

MIT © Vic

---
<div align="center">Made by <b>Vic</b> · 外贸 · 国际物流 · AI 一人公司 · 全球自媒体</div>
