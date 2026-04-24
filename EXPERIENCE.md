# 开发日志与踩坑记录

## 环境配置

### PyTorch 与 MKL 冲突
症状: `ImportError: undefined symbol: iJIT_NotifyEvent`
原因: conda 环境里 MKL 版本和 PyTorch 不匹配
解法: 改用 pip 装 PyTorch (`pip install torch --index-url https://download.pytorch.org/whl/cu121`), 不走 conda channel
启发: 带 C 扩展的包, conda 官方 channel 和 conda-forge 经常版本错位. 现代推荐全用 pip, conda 只管 Python 解释器.

### Gradio 6.0 API 变动
症状: `ChatInterface.__init__() got an unexpected keyword argument 'type'`
原因: Gradio 5.x 里的 type="messages" 在 6.0 里变成默认值, 参数被移除
解法: 删掉该参数; theme 参数从 Blocks() 移到 launch()
启发: 开源项目升级大版本经常破坏 API. requirements.txt 应该锁上界避免意外升级.

### Chroma 数据库版本不兼容
症状: `KeyError: '_type'`
原因: 升级 Chroma 后, 旧版本创建的 sqlite 里的 collection 配置 JSON 格式变了
解法: 删掉 data/chroma/ 重建; 代码里加 try/except 给出友好错误提示
启发: 持久化数据格式升级是个难题. 小项目直接重建, 真实生产要做 schema migration.

## 实验结论

### Reranker 的效果
测试集: 15 道问答 (基于 rag-intro.md)
| 配置 | 检索命中率 | 关键词召回 |
|---|---|---|
| 纯向量检索 (top-3) | XX% | XX% |
| 向量检索 top-20 + reranker top-3 | XX% | XX% |

观察:
- xxx
- xxx

### Chunk size 的影响
xxx

## 后续想改进的点
- [ ] 支持 OCR (现在扫描版 PDF 提不出文字)
- [ ] 加入混合检索 (向量 + BM25)
- [ ] 用 ragas 做更系统的评估
