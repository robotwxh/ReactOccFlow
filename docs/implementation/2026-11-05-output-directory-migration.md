# 2026-11-05：将 output 产物统一迁入 outputs

按用户要求，已将旧`output/`中的全部196个普通文件迁入`outputs/`，总计97,493,685,222字节（约90.8 GiB）。后续评测对比需要的文件全部保留，未删除任何文件；旧目录清空后通过`rmdir`移除。

| 旧位置 | 现位置 | 内容 |
|---|---|---|
| `output/pdf/` | [`outputs/pdf/`](../../outputs/pdf/) | 项目一页A4概览PDF及可编辑Markdown，2个文件 |
| `output/audits/OccFM_2026-11-01/` | [`outputs/audits/OccFM_2026-11-01/`](../../outputs/audits/OccFM_2026-11-01/核查报告.md) | OccFM论文、权重、轨迹和本地材料核查与证据 |
| `output/audits/OccFM_weights_2026-11-02/` | [`outputs/audits/OccFM_weights_2026-11-02/`](../../outputs/audits/OccFM_weights_2026-11-02/权重评估与论文对照.md) | 10个完整检查点评测结果，六份全量预测数组、FVD特征、对照报告、诊断及运行日志 |

`audits/`是OccFM基线评测资料，不是标准DiT参数目录。标准DiT参数与恢复状态仍在`outputs/reactflow_dit/seed_2026/`；原U形CFM参数仍在`outputs/reactflow_joint/seed_2026/`，与本次搬迁无关。

两个子目录在同一文件系统内直接重命名到`outputs/`，未重新复制91GB数据、未覆盖目标文件，也未创建链接。196个文件的设备号、inode、大小、mtime、链接数全部保持一致；小于等于64MiB的文件核对完整SHA-256，六个大预测文件核对相同inode及首尾各1MiB摘要。原评测数组、特征、指标和日志内容均保持不变。

PDF生成器和OccFM核查工具中的六处当前路径已改为`outputs/`，防止后续继续使用旧目录。历史ledger、来源记录、JSON和冻结源码中的原执行路径保留，通过[目录说明](../../outputs/audits/README.md)和本次迁移清单映射到当前位置，避免改写已有校验链。ReactFlow既有对照文件内的历史基线路径同样使用此前缀映射，后续报告命令明确传入新`--baseline-root`。

逐文件清单与核验结果：`outputs/directory_migrations/2026-11-05-output-to-outputs/migration.json`；来源与验收摘要：[output_directory_migration_20261105.json](../provenance/output_directory_migration_20261105.json)。两个参考仓库及三份训练/评测冻结源码均保持原状。

训练继续按用户指示保持暂停；本次没有加载模型进行训练、推理或重新评测。
