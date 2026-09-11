# XGO Lite2归档说明

当前有效内容在项目根目录、`docs/`与`assets/`维护。归档包是日期快照，不代表整机已经购买、验收或完成策略部署。

## 归档索引

| 日期 | 归档文件 | 类型与版本 |
| --- | --- | --- |
| 2026-09-11 | `XGO-Lite2_2026-09-11_plan-v0.1.zip` | 购买、建模、RL与接管规划快照 |

ZIP同目录附带同名`.manifest.json`和`.sha256`。包内以`XGO-Lite2/`为顶层目录，并附`ARCHIVE-MANIFEST.json`。

## 包含范围

- `README.md`、`project.json`和`CHANGELOG.md`。
- `docs/`全部当前文档。
- `assets/README.md`及未来已进入Git的轻量资产。
- 本归档说明。

不包含外部STEP、CM4镜像、训练数据、视频、长日志、其他归档包或Git数据库。外部资产的位置与哈希记录在项目元数据和资产清单中。

## 校验与恢复

1. 使用`Get-FileHash -Algorithm SHA256`计算ZIP哈希，与同名`.sha256`比较。
2. 解压到新的空目录，不覆盖当前工作区。
3. 使用包内`ARCHIVE-MANIFEST.json`核对每个文件的相对路径、字节数和SHA-256。
4. 从README进入，先核对项目状态和未完成项。

哈希用于一致性检查，不等同于数字签名。后续归档使用新的日期或版本文件名，不能覆盖历史快照。
