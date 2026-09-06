# 资产说明

此目录只存放可提交、可追溯的轻量化资产和生成说明。原始大型CAD、系统镜像、训练数据和视频不直接提交到Git。

当前外部源资产：

| 资产 | 位置 | 处理要求 |
| --- | --- | --- |
| Lite2完整STEP | `D:\BaiduNetdiskDownload\XGO-lite2总装2.STEP` | SHA-256：`523FC6C5475CF37122C3C48C07B5B8B22A6446ABE4D89B89EF080426E6B49609` |

后续建议目录：

```text
assets/
├─ manifests/       # 源文件哈希、工具版本和许可说明
├─ meshes/visual/   # 视觉网格
├─ meshes/collision/# 简化碰撞网格
├─ urdf/            # URDF/Xacro
└─ mjcf/            # MuJoCo模型和场景
```

任何派生文件都必须说明单位、坐标系、源部件和生成步骤。
