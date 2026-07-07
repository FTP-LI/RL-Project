# learn 工作区

当前主项目：

```text
learn/
├── MuJoCo-Train/   # 本机建模、仿真、训练
├── MuJoCo-Lab/     # 树莓派部署验证
└── tools/          # 同步脚本
```

## 工作流

先在本机训练目录工作：

```powershell
cd C:\Users\27348\Desktop\learn\MuJoCo-Train
powershell -ExecutionPolicy Bypass -File .\scripts\setup_windows.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\check_model.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\random_rollout.ps1
```

把训练目录里的运行时代码更新到部署验证目录：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\update_lab_runtime.ps1
```

同步 `MuJoCo-Lab` 到树莓派：

```powershell
cd C:\Users\27348\Desktop\learn
powershell -ExecutionPolicy Bypass -File .\tools\sync_to_pi.ps1
```

第一次连接这台树莓派时，脚本会自动生成本机 SSH key，并把公钥安装到树莓派上。这一步需要输入一次 `ftp` 用户密码。

之后再次同步同一台树莓派，就不需要再输入密码。

同步目标固定为：

```text
ftp@192.168.8.202:/home/ftp/MuJoCo-Lab
```

如果换了树莓派、IP 或用户名变化，直接修改：

```text
tools/sync_to_pi.ps1
```

文件顶部这几行：

```powershell
$PiHostName = "192.168.8.202"
$PiUser = "ftp"
$RemoteDir = "/home/ftp/MuJoCo-Lab"
```

然后重新运行同步脚本。新设备第一次连接时仍然只需要输入一次密码。
