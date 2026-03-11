# EEG-fMRI-Contrastive

一个用于 EEG-fMRI 配对样本学习的工程。当前仓库内置 EEG 编码器 CBraMod 和 fMRI 编码器 NeuroSTORM，支持以下工作流：

- EEG-fMRI 双塔对比学习预训练
- 基于对比学习骨干的分类微调
- ds002336 与 ds002739 的数据预处理
- 基于被试的 LOSO 交叉验证
- 基于已有 finetune checkpoint 的逐折 TestOnly 评估

当前默认流程已经统一到本仓库内部，不再依赖外部 NeuroSTORM 仓库路径。

## 目录结构

```text
EEG-fMRI-Contrastive/
  README.md
  requirements.txt
  run_train.py
  run_finetune.py
  configs/
    train_ds002336.yaml
    finetune_ds002336.yaml
    train_ds002739.yaml
    finetune_ds002739.yaml
  preprocess/
    prepare_ds002336.py
    prepare_ds002739.py
    preprocess_common.py
  mmcontrast/
    datasets/
    models/
    trainer.py
    finetune_trainer.py
    losses.py
  scripts/
    run_prepare_all.ps1
    ds002336/
      prepare_ds002336.ps1
      prepare_ds002336_spm.ps1
      run_ds002336.ps1
    ds002739/
      prepare_ds002739.ps1
      run_ds002739.ps1
    test/
      run_ds002336_contrastive.ps1
      run_ds002336_finetune.ps1
      run_ds002739_contrastive.ps1
      run_ds002739_finetune.ps1
```

## 默认约定

- 数据缓存目录：cache/ds002336、cache/ds002739
- 训练输出目录：outputs/ds002336、outputs/ds002739
- 主批量入口：scripts/ds002336/run_ds002336.ps1、scripts/ds002739/run_ds002739.ps1
- 主评估方式：LOSO subject-wise
- scripts/test 下的脚本只用于单次调试，不是主批量入口
- PowerShell 脚本默认直接调用当前终端里的 python

## 环境准备

```powershell
pip install -r requirements.txt
```

## 输入形状

### EEG

- 输入形状为 [C, S, P]
- C 是通道数
- S 是序列块数
- P 是 patch 长度

### fMRI

- 当前默认走 volume 路径
- 单样本原始数组可以是 [H, W, D, T]
- 模型输入整理为 [B, H, W, D, T]

当前本地常用导出形状：

- ds002336：EEG [63, 20, 200]，fMRI [48, 48, 48, 10]
- ds002739：EEG [53, 2, 200]，fMRI [48, 48, 48, 3]

对应默认值写在以下配置文件中：

- configs/train_ds002336.yaml
- configs/finetune_ds002336.yaml
- configs/train_ds002739.yaml
- configs/finetune_ds002739.yaml

## 数据准备

下面命令默认都在仓库根目录执行。

### ds002739

被试预处理：

```powershell
.\scripts\ds002739\prepare_ds002739.ps1 -NumWorkers 8 -OutputRoot cache\ds002739_parallel
```

### ds002336

直接使用原始 fMRI：

```powershell
.\scripts\ds002336\prepare_ds002336.ps1
```

使用 SPM12 预处理后的 fMRI，这是当前更推荐的路径：

```powershell
.\scripts\ds002336\prepare_ds002336_spm.ps1 -ParallelJobs 4
```

说明：

- 该脚本要求本机可直接调用 matlab
- SPM12 需要已配置好
- Python 侧默认读取 derivatives/spm12_preproc 下的最终 NIfTI

### 顺序处理两个数据集

```powershell
.\scripts\run_prepare_all.ps1
```

## Manifest 与样本组织

训练与评估都由 manifest CSV 驱动。常见列包括：

- eeg_path
- fmri_path
- sample_id
- label

如果 manifest 包含 subject_path 和 sample_count，则表示这是 subject-packed 形式。当前默认会把每个被试写成一个目录，目录内包含 eeg.npy、fmri.npy、labels.npy 等 memmap 友好文件。运行时会先展开成逐样本索引，再由 DataLoader 按 batch 取样。

## 训练机制

### 对比学习

- 训练对象是 EEG-fMRI 配对样本
- 每个 batch 同时取出 EEG 和 fMRI
- 双塔编码器得到 eeg_embed 和 fmri_embed
- 使用对称 InfoNCE 做 EEG->fMRI 与 fMRI->EEG 两个方向的对比损失
- 正样本是同一索引位置的 EEG-fMRI 配对
- 负样本是当前全局 batch 中的其他样本

### 微调

- 读取 train、val、test manifest
- 在对比学习骨干上接分类头
- 默认按 validation 指标选 best.pth
- 最终对 test split 评估，并写出 test_metrics.json、test_logits.csv、test_logits_summary.json

## 主工作流：LOSO

主脚本：

- scripts/ds002336/run_ds002336.ps1
- scripts/ds002739/run_ds002739.ps1

这两个脚本会自动遍历 cache/<dataset>/loso_subjectwise/fold_*，为每一折构造 train、val、test manifest，执行训练或测试，并汇总结果到 loso_finetune_summary.csv。

### 完整 LOSO

```powershell
.\scripts\ds002336\run_ds002336.ps1
```

```powershell
.\scripts\ds002739\run_ds002739.ps1
```

### 只做微调，不跑对比学习

如果某折已有 contrastive checkpoint，会自动复用；没有则从随机初始化开始微调。

```powershell
.\scripts\ds002336\run_ds002336.ps1 -SkipContrastive
```

```powershell
.\scripts\ds002739\run_ds002739.ps1 -SkipContrastive
```

### 只做测试

主 LOSO 脚本支持 TestOnly。它会为每一折自动查找：

- outputs/<dataset>/finetune/<fold_dir>/checkpoints/best.pth

然后直接做 test-only 评估。

```powershell
.\scripts\ds002336\run_ds002336.ps1 -TestOnly
```

```powershell
.\scripts\ds002739\run_ds002739.ps1 -TestOnly
```

说明：

- TestOnly 不会重新训练
- 如果某一折缺少 finetune checkpoint，脚本会直接报错
- 可以叠加 -ForceCpu、-EvalBatchSize、-NumWorkers 等参数

## 单次调试入口

scripts/test 下保留了单次运行脚本，适合调试单个配置或单个 checkpoint：

- scripts/test/run_ds002336_contrastive.ps1
- scripts/test/run_ds002336_finetune.ps1
- scripts/test/run_ds002739_contrastive.ps1
- scripts/test/run_ds002739_finetune.ps1

这些脚本不是主批量入口。日常批量训练与测试应优先使用 scripts/ds002336 和 scripts/ds002739 下的 LOSO 主脚本。

## 直接调用 Python

如果不想走 PowerShell 包装脚本，也可以直接调用 Python。

### 对比学习

```powershell
python run_train.py --config configs/train_ds002336.yaml
```

```powershell
python run_train.py --config configs/train_ds002739.yaml
```

### 微调

```powershell
python run_finetune.py --config configs/finetune_ds002336.yaml
```

```powershell
python run_finetune.py --config configs/finetune_ds002739.yaml
```

### 从已有 checkpoint 做 test-only

```powershell
python run_finetune.py --config configs/finetune_ds002336.yaml --finetune-checkpoint outputs/ds002336/finetune/fold_sub-xp101/checkpoints/best.pth --test-only
```

```powershell
python run_finetune.py --config configs/finetune_ds002739.yaml --finetune-checkpoint outputs/ds002739/finetune/fold_sub-01/checkpoints/best.pth --test-only
```

## 输出文件

典型输出结构如下：

```text
outputs/
  ds002336/
    contrastive/
      fold_sub-xp101/
        checkpoints/best.pth
    finetune/
      fold_sub-xp101/
        checkpoints/best.pth
        test_metrics.json
        test_logits.csv
        test_logits_summary.json
      loso_finetune_summary.csv
  ds002739/
    contrastive/
    finetune/
```

其中：

- test_metrics.json 保存单折测试指标
- test_logits.csv 保存逐样本 logits 明细
- test_logits_summary.json 保存 logits 数值范围、NaN/Inf 检查和绝对值最大的若干样本
- loso_finetune_summary.csv 保存跨折汇总结果

## 配置说明

四个主配置文件里的默认 manifest 路径都指向某一个具体 fold，这只是为了让单次运行和直接 Python 调用有一个可启动的默认值。真正使用 LOSO 主脚本时，train、val、test manifest 和 output_dir 都会在运行时按折动态覆盖。

## 已知边界

- 当前默认 fMRI 路径是 NeuroSTORM volume，不再以旧的 Brain-JEPA 入口为主
- 在线 pad、crop、interpolate 只能解决尺寸不一致，不能修复采集阶段的信息缺失
- Windows 下某些加速依赖不一定总能安装，但当前默认流程通常仍可运行

## 最小工作流

如果只想复现当前默认流程：

1. 激活 mamba 环境并安装 requirements
2. 运行对应数据集的预处理脚本
3. 运行对应数据集的 LOSO 主脚本
4. 在 outputs/<dataset>/finetune 下查看各折结果和 loso_finetune_summary.csv

例如 ds002336：

```powershell
conda activate mamba
cd D:\OpenNeuro\EEG-fMRI-Contrastive
.\scripts\ds002336\prepare_ds002336_spm.ps1 -ParallelJobs 4
.\scripts\ds002336\run_ds002336.ps1
```

如果只想复查已有模型：

```powershell
.\scripts\ds002336\run_ds002336.ps1 -TestOnly
```
