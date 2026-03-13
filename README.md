# EEG-fMRI-Contrastive

当前仓库只保留一条正式工作流：

1. 选择一个或多个数据集做 EEG-fMRI 联合对比学习预训练。
2. 预训练阶段先做时间对齐和通道统一，再把全部预训练样本合并成一个训练集。
3. 选择单个目标数据集做分类微调。
4. 微调阶段只使用目标数据集自己的 trial 或 block 切窗逻辑，并按 LOSO 划分 train、val、test。

旧的“按单数据集单独跑 contrastive，再逐折做 contrastive 验证”的流程已经删除。当前不再支持旧的 per-dataset contrastive 运行脚本，也不再保留旧的单数据集 contrastive 配置文件。

## 当前目录

```text
EEG-fMRI-Contrastive/
  README.md
  requirements.txt
  run_train.py
  run_finetune.py
  run_optuna_search.py
  configs/
    train_joint_contrastive.yaml
    finetune_ds002336.yaml
    finetune_ds002338.yaml
    finetune_ds002739.yaml
    optuna_loso_ds002336.yaml
    optuna_loso_ds002338.yaml
    optuna_loso_ds002739.yaml
  preprocess/
    prepare_ds00233x.py
    prepare_ds002739.py
    prepare_joint_contrastive.py
    run_spm_preproc_ds00233x.m
    preprocess_common.py
  mmcontrast/
    contrastive_runner.py
    contrastive_trainer.py
    finetune_runner.py
    finetune_trainer.py
    datasets/
    models/
    losses.py
    metrics.py
  scripts/
    run_pretrain_and_finetune.ps1
    run_optuna_pretrain_and_finetune.ps1
    prepare_joint_contrastive.ps1
    ds00233x/
      prepare_ds00233x.ps1
      prepare_ds00233x_spm.ps1
    ds002739/
      prepare_ds002739.ps1
```

## 核心规则

### 1. 联合预训练和微调已经解耦

联合预训练使用 [preprocess/prepare_joint_contrastive.py](preprocess/prepare_joint_contrastive.py)。

- 以单个 fMRI TR 为锚点。
- 对每个 TR 提取该时刻之前固定长度的连续 EEG 窗口，默认 8 s。
- 步长固定为 1 TR。
- 允许跨 trial 提取 EEG，只要 run 内时间轴连续。
- 不再按 EEG event index 和 fMRI event index 直接一一配对。
- 预训练阶段不再划分 train、val、test，全部样本直接写入一个 manifest_all.csv，全部作为训练集。

单数据集微调使用 [preprocess/prepare_ds00233x.py](preprocess/prepare_ds00233x.py) 或 [preprocess/prepare_ds002739.py](preprocess/prepare_ds002739.py)。

- 仍然是目标数据集自己的分类切窗逻辑。
- EEG 不允许跨 trial。
- 标签来自当前 trial 或 block 的分类定义。
- 微调阶段才做 LOSO 划分。
- 微调读取阶段会再次按有序目标通道子集映射 EEG（`data.eeg_channel_subset: auto`），最终送入模型的是通道子集后的张量。

### 2. 可选数据集联合预训练

如果只选择 ds002336：

- 只处理 ds002336。
- 生成 ds002336 的联合预训练样本。
- 直接把这些样本全部送入 contrastive 训练。

如果同时选择 ds002336 和 ds002739：

- 分别处理两个数据集。
- 在预处理阶段完成时间对齐和通道统一。
- 把两个数据集的预训练样本拼接成一个 joint manifest。
- 再统一送入 contrastive 训练。

现在也支持 ds002338：

- 可单独把 ds002338 预处理并追加到已有 joint cache。
- 可与 ds002336、ds002739 任意组合做联合预训练。
- ds002338 走 ds002336 同模板流程，且 fMRI 仍要求先经 MATLAB SPM 预处理。

### 3. 联合 cache 增量更新

- `scripts/prepare_joint_contrastive.ps1` 新增默认增量模式：cache 已有数据集时，默认跳过该数据集重算。
- 仅当你显式传 `-ForceRefreshDatasets` 时才会重算指定数据集。
- `prepare_joint_contrastive.py` 支持 ds002336 与 ds002338 数据集级并行（`--num-workers`）。
- 新增数据集接入时，会只处理新增数据集，并更新：
  - 通道名称规范化
  - 公共通道交集
  - 通道映射关系
- 如果目标通道交集缩小，脚本会对已缓存 subject pack 的 EEG 通道做增量重映射，不会强制全量重导所有数据集。

### 4. subject 命名统一

- 所有数据集导出的标准 subject 统一为 sub01、sub02 这种格式。
- 多数据集场景下再加 subject_uid，格式为 <dataset>_<subject>，例如 ds002336_sub01、ds002739_sub01。
- original_subject 只保留在映射表中做追踪。
- 每次预处理都会导出 subject_mapping.csv。

### 5. EEG 通道统一

- 跨数据集联合预训练前，会先规范化各数据集 EEG 通道名称。
- 公共通道交集按名称求，不按索引求。
- 所有数据集在联合预训练阶段映射到相同通道集合和相同顺序。
- 预处理会导出：
  - eeg_channels_dataset.csv
  - eeg_channels_target.csv
  - eeg_channel_mapping.csv
- 目标数据集微调如果要复用 joint pretrain backbone，应传入 joint 导出的 eeg_channels_target.csv，保证输入通道顺序一致。

## 环境准备

```powershell
conda activate mamba
cd D:\OpenNeuro\EEG-fMRI-Contrastive
pip install -r requirements.txt
```

如果出现 `ModuleNotFoundError: No module named 'nibabel'`，先确认当前环境是 `mamba`，再执行：

```powershell
python -m pip install -r requirements.txt
```

## 数据预处理入口（推荐先执行）

联合数据集预处理（只构建联合缓存，不启动对比学习和微调）：

```powershell
.\scripts\prepare_joint_contrastive.ps1 -Datasets ds002338 -OutputRoot cache\joint_contrastive
```

单数据集微调预处理：

```powershell
.\scripts\ds00233x\prepare_ds00233x.ps1 -DatasetName ds002338 -DsRoot ..\ds002338 -OutputRoot cache\ds002338
```

现在文档顺序为：先预处理，再训练入口。

## 唯一主入口

正式入口只有一个：

```powershell
.\scripts\run_pretrain_and_finetune.ps1
```

这个脚本会做两件事：

1. 根据 PretrainDatasets 生成联合预训练缓存并训练 contrastive backbone。
2. 根据 TargetDataset 生成单数据集微调缓存并执行 LOSO 微调。

### 常用参数

- PretrainDatasets：预训练使用哪些数据集，可选 ds002336、ds002338、ds002739。
- TargetDataset：微调目标数据集，只能选一个。
- JointTrainConfig：联合预训练配置，默认 [configs/train_joint_contrastive.yaml](configs/train_joint_contrastive.yaml)。
- Ds002336FinetuneConfig：ds002336 微调配置。
- Ds002739FinetuneConfig：ds002739 微调配置。
- JointCacheRoot：联合预训练缓存目录，默认 cache/joint_contrastive。
- JointOutputRoot：联合预训练输出目录，默认 outputs/joint_contrastive。
- JointEegWindowSec：联合预训练的 EEG 连续上下文窗口长度，默认 8 秒。
- PretrainEpochs：覆盖 contrastive epoch。
- FinetuneEpochs：覆盖 finetune epoch。
- BatchSize：覆盖训练 batch size。
- EvalBatchSize：覆盖微调评估 batch size。
- NumWorkers：覆盖 DataLoader worker 数。
- NumWorkers：也会传递给联合预处理脚本，用于 ds002336 / ds002338 并行处理。
- SkipPretrain：跳过联合预训练，只做微调。
- SkipFinetune：跳过微调，只做联合预训练。
- TestOnly：只加载已有微调 checkpoint 做测试。
- ForceCpu：强制 CPU。

### 示例 1：只用 ds002336 预训练，并在 ds002336 上微调

```powershell
.\scripts\run_pretrain_and_finetune.ps1 -PretrainDatasets ds002336 -TargetDataset ds002336
```

### 示例 2：用 ds002336 和 ds002739 联合预训练，再在 ds002739 上微调

```powershell
.\scripts\run_pretrain_and_finetune.ps1 -PretrainDatasets ds002336,ds002739 -TargetDataset ds002739
```

### 示例 2b：先追加 ds002338 到 joint cache，再在 ds002338 上微调

```powershell
.\scripts\run_pretrain_and_finetune.ps1 -PretrainDatasets ds002338 -TargetDataset ds002338
```

### 示例 3：跳过预训练，只复用已有 joint checkpoint 做 ds002336 微调

```powershell
.\scripts\run_pretrain_and_finetune.ps1 -SkipPretrain -TargetDataset ds002336
```

## 低层入口

如果你只想单独运行某一步，也可以直接调用低层脚本。

### 联合预训练数据预处理

```powershell
.\scripts\prepare_joint_contrastive.ps1
```

输出目录默认是 cache/joint_contrastive，关键文件包括：

- manifest_all.csv
- run_summary.csv
- subject_mapping.csv
- eeg_channels_dataset.csv
- eeg_channels_target.csv
- eeg_channel_mapping.csv

常用增量参数：

- `-SkipExistingDatasets $true`：默认启用，已存在数据集跳过重算。
- `-ForceRefreshDatasets ds002336`：仅重算指定数据集。

### 单数据集微调预处理

ds002336 / ds002338（通用 233x 入口）：

```powershell
.\scripts\ds00233x\prepare_ds00233x.ps1 -DatasetName ds002336 -DsRoot ..\ds002336 -OutputRoot cache\ds002336
.\scripts\ds00233x\prepare_ds00233x.ps1 -DatasetName ds002338 -DsRoot ..\ds002338 -OutputRoot cache\ds002338
```

ds002336 / ds002338（先跑 MATLAB SPM，再跑 Python 预处理）：

```powershell
.\scripts\ds00233x\prepare_ds00233x_spm.ps1 -DatasetName ds002336 -DsRoot ..\ds002336 -OutputRoot cache\ds002336 -ParallelWorkers 4
.\scripts\ds00233x\prepare_ds00233x_spm.ps1 -DatasetName ds002338 -DsRoot ..\ds002338 -OutputRoot cache\ds002338 -ParallelWorkers 4
```

`prepare_ds00233x_spm.ps1` 的 `-ParallelWorkers` 可直接控制 MATLAB 并行核心数，会调用 [preprocess/run_spm_preproc_ds00233x.m](preprocess/run_spm_preproc_ds00233x.m)。

ds002739：

```powershell
.\scripts\ds002739\prepare_ds002739.ps1
```

如果目标数据集微调要复用联合预训练的 backbone，建议把 joint 导出的 eeg_channels_target.csv 传给对应 prepare 脚本的 TargetChannelManifest。

## 训练入口

- [run_train.py](run_train.py) 只用于联合对比学习预训练，只接受一个 manifest。
- [run_finetune.py](run_finetune.py) 只用于单目标数据集微调，仍然接受 train、val、test 三个 manifest。

## 当前配置约定

### 联合预训练

配置文件： [configs/train_joint_contrastive.yaml](configs/train_joint_contrastive.yaml)

- data.manifest_csv 指向 joint manifest_all.csv。
- 预训练阶段不再使用 val_manifest_csv 或 test_manifest_csv。
- contrastive 训练不再执行验证逻辑，checkpoint 仅按 train loss 选择。

### 单数据集微调

配置文件：

- [configs/finetune_ds002336.yaml](configs/finetune_ds002336.yaml)
- [configs/finetune_ds002338.yaml](configs/finetune_ds002338.yaml)
- [configs/finetune_ds002739.yaml](configs/finetune_ds002739.yaml)

这些配置仍然保留 LOSO 微调所需的 train、val、test manifest 结构。

## Optuna

Optuna 现在仍然保留，但已经切换到当前的新主流程，不再依赖旧的单数据集 contrastive 运行脚本。

你可以先用 dry-run 快速确认 study 配置是否真实被执行链路消费：

```powershell
python .\run_optuna_search.py --study-config configs\optuna_loso_ds002336.yaml --mode full --dry-run
python .\run_optuna_search.py --study-config configs\optuna_loso_ds002739.yaml --mode full --dry-run
```

### 入口

- [run_optuna_search.py](run_optuna_search.py)：通用 Optuna 搜索入口。
- [scripts/run_optuna_pretrain_and_finetune.ps1](scripts/run_optuna_pretrain_and_finetune.ps1)：把单个 trial 的输出目录映射到当前唯一主流程的包装脚本。

### Study 配置

- [configs/optuna_loso_ds002336.yaml](configs/optuna_loso_ds002336.yaml)
- [configs/optuna_loso_ds002338.yaml](configs/optuna_loso_ds002338.yaml)
- [configs/optuna_loso_ds002739.yaml](configs/optuna_loso_ds002739.yaml)

这两个配置已经适配当前工作流：

- full：联合预训练 + 单目标数据集微调。
- finetune_only：跳过预训练，只做目标数据集微调。
- pretrain_only：跳过微调，只做联合预训练。

### 当前保留的搜索参数

参数选择已经按你之前那套恢复进来了：

- train_epochs
- batch_size
- lr
- weight_decay
- min_lr
- hidden_dim
- grad_clip
- early_stop_patience

其中 ds002336 和 ds002739 各自的候选值也保留了之前的区别，例如 batch_size、train_epochs 和 early_stop_patience 的搜索范围仍然沿用原来的两套配置。

### 使用示例

搜索 ds002336 的完整流程：

```powershell
python .\run_optuna_search.py --study-config configs\optuna_loso_ds002336.yaml --mode full
```

只搜索 ds002739 的微调阶段：

```powershell
python .\run_optuna_search.py --study-config configs\optuna_loso_ds002739.yaml --mode finetune_only
```

只搜索联合预训练阶段：

```powershell
python .\run_optuna_search.py --study-config configs\optuna_loso_ds002336.yaml --mode pretrain_only
```
