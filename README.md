# CMCL-EEG

本项目用于 EEG-fMRI 联合预训练与 EEG-only 下游 LOSO 微调，当前支持：

- 联合预训练目标：`contrastive`、`infonce`、`barlow_twins`
- 下游数据集：`ds002336`、`ds002338`、`ds002739`、`ds009999`
- 微调方式：仅保留 `LOSO`
- 预训练模式：`full`、`strict`

README 已按主流程重写，只保留 LOSO 相关内容。单 fold 微调、历史脚本索引、重复说明已移除。

原 README 备份在：

- `README.backup_before_rewrite.md`

## 1. 项目主流程

标准流程只有四步：

1. 预处理各数据集，生成各自 cache。
2. 预处理联合预训练 cache，生成 `cache/joint_contrastive`。
3. 先单独完成离线预训练，得到预训练权重。
4. 再针对目标数据集运行 LOSO 微调，并按需要做离线评估与可视化。

预训练和微调是解耦的，不要求联动运行。

## 2. 目录约定

- 各数据集 cache：
  - `cache/ds002336`
  - `cache/ds002338`
  - `cache/ds002739`
  - `cache/ds009999`
- 联合预训练 cache：
  - `cache/joint_contrastive`
- 预训练权重：
  - `pretrained_weights/pretrain_full/<objective>/checkpoints/best.pth`
  - `pretrained_weights/pretrain_strict/<dataset>/<subject>/<objective>/checkpoints/best.pth`
- 微调输出：
  - `outputs/<dataset>/finetune/...`
- LOSO 离线评估输出：
  - `outputs/<dataset>/offline_eval/...`

其中 `<objective>` 可选：

- `contrastive`
- `infonce`
- `barlow_twins`

## 3. 数据集用途

- `ds002336`：XP1，参与联合预训练，也用于 LOSO 微调。
- `ds002338`：XP2，参与联合预训练，也用于 LOSO 微调。
- `ds002739`：PDC，参与联合预训练；如需下游，可按 LOSO 配置运行。
- `ds009999`：SEED，不参与联合预训练 cache 构建，只用于 EEG-only LOSO 微调。

## 4. 预处理

### 4.1 各数据集 cache

#### Windows

```powershell
.\scripts\ds00233x\prepare_ds00233x.ps1 -DatasetName ds002336 -DsRoot ..\ds002336 -OutputRoot cache\ds002336
.\scripts\ds00233x\prepare_ds00233x.ps1 -DatasetName ds002338 -DsRoot ..\ds002338 -OutputRoot cache\ds002338
.\scripts\ds002739\prepare_ds002739.ps1 -DsRoot ..\ds002739 -OutputRoot cache\ds002739
.\scripts\ds009999\prepare_ds009999.ps1 -DsRoot ..\SEED -OutputRoot cache\ds009999
```

#### Linux

```bash
./scripts_linux/ds00233x/prepare_ds00233x.sh --dataset-name ds002336 --ds-root ../ds002336 --output-root cache/ds002336
./scripts_linux/ds00233x/prepare_ds00233x.sh --dataset-name ds002338 --ds-root ../ds002338 --output-root cache/ds002338
./scripts_linux/ds002739/prepare_ds002739.sh --ds-root ../ds002739 --output-root cache/ds002739
./scripts_linux/ds009999/prepare_ds009999.sh --ds-root ../SEED --output-root cache/ds009999
```

### 4.2 联合预训练 cache

联合预训练 cache 只需要为 `ds002336`、`ds002338`、`ds002739` 构建一次。

#### Windows

```powershell
.\scripts\prepare_joint_contrastive.ps1
```

#### Linux

```bash
./scripts_linux/prepare_joint_contrastive.sh
```

生成结果位于：

- `cache/joint_contrastive/manifest_all.csv`

## 5. 离线预训练

### 5.1 `full` 与 `strict`

支持两种离线预训练模式：

- `full`
  - 直接使用 `cache/joint_contrastive/manifest_all.csv` 中全部样本。
  - 不排除任何目标数据集或目标被试。
  - 不需要提供 `--target-dataset`。
- `strict`
  - 仍然直接使用现有 `cache/joint_contrastive`。
  - 只在启动预训练时，临时从 `manifest_all.csv` 中排除“目标数据集中的目标测试被试”。
  - 不修改原始 cache。
  - 只传 `--target-dataset` 时，会自动遍历该数据集全部被试，并连续跑完整套 strict 预训练。

### 5.2 预训练命令

#### 5.2.1 主方法 `contrastive`

##### Windows

```powershell
python run_pretrain.py --config configs\train_joint_contrastive.yaml --pretrain-mode full
python run_pretrain.py --config configs\train_joint_contrastive.yaml --pretrain-mode strict --target-dataset ds002336
python run_pretrain.py --config configs\train_joint_contrastive.yaml --pretrain-mode strict --target-dataset ds002338
```

##### Linux

```bash
python run_pretrain.py --config configs/train_joint_contrastive.yaml --pretrain-mode full
python run_pretrain.py --config configs/train_joint_contrastive.yaml --pretrain-mode strict --target-dataset ds002336
python run_pretrain.py --config configs/train_joint_contrastive.yaml --pretrain-mode strict --target-dataset ds002338
```

说明：

- `full` 不需要 `--target-dataset`。
- `strict --target-dataset ds002336` 会自动遍历 `ds002336` 的全部被试，每个被试各训练一套 strict 预训练权重。

#### 5.2.2 Pure InfoNCE baseline

##### Windows

```powershell
python run_pretrain.py --config configs\train_joint_infonce.yaml --pretrain-mode full
python run_pretrain.py --config configs\train_joint_infonce.yaml --pretrain-mode strict --target-dataset ds002336
```

##### Linux

```bash
python run_pretrain.py --config configs/train_joint_infonce.yaml --pretrain-mode full
python run_pretrain.py --config configs/train_joint_infonce.yaml --pretrain-mode strict --target-dataset ds002336
```

#### 5.2.3 Barlow Twins baseline

##### Windows

```powershell
python run_pretrain.py --config configs\train_joint_barlow_twins.yaml --pretrain-mode full
python run_pretrain.py --config configs\train_joint_barlow_twins.yaml --pretrain-mode strict --target-dataset ds002336
```

##### Linux

```bash
python run_pretrain.py --config configs/train_joint_barlow_twins.yaml --pretrain-mode full
python run_pretrain.py --config configs/train_joint_barlow_twins.yaml --pretrain-mode strict --target-dataset ds002336
```

### 5.3 预训练权重命名

#### `full`

- `pretrained_weights/pretrain_full/contrastive/checkpoints/best.pth`
- `pretrained_weights/pretrain_full/infonce/checkpoints/best.pth`
- `pretrained_weights/pretrain_full/barlow_twins/checkpoints/best.pth`

#### `strict`

- `pretrained_weights/pretrain_strict/ds002336/ds002336_sub01/contrastive/checkpoints/best.pth`
- `pretrained_weights/pretrain_strict/ds002338/ds002338_sub01/contrastive/checkpoints/best.pth`
- 其他 objective 同理

### 5.4 在线监控

预训练支持在线监控页面，默认输出到对应预训练目录下的：

- `online_monitor/index.html`

当前在线监控默认显示：

- PCA 趋势图
- 总损失曲线
- `R@1 / R@5` 曲线

如需启动本地网页控制台：

#### Windows / Linux

```bash
python server/app.py
```

浏览器访问：

```text
http://127.0.0.1:8765
```

## 6. LOSO 微调

### 6.1 说明

本项目 README 只保留 LOSO 微调，不再介绍单 fold 微调。

如果 `finetune.contrastive_checkpoint_path` 为空，程序会按配置中的以下字段自动定位预训练权重：

- `finetune.pretrain_mode`
- `finetune.pretrain_objective`
- `finetune.pretrain_output_root`

当前默认配置：

- `ds002336`：`strict`
- `ds002338`：`strict`
- `ds002739`：`full`
- `ds009999`：`full`

### 6.2 LOSO 微调命令

#### XP1 `ds002336`

##### Windows

```powershell
python run_finetune.py --config configs\finetune_ds002336.yaml --loso --root-dir cache\ds002336 --output-dir outputs\ds002336\finetune
```

##### Linux

```bash
python run_finetune.py --config configs/finetune_ds002336.yaml --loso --root-dir cache/ds002336 --output-dir outputs/ds002336/finetune
```

#### XP2 `ds002338`

##### Windows

```powershell
python run_finetune.py --config configs\finetune_ds002338.yaml --loso --root-dir cache\ds002338 --output-dir outputs\ds002338\finetune
```

##### Linux

```bash
python run_finetune.py --config configs/finetune_ds002338.yaml --loso --root-dir cache/ds002338 --output-dir outputs/ds002338/finetune
```

#### PDC `ds002739`

##### Windows

```powershell
python run_finetune.py --config configs\finetune_ds002739.yaml --loso --root-dir cache\ds002739 --output-dir outputs\ds002739\finetune
```

##### Linux

```bash
python run_finetune.py --config configs/finetune_ds002739.yaml --loso --root-dir cache/ds002739 --output-dir outputs/ds002739/finetune
```

#### SEED `ds009999`

##### Windows

```powershell
python run_finetune.py --config configs\finetune_ds009999.yaml --loso --root-dir cache\ds009999 --output-dir outputs\ds009999\finetune
```

##### Linux

```bash
python run_finetune.py --config configs/finetune_ds009999.yaml --loso --root-dir cache/ds009999 --output-dir outputs/ds009999/finetune
```

### 6.3 EEG baseline 的 LOSO 微调

如果要切换为 EEG baseline，可直接在 LOSO 命令上叠加参数。

#### Windows

```powershell
python run_finetune.py --config configs\finetune_ds002336.yaml --loso --root-dir cache\ds002336 --output-dir outputs\ds002336\eegnet_loso --eeg-baseline-category classical --eeg-baseline-model eegnet
```

#### Linux

```bash
python run_finetune.py --config configs/finetune_ds002336.yaml --loso --root-dir cache/ds002336 --output-dir outputs/ds002336/eegnet_loso --eeg-baseline-category classical --eeg-baseline-model eegnet
```

## 7. LOSO 离线评估与可视化

### 7.1 离线复现已训练好的 LOSO 权重

统一入口：

- `python run_visualize.py offline-loso`

它会：

1. 遍历 `checkpoints-root` 下的 `fold_*`
2. 对每个 fold 加载 `checkpoints/best.pth`
3. 按 LOSO 测试集复现结果
4. 汇总输出 `loso_test_summary.csv`
5. 输出按数据集命名的 LOSO 混淆矩阵

#### XP1 `ds002336`

##### Windows

```powershell
python run_visualize.py offline-loso --dataset-name ds002336 --config configs\finetune_ds002336.yaml --checkpoints-root outputs\ds002336\finetune --output-dir outputs\ds002336\offline_eval
```

如果当前配置会自动解析预训练权重、但本地没有对应文件，可改为：

```powershell
python run_visualize.py offline-loso --dataset-name ds002336 --config configs\finetune_ds002336.yaml --checkpoints-root outputs\ds002336\finetune --output-dir outputs\ds002336\offline_eval --allow-missing-pretrain-checkpoint
```

##### Linux

```bash
python run_visualize.py offline-loso --dataset-name ds002336 --config configs/finetune_ds002336.yaml --checkpoints-root outputs/ds002336/finetune --output-dir outputs/ds002336/offline_eval
```

#### XP2 `ds002338`

##### Windows

```powershell
python run_visualize.py offline-loso --dataset-name ds002338 --config configs\finetune_ds002338.yaml --checkpoints-root outputs\ds002338\finetune --output-dir outputs\ds002338\offline_eval
```

##### Linux

```bash
python run_visualize.py offline-loso --dataset-name ds002338 --config configs/finetune_ds002338.yaml --checkpoints-root outputs/ds002338/finetune --output-dir outputs/ds002338/offline_eval
```

#### PDC `ds002739`

##### Windows

```powershell
python run_visualize.py offline-loso --dataset-name ds002739 --config configs\finetune_ds002739.yaml --checkpoints-root outputs\ds002739\finetune --output-dir outputs\ds002739\offline_eval
```

##### Linux

```bash
python run_visualize.py offline-loso --dataset-name ds002739 --config configs/finetune_ds002739.yaml --checkpoints-root outputs/ds002739/finetune --output-dir outputs/ds002739/offline_eval
```

#### SEED `ds009999`

##### Windows

```powershell
python run_visualize.py offline-loso --dataset-name ds009999 --config configs\finetune_ds009999.yaml --checkpoints-root outputs\ds009999\finetune --output-dir outputs\ds009999\offline_eval
```

##### Linux

```bash
python run_visualize.py offline-loso --dataset-name ds009999 --config configs/finetune_ds009999.yaml --checkpoints-root outputs/ds009999/finetune --output-dir outputs/ds009999/offline_eval
```

离线评估结果包含：

- `loso_test_summary.csv`
- `confusion_matrix_<dataset>_loso.png`
- `confusion_matrix_<dataset>_loso.svg`
- `confusion_matrix_<dataset>_loso.json`

其中混淆矩阵标题映射为：

- `ds002336 -> XP1`
- `ds002338 -> XP2`
- `ds009999 -> SEED`

### 7.2 预训练表征可视化

如果需要查看联合预训练 checkpoint 的表征分布，可使用：

#### Windows

```powershell
python run_visualize.py contrastive --config configs\train_joint_contrastive.yaml --checkpoint pretrained_weights\pretrain_full\contrastive\checkpoints\best.pth --output-dir outputs\visualizations\contrastive --batch-size 128 --max-samples 1000 --tsne-max-points 200 --heatmap-max-points 128
```

#### Linux

```bash
python run_visualize.py contrastive --config configs/train_joint_contrastive.yaml --checkpoint pretrained_weights/pretrain_full/contrastive/checkpoints/best.pth --output-dir outputs/visualizations/contrastive --batch-size 128 --max-samples 1000 --tsne-max-points 200 --heatmap-max-points 128
```

说明：

- `--max-samples` 会在读取阶段截取前部连续批次样本，而不是随机打乱全数据集。
- `--sample-seed` 可用于随机选择一段连续批次起点。

## 8. 关键配置项

### 8.1 预训练配置

最常用的是：

```yaml
train:
  pretrain_objective: contrastive

data:
  manifest_csv: cache/joint_contrastive/manifest_all.csv

train:
  output_dir: pretrained_weights/pretrain_full/contrastive
```

其中：

- `pretrain_objective` 可选：`contrastive`、`infonce`、`barlow_twins`
- 实际运行时，如使用 `--pretrain-mode full|strict`，输出目录会按模式自动改写到：
  - `pretrained_weights/pretrain_full/...`
  - `pretrained_weights/pretrain_strict/...`

### 8.2 微调配置

以 `configs/finetune_ds002336.yaml` 为例：

```yaml
finetune:
  contrastive_checkpoint_path: ""
  pretrain_mode: strict
  pretrain_objective: contrastive
  pretrain_output_root: pretrained_weights
  fusion: eeg_only
  classifier_mode: add
  selection_metric: accuracy
```

说明：

- `contrastive_checkpoint_path: ""`
  - 表示不手动写死路径
  - 而是按 `pretrain_mode + pretrain_objective + pretrain_output_root` 自动定位权重
- `allow_missing_pretrain_checkpoint: true`
  - 表示如果自动定位到的预训练权重不存在，则跳过该权重并使用随机初始化
- `pretrain_mode`
  - `ds002336`、`ds002338` 默认是 `strict`
  - `ds002739`、`ds009999` 默认是 `full`
- `classifier_mode`
  - 当前支持 `shared`、`private`、`concat`、`add`
  - 默认配置使用 `add`

## 9. 输出结果说明

### 9.1 预训练

- `checkpoints/best.pth`
- `checkpoints/last.pth`
- `train_history.json`
- `online_monitor/index.html`

### 9.2 微调

- `fold_*/checkpoints/best.pth`
- `fold_*/checkpoints/last.pth`
- `fold_*/test_metrics.json`
- `loso_finetune_summary.csv`

### 9.3 离线评估

- `loso_test_summary.csv`
- `confusion_matrix_<dataset>_loso.png`
- `confusion_matrix_<dataset>_loso.svg`
- `confusion_matrix_<dataset>_loso.json`

## 10. 常见问题

### 10.1 `full` 和 `strict` 有什么区别

- `full`：联合预训练时不排除任何人。
- `strict`：联合预训练时排除“目标数据集中的目标测试被试”。

### 10.2 `strict` 会不会修改 joint cache

不会。  
`strict` 只会临时过滤 `cache/joint_contrastive/manifest_all.csv` 的使用范围，不会改磁盘上的 cache 文件。

### 10.3 `full` 是否需要 `--target-dataset`

不需要。  
只有 `strict` 需要 `--target-dataset`。

### 10.4 `common-channel overlap=40/40` 是什么意思

它表示当前数据集的通道名称与联合预训练公共通道清单完全重合。  
它不表示当前输入只有 40 个通道。
