# UNet 遥感影像建筑提取

## 项目概述

本项目实现了一个完整的二分类语义分割流程——从大幅面遥感影像（11256×10000 像素）中提取建筑区域。核心模型为基于 ResNet34 encoder 的 UNet，对 `img_test.png`（10000×5000 像素）执行滑窗推理并输出建筑预测结果。

项目严格按三个递进子任务组织：数据构建 → 训练验证 → 大图推理与改进实验。每个子任务有独立的测试套件，可独立验证闭环是否正确。

---
## 项目结构

```
UNet/
├── code/
│   ├── step1_crop_dataset.py    # Task 1: 大图裁剪与数据集构建
│   ├── dataset.py               # PyTorch Dataset 数据加载与增强
│   ├── unet_model.py            # 基线 Res34UNet_light 模型
│   ├── eval.py                  # IoU 评价
│   ├── step2_train.py           # Task 2: 训练主循环与模型选择
│   ├── step3_predict.py         # Task 3: 滑窗推理、融合、评价、可视化
│   ├── cbam.py                  # CBAM 注意力模块
│   ├── unet_cbam.py             # CBAM-enhanced UNet 变体
│   ├── unet_efficientnet.py     # EfficientNet encoder UNet 变体
│   ├── weight/                  # 预训练权重目录（需手动下载 resnet34 权重）
│   ├── checkpoints/             # 训练产出：最佳模型权重 + 训练历史
│   ├── dataset/                 # Task 1 产出：裁剪后的训练/验证数据
│   └── img_*.png / label_*.png  # 原始数据
├── test/
│   ├── task1/                   # Task 1 测试套件
│   ├── task2/                   # Task 2 测试套件
│   └── task3/                   # Task 3 测试套件
├── compare_all.sh               # 全组合方案对比脚本
├── comparison_results/          # compare_all.sh 产出：对比表格与各模型结果
├── requirements.txt             # Python 依赖
├── README.md                    # 本文件
├── Project.md                   # 原始项目执行简报
└── 任务二：UNet语义分割实现.docx  # 原始作业文档
```

---


## 环境准备与运行方式

### 1. 环境创建

```bash
cd UNet

# CPU 环境（推理可用，训练较慢）
pip install -r requirements.txt

# GPU 环境（推荐，训练需要 CUDA）
# 先装 GPU 版 PyTorch，再装其余依赖
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

**依赖清单**：`torch` / `torchvision` / `numpy` / `Pillow` / `scikit-image` / `tqdm`。`imgaug` 在当前代码中已不再使用。

### 2. 准备预训练权重

基线模型 `Res34UNet_light` 依赖 ImageNet 预训练的 ResNet34 权重。需下载 `resnet34-b627a593.pth` 并放置到 `code/weight/` 目录下（文件名不可更改）。

```
code/weight/resnet34-b627a593.pth
```

该文件是 `torchvision.models.resnet34` 的标准预训练权重。若缺失，训练脚本会在启动时报 `FileNotFoundError`。

### 3. 整体管线

整个项目的运行流程为三步，必须按顺序执行，前一步的输出是后一步的输入：

```
┌─────────────────────────────────────────────────────────┐
│  Step 1: 数据构建                                       │
│  python step1_crop_dataset.py                           │
│  输入: img_trainval.png + label_trainval.png            │
│  输出: dataset/train/image/*.png                         │
│        dataset/train/label/*.png                         │
│        dataset/val/image/*.png                           │
│        dataset/val/label/*.png                           │
├─────────────────────────────────────────────────────────┤
│  Step 2: 训练                                           │
│  python step2_train.py                                  │
│  输入: dataset/{train,val}/{image,label}/  ← Step 1 输出│
│  输出: checkpoints/UNet_res34_best.pth                   │
│        checkpoints/UNet_res34_history.json               │
├─────────────────────────────────────────────────────────┤
│  Step 3: 推理与评价                                     │
│  python step3_predict.py [--mode no_overlap|overlap]     │
│  输入: img_test.png + label_test.png                    │
│        checkpoints/UNet_res34_best.pth  ← Step 2 输出   │
│  输出: outputs/predict.png                               │
│        outputs/visualization.png                         │
│        outputs/predict_prob.npy                          │
│        outputs/predict_results.json                      │
└─────────────────────────────────────────────────────────┘
```

**所有命令默认工作目录均为 `code/`**，因为代码内部大量使用相对路径（如 `./dataset/train/image/`）。

### 4. 各步详细命令

#### Step 1：数据构建（基线无重叠裁剪）

```bash
cd code
python step1_crop_dataset.py
```

将自动输出：
- 图像尺寸、tile size、网格数量、train/val patch 数量
- 验证块所处的行列范围
- 随机种子（42，确保可复现）

带重叠训练扩增的 v2 版本（详见"改进 1"）：

```bash
python step1_crop_dataset.py --train-stride 384   # 25% 重叠
python step1_crop_dataset.py --train-stride 256   # 50% 重叠
```

#### Step 2：基线训练

```bash
cd code
python step2_train.py
```

训练配置（`step2_train.py` 中的 `CONFIG` 字典）：
- 模型：`Res34UNet_light`（ResNet34 encoder + UNet decoder）
- 损失：`BCEWithLogitsLoss`
- 优化器：Adam（lr=1e-3, weight_decay=1e-5）
- 学习率调度：StepLR（每 8 轮降为 0.3 倍）
- Batch size：8
- Epoch：15
- 最佳模型选择：校准 mIoU（在 0.2–0.8 区间搜索最佳阈值）
- 归一化：使用固定 mean/std（从训练数据统计得出）

训练完成后输出 `checkpoints/UNet_res34_best.pth` 和 `checkpoints/UNet_res34_history.json`。

若需换用 EfficientNet 或 CBAM 变体，修改 `CONFIG["model_type"]`：

```python
# 在 step2_train.py 顶部修改 CONFIG：
CONFIG["model_type"] = "res34_cbam_enc_dec"    # CBAM 变体
CONFIG["model_type"] = "efficientnet_b0"        # EfficientNet-B0
```

#### Step 3：测试推理

```bash
cd code

# 基线：无重叠滑窗推理
python step3_predict.py --mode no_overlap

# 带重叠滑窗推理 + 高斯融合（推荐）
python step3_predict.py --mode overlap --fusion gaussian

# 运行完整实验对比套件（4 组预设实验）
python step3_predict.py --run-suite

# 使用训练历史中校准的最佳阈值
python step3_predict.py --mode overlap --use-history-threshold
```

输出文件位于 `outputs/`（单次运行）或 `outputs/{experiment_id}/`（实验套件）。

### 5. 运行测试

测试套件位于 `test/` 目录，按任务组织：

```bash
cd UNet

# Task 1 测试：数据裁剪、对齐、划分可复现性
python test/task1/test_crop.py
python test/task1/test_crop_overlap.py
python test/task1/test_spatial_block_split.py

# Task 2 测试：过拟合、模型选择、checkpoint、augmentation 同步等
python test/task2/run_all.py

# Task 3 测试：推理逻辑、实验套件、小裁剪端到端测试
python test/task3/run_all.py
```

### 6. 全组合方案对比

完成单模型基线训练后，可使用 `compare_all.sh` 对所有模型变体与推理策略的组合进行系统对比。

```bash
cd UNet

# 核心模型快速对比（res34 + 4×CBAM + enet-b0）
bash compare_all.sh --quick

# 全部 11 模型完整对比
bash compare_all.sh

# 仅推理，跳过训练（checkpoint 已就绪）
bash compare_all.sh --skip-train

# 仅对比 CBAM 插入位置
bash compare_all.sh --cbam-only
```

**对比矩阵**：

| 维度 | 变体 | 数量 |
|------|------|------|
| 模型 | `res34` + 4 种 CBAM 插入模式 + 5 种 EfficientNet 尺度 + auto | 11 |
| 推理策略 | baseline / overlap+uniform / overlap+gaussian / overlap+gaussian(stride=256) | 4 |
| 合计 | | 44 组 |

**脚本执行三阶段**：

1. **训练** — 遍历模型列表，对每个通过环境变量注入 `model_type` 调用 `step2_train.py`。已存在的 checkpoint 自动跳过。
2. **推理** — 对每个已训练 checkpoint 调用 `step3_predict.py --run-suite`，一次性跑完 4 组推理实验，输出 `experiment_records.json`。
3. **汇总** — 读取所有模型的实验记录，编译为 `comparison_results/records.json`，输出按 mIoU 排序的对比表格，并生成四项推荐：
   - 全局最优（mIoU 最高）
   - 各模型家族最优（baseline / CBAM 变体 / EfficientNet 变体）
   - 最优推理策略（按所有模型聚合平均）
   - 性价比排名（mIoU 每推理秒）

最终对比表格与推荐写入 `comparison_results/table.txt`，结构化数据保存在 `comparison_results/records.json`。

---

## 代码如何完成 docx 任务要求（v1 基线）

### 任务一：数据集构建

docx 要求将 `img_trainval.png`（11256×10000）按 512×512 stride 裁剪，70%/30% 划分训练/验证集。

**`step1_crop_dataset.py` 的实现**：

1. **尺寸校验**：读取图像与标签后立即比对空间尺寸，不一致则断言失败，阻塞后续执行。

2. **完整边界覆盖**：使用 `compute_axis_positions` 计算所有合法滑动位置，最后一行/列的窗口采取"贴边"策略（`last_start = length - tile_size`），确保原图每个像素至少被一个块覆盖，不会因尺寸不能整除而丢弃边界区域。

3. **同步裁剪**：图像块与标签块使用完全相同的窗口坐标 `(y, x)` 提取，通过 `save_patch` 函数保证二者一一对应。

4. **可追溯命名**：基线模式下文件名为 `patch_r{row:04d}_c{col:04d}.png`，可从文件名直接回溯至原图网格位置。重叠模式下使用坐标命名 `patch_y{yyyyy}_x{xxxxx}.png`，防止与基线块混淆。

5. **划分策略**：采用紧凑的 2D 验证块（而非随机打散或行级划分），优先选择靠近图像中心的连续区域。这比按行划分或随机划分更严格地减少了训练/验证集之间的空间泄漏。划分完全由确定性算法决定，固定种子 42 保证可复现。

6. **统计输出**：打印总 patch 数、train/val patch 数、网格参数、验证块范围。

### 任务二：基线训练与验证

docx 要求理解训练、推理、数据集读取逻辑，完成 UNet 模型训练，收敛后保存最佳模型。

**`step2_train.py` 的实现**：

1. **数据读取**：`MyDataset`（`dataset.py`）从 Step 1 输出的目录加载数据，标签通过 `>0` 转为二值图，使用与训练阶段一致的 mean/std 做归一化。增强仅应用于训练集（Rot90、垂直/水平翻转），且通过 epoch 感知的随机种子确保 image-mask 完全同步。

2. **模型**：`Res34UNet_light`（`unet_model.py`）使用 ResNet34 前 5 个 stage 作为 encoder（含 ImageNet 预训练权重），decoder 包含 4 个上采样 block + skip connections，最后通过 1×1 卷积输出单通道 logits。训练时返回 `(loss, out)`，推理时仅返回 `out`。

3. **训练循环**：Adam 优化器 + StepLR 调度 + 梯度裁剪（max_norm=1.0）。使用 `BCEWithLogitsLoss`（内置 sigmoid + BCE，数值更稳定）。

4. **验证与模型选择**：每个 epoch 后执行 `evaluate_validation`，不仅计算固定阈值 0.5 下的 IoU，还在 0.2–0.8 区间搜索最佳阈值（601 步搜索），以校准 mIoU 作为模型选择标准。选择验证集上校准 mIoU 最高的 epoch 权重保存为 `UNet_res34_best.pth`。

5. **可复现性**：固定所有随机种子（Python、NumPy、PyTorch、CUDA），启用 cuDNN deterministic 模式。训练历史完整保存为 JSON。

**`eval.py`** 使用混淆矩阵计算二分类 IoU，与 `step2_train.py` 中的评价逻辑共享相同的底层函数，确保语义统一。

### 任务三：大图推理与评价

docx 要求对 `img_test.png`（10000×5000）实现"边裁剪、边预测、边拼接"，计算 IoU 并可视化。

**`step3_predict.py` 的实现**：

1. **滑窗推理**：`compute_grid` 计算所有滑窗位置（支持自定义 tile_size 和 stride），批量化推理以提高效率（GPU 默认 batch_size=8，CPU 默认 batch_size=4）。

2. **概率图累积融合**：不对每个 patch 独立二值化再拼接，而是先输出概率图。在无重叠模式下，将各块概率直接拼回全图；在重叠模式下，维护累加图（accum）和权重图（weight_sum），最终逐像素除以权重得到融合概率图。这避免了"先二值化再拼接"导致的块边界伪影。

3. **全图统一阈值化**：仅在概率图拼接完成后执行一次 `threshold_map`，保证阈值语义全局一致。

4. **IoU 评价**：`compute_test_iou` 基于混淆矩阵（`fast_hist`）计算 mIoU 和各类别 IoU，与训练阶段的 IoU 定义完全一致。

5. **误差可视化**：生成 `visualization.png`：
   - 红色（False Positive）：背景被误判为建筑
   - 绿色（False Negative）：建筑被漏检
   - 白色（True Positive）：正确预测的建筑

6. **输出文件**：`predict.png`（二值预测图）、`predict_prob.npy`（概率图，保留完整软信息）、`predict_results.json`（结构化评价结果）、`visualization.png`。

---

## 四项改进的实现与思考

docx 提出了四个改进方向，我们在代码中全部予以实现，并通过测试验证了各自的正确性。

### 改进 1：带重叠率的训练集裁剪扩增

**问题**：无重叠裁剪导致训练样本数量有限，且 patch 边缘区域缺乏上下文。

**实现**：`step1_crop_dataset.py` 的 `--train-stride` 参数（如 `--train-stride 384` 产生约 25% 重叠，`--train-stride 256` 产生约 50% 重叠）。重叠仅应用于训练集——使用更小的 stride 生成更多训练 patch，每个 patch 仍为 512×512，但相邻块之间存在像素重叠。



**权衡**：训练集扩增增加了训练时间（更多样本），但提高了模型对边界区域的鲁棒性。代价是若过度扩增（stride 过小），文件数量会急剧增加。

### 改进 2：带重叠率的测试滑窗融合

**问题**：无重叠滑窗推理导致每个 patch 的边缘约 10 个像素预测质量较差（感受野不足），块与块之间出现明显的拼接网格缝。

**实现**：`step3_predict.py` 中的 `sliding_window_infer` 函数支持两种融合策略：

1. **均匀平均融合（uniform）**：在重叠区域对多个 patch 的概率取算术平均。维护一个累加图和一个计数图，最终逐像素 `accum / weight_sum`。

2. **高斯加权融合（gaussian）**：每个 patch 使用中心高、边缘低的 2D 高斯权重核（σ=0.25×tile_size）。距离 patch 中心越近的像素权重越高，边缘像素权重接近零。这样在重叠区域，更靠近某个 patch 中心的预测贡献更大，自然消除网格缝。

**关键时序**：先在概率图上做融合，再全图统一阈值化。如果先独立二值化再拼接，会导致硬边界无法消除。

**测试验证**：`test_effectiveness.py` 断言 `overlap + gaussian > overlap + uniform > baseline`（按 mIoU 和 building IoU 递增），且总误差像素数递减。`test_small_crop_smoke.py` 在真实模型上验证推理的确定性和数值稳定性。

### 改进 3：EfficientNet backbone 替换

**问题**：ResNet34 作为 encoder 表达能力有限，EfficientNet 系列通过神经架构搜索（NAS）在相同参数预算下通常获得更高精度。

**实现**：`unet_efficientnet.py` 中的 `EfficientUNet` 类支持 EfficientNet-B0 至 B4 五种 backbone。

**关键设计**：
1. **动态 stage 检测**：不对每个 EfficientNet 变体硬编码 stage 划分。`_resolve_efficientnet_stages` 通过实际前向传播追踪空间分辨率变化，自动识别 5 个 encoder stage（UNet 需要恰好 5 级分辨率）。这意味着添加新 EfficientNet 变体无需手动调整 stage 切片。

2. **通道数自适应**：`_probe_channels` 在初始化时前向一次随机张量，自动获取各 stage 的输出通道数，decoder 的 skip connection 通道数据此自动适配，无需手工配置。

3. **模型注册**：通过 `MODEL_REGISTRY` 统一管理，在 `step2_train.py` 和 `step3_predict.py` 中均可通过修改 `model_type` 参数切换。

4. **Encoder 冻结**：支持 `freeze_encoder` 参数，允许冻结 encoder 权重做迁移学习实验。

**权衡**：EfficientNet 在同等参数下通常精度更高，但 B3/B4 变体显存占用显著增加。动态 stage 检测带来灵活性，但增加了初始化时的一次性开销。当前默认使用随机初始化权重（`pretrained=False`），因为我们依赖 checkpoint 加载训练好的模型权重而非 ImageNet 预训练权重——但如果从头训练，建议开启 `pretrained=True`。

### 改进 4：CBAM 注意力嵌入

**问题**：标准 UNet 对通道和空间特征的关注是均匀的。CBAM（Convolutional Block Attention Module）通过通道注意力和空间注意力两个子模块增强特征表达——通道注意力关注"哪些特征重要"，空间注意力关注"特征中哪些位置重要"。

**实现**：`cbam.py` 实现了标准的 CBAM 模块（`ChannelAttention` + `SpatialAttention`），`unet_cbam.py` 中的 `Res34UNet_CBAM` 继承自 `Res34UNet_light`，支持 6 种插入模式：

| cbam_mode | encoder | decoder | skip | 说明 |
|-----------|---------|---------|------|------|
| `encoder` | ✓ | | | 在 encoder 各 stage 之后插入 |
| `decoder` | | ✓ | | 在 decoder 各 stage 之后插入 |
| `skip` | | | ✓ | 在 skip connection 传给 decoder 之前 |
| `enc_dec` | ✓ | ✓ | | encoder + decoder |
| `adaptive` | | ✓ | ✓ | decoder + skip（默认推荐） |
| `manual` | 可自由组合三个开关 |

**回应 docx 的核心问题——"嵌在哪？"**：docx 明确要求不拍脑袋决定 CBAM 位置，而是将 encoder、decoder、skip connection 视为待比较的设计点。我们的实现通过 `cbam_mode` 参数化，使消融实验变得简单——只需更改一个字符串即可切换插入位置，无需修改模型结构代码。

**初始化可复现**：CBAM 模块的初始化使用 `torch.random.fork_rng` 隔离随机数生成器，确保不同 CBAM 模式之间初始化一致且可复现，这是公平对比的前提。

**测试验证**：`test_cbam_variants.py` 覆盖所有 `cbam_mode`，验证前向传播成功、输出 shape 正确、以及不同模式之间确实产生不同输出（证明 CBAM 确实改变了特征流）。

---

## 当前不足与待解决问题

以下是经过静态代码审查和测试覆盖分析后，诚实记录的问题与不足：

### 1. 全组合对比已脚本化，但训练成本高

`compare_all.sh` 已覆盖全部模型变体（11 种）× 推理策略（4 种）的对比。脚本通过环境变量注入 `model_type` 与 `step2_train.py` 和 `step3_predict.py` 的 `MODEL_REGISTRY` 接口对齐，可自动完成训练、推理、汇总三步，输出排序对比表与推荐。





### 2. 超参数未针对改进模型单独调优

当前 CONFIG（lr=1e-3, epochs=15, batch_size=8, StepLR step=8, gamma=0.3）是为 Res34UNet_light 设计的。当切换为 EfficientNet-B3（参数量大得多）或 CBAM 变体（额外参数 + 不同的收敛特性）时，这些超参数未必最优。控制变量对比要求所有模型使用相同的训练预算，但这可能对某些模型不公平（训练不足或过拟合）。

**建议**：为每个模型变体至少尝试 2 组学习率，在统一对比表中标注是否为模型的最优配置。

### 3. 验证集空间泄漏的残余风险

`choose_val_block` 通过选择紧凑的 2D 块减少了训练/验证之间的空间泄漏，但：
- 空间自相关在遥感影像中通常跨越几十到上百米（对应数十到上百像素），一个 512×512 的验证块即使与训练块无像素重叠，周边训练块仍可能与验证块共享高度相关的纹理和建筑模式
- 对于大尺度同质纹理（如大片林地、水域背景），空间泄漏的影响可能被低估

**建议**：若要严格评估泛化性能，应使用空间上完全独立的测试图（如 `img_test.png` 作为最终 hold-out），而非仅依赖从 `img_trainval.png` 划分的验证集。

### 4. 无 EfficientNet 与 CBAM 组合的代码路径

当前 `EfficientUNet` 和 `Res34UNet_CBAM` 是两个独立的模型类，没有将 CBAM 嵌入 EfficientNet-based UNet 的模型定义。如果希望同时享受 EfficientNet encoder 的能力和 CBAM 的注意力增强，需要额外编写组合模型。

**建议**：或者在 `unet_efficientnet.py` 中增加 CBAM 注入接口（类似 `Res34UNet_CBAM` 的模式），或者在 README 中明确当前不支持组合变体。

### 5. 归一化参数的来源未文档化

`CONFIG["data_mean"]` 和 `CONFIG["data_std"]` 是硬编码的固定值。这些值理应从 `img_trainval.png` 统计得出。当前没有脚本验证这些统计量与当前数据的一致性——如果数据被替换或修改，这些硬编码值可能不再正确。

**建议**：增加一个 `compute_dataset_stats.py` 脚本，从实际训练数据计算 mean/std，并作为文档记录到 README 中，以避免后续数据变更导致归一化不匹配。

### 6. 缺少对大图推理边界行为的显式测试

`step3_predict.py` 中 `compute_axis_positions` 的边界贴边逻辑虽然与 `step1_crop_dataset.py` 共享相同策略，但没有独立的单元测试验证：最右侧/最下侧窗口的预测结果、以及当 `stride` 不能整除原图尺寸时融合权重的正确性。`test_small_crop_smoke.py` 仅在一个 640×640 的小裁剪上测试，不足以覆盖大图边界场景。

### 7. weight 目录下的文件管理

`code/weight/resnet34-b627a593.pth` 不在版本控制中（需用户自行下载），但代码在启动时会强制检查该文件是否存在。README 中已说明下载要求，但若用户误操作或遗漏，报错信息虽然清晰，仍需用户手动搜索权重文件。

**建议**：可在 `step2_train.py` 中增加一个自动下载逻辑（如从 torchvision 官方 URL 下载），作为备选方案。

### 8. 线程与进程安全

当前代码使用 `num_workers=0`（DataLoader 单进程），这对调试友好但训练效率较低。而且PyTorch 的 `worker_init_fn` 机制在此处未显式配置。



---


## 技术约定

- **二分类**：背景（0）vs 建筑（≥1，二值化为 1）
- **标签语义**：`>0` 即为前景（建筑）
- **推理阈值**：训练校准阶段在 0.2–0.8 搜索最佳阈值，测试默认 0.5
- **归一化**：mean=`[0.438, 0.446, 0.412]`, std=`[0.197, 0.185, 0.193]`，训练和推理必须一致
- **数据格式**：普通 PNG 图像，无地理坐标或投影信息
- **工作目录**：所有脚本默认在 `code/` 下运行
- **随机种子**：全局使用 42，确保数据集划分和训练初始化可复现
