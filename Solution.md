# UNet Assignment Solution Notes

## 文档定位

这份文档是给你自己学习、review AI 产出、以及最终独立完成整个 assignment 用的。

它解决三个问题：

1. 现有每个文件到底在做什么。
2. 每个代码文件里的函数、类、算法分别是什么，它们的优点和缺点是什么。
3. 如果没有 AI 辅助，你要在 3 天内分别学习哪些知识、review 哪些关键实现、如何独立完成每一个 task。

这份文档不讲文件树，不重复 `Project.md` 的 AI 执行语气，而是偏向“你应该掌握什么、怎么核验、怎么自己做出来”。

---

## 1. 逐文件学习与代码解析

## `任务二：UNet语义分割实现.docx`

### 作用

这是题目来源文件，不是代码文件，没有函数。

### 题目要求对应的核心算法

- 大图裁剪构建训练/验证集
- UNet 训练与验证
- 大图滑窗预测与拼接
- IoU 精度计算
- 错分与漏分可视化
- 改进实验：重叠裁剪、重叠预测融合、EfficientNet、CBAM

### 优点

- 任务闭环完整，从数据构建到实验改进都覆盖到了。
- 改进方向明确，便于做分层实验。

### 缺点

- 题目描述给的是任务目标，不是工程实现说明。
- 尺寸描述和真实文件可能不完全一致，不能直接硬编码。
- 没有告诉你现有代码哪些已经写完，哪些只是骨架。

---

## `Project.md`

### 作用

这是给 AI 代理执行项目时用的约束型说明文件，没有函数。

### 它提供的“算法价值”

- 帮 AI 确认任务拆分顺序
- 帮 AI 明确每个 task 的验收目标
- 帮 AI 避免脱离当前工程现状瞎改

### 优点

- 适合作为后续 AI 的执行上下文。
- 已经把项目拆成 3 个递进 task。

### 缺点

- 它不是面向“你自己学习”的材料。
- 它不会替你解释代码细节，也不会替你掌握 PyTorch 和分割算法。

---

## `code/step1_crop_dataset.py`

### 当前状态

这是一个**未完成**的数据预处理脚本。目前没有定义任何函数，只有：

- `io.imread('img_trainval.png')`
- `io.imread('label_trainval.png')`
- 一组 `os.makedirs(...)`

### 本文件本应承担的算法职责

- 读取训练大图和标签大图
- 以 `512 x 512` 为窗口裁剪
- 以 `512` 为步长做无重叠裁剪
- 处理右边界和下边界，保证全图覆盖
- 生成图像块和标签块
- 按 `70% / 30%` 划分训练集和验证集
- 保存到：
  - `./dataset/train/image/`
  - `./dataset/train/label/`
  - `./dataset/val/image/`
  - `./dataset/val/label/`

### 当前已有“语句”用到的函数/接口

虽然没有自定义函数，但用到了：

- `skimage.io.imread`
  - 作用：读取图像文件到数组
  - 优点：使用简单，适合读 PNG
  - 缺点：环境依赖 `skimage`，当前机器未必已安装
- `os.makedirs(path, exist_ok=True)`
  - 作用：创建目录
  - 优点：幂等，重复执行不容易报错
  - 缺点：只能建目录，不能完成任何裁剪逻辑

### 这个文件需要用到的核心算法

如果你自己实现，这里最关键的是**滑窗裁剪算法**：

1. 设定窗口大小 `tile_size = 512`
2. 设定步长 `stride = 512`
3. 沿宽和高方向生成所有起点坐标
4. 对每个坐标执行数组切片
5. 保证最后一行和最后一列窗口贴边
6. 图像和标签必须共享完全相同的坐标
7. 保存时保持文件名一一对应

### 优点

- 已经把目标输出目录约定清楚了。
- 与后续训练脚本的数据目录约定是一致的。

### 缺点

- 没有任何真正的裁剪实现。
- 没有随机种子控制。
- 没有文件命名规则。
- 没有边界覆盖策略。
- 没有可视化检查或统计输出。
- 没有为后续重叠裁剪扩增预留参数化设计。

### 你 review 这个文件时必须重点看什么

- AI 是否真的生成了所有裁剪坐标，而不是只裁整除部分。
- AI 是否保证图像块和标签块完全对齐。
- AI 是否对训练/验证划分设置了固定随机种子。
- AI 是否保存成了训练脚本真正能读到的路径。
- AI 是否避免了重叠扩增泄漏到验证集。

---

## `code/dataset.py`

### 文件作用

这个文件定义了训练/验证时使用的数据集类 `MyDataset`。

### 包含的类和方法

#### `class MyDataset(Dataset)`

继承自 `torch.utils.data.Dataset`。

#### `__init__(self, imgs_dir, masks_dir, mean, std, is_train)`

### 作用

- 记录图像目录和标签目录
- 记录归一化参数
- 记录当前是否是训练模式
- 列出所有图像文件名
- 定义增强流水线

### 使用的算法

- 数据集索引管理
- 基础随机增强
- 归一化参数配置

### 使用的函数/接口

- `os.listdir(imgs_dir)`
- `iaa.Sequential([...])`
- `iaa.Rot90`
- `iaa.VerticalFlip`
- `iaa.HorizontalFlip`

### 优点

- 数据增强和读取逻辑集中在一起，训练脚本调用简单。
- 训练增强是图像和 mask 同步做的，思路是对的。
- 归一化参数可配置，不是写死在 `__getitem__` 里。

### 缺点

- `os.listdir` 未排序，样本顺序不稳定，复现性一般。
- 没有检查图像文件和标签文件是否一一匹配。
- 对 `imgaug` 依赖较重，环境和版本敏感。
- 没有提供更细粒度的增强开关。

#### `__len__(self)`

### 作用

返回数据集样本数。

### 使用的算法

- 无复杂算法，本质是容器长度查询。

### 优点

- 简单直接。

### 缺点

- 完全依赖 `self.ids` 的正确性，如果目录里混入无关文件也会被统计进去。

#### `__getitem__(self, i)`

### 作用

给定索引 `i`，返回一个样本字典：

- `image`
- `mask`
- `name`

### 具体执行流程

1. 根据 `i` 取得文件名 `idx`
2. 拼出图像路径和标签路径
3. 用 `PIL.Image.open` 读取图像和标签
4. 将图像转成 `numpy` 数组
5. 将标签转成 `numpy` 数组，并通过 `>0` 二值化
6. 如果是训练模式，则执行同步增强
7. 用 `torchvision.transforms.functional.to_tensor` 转张量
8. 对图像做归一化
9. 返回字典

### 使用的算法

- 二值标签构建
- 数据增强
- 图像张量化
- 通道归一化

### 具体使用的函数/接口

- `Image.open`
- `np.array`
- `(np.array(mask) > 0).astype(np.uint8)`
- `self.transform(...)`
- `transF.to_tensor`
- `transF.normalize`

### 优点

- 明确把标签统一成二值 mask，这对二分类分割是必要的。
- 训练增强和验证不增强的逻辑分开了。
- 返回字典结构，和训练脚本对接方便。
- 图像标准化已经接入训练流程。

### 缺点

- `imgaug` 的 `segmentation_maps` 用法对版本较敏感，后续运行时可能需要排查。
- `img.squeeze()` 和 `mask.squeeze()` 存在潜在风险：
  - 对 RGB 图像通常没问题
  - 但 `squeeze()` 这种写法不够稳健，容易在边界情况下导致维度语义不清晰
- 没有显式检查 `mask` 是否真的是单通道图。
- 没有异常处理，如果图像损坏会直接报错。

### 你 review 这个文件时必须重点看什么

- `mask > 0` 是否符合标签真实语义。
- 增强是否对 image 和 mask 同步执行。
- 返回的 `image` shape 是否是 `[3, H, W]`。
- 返回的 `mask` 是否与 loss 计算期望一致。
- 归一化参数是否与训练、预测保持一致。

---

## `code/unet_model.py`

### 文件作用

这个文件定义了当前的基线分割模型 `Res34UNet_light`。

### 包含的类和方法

#### `class Res34UNet_light(nn.Module)`

这是一个**ResNet34 编码器 + UNet 风格解码器**的轻量化二分类分割网络。

#### `__init__(self)`

### 作用

- 加载 `resnet34` 预训练 backbone
- 拆出 encoder 各层
- 搭建 decoder 上采样模块
- 定义最终二分类输出头
- 定义损失函数 `BCEWithLogitsLoss`

### 使用的算法

- 迁移学习：用预训练 `resnet34` 作为 encoder
- 编码器-解码器结构
- skip connection 特征拼接
- 双线性插值上采样
- 二分类 BCE loss

### 结构说明

- encoder：
  - `conv1 + bn1 + relu`
  - `maxpool`
  - `layer1`
  - `layer2`
  - `layer3`
  - `layer4`
- decoder：
  - `up1` 到 `up5`
  - 每一级都是卷积块 + 与 encoder 特征拼接
- classifier：
  - `Conv2d(64 -> 32)`
  - `ReLU`
  - `Conv2d(32 -> 1)`

### 优点

- 利用了预训练 ResNet34，特征提取能力比纯手写小网络强。
- UNet 的 skip connection 对保留建筑边缘细节有帮助。
- 输出单通道，适合二分类任务。
- 解码器较轻，计算量相对可控。

### 缺点

- `models.resnet34(pretrained=True)` 是旧写法，新版 torchvision 中会有弃用风险。
- `forward` 里直接把 loss 计算耦合进模型，不够“工程解耦”。
- `squeeze()` 写法存在 batch size 为 1 时的维度风险。
- `torch.nn.functional as F` 被导入但没有使用。
- 没有 deep supervision、attention、multi-scale 等增强能力。

#### `forward(self, x, gts=None)`

### 作用

- 训练时：
  - 前向传播
  - 计算 loss
  - 返回 `(loss, out)`
- 推理时：
  - 只返回 `out`

### 使用的算法

- encoder 多层特征提取
- decoder 逐层上采样与 skip 拼接
- logits 输出
- BCEWithLogitsLoss

### 关键函数/接口

- `self.maxpool`
- `torch.cat`
- `self.up(...)`
- `self.criterion(...)`

### 优点

- 训练和推理共用一个 forward，调用简单。
- skip 连接写得直接，结构比较容易读懂。

### 缺点

- 训练逻辑和模型逻辑耦合，后续若想换 loss 或做多任务会不方便。
- 依赖 `self.training` 来决定返回内容，调用者必须非常清楚模型当前处于什么模式。
- `out.squeeze()`、`gts.squeeze()` 会让张量维度变得不稳定。

### 你 review 这个文件时必须重点看什么

- 每一级 encoder 和 decoder 的通道数是否匹配。
- `torch.cat` 前后的空间尺寸是否匹配。
- 输出是否为单通道 logits，而不是已经 sigmoid 后的概率。
- loss 是否确实用的是 logits 输入。
- 训练和推理两种分支是否被调用正确。

---

## `code/eval.py`

### 文件作用

这个文件负责验证集 IoU 评价。

### 包含的函数

#### `fast_hist(label_pred, label_true, num_classes)`

### 作用

根据预测标签和真实标签快速计算混淆矩阵。

### 使用的算法

- 利用 `numpy.bincount` 向量化统计混淆矩阵

### 核心公式

对于二分类：

- 行或列分别表示真实类别和预测类别
- 最终得到一个 `2 x 2` 的混淆矩阵

### 优点

- 向量化实现，速度快。
- 写法简洁，适合像素级分类统计。

### 缺点

- 假设输入标签已经是离散整数类别。
- 不能直接处理 logits 或概率，阈值化必须在外部完成。

#### `eval_net(net, loader, device)`

### 作用

- 切换模型到评估模式
- 遍历验证集
- 生成预测 mask
- 统计混淆矩阵
- 计算 IoU

### 使用的算法

- `torch.no_grad()` 无梯度推理
- `sigmoid + threshold(0.5)` 二值化
- 混淆矩阵累计
- IoU 计算

### IoU 公式

`IoU = TP / (TP + FP + FN)`

当前代码实际计算的是**两类的平均 IoU**：

- 背景类 IoU
- 建筑类 IoU
- 最后取平均

### 优点

- 验证流程清楚，适合作为训练时的模型选择依据。
- 二值化逻辑比较直接。

### 缺点

- `net(imgs, batch)` 这种调用方式语义不够好，因为模型第二参数理论上应该是 `gts`，这里只是因为 `eval()` 模式下不会用到才没报错。
- 固定阈值 `0.5`，没有为后续分析保留阈值对比接口。
- 取“背景和建筑平均 IoU”，有时会弱化你真正关心的建筑类表现。

### 你 review 这个文件时必须重点看什么

- AI 是否保持了和这里一致的阈值语义。
- AI 是否在测试集评价时复用了同样的 IoU 定义。
- AI 是否误把概率直接送进 `fast_hist`。
- AI 是否清楚当前算的是 mean IoU，而不是只算建筑类 IoU。

---

## `code/step2_train.py`

### 文件作用

这是当前项目里最接近“可运行主流程”的脚本，用于训练和验证基线模型。

### 主要组成

- 模块级超参数定义
- `train_net(...)`
- `if __name__ == '__main__':` 主入口

### 模块级配置

包括：

- `lr = 1e-3`
- `batchsize = 8`
- `epochs = 50`
- `num_workers = 0`
- `read_name = ''`
- `save_name = 'UNet'`

以及数据路径、checkpoint 路径和模型实例化。

### 优点

- 超参数集中在文件顶部，便于修改。
- 已接通 dataset、model、eval 之间的主干流程。

### 缺点

- 没有命令行参数接口。
- 没有随机种子控制。
- 没有实验记录系统。
- 没有自动判断路径是否存在。

#### `train_net(net, device, epochs=5, batch_size=1, lr=0.001, save_cp=True)`

### 作用

- 创建训练集和验证集
- 构建 DataLoader
- 定义优化器和学习率调度器
- 执行训练循环
- 每个 epoch 后做验证
- 保存最佳模型
- 训练结束后重新加载最佳模型并做最终验证

### 使用的算法

- PyTorch 标准训练循环
- Adam 优化
- StepLR 学习率衰减
- 验证集最优 checkpoint 保存

### 关键函数/接口

- `DataLoader`
- `optim.Adam`
- `optim.lr_scheduler.StepLR`
- `optimizer.zero_grad()`
- `loss.backward()`
- `optimizer.step()`
- `torch.save`
- `torch.load`
- `eval_net`

### 优点

- 是标准、容易理解的训练结构。
- 每个 epoch 后验证一次，符合模型选择逻辑。
- 保存的是“验证最好”的模型，而不是最后一个 epoch。

### 缺点

- `read_name` 检查路径用了 `../checkpoints`，但真正保存和加载主要用的是 `./checkpoints`，路径不一致，有 bug 风险。
- `drop_last=True` 会丢掉训练集尾部样本。
- `global_step` 定义了但没有实际用途。
- `epoch_loss` 只累计不打印 epoch 平均值。
- 没有混合精度、梯度裁剪、早停等稳定性设计。
- 引入了 `matplotlib.pyplot as plt` 和 `torch.nn as nn`，但未使用。

#### `if __name__ == '__main__':`

### 作用

- 选择 `cuda` 或 `cpu`
- 如果设置了 `read_name`，则尝试加载已有权重
- 调用 `train_net(...)`

### 优点

- 方便直接用脚本运行。

### 缺点

- 对路径、权重文件存在性和环境依赖都比较乐观。
- `read_name` 的逻辑没有完全设计干净。

### 你 review 这个文件时必须重点看什么

- 数据路径是否和裁剪脚本输出路径一致。
- 模型保存与加载路径是否一致。
- 训练时 `imgs` 和 `true_masks` 的 shape 是否正确。
- 训练模式和验证模式是否切换正确。
- 保存的 best model 是否真的来自验证集最好结果。

---

## `code/step3_predict.py`

### 当前状态

这是一个**未完成**的测试脚本。目前没有定义任何函数，只有：

- 读取测试图像
- 占位权重加载语句
- 两段任务说明注释
- 读取测试标签

### 本文件本应承担的算法职责

- 加载训练好的模型权重
- 切换到 `eval()` 模式
- 对测试大图做滑窗预测
- 将 patch 预测结果拼回全图
- 保存 `predict.png`
- 和 `label_test.png` 比较并计算 IoU
- 将错分和漏分着色保存为 `visualization.png`
- 在基线闭环完成后，扩展到重叠预测融合

### 当前已有“语句”用到的函数/接口

- `skimage.io.imread`
- `torch.load`

### 优点

- 任务输出要求写得比较明确。

### 缺点

- 完全不能运行。
- checkpoint 路径还是占位符 `???_best.pth`。
- 没有任何滑窗、拼接、阈值化、可视化实现。

### 你 review 这个文件时必须重点看什么

- AI 是否先做概率图融合，再做全图阈值化。
- AI 是否复用了训练阶段相同的归一化。
- AI 是否正确处理右边界和下边界 patch。
- AI 是否输出与标签同尺寸的预测图。
- AI 是否把 FP 标成红色、FN 标成绿色。

---

## `code/img_trainval.png` 与 `code/label_trainval.png`

### 作用

这是训练/验证的原始大图和标签图，没有函数。

### 算法意义

- 它们是 Task 1 的裁剪来源。
- `label_trainval.png` 的语义决定了 `mask > 0` 是否合理。

### 你必须关注的点

- 标签是否真的是“建筑非零，背景为零”。
- 图像和标签的空间尺寸是否一致。
- 裁剪后是否保持严格对齐。

---

## `code/img_test.png` 与 `code/label_test.png`

### 作用

这是测试阶段的大图和评价标签，没有函数。

### 算法意义

- `img_test.png` 是 Task 3 滑窗推理对象。
- `label_test.png` 是 Task 3 IoU 评价和误差可视化的基准。

### 你必须关注的点

- 预测输出必须和 `label_test.png` 尺寸一致。
- 评价前二者必须采用同一标签语义。

---

## 2. 三天学习与 review 方案

你现在最合理的节奏不是“同时学完全部”，而是：

- 第 1 天：学 Task 1，并 review 数据构建
- 第 2 天：学 Task 2，并 review 训练闭环
- 第 3 天：学 Task 3，并 review 推理、评价和改进实验

---

## Day 1：Task 1 数据构建与裁剪 review

### 你必须掌握的知识

- `numpy` 数组切片
- 图像读写
- `os` / `path` 路径管理
- 随机种子与可复现划分
- 训练集/验证集拆分原则
- 二值标签处理
- 边界覆盖策略

### 你必须会的关键函数/API

- `imread` / `Image.open`
- `imsave` / `Image.save`
- `array[y:y+h, x:x+w]`
- `os.makedirs`
- `os.path.join`
- `random.seed` 或 `np.random.seed`
- `random.shuffle` 或等价划分方法

### 如果你独立实现这个 task，应如何做

1. 读取训练图像和标签图。
2. 检查两者尺寸是否一致。
3. 设定 `tile_size=512`、`stride=512`。
4. 生成横向和纵向所有窗口起点。
5. 对于最后一列和最后一行，保证窗口贴边。
6. 对每个窗口同时裁图像和标签。
7. 给每个块命名，最好带坐标。
8. 固定随机种子，对样本做 `70/30` 划分。
9. 分别保存到 train/val 的 image/label 目录。
10. 抽样可视化检查几个块是否对齐。

### 你 review AI 结果时必须核验的步骤

- 是否真的覆盖了原图全部区域。
- 是否没有丢失右边缘和下边缘。
- 是否图像块和标签块完全一一对应。
- 是否 train 和 val 文件数匹配。
- 是否固定了随机种子。
- 是否命名可追溯。

### 你 review AI 结果时必须核验的算法

- 坐标生成算法
- 边界补齐算法
- 数据集划分算法
- 标签二值化或标签保持算法

### 你 review AI 结果时必须核验的“函数实现”

AI 很可能会新增类似函数，即使函数名不同，你也要检查这些职责是否被正确实现：

- 生成裁剪坐标的函数
- 执行 crop 的函数
- 保存图像和标签块的函数
- 划分 train/val 的函数

### 这一天学习结束后，你应该能回答的问题

- 为什么不能只按 `range(0, W, 512)` 直接裁？
- 为什么最后一列窗口通常要贴边？
- 为什么重叠裁剪不能直接和验证集随机混合？
- 为什么图像块与标签块必须共享完全相同的切片坐标？

---

## Day 2：Task 2 训练与验证闭环 review

### 你必须掌握的知识

- PyTorch 的 `Dataset`、`DataLoader`
- Tensor shape 约定
- 图像归一化
- UNet 基本结构
- ResNet34 backbone 基础
- `BCEWithLogitsLoss`
- `sigmoid` 与阈值化
- IoU 指标
- checkpoint 保存与加载

### 你必须会的关键函数/API

- `torch.utils.data.Dataset`
- `DataLoader`
- `torch.tensor` / `to(device)`
- `torch.sigmoid`
- `nn.BCEWithLogitsLoss`
- `optimizer.zero_grad`
- `loss.backward`
- `optimizer.step`
- `torch.save`
- `torch.load`
- `model.train()`
- `model.eval()`
- `torch.no_grad()`

### 如果你独立实现这个 task，应如何做

1. 写好 `Dataset`，能够正确读取图像和标签。
2. 确保训练时增强同步施加在 image 和 mask 上。
3. 将图像转成 `[C, H, W]`，标签转成二值 mask。
4. 对图像应用和训练一致的归一化。
5. 实例化 `Res34UNet_light`。
6. 定义 `BCEWithLogitsLoss`、优化器、学习率调度器。
7. 写训练循环：
   - 前向
   - 反向
   - 更新参数
8. 每个 epoch 后在验证集上计算 IoU。
9. 保存验证最好模型。
10. 重新加载最佳模型并再次验证。

### 你 review AI 结果时必须核验的步骤

- 数据集是否返回了正确 shape。
- image 和 mask 是否还保持同步。
- 模型输出是不是 logits。
- loss 是否直接吃 logits。
- 评估时是否做了 `sigmoid + 0.5 threshold`。
- best checkpoint 是否真的按验证集 IoU 保存。
- train 和 eval 模式是否切换正确。

### 你 review AI 结果时必须核验的算法

- 数据增强同步算法
- 编码器-解码器前向传播
- BCE loss 计算
- IoU 评价算法
- checkpoint 最优保存策略

### 你 review AI 结果时必须核验的“函数实现”

当前你最该读懂的真实函数是：

- `MyDataset.__init__`
- `MyDataset.__getitem__`
- `Res34UNet_light.__init__`
- `Res34UNet_light.forward`
- `train_net`
- `fast_hist`
- `eval_net`

### 这一天学习结束后，你应该能回答的问题

- 为什么二分类输出头只需要 1 个通道？
- 为什么 loss 里要用 logits，而不是先 sigmoid 再算 BCE？
- 为什么验证阶段必须 `torch.no_grad()`？
- 为什么 `eval()` 模式会影响 BatchNorm 和 Dropout？
- 为什么 mean IoU 和建筑类 IoU 不是一回事？

---

## Day 3：Task 3 大图推理、评价与改进实验 review

### 你必须掌握的知识

- 推理阶段模型加载
- 滑窗推理
- patch 拼接
- 概率图融合
- 重叠区域均值融合
- 阈值化时机
- 全图 IoU 评价
- FP/FN 可视化
- 控制变量实验设计
- EfficientNet 和 CBAM 的基本概念

### 你必须会的关键函数/API

- `model.load_state_dict`
- `model.eval`
- `torch.no_grad`
- `torch.sigmoid`
- 数组切片与 patch 回填
- `np.zeros`
- 逐像素加和与逐像素除法
- 掩码逻辑运算
- 图像保存函数

### 如果你独立实现这个 task，应如何做

1. 加载训练好的最佳模型。
2. 切换到 `eval()` 模式。
3. 用与训练完全一致的归一化处理测试图像 patch。
4. 生成测试图的滑窗坐标。
5. 对每个 patch 前向推理，得到 logits 或概率图。
6. 将 patch 结果累积回全图。
7. 如果存在重叠，则同时维护计数图或权重图。
8. 全图融合结束后，再统一做阈值化。
9. 保存 `predict.png`。
10. 与 `label_test.png` 对比，计算 IoU。
11. 构建 FP/FN 可视化并保存 `visualization.png`。
12. 在基线完成后，再做重叠推理、EfficientNet、CBAM 等改进实验。

### 你 review AI 结果时必须核验的步骤

- patch 坐标是否完整覆盖全图。
- 推理时是否用了训练同样的预处理。
- 融合时是否累积的是概率而不是已经二值化的结果。
- 重叠区域是否正确平均。
- 输出图尺寸是否与标签图完全一致。
- FP 是否标红，FN 是否标绿。
- 改进实验是否保持了控制变量。

### 你 review AI 结果时必须核验的算法

- 滑窗遍历算法
- patch 拼接算法
- 重叠融合算法
- 全图阈值化算法
- 错误类型可视化算法
- 实验对比设计

### 你 review AI 结果时必须核验的“函数实现”

AI 后续大概率会新增以下职责的函数，你要重点读：

- 生成测试滑窗坐标的函数
- patch 推理函数
- 概率图拼接函数
- 计数图或权重图融合函数
- 计算测试 IoU 的函数
- 生成可视化图的函数

### 这一天学习结束后，你应该能回答的问题

- 为什么最好先融合概率图，再阈值化？
- 为什么无重叠推理会出现明显网格缝？
- 为什么重叠区域不能简单覆盖？
- 为什么要用控制变量方法比较 EfficientNet 和 CBAM？

---

## 3. 你必须掌握的“高频 review 标准”

无论 AI 完成哪个 task，你都要反复检查这 6 件事：

1. 路径是否真的对得上当前工程，而不是只在作者本地环境可用。
2. 图像和标签是否始终语义一致、空间对齐。
3. 训练、验证、测试是否使用了同一套预处理和阈值逻辑。
4. 输出尺寸是否和真实标签尺寸一致。
5. 评价指标是否前后一致，没有偷偷换公式。
6. 改进实验是否真的只改了一个关键因素。

## 4. 三天结束后你应达到的能力

如果这三天学得扎实，到最后你应该能独立做到：

- 自己写出大图裁剪和 train/val 划分脚本
- 自己解释 `Dataset`、`DataLoader`、UNet、BCEWithLogitsLoss、IoU 的实现逻辑
- 自己检查训练代码中的 shape、路径、loss 和 checkpoint 问题
- 自己写出滑窗预测、重叠融合和误差可视化
- 自己判断一个改进实验是否公平、是否可信

## 一句话总结

对你来说，这个项目真正要学会的不是“把 AI 写的代码跑起来”，而是掌握三件事：如何正确构建数据、如何正确训练分割模型、以及如何正确地在大图上推理和做实验对比。
