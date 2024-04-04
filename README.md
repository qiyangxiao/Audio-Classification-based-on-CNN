# 基于 CNN 在数据集 UrbanSound8K 上的音频分类


## 概述

本项目是在数据集 UrbanSound8K 上进行的音频分类任务，训练 60 个 epoch，最终模型在 10 个类别的整体准确率达到了 98% ，取得了较好的效果。模型的配置参数在 config.yaml 文件中。

## 音频特征

本项目提取的是音频 MFCC 特征 (n_mfcc, time_length)，每个时间片的 MFCC 特征维度为 n_mfcc。我们将预处理过程封装在 utils.py/preProcess 函数中，你可以通过修改通道数，采样率等来适应你的数据集（当然这也可能导致模型输入输出的修改，见 config.yaml ）；

## 训练

切换到项目根目录，在终端输入如下命令进行训练。

```
python train.py -c config.yaml
```
+ 你可以通过调整 config.yaml 文件的参数来调整你的模型；

+ dataset.py 中末尾注释部分是预处理数据集的操作，最终生成的 npy 文件存储在 npy_data 文件夹下；

+ 你可以修改 train_main.py 中的 split_size 来调整你的训练集和验证集大小；

```
train_loader, valid_loader = load_torch_data(X, y, split_size=0.9) # change split_size to define train and valid size 
```

+ 本项目使用 Tensorboard 来保存训练过程中的评价指标，你可以通过修改 train_main.py 中的 train 和 valid 来设置你需要观测的数据；

+ 你可以在 train_main.py/train 中按照模型的实际效果，来保存模型；

## 验证

interface.py 提供了简单的验证集操作，可以通过观察输出的混淆矩阵来验证模型效果；

