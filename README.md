# SLFCD

> Supervised Learning Framework for Cancer Detection.
>
> 基于监督学习的癌症检测框架 , 该框架在根据全幻灯片癌症病理图像（Whole slide images, WSI）作为输入数据，实现了癌区域定位，以及基于WSI的图像分类。

## 基本概述

这个仓库主要包含了基于深度学习的癌症检测的源代码框架，开发用于从全幻灯片癌症病理图像（WSI）中识别癌转移。该框架成功应用于Camelyon‘16 grand challenge的数据集。我开发本项目主要是提供一个关于WSI检测的解决方案，以及超简优化代码，并没有在性能上做出较多的探索。

![](https://github.com/ilikewind/CAMELYON16/blob/master/docs/test_own.jpg)

## Notes

- [x] **extras/CNNRF 是使用keras进行建模的相近项目**
- [ ] train.py
- [ ] probs_map.py
- [ ] nms.py
- [ ] Evalution_FROC.py
- [ ] extract_feature_probsmap.py
- [ ] wsi_classification.py

## Requisetes

- Python (3.6)
- Numpy 
- PyTorch
- Openslide 3.4.1
- etc.

## 数据

### 全幻灯片图像 (Whole slide images)

数据主要是来自[Camelyon16](https://camelyon16.grand-challenge.org/)挑战赛的*.tif格式的全幻灯片图像 (WSI)。您可以在[Camelyon16](https://camelyon16.grand-challenge.org/)上下载免费数据，不过建议您应当在获得官方批准之后再使用该数据。注意，一张全幻灯片图像通常在0级为~100k×100k像素，在磁盘上有1GB+的储存量。比赛数据一共有700张WSI，总计约700GB+数据，所以要确保有足够的磁盘空间。用于训练的肿瘤WSI命名为Tumor\_×××.tif，其中xxx的范围是001到110。用于训练的正常WSI名为Normal\_xxx.tif，其中xxx的范围是从001到160。用于测试的WSI名为Test\_xxx.tif，其中xxx的范围从001到130。

![](https://github.com/ilikewind/CAMELYON16/blob/master/docs/datavisul.png)

### 标注 (Annotations)

Camelyon16组织官方还以xml的格式为每张肿瘤WSI提供肿瘤区域的注释。在本阶段，我已经将它转换成一些更简单的json格式。每个注释都是多边形列表，其中每个多边形都由其顶点表示。特别是阳性多边形表示肿瘤区域，阴性多边形表示正常区域。可是使用如下命令将xml格式转换为json格式。

```shell
python CAMELYON16/camelyon16/bin/camelyon16xml2json.py Tumor_001.xml Tumor_001.json
```

### 补丁图像 (Patch images)

虽然最初的400张WSI文件包含了所有必要的信息，但是由于高水平分辨率较高引起机器内存不足以及多分辨率的问题，它们并不适用于直接训练一个深度卷积神经网络 (CNN)。因此，我们必须采样更小的补丁图像，例如256×256，使得典型的CNN能够处理。**有效的采样信息和代表性的补丁图像是实现良好肿瘤检测性能的关键部分之一。**为了简化这个过程，本项目并为采用硬挖掘的信息挖掘技术进行提升模型性能。补丁图像的获取应当包含三个部分，掩码的生成、补丁图像中心坐标的筛选和补丁的获取。

1. 使用```python CAMELYON16/camelyon16/bin/tissue_mask.py Tumor_001.tif Tumor_001_tissue.npy 6```获得Tumor_001.tif在level_6下的组织掩码Tumor_001_tissue.npy；

2. 使用```python CAMELYON16/camelyon16/bin/tumor_mask.py Tumor_001.json Tumor_001.tif Tumor_001_tumor.npy 6```获得癌症掩码；

3. 使用```python CAMELYON16/camelyon16/bin/non_tumor_mask.py Tumor_001_tumor.npy Tumor_001_tissue.npy Tumor_001_normal.npy 6```获得癌症WSI中除去背景正常区域的掩码；

4. ```python python CAMELYON16/camelyon16/bin/sampled_spot_gen.py Turmor_001_tumor.npy train_spot.txt 1000 6```获取随机的癌症区域中的坐标点。

5. 获取补丁数据集

   ```python
   python CAMELYON16/camelyon16/bin/patch_gen.py /WSI_TRAIN/ NCRF/coords/tumor_train.txt /PATCHES_TUMOR_TRAIN/
   python CAMELYON16/camelyon16/bin/patch_gen.py /WSI_TRAIN/ NCRF/coords/normal_train.txt /PATCHES_NORMAL_TRAIN/
   python CAMELYON16/camelyon16/bin/patch_gen.py /WSI_TRAIN/ NCRF/coords/tumor_valid.txt /PATCHES_TUMOR_VALID/
   python CAMELYON16/camelyon16/bin/patch_gen.py /WSI_TRAIN/ NCRF/coords/normal_valid.txt /PATCHES_NORMAL_VALID/
   ```

   其中```/WSI_TRAIN/```是存放所有用于训练的WSI文件目录路径，``/PATCHES_TUMOR_TRAIN/``是存储生成的用于培训的肿瘤补丁的目录的路径。同样的命名也适用于``/PATCHES_NORMAL_TRAIN/``、``/PATCHES_TUMOR_VALID/``和``/PATCHES_NORMAL_VALID/``。

## 模型 (Model)

![](https://github.com/ilikewind/CAMELYON16/blob/master/docs/framework.png)

框架主要包含两部分，一是基于CNN的补丁图像分类（肿瘤定位），另外一个是基于机器学习，比如SVM, 贝叶斯分类器等算法的WSI分类。本部分讲解概述性内容，具体问题请参考代码。

### CNN

使用Inception_v3、ResNet、以及DenseNet模型作为特征提取器，改写模型最后一层全连接层，成为（1000-2）。建立补丁两分类预测模型，使得模型能够识别补丁为癌（1）的概率，输出为0-1。

### WSI分类

本阶段主要通过各个热力图谱中提取的特征指标作为判断WSI是否是癌图像的依据，因此利用作为输入特征，通过机器学习方法，包括但不限于XGBoost、SVM、贝叶斯等作为WSI的分类器。

## 训练 (Training)

1. **CNN**

利用生成的补丁图像，我们现在可以使用下面的命令训练CNN模型用来识别补丁图像；

```shell
python CAMELYON16/camelyon16/bin/train.py /cnn_path/cnn.json /SAVE_PATH/
```

其中``/cnn_path/cnn.json``包含了训练所需要的配置文件，`/SCVE_PATH`是您保存训练结果的文件夹。

2. **WSI分类**

为了训练WSI分类器，首先需要将所有的训练集通过CNN模型得到概率热力图谱，[如下操作](#概率热力图谱-(Probability map))；

然后在概率热力图中使用以下命令提取的特征：

```shell
python CAMELYON16/camelyon16/bin/extract_feature_robsmap.py probs_map_path wsi_path, feature_path
```

其中，`probs_map_path`为热力图路径, `wsi_path`为与热力图相对于的WSI路径，`feature_path`为需要保存的包好features的csv文件。

最终使用特征训练模型得到WSI分类器，以及对分类器进行ROC评价，都是使用下面的命令进行的：

```python
python CAMELYON16/camelyon16/bin/wsi_classification.py probs_map_features_train.cvs probs_map_features_test.cvs TEST_CSV_GT
```

其中，`probs_map_features_train.cvs`为训练集， `probs_map_features_test.cvs`为测试集，`TEST_CSV_GT`为测试集WSI的label。

## 测试 (Testing)

### 组织掩码

经过训练的模型对WSI分析的主要测试结果是表示模型认为WSI上的肿瘤区域的概率图。当然，我们可以使用一种滑动窗口的方式来预测所有补丁为肿瘤的概率。**但是由于WSI的大部分实际上是白色背景区域，所以这种滑动窗口的方式浪费了大量的计算**。取而代之的是，我们首先计算一个二元组织掩模，表示每个patch是组织或背景，然后只对组织区域进行肿瘤预测。如下所示

![](https://github.com/ilikewind/CAMELYON16/blob/master/docs/tissue_mask_wsi.png)

获取给定输入WSI的组织掩码图像，请使用下面的命令行

```shell
python CAMELYON16/camelyon16/bin/tissue_mask.py Tumor_001.tif Tumor_001_tissue.npy 6
```

### 概率热力图谱 (Probability map)

使用生成的组织掩码，利用训练好的CNN模型获得概率热力图：

```shell
python CAMELYON16/camelyon16/bin/probs_map.py /WSI_PATH/Test_001.tif /CKPT_PATH/best.ckpt /cnn_path/cnn.json /MASK_PATH/Test_001.npy /PROBS_MAP_PATH/Test_001.npy
```

利用该模型生成癌区域热力图谱，如下所示：

![](https://github.com/ilikewind/CAMELYON16/blob/master/docs/heatmap.png)

### 肿瘤定位 (Tumor localization)

我们使用非最大抑制(nms)算法，在给定概率图的情况下，获得每个检测到的肿瘤区域在level 0 级的坐标。

```shell
python CAMELYON16/camelyon16/bin/nms.py /PROBS_MAP_PATH/Test_001.npy /COORD_PATH/Test_001.csv
```

其中`/PROBS_MAP_PATH/`是您保存生成的概率图的位置，`/COORD_PATH/`是您希望将每个肿瘤区域生成的坐标以csv格式保存为0级的位置。有一个可选的命令——level，默认值为6，并确保它与对应的组织掩模和概率图使用的级别一致。

### FROC 评价 (FROC evalution)

利用每个检测WSI的肿瘤区域坐标，我们最终可以评估肿瘤定位的平均FROC评分。

```shell
python CAMELYON16/camelyon16/bin/Evaluation_FROC.py /TEST_MASK/ /COORD_PATH/
```

`/TEST_MASK/`是放置测试集的ground truth tif掩码文件的位置，`/COORD_PATH/`是保存生成的肿瘤坐标的位置。Evaluation_FROC.py基于Camelyon16组织者提供的评估代码，只做了少量修改。注意，正如Camelyon16组织者指出的，Test_049和Test_114被排除在评估之外。

## 贡献

