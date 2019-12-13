
#1、数据
数据尺寸：不形变，缩放不好，resize done
数据平衡：done
数据预处理：平衡分布 是否灰度图，rgb特征丰富 done 
数据增强：随机裁剪和翻转 done

#2、模型
baseline：resnet50
(1)resnet浅层特征 done 效果不好
(2)注意力 空间，准备试试只用通道注意力 done 
(3)模型集成，能解决0，1混淆的问题，但是和2混淆就难解决。 二阶段级联分类器
* 分类01，99 
  分类02，99  然后结合, 如果1&2 取置信度高的
* 分类01，其他 -> 分类02，其他
然后后处理
(4)可变卷积，增加有效采样点
(5)增大感受野
(6)增大模型容量，使用注意力模型时，数据量大的时候损失波动很大。done
(7)更好的特征提取网络，resnext ing
(8)改进损失，focal loss用于多分类任务 done

#3、细粒度分类
(1)BCNN done

#4、分析错误样本
可视化特征与激活图 grad-cam


# 实验数据

三分类

| model | val_loss | val_acc | test(32130) | testA(3257) | testB(200) | testC(56) | description |
| --- | --- | --- | --- | --- | --- | --- | --- |
| resnet50(224x224+train) | 0.176 | 0.939 | 0.936 | 0.863 | 0.705 | 0.893 | baseline |
| resnet50b(block3+train) | 0.495 | 0.858 | 0.853 | 0.202 | 0.304 | **0.964** | 偶然性 |
| resnet50c(keep_size+train) | 0.108 | **0.996** | **0.955** | **0.896** | 0.600 | **0.964** | # |
| resnet50d(ks+all) | 0.003 | **0.999** | **0.999** | 0.798 | 0.515 | **0.946** |
| resnet50(ks+cbam+all) | 0.498 | 0.955 | 0.956 | 0.460 | 0.385 | 0.625 |
| resnet50(ks+se+all) | 0.429 | 0.977 | 0.978 | **0.896** | 0.600 | **0.964** | 泛化性能差,没有学到通用特征 |
| resnet50(ks+se+all+fl) | 0.038 | 0.997 | 0.990 | **0.889** | 0.575 | **0.911** | 增强hard样本损失,能提高hard样本分对置信度 |
| resnet50(ks+all+fl) | 0.001 | 0.998 | 0.998 | **0.891** | 0.580 | **0.964** | 增强hard样本损失,能提高hard样本分对置信度 |
| resnet50(ks+all+fl+aug) | 0.001 | 0.998 | 0.998 | **0.926** | 0.705 | 0.750 | 增强hard样本损失,能提高hard样本分对置信度 |
| resnet20(ks+se+all+fl+aug) | 1.988 | 0.827 | # | 0.6875 | # | # |
| resnet101(ks+se+train) | 1.086 | 0.836 | 0.920 | **0.830** | **0.840** | 0.500 | 增大模型容量 |
| resnet101(ks+all+fl+aug) | 0.023 | 0.910 | # | 0.928 | 0.685 | 0.589 | 能提高1个点 |
| inception_resnet(ks+train+fl+aug+320,80) | 0.007 | 0.988 | # | 0.971 | 0.750 | 0.929 | testA提高 |
| inception_resnet(ks+train+fl+aug+480,80) | 0.007 | 0.988 | # | 0.971 | 0.750 | 0.929 | testA提高 |
| xception(ks+train+fl+aug+240,40) | 0.037 | **0.863** | # | **0.876** | 0.670 | **0.911** | testA提高 |
| xception(ks+train+fl+aug+320,80) | 0.013 | **0.984** | # | **0.971** | 0.745 | **0.946** | testA提高 |
| xception(ks+train+fl+aug+400,80) | 0.014 | 0.947 | # | 0.961 | 0.745 | 0.839 | # |
| xception(ks+train+fl+aug+480,80) | 0.018 | 0.938 | # | 0.951 | 0.730 | 0.768 | # |
| snet | # | # |
| snetplus | # | # |
| BCNN | 0.714 | 0.814 | 
| inception_resnet(ks+train+fl+aug+320,80+gray) | 0.031 | 0.874 | # | 0.894 | 0.660 | 0.589 | # |

二分类

| model | train_loss | train_acc | val_loss | val_acc | 
| --- | --- | --- | --- | --- |
|resnet50_1|0.008|0.952|0.005|0.977|
|resnet50_2|0.006|0.966|0.036|0.827|


| model | testA | testB | testC | des |
| --- | --- | --- | --- | --- |
| resnet50(ks+all+fl+aug) | 0.926 | **0.705** | 0.750 | # |
| 并行集成(12) | **0.941** | 0.665 | **0.821** | testa提高1-2个点,b降低, c提高 |
| 二阶段级联(1&2) | 0.939 | 0.665 | 0.821 | # |
| 二阶段级联(2&1) | 0.942 | 0.660 | 0.821 | # |


# 目前问题
 
testA，有部分1、2混淆，精度接近90
testB，主要是1、2混淆，精度有待提高。
解决1、2混淆问题，集成模型。

# 结论
1、全数据集相对半数据集，没有提升，说明数据多样性不够
2、预处理后提升不大，数据增强有效，不形变训练有效
3、注意力理论有效
4、模型集成有效
5、改进损失效果不大，增强骨干网有效。
6、只用浅层特征效果不好。
7、使用细粒度分类效果不明显，应该是监督信息不够。

