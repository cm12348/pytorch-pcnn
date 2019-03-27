2019.3.27更新
## 添加新数据集medical
需先运行dataset/medical.py
修改config.py中data = 'MED'
然后运行 "python main_med.py train"

2019.3.15

- From 赵立行
- 环境搭建: ubuntu 16.04 + CUDA 9.0 + python 2.7 + torch 0.4

## Notation
运行dataset/semeval.py会报错，原因是SemEval/train和SemEval/test下没有npy文件夹，只需要新建两个npy文件夹并各添加一个word_feautre.npy文件就行了。

## main_sem.py中
import了models，dataset。
models里是定义了的PCNN模型，dataset里是测试和训练数据

criterion = nn.CrossEntropyLoss
CrossEntropyLoss包含了The negative log likelihood loss和LogSoftmax.

## config.py中
设置了所有超参数的值，同时设置了多个训练样本的接口，当前使用的是“SEM”，之后可以添加其他样本！
