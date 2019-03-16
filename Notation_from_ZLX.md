2019.3.15

- From 赵立行
- 环境搭建: ubuntu 16.04 + CUDA 9.0 + python 2.7 + torch 0.4

## Notation
运行dataset/semeval.py会报错，原因是SemEval/train和SemEval/test下没有npy文件夹，只需要新建两个npy文件夹并各添加一个word_feautre.npy文件就行了。