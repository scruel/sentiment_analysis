# 基于 SVM 的情感分析
## 项目背景(赛题内容)
官方给出的数据集共包含6大类20个细粒度要素的情感倾向。根据标注的细粒度要素的情感倾向建立算法，对用户评论进行情感挖掘。

[比赛链接](https://challenger.ai/competition/fsauor2018?from=groupmessage)

## 说明
由于数据量较大, 在框架上采用 liblinear 而不是 libsvm, 通过预处理等操作后单模计算 marco F1 score, 验证集上的预测评分最高为 0.53274.

由于数据集文件较大, 不适合通过 git 提交, 部分数据集为内容截取. **因此直接执行时会报错, 补全训练集数据后才可正常执行.**

## 配置
- Intel Xeon E5-2680 v4
- 64 GB RAM

## 后续方向
- BERT
- RNN+ATT
- ...
