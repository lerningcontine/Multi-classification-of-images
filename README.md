# Multi-classification-of-images
### 图片多分类系统，这里是找到了一个奥特曼的数据集，针对这个数据集进行的一个分类模型
#### 主要分类有：["dijia", "jiake", "saiwen", "tailuo"] 四分类
该代码也可以运用你自己的数据，训练出你自己的模型

### 代码结构
主要是通过tensorflow构建了三个比较好的分类模型
模型的结构如下
```
data 放数据
models 目录下放置训练好的模型
results 目录下放置的是训练的训练过程的一些可视化的图
cnn.py: cnn模型训练文件
mobilenet.py: mobilenet模型训练文件
vgg16.py: vgg16模型训练文件
config.py 一些配置文件，包括加载数据，可视化图分析等
predict.py 运用训练好的模型进行数据的预测
requirements.txt 是本项目所需要的第三方库文件，其中最主要是tensorflow 需要对应 不然会出问题
```

