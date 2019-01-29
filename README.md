# bilstm-crf_ner
bilstm+crf algorithm

数据集：存放在data文件夹中

    1998年人民日报数据集
 
运行代码：存放在code文件夹中

1.  数据处理：

    data_preprocess.py  对数据进行实体标签化及one-hot编码等处理
  
    执行命令： python data_preprocess.py 
  
2. .py代码注解：

   （1）Batch.py: 训练批处理
  
    （2）bilstm_crf.py: bilstm_crf算法模型
  
    （3）utils.py: 训练预测及评估指标存储模块
  
    （4）train.py: 功能块封装接口，训练测试执行
  
3. 运行代码命令：

   python train.py
