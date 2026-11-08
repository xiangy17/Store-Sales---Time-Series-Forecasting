# Store-Sales---Time-Series-Forecasting

- ## 数据来源：
     kaggle：(https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data?select=train.csv)

- ## 目标：
     通过日期、商店和商品信息、促销和单位销售等数据,准确地预测在不同Favorita商店销售的数千件商品的单位销售额

- ## 使用：
     直接用program.py

- ## 项目流程精简总结：
     1. 数据读取与初探：

        加载训练集、测试集、油价、门店、交易和节假日等6个CSV文件，打印形状和头部数据。
     
     2. 可视化与初步清洗：
     
        绘制油价走势图。对油价数据使用后向填充（bfill）处理缺失值。

     3. 特征工程（合并多表）：
     
        重命名多表中的列名（如给节假日、门店的特征加上前缀），防止merge合并时出现重名冲突。以date和store_nbr为键，将训练集/测试集与节假日、油价、门店信息进行左连接（left join）。
     
     6. 缺失值处理：
     
        删掉测试集中没有的特征（Daily_transactions）以保证训练/测试集特征一致性。对油价缺失值使用前向填充（ffill）。 剩余缺失值（通常是节假日特征）统一填充为字符串'no'。

     5. 数据编码与预处理：
   
        将日期（date）字符串拼接转化为整型（如20170101）.对类别特征（节假日类型、城市、州、门店类型等）使用 pd.factorize() 进行标签编码。
  
     6. 模型训练与评估：

        划分训练/验证集（train_test_split）。分别训练决策树（DecisionTree）和随机森林（RandomForest）。使用平均绝对误差（MAE）作为评估指标。
  
     7. 预测与提交：
        
        使用表现更好的随机森林模型，对测试集进行预测，按id和sales格式化结果，导出为submission.csv。


