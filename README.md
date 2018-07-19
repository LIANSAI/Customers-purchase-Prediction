# Customers-purchase-Prediction-Competition-
招商消费金融场景下的用户购买预测 34th 方案

思路：

特征拆分为两个子群，子群一以统计特征为主，子群二以对业务理解所捏造的特征为主。为了让两个子群的成绩都足够好，用Level1点击分布的特征作为公共特征群

模型为 LGB,XGB 双模型rank加权融合
