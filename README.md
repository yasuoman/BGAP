## Bottleneck Generalized Assignment Problems

**参考文献**：Mazzola J B, Neebe A W. Bottleneck generalized assignment problems[J]. Engineering Costs and Production Economics, 1988, 14(1): 61-65.

###  实现的总体思路：

####     1初始化相关的输入数据

####     2 根据cij与ck的关系建立新的TGBAP(K)问题

####     3 找到Z的下限，从这个下限开始往更大的数方向寻找

####     4 TGBAP(K)是否存在可行解，如果不存在的话，继续往下个数找，直到找到一个可行的TGBAP(K)

####     5 输出这个可行方案和对应的最小最大时间
