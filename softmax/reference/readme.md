参考https://zhuanlan.zhihu.com/p/668888063
navie softmax:
$$y_{i}=\frac{exp^{(x_{i})}}{\sum_{i} exp^{(x_{i})}} $$
safe softmax
$$y_{i}=\frac{exp^{(x_{i}-max)}}{\sum_{i} exp^{(x_{i}-max)}} $$
一维的safe softmax其实就是两次规约,两个element wise操作
1.规约遍历向量$x_{i}$，得到全局最大值max.

2.遍历向量$x_{i}$，对每个向量取${exp(x_{i}-max)}$

3.规约遍历计算数值和$S = \sum_{i}x_{i}$

4.遍历向量$x_{i}$，对每个向量做DIV S
online softmax