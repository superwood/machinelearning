<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>


## 概念
1. 常见损失函数
	
	1) 0-1损失函数(0-1 loss function)
	$$
		L(Y, f(X) ) = \begin{cases}  
			1, & Y \neq f(X) \\\ 
			0, & Y  = f(X)
		\end{cases} \tag{1.1}
	$$
	
	2) 平方损失函数 (quadratic loss function)
	$$
		L( Y, f(X) ) = (Y-f(x) )^2 \tag{1.2}
	$$
	
	3) 绝对损失函数(absolute loss function)
	$$
		L( Y, f(X) )  = | Y - f(X) | \tag{1.3}
	$$
	
	4) 对数损失函数( logarithmic loss function) 对数似然损失函数(log-likehood loss function)
	$$
		L(Y, P(Y|X))  = -logP(Y|X) \tag{1.4}
	$$

所有损失函数中， 1) 和4)的损失函数避开了与真实Y值直接的减法操作(既避免了数值性和有序性)。


* 对数损失函数各种解析理解
>解答
	$$
		L(Y, P(Y|X))  = -logP(Y|X) \tag{1.5}
	$$
	

	1) 概率视角理解损失。分类问题: 设单个样本X在Y分类(实际分类)时， 预测的概率为 \\(P(Y|X) \\). 考察损失值 \\( -logP(Y|X) \\), 当\\( P(Y|X) = 1\\) 既学习到的预测概率为1，则\\( -logP(Y|X)  = 0\\)没有损失。当\\( P(Y|X) = 0\\) 既学习到的预测概率为0，则\\( -logP(Y|X)  =  \infty \\) 损失为无穷。
	
	
		
	2) 概率模型中(包含二项分类)中，使用数损失函数，经验风险最小化的训练就是极大似然法。
	
	
	证明:
	* 极大似然法 
	
	不妨设 \\(Y = h_\theta(X) = P(Y|X) \\) 为模型目标函数(或者概率分布函数?),\\(\theta\\)为参数.很久极大似然思想(假设训练的样本是独立同分布的)，得到似然函数为
$$
	\begin{aligned}
	l(\theta) = \prod\_{i=0}^m h\_\theta(x\_{i})
	\end{aligned} \ x\_i 为第i个样本 \tag{1.6}
$$

	极大似然法求解\\(\theta\\) 参数，使得1.6式子(概率)值最大。由于在实际求解过程中,就有求解方便(连乘容易溢出,以及做求导、微分不方便)，而且求解\\(l(\theta)\\)最大等价于\\(log l(\theta)\\)最大。
$$
		\begin{aligned}
	log l(\theta) &= log \prod\_{i=0}^m h\_\theta(x\_{i}) \\\ &= \sum_{i=0}^m log h\_\theta(x\_{i}) \\\  
	\end{aligned}  \tag{1.7}
$$	
	式子1.7 是一个关于参数\\(\theta \\)的函数\\(log l(\theta)\\)， 求函数 \\(log l(\theta)\\)的极大值处的 \\(\theta \\)参数值
	* 对数损失函数求解法
	
	不妨设 \\(Y = h_\theta(X) = P(Y|X) \\) 为模型目标函数(或者概率分布函数?),\\(\theta\\)为参数。损失函数则如下
	$$
		\begin{aligned}
		lossfunction &= \sum\_{i=0}^m -logP(Y|X) \\\ &= \sum\_{i=0}^m  -log h\_\theta(x\_{i})
		\end{aligned} \tag{1.8}
	$$
	式子1.8 是一个关于参数\\(\theta \\)的函数\\(log l(\theta)\\)， 求函数 \\(log l(\theta)\\)的极小值处的 \\(\theta \\)参数值  等价于 式子 1.7的极大值的求解
	
	* 逻辑斯蒂回归的例子(二分类)
	
	在逻辑斯蒂回归中 \\(h_{\theta}\\)如下
	$$
			h\_\theta (X) = \begin{cases}  
			\sigma\_\theta(X) = \frac{1}{1+e^-\theta \cdot X} = \frac{e^\theta \cdot X}{1+e^\theta \cdot X}, & Y =  1 \\\ 
			1- \sigma\_\theta(X) = \frac{e^-\theta \cdot X}{1+e^-\theta \cdot X} = \frac{1}{1+e^\theta\cdot X}, & Y  = 0
		\end{cases} \tag{1.9}
	$$
	1.9分段函数可以重写成如下形式
	$$
		h\_\theta (X) = \sigma\_\theta(X)^Y \cdot (1 - \sigma\_\theta(X) )^{1-Y} \tag{1.10}
	$$
	或者
	$$
		h\_\theta (X) = Y \cdot \sigma\_\theta(X) + (1-Y)\cdot (1 - \sigma\_\theta(X)) \tag{1.11}
	$$
	
	将式子1.10 代入1.8得到
		$$
		\begin{aligned}
		lossfunction &= \sum\_{i=0}^m -logP(Y|X) \\\ &= \sum\_{i=0}^m  -log h\_\theta(x\_{i})\\\ &= -\sum_{i=0}^m Y_i \cdot log \sigma\_\theta(X_i) + (1-Y_i)log( 1- \sigma\_\theta(X_i) ) \\\ &= -\sum\_{i=0}^m Y_i log \frac{e^{\theta \cdot x_i}}{1+e^{\theta \cdot X_i}} + (1-Y_i)log \frac{1}{1+e^{\theta \cdot X_i}} \\\ &= \sum\_{i=0}^m Y_i \cdot \theta \cdot X_i - Y_i \cdot log (1+e^{\theta \cdot X_i})
		\end{aligned} \tag{1.12}
	$$
	采用牛顿法、随机梯度法、标准梯度下降法 求式子1.12的极小值即可。

* 监督学习模型的分类
	* 判别模型 VS 生成模型
	* 非概率模型 VS 概率模型
	* 参数模型 VS 非参数模型

* 极大似然估计 (损失函数)、贝叶斯估计 (损失函数)、 极大后验概率估计 (损失函数)、EM、结构风险、经验风险、概率模型、非概率模型
	
	* 经验风险: 模型对训练数据的平均损失(所有训练样本的损失加和的平均)

