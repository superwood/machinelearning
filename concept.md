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

* 极大似然估计 (损失函数)、贝叶斯估计 (损失函数)、最大熵、 极大后验概率估计 (损失函数)、EM、结构风险、经验风险、概率模型、非概率模型
	
	* 经验风险: 模型对训练数据的平均损失(所有训练样本的损失加和的平均)
	* 极大似然估计(MLE)、极大后验估计(MAP)、贝叶斯估计(Bayes) 区别和联系
		* 背景: 已知训练集(抽样)D [已知发生事件集合D]。假设事件都是独立同分布，且其分布函数为\\(h(X ;\theta)\\)【\\(P(X;\theta)\\)】。已经发生事件的分布函数\\(L(X_1,X_2,\cdots ;\theta) =\prod\_{X_i \in D} h(X_i;\theta) \\)。当固定\\(\theta\\), \\(L(X_1,X_2,\cdots;\theta) \\)可以看成\\(P(X_1,X_2,\cdots|\theta)\\) 既\\(theta\\)下的条件概率。
		* 极大似然估计(MLE): 已知抽样D,假定模型\\(h(X ;\theta)\\)。 目前任务就是估计\\(h(X ;\theta)\\)中\\(\theta\\)参数 【LR中假定的模型为逻辑斯蒂回归函数 \\(h(x;\theta) = \frac{1}{1+e^{-x\cdot \theta}} \\)】。因为已有抽样D已经出现，有理由去找一个\\(\theta\\), 使得似然函数【条件概率/分布】\\(L(X_1,X_2,\cdots;\theta)\\)【\\( log P(D|\theta) \\)】最大。 这就是极大似然法
		$$\theta = argmax(  P(D|\theta))  \tag{g0}$$
		等价于log形式下
		$$\theta = argmax( log P(D|\theta))  \tag{g0}$$
		* 极大后验概率最大(MAP): 根据背景,已知抽样\\(D={X_1,X_2,X_3,\cdots}\\),假定模型\\(h(X ;\theta)\\) 【\\(P(X;\theta)\\)】则可以得到\\(L(D;\theta)= P(D|\theta) = \prod_{X_i \in D}  \\)。极大后验的思想是要找出已经存在D下的最大概率的\\(\theta_k\\),也既就是求条件分布函数 \\(P(\theta|D)\\)的最大值【连续的情况下】，或者是求\\(\theta_k\\),使得\\(P(\theta_k|D)\\)是最大值。
		
			根据贝叶斯公式
			$$ P(\theta|D) = \frac{P(D|\theta)\cdot P(\theta) }{P(D)} \tag{g1} $$
			MAP等价于求解 \\(\theta\\)
			$$\theta = argmax(\frac{P(D|\theta)\cdot P(\theta) }{P(D)}) \tag{g2}$$
			在目前估计算法中，\\(\theta\\)与\\(P(D)\\)无关，式子g2 等价于
			$$\theta = argmax(P(D|\theta)\cdot P(\theta)) \tag{g3}$$
			转成log形式, g3可以表示为
			$$\theta = argmax( log P(D|\theta)+log P(\theta)) \tag{g4}$$
			g4中\\(log P(D|\theta) \\)部分和式子 g0完全一样，MAP比EML多考虑了\\(\theta\\)本身的分布\\(P(\theta)\\)。
			
		*贝叶斯估计(Bayes):	 
	
-----------------------------------------------------------------------------------------
##信息概念

1. 信息量、信息熵、信息增益、信息增益率、GINI系数

	* 信息量

##正则化相关概念

1. 正则项类似
