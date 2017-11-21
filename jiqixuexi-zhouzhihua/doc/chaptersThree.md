<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

1. 试分析在什么情况下，在3.2式子中不比考虑偏置项b

>解答：

*  式子 3.2  \\(f(x) = w^T x + b \\)

	
命题： \\((x_1,y_1), (x_2,y_2) \\) 是式子3.2的解， 则 \\( (x_2-x_1, y_2-y_1) \\)是 \\(f(x) = w^T x \\)的解


因此，训练集合任意取一项令其为 \\((x_0, y_0)\\), 重新构成训练集\\( D' = \lbrace  (x_i-x_0, y_i-y_0)  | i \neq  0  \rbrace  \\) 

即可。

---------------------------------------------------------------

2. 试证明，对于参数w，对率回归（logistics回归）的目标函数（3.18）是非凸的，但其对数似然函数（3.27）是凸的。

>解答：

* 目标函数  \\( y = \frac{1}{1+e^{-(w^Tx+b) }} \\)



命题: 设\\( f(w) 是定义在非空的开集合 D \in R^n 是二次可微， 则f(w)是凸函数的充要条件是在任意点 w_0 \in D , f(w)的Hessian 矩阵 \\)

令\\( z = w^Tx+b, 则 y =  \frac{1}{1+e^{-z }} \\)

$$ \frac{ \mathrm{d}y}{dw} =  \frac{-1\times (e^{-z}) \times (-x) }{(1+e^{-z})^2} 
\\\\ = \frac{1}{1+e^{-z}} \times ( 1 - \frac{1}{1+e^{-z} }) \times x 
\\\\ = y(1-y)x  = xy - xy^2
$$

继续计算二次导数

$$\frac{ \mathrm{d}}{dw^T} (\frac{\mathrm{d}y}{dw}) = \frac{ \mathrm{d}}{dw^T} (xy - xy^2)  = x \frac{ \mathrm{d}y}{dw} -2xy \frac{ \mathrm{d}y}{dw} = x(1-2y)(xy-xy^2)
\\\\ = x^2y(1-2y)(1-y)
$$

多维度情况下 \\( x^2  = x x^T \\)形成的矩阵是半正定, 对任意向量x成立。现在考虑 $$y(1-2y)(1-y) = y(2y-1)(y-1)  \ \ \ y \in (0 , 1)$$  

明显在\\( y \in ( \frac{1}{2} , 1) \\)  时候 \\( \frac{ \mathrm{d}}{dw^T} (\frac{\mathrm{d}y}{dw}) < 0 \\), 因此 \\( y = \frac{1}{1+e^{-(w^Tx+b) }} \\) 非凸。


* 对数似然函数 \\( l(\beta) = \sum_{i=1}^{m} \left(  -y_i \beta^T  \hat{x_i}  + ln(1+ e^{\beta^T \hat{x_i}}) \right) \\)

$$\frac{ \mathrm{d}l(\beta)}{d\beta} = \sum_{i=1}^m ( -y_i \hat{x_i } + \frac{\hat x_i e^{\beta^T \hat{x_i}} }{1+e^{\beta^T \hat {x_i}}} )  = \hat{x_i} - \frac{\hat x_i}{ 1+e^{\beta^T \hat {x_i}} } $$

$$\frac{ \mathrm{d}}{d\beta} (\frac{\mathrm{d}l(\beta)}{d\beta}) = \sum_{i=1}^m   = \frac{\hat x_i {\hat x_i}^T ( 1+e^{\beta^T \hat {x_i}  }) }{ (1+e^{\beta^T \hat {x_i} })^2 }$$

\\(1+e^{\beta^T \hat {x_i}   } \gt 0 \\), 所以
$$\frac{ \mathrm{d}}{d\beta} (\frac{\mathrm{d}l(\beta)}{d\beta})  \gt 0 $$
得证。
  
----------------------------------------------------------------------------------------
3. 编程实现对率回归，并给出西瓜数据集3.0α上的结果

	See [code] 