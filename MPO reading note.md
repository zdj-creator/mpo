# Maximum a Posteriori Policy Optimisation

## Abstract

​	We introduce a new algorithm for reinforcement learning called Maximum aposteriori Policy Optimisation (MPO) based on coordinate ascent on a relative entropy objective. We show that several existing methods can directly be related to our derivation. We develop two off-policy algorithms and demonstrate that they are competitive with the state-of-the-art in deep reinforcement learning. In particular, for continuous control, our method outperforms existing methods with respect to sample efficiency, premature convergence and robustness to hyperparameter settings while achieving similar or better final performance.

## Preliminaries

#### 1. 先验分布和后验分布

贝叶斯定理：
$$
p(\theta|X)=\frac{p(X|\theta)p(\theta)}{p(X)}
$$
p(X)作为归一化因子（边缘概率），与θ无关，故一般略去。上式写作：
$$
p(\theta|X) \propto  p(X|\theta)p(\theta)
$$
p(θ)为先验分布(prior)，反应的是对估计参数θ的先验知识。

p(X|θ)为基于先验分布（参数θ）的观测数据的概率分布。

p(θ|X)为后验分布(posterior)，即通过贝叶斯定理得到的最终的分析结果。

###### 

确定先验分布p(θ)的方法：

a) 有信息先验分布

可以使用共轭先验分布，如伯努利分布的共轭分布族为Beta分布，正太分布的共轭分布族仍为正态分布。

b) 无信息先验分布

包括laplace先验，invariant先验，Jeffreys'先验等。



本文中确定p(θ)的方式如下：

> In this paper we set p(θ) to a Gaussian prior around the current policy, i.e, p(θ) ≈ N ( μ = θi, Σ = Fθi λ ) , where θi are the parameters of the current policy distribution, Fθi is the empirical Fisher information matrix and λ is a positive scalar.

来自StatLect的定义：

The information matrix (also called Fisher information matrix) is the matrix of second cross-moments of the score vector. The latter is the vector of first partial derivatives of the log-likelihood function with respect to its parameters.

令$L(\theta;\xi)$表示ξ的似然函数，令$l(\theta;\xi)=ln[L(\theta;\xi)]$，则score vector为前者的梯度，即
$$
s(\theta)=\nabla_\theta l(\theta;\xi)
$$
Information matrix 是 score 的二阶交叉矩：
$$
I(\theta)=E_\theta [\nabla_\theta l(\theta;\xi) \nabla l(\theta;\xi)^T]
$$
由该定义可知，当 score 的期望为0时，即当
$$
E_\theta[\nabla_\theta l(\theta;\xi)]=0
$$
时，the information matrix 是 score 的协方差矩阵，即
$$
\begin{aligned}
I(\theta) &=\mathrm{E}_{\theta}\left[\nabla_{\theta} l(\theta ; \xi) \nabla_{\theta} l(\theta ; \xi)^{\top}\right] \\
&=\mathrm{E}_{\theta}\left[\left\{\nabla_{\theta} l(\theta ; \xi)-\mathrm{E}_{\theta}\left[\nabla_{\theta} l(\theta ; \xi)\right]\right\}\left\{\nabla_{\theta} l(\theta ; \xi)-\mathrm{E}_{\theta}\left[\nabla_{\theta} l(\theta ; \xi)\right]\right\}^{\top}\right] \\
&=\operatorname{Var}_{\theta}\left[\nabla_{\theta} l(\theta ; \xi)\right]
\end{aligned}
$$
另外，当 score 期望为0时，满足
$$
I(\theta)=-\mathrm{E}_{\theta}\left[\nabla_{\theta \theta}^{2} l(\theta ; \xi)\right]
$$
即 the information matrix 是 $l(\theta;\xi)$ 的Hesse矩阵的期望的负数。上式被称作 information equality。

###### ref:[1](https://www.zhihu.com/question/24261751/answer/2355943888)   [2](https://zhuanlan.zhihu.com/p/164377502)   [3](https://statlect.com/glossary/information-matrix#:~:text=The%20information%20matrix%20%28also%20called%20Fisher%20information%20matrix%29,Definition%20The%20information%20matrix%20is%20defined%20as%20follows)



#### 2. EM算法

###### 推导过程

a) Jensen不等式

若f是凸函数，x是随即变量，有
$$
E[f(x)] \geq f(E(x))
$$
特别地，当f为严格凸函数时，当且仅当$p(x=E(x))=1$时等号成立。

若f是凹函数，不等号取反。

b) EM算法

前提：已知数据总体的分布

对于m个独立样本，观测值为$$
x=\left(x^{(1)}, x^{(2)}, \ldots x^{(m)}\right)
$$，对应的隐含数据为$$
z=\left(z^{(1)}, z^{(2)}, \ldots z^{(m)}\right)
$$，此时$(x,z)$即为完全数据，样本的模型参数为θ。

由MLE可得
$$
\theta, z=\arg \max _{\theta, z} L(\theta, z)=\arg \max _{\theta, z} \sum_{i=1}^{m} \log \sum_{z^{(i)}} P\left(x^{(i)}, \quad z^{(i)} \mid \theta\right)
$$
显然，若对上式的各参数求偏导，形式会非常复杂。考虑如下形式的放缩。
$$
\begin{aligned}
\sum_{i=1}^{m} \log \sum_{z^{(i)}} P\left(x^{(i)}, z^{(i)} \mid \theta\right) &=\sum_{i=1}^{m} \log \sum_{z^{(i)}} Q_{i}\left(z^{(i)}\right) \frac{P\left(x^{(i)}, z^{(i)} \mid \theta\right)}{Q_{i}\left(z^{(i)}\right)} \\
& \geq \sum_{i=1}^{m} \sum_{z^{(i)}} Q_{i}\left(z^{(i)}\right) \log \frac{P\left(x^{(i)}, z^{(i)} \mid \theta\right)}{Q_{i}\left(z^{(i)}\right)}
\end{aligned}
$$
其中，$Q_i(z^{(i)})$为一未知分布，满足归一性。放缩用到Jensen不等式。

上式求得的是$L(\theta,z)$的下界，即
$$
L(\theta,z) \ge \sum_{i=1}^m 
E_{Q_i(z^{(i)})}[log\frac{P(x^{(i)},z^{(i)}|\theta)}{Q_i(z^{(i)})}]
$$
假设θ已给定，那么$logL(\theta)$的值仅取决于$Q_i(z)$和$p(x^{(i)},z^{(i)})$，根据Jensen不等式取等条件，当
$$
Q_{i}\left(z^{(i)}\right)=P\left(z^{(i)} \mid x^{(i)}, \quad \theta\right)
$$
时，ELBO取到等号。此即为E步(Expectation)。

此时，若我们能够极大化ELBO，则$L(\theta,z)$也被极大化，即
$$
\arg \max _{\theta} \sum_{i=1}^{m} \sum_{z^{(i)}} Q_{i}\left(z^{(i)}\right) \log \frac{P\left(x^{(i)}, z^{(i)} \mid \theta\right)}{Q_{i}\left(z^{(i)}\right)}
$$
此即为M步。



###### EM算法流程

Input: 观察数据$$
x=\left(x^{(1)}, x^{(2)}, \ldots x^{(m)}\right)
$$，联合分布$p(x,z|\theta)$，条件分布$p(z|x.\theta)$，极大迭代次数J。

STEP1: 随机初始化模型参数θ的初值$\theta^0$。

STEP2: for j = 1 to J:

E-STEP: 计算联合分布的条件概率期望：
$$
\left.Q_{i}\left(z^{(i)}\right):=P\left(z^{(i)} \mid x^{(i)}, \quad \theta\right)\right)
$$
M-STEP: 极大化L(θ)，得到θ：
$$
\theta:=\arg \max _{\theta} \sum_{i=1}^{m} \sum_{z^{(i)}} Q_{i}\left(z^{(i)}\right) \log P\left(x^{(i)}, \quad z^{(i)} \mid \theta\right)
$$
Output: 模型参数θ



###### EM算法收敛性

可以保证稳定收敛到局部最优，当目标函数$L(\theta,\theta^j)$为凸函数时可以收敛到全局最优值。



###### ref:[1](https://zhuanlan.zhihu.com/p/36331115)   [2](https://zhuanlan.zhihu.com/p/365641813)



#### 3. KL散度

1. 离散情形

   X,Y是两个离散随机变量，取值范围为$R_x,R_Y$，概率密度函数为$p_X(x),p_Y(y)$。设$R_{X} \subseteq R_{Y}$，则
   $$
   p_{X}(x) \neq 0 \Rightarrow p_{Y}(x) \neq 0
   $$
   则$p_X(x)$对$p_Y(x)$的KL散度为
   $$
   D_{K L}\left(p_{X,} p_{Y}\right)=-\sum_{x \in R_{X}} p_{X}(x) \ln \left(\frac{p_{Y}(x)}{p_{X}(x)}\right)
   $$
   $D_{K L}\left(p_{X}, p_{Y}\right)$是衡量$p_Y(y)$与参考分布$p_X(x)$的不相似度的。

2. 连续情形

   X,Y是两个连续随机变量，取值范围为$R_x,R_Y$，概率密度函数为$f_X(x),f_Y(y)$。满足
   $$
   \int_{A} f_{X}(x) d x \neq 0 \Rightarrow \int_{A} f_{Y}(x) d x \neq 0
   $$
   则$f_X(x)$对$f_Y(x)$的KL散度为
   $$
   D_{K L}\left(f_{X,} f_{Y}\right)=-\int_{x \in R_{X}} f_{X}(x) \ln \left(\frac{f_{Y}(x)}{f_{X}(x)}\right) d x
   $$

##### Reverse & Forward KL

<img src="https://picbed-1310993658.cos.ap-guangzhou.myqcloud.com/picture/image-20220704205745024.png" alt="image-20220704205745024"  />

![image-20220704210924310](https://picbed-1310993658.cos.ap-guangzhou.myqcloud.com/picture/image-20220704210924310.png)

###### Reverse KL: Zero-Forcing/Mode-Seeking

KL(q||p): 当最小化该式时，若p(x)=0，则q(x)=0，这意味着一种强制为0的独占模式。

###### Forward KL: Mass-Covering/Mean-Seeking

KL(p||q): 当最小化该式时，必须保证当p(x)≠0时，q(x)≠0，这意味着这是一种大规模覆盖的模式。



###### ref:[1](https://statlect.com/fundamentals-of-probability/Kullback-Leibler-divergence#:~:text=The%20KL%20divergence%20measures%20how%20much%20the%20distribution,The%20definition%20for%20continuous%20random%20variables%20is%20analogous.) [2]([Tuan Anh Le](https://www.tuananhle.co.uk/notes/reverse-forward-kl.html#:~:text=The reverse KL is called the exclusive KL,ϕ wherever there is some mass under p.)) [3](https://zhuanlan.zhihu.com/p/372835186)



#### 4. MAP

Maximum A Posteriori，最大后验估计

假设数据$x_1,x_2,...,x_n$是i.i.d.的一组抽样，$X=(x_1,x_2,...,x_n)$。则有
$$
\begin{aligned}
\hat{\theta}_{\mathrm{MAP}} &=\arg \max P(\theta \mid X) \\
&=\arg \min -\log P(\theta \mid X) \\
&=\arg \min -\log P(X \mid \theta)-\log P(\theta)+\log P(X) \\
&=\arg \min -\log P(X \mid \theta)-\log P(\theta)\\
&=\arg \max \ \log P(X|\theta)+ \log P(\theta)
\end{aligned}
$$


###### ref: [1]([聊一聊机器学习的MLE和MAP：最大似然估计和最大后验估计 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/32480810))



#### 5. Bellman Operator

定义$R_s^a$是在状态s做动作a得到奖励的期望。

定义$P_{s,s'}^a$是在状态s做动作a到达状态s'的概率。

有
$$
\mathbf{R}_{\pi}(s)=\sum_{a \in \mathcal{A}} \pi(a \mid s) \cdot \mathcal{R}_{s}^{a}
$$
表示对于这个策略$\pi$，在状态s下可能获得的“奖赏的期望”的期望（第一个期望来自环境的不确定性，第二个期望来自策略的不确定性）。

用$P_\pi(s,s')$表示对于这个策略$\pi$，从状态s转移到s'的概率：
$$
\mathbf{P}_{\pi}\left(s, s^{\prime}\right)=\sum_{a \in \mathcal{A}} \pi(a \mid s) \cdot \mathcal{P}_{s, s^{\prime}}^{a}
$$
用$\mathbf{R}_{\pi}$表示向量$\left[\mathbf{R}_{\pi}\left(s_{1}\right), \mathbf{R}_{\pi}\left(s_{2}\right), \ldots, \mathbf{R}_{\pi}\left(s_{n}\right)\right]$，用$P_\pi$表示矩阵$\left[\mathbf{P}_{\pi}\left(s_{i}, s_{i^{\prime}}\right)\right], 1 \leq i, i^{\prime} \leq n$。



没啥好说的，看知乎吧



下面给出几个重要结论：

1. $B_\pi$和$B_*$都含有唯一的不动点。

2. $B_\pi$和$B_*$都是单调的。
   $$
   \begin{aligned}
   &\mathbf{v}_{1} \leq \mathbf{v}_{2} \Rightarrow \mathbf{B}_{\pi} \mathbf{v}_{1} \leq \mathbf{B}_{\pi} \mathbf{v}_{2} \\
   &\mathbf{v}_{1} \leq \mathbf{v}_{2} \Rightarrow \mathbf{B}_{*} \mathbf{v}_{1} \leq \mathbf{B}_{*} \mathbf{v}_{2}
   \end{aligned}
   $$

3. Greedy Policy from Optimal VF(Value Function) is an Optimal Policy

   当得到最优值函数$v_*$时，选择的是贪心策略，即
   $$
   \pi_*=G(v_*)
   $$
   则此时有
   $$
   v_{\pi_*}=v_*
   $$
   

A. Policy Evaluation

对于任何策略$\pi$，我们可以得到$B_\pi$，那么策略评估就是在找$B_\pi$的不动点。

根据Contraction Mapping Theorem，对任何值函数v都有$\lim _{N \rightarrow \infty} \mathbf{B}_{\pi}^{N} \mathbf{v}=\mathbf{v}_{\pi}$，这保证了我们可以从一个随机的起点出发，不断按照Bellman方程更新值函数v，从而收敛到$v_\pi$。

B. Policy Improvement

令$\pi_{k+1}$满足
$$
\pi_{k+1}=G(v_{\pi_k})
$$
有
$$
\mathbf{v}_{\pi_{\mathrm{k}+1}}=\lim _{N \rightarrow \infty} \mathbf{B}_{\pi_{k+1}}^{N} \mathbf{v}_{\pi_{\mathrm{k}}} \geq \mathbf{v}_{\pi_{\mathrm{k}}}
$$
C. Policy Iteration

当$v_{\pi_{k+1}}=v_{\pi_{k}}$时，则有$v_*=v_{\pi_k}$。

D. Value Iteration

绕开$B_\pi$，直接使用$B_*$参与运算。

$B_*$有唯一的不动点$v_*$满足$\mathbf{B}_{*} \mathbf{v}_{*}=\mathbf{v}_{*}$，而$B_*v$又是一个单调递增的运算，因为对于任意的值函数v，有
$$
\lim _{N \rightarrow \infty} \mathbf{B}_{*}^{N} \mathbf{v}=\mathbf{v}_{*}
$$

ref: [1]([用Bellman算子理解动态规划 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/137980157))



#### 6. Cholesky 分解

当A是一个实对称正定矩阵时，可以分解为一个对角元为正数的下三角矩阵L和它的转置（也是一个上三角矩阵）$L^T$。

<img src="https://picbed-1310993658.cos.ap-guangzhou.myqcloud.com/picture/v2-7fb9749b6ff22cc6d294e4553b4446fc_720w.jpg" alt="img" style="zoom:67%;" />

ref: [1](https://blog.csdn.net/acdreamers/article/details/44656847) [2](https://www.statlect.com/matrix-algebra/Cholesky-decomposition)



#### 7. Lagrange 对偶函数

$$
g(\lambda,\nu)=\inf_{x}L(x,\lambda,\nu)
$$

ref:[1]([(9条消息) 【优化】对偶上升法(Dual Ascent)超简说明_shenxiaolu1984的博客-CSDN博客_对偶上升法](https://blog.csdn.net/shenxiaolu1984/article/details/78175382))



#### 8. 多维高斯分布概率密度





ref: [1]([多元/多维高斯/正态分布概率密度函数推导 (Derivation of the Multivariate/Multidimensional Normal/Gaussian Density) - 凯鲁嘎吉 - 博客园 (cnblogs.com)](https://www.cnblogs.com/kailugaji/p/15542845.html))



#### 9.  Likelihood Ratio Gradient





ref:[1]([The likelihood-ratio gradient — Graduate Descent (timvieira.github.io)](https://timvieira.github.io/blog/post/2019/04/20/the-likelihood-ratio-gradient/))



#### 10. On-policy and Off-policy



ref:[1]([强化学习中的奇怪概念(一)——On-policy与off-policy - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/346433931))



#### 11. Retrace

Off-policy 中由于存在不同策略采样和更新，故在

通用算子$R$:

![[公式]](https://www.zhihu.com/equation?tex=R+Q%28x%2Ca%29+%3A%3D+Q%28x%2Ca%29+%2B+E_%5Cmu%5B%5Csum_%7Bt+%5Cgeq+0%7D%7B%5Cgamma%5Et%28%5CPi_%7Bs%3D1%7D%5E%7Bt%7D%7Bc_s%7D%29%28r_t+%2B+%5Cgamma+E_%5Cpi+Q%28x_%7Bt%2B1%7D%2C%5Ccdot%29-Q%28x_t%2C+a_t%29%29%7D%5D%2C+%284%29)

1. Importance Sampling

   ![[公式]](https://www.zhihu.com/equation?tex=c_s+%3D+%5Clambda%5Ccdot%5Cfrac%7B%5Cpi%28a_s%7Cx_s%29%7D%7B%5Cmu%28a_s%7Cx_s%29%7D%5C%5C)

2. off-policy $Q^\pi(\lambda) \  and \  Q^*(\lambda)$

   ![[公式]](https://www.zhihu.com/equation?tex=c_s+%3D+%5Clambda%5C%5C)

3. $TB(\lambda)$ (Tree-backup)

   ![[公式]](https://www.zhihu.com/equation?tex=c_s%3D%5Clambda+%5Cpi%28a_s%7Cx_s%29)

4. $Retrace(\lambda)$

   ![[公式]](https://www.zhihu.com/equation?tex=c_s+%3D+%5Clambda+%5Ccdot+min%5Cbigg%281%2C%5Cfrac%7B%5Cpi%28a_s%7Cx_s%29%7D%7B%5Cmu%28a_s%7Cx_s%29%7D%5Cbigg%29%5C%5C)

ref:[1]([【Typical RL 19】Retrace - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/419917893)) [2](https://zhuanlan.zhihu.com/p/56391653)



#### 12. State Visitation Distribution





ref:[1](https://www.alexirpan.com/rl-derivations/#state-visitation--occupency-measure)



#### 13. Categorical Distribution



ref:[1](https://zhuanlan.zhihu.com/p/59550457)

## Maximum a Posteriori Policy Optimisation

MPO 是一种 off-policy 的算法。

> It exhibits the scalability, robustness and hyperparameter insensitivity of on-policy algorithms, while offering the data-efficiency of off-policy, value-based methods.
>
> We leverage the fast convergence properties of EM-style coordinate ascent by alternating a **nonparametric data-based E-step** which re-weights state-action samples, with a **supervised, parametric M-step** using deep neural networks.
>
> In contrast to typical off-policy value-gradient algorithms, the new algorithm does not require gradient of the Q-function to update the policy. Instead it uses samples from the Q-function to compare different actions in a given state. And subsequently it updates the policy such that better actions in that state will have better probabilities to be chosen.

假设事件$O=1$表示选择使奖励最大的action，或者在RL任务中取得成功，则有
$$
p(O=1 \mid \tau) \propto \exp \left(\sum_{t} r_{t} / \alpha\right)
$$
其中α为温度系数。

我们定义如下似然函数的下界作为优化目标函数：
$$
\begin{aligned}
\log p_{\pi}(O=1) &=\log \int p_{\pi}(\tau) p(O=1 \mid \tau) d \tau \geq \int q(\tau)\left[\log p(O=1 \mid \tau)+\log \frac{p_{\pi}(\tau)}{q(\tau)}\right] d \tau \\
&=\mathbb{E}_{q}\left[\sum_{t} r_{t} / \alpha\right]-\operatorname{KL}\left(q(\tau) \| p_{\pi}(\tau)\right)=\mathcal{J}(q, \pi),
\end{aligned}
$$
其中，$q(\tau)$是一个辅助分布，$\mathcal{J}(q,\pi)$是原优化目标的ELBO。

EM算法可以用于逐步优化$\mathcal{J}(q,\pi)$，其中，E-STEP通过改变q增大J，传统的EM算法一般通过使用采样对轨迹重新加权或者通过本地轨迹优化来进行策略搜索。本文通过使用 off-policy deep RL 和值函数逼近来进行E-STEP。M-STEP通过监督学习E-STEP中的重新加权的状态-动作采样来更新参数化策略。



#### Policy Improvement

显然，$q(\tau)$可以写作
$$
q(\tau)=p\left(s_{0}\right) \prod_{t>0} p\left(s_{t+1} \mid s_{t}, a_{t}\right) q\left(a_{t} \mid s_{t}\right)
$$
则$J(q,\theta)$可写作
$$
\mathcal{J}(q, \boldsymbol{\theta})=\mathbb{E}_{q}\left[\sum_{t=0}^{\infty} \gamma^{t}\left[r_{t}-\alpha \operatorname{KL}\left(q\left(a \mid s_{t}\right) \| \pi\left(a \mid s_{t}, \boldsymbol{\theta}\right)\right)\right]\right]+\log p(\boldsymbol{\theta})
$$
其中，$logp(\theta)$是策略参数的先验分布，可由MAP获得。

由上式可以定义正则化的Q函数：
$$
Q_{\theta}^{q}(s, a)=r_{0}+\mathbb{E}_{q(\tau), s_{0}=s, a_{0}=a}\left[\sum_{t \geq 1}^{\infty} \gamma^{t}\left[r_{t}-\alpha \operatorname{KL}\left(q_{t} \| \pi_{t}\right)\right]\right]
$$
其中$\left.\mathrm{KL}\left(q_{t} \| \pi_{t}\right)=\mathrm{KL}\left(q\left(a \mid s_{t}\right)\right) \| \pi\left(a \mid s_{t}, \boldsymbol{\theta}\right)\right)$。

注意到，通过q优化J等价于解决一个期望回报RL问题，后者的增广奖励$\tilde{r}_{t}=r_{t}-\alpha \log \frac{q\left(a_{t} \mid s_{t}\right)}{\pi\left(a_{t} \mid s_{t}, \theta\right)}$。

对于无信息先验分布的$p(\theta)$，在附录A中证明了本文算法可以单调改进。



#### E-STEP

在第i次迭代中，$J(q,\theta)$是在给定$\theta=\theta_i$时关于q的局部最优。

首先设$q=\pi_{\theta_i}$，则未正则化的Q函数为：
$$
Q_{\boldsymbol{\theta}_{i}}^{q}(s, a)=Q_{\boldsymbol{\theta}_{i}}(s, a)=\mathbb{E}_{\tau_{\pi_{i}}, s_{0}=s, a_{0}=a}\left[\sum_{t}^{\infty} \gamma^{t} r_{t}\right]
$$
实际中，我们从 off-policy data 中估计$Q_{\boldsymbol{\theta}_{i}}$。

给定$Q_{\boldsymbol{\theta}_{i}}$时，我们首先通过正则化Bellman算子
$$
T^{\pi, q}=\mathbb{E}_{q(a \mid s)}\left[r(s, a)-\alpha \operatorname{KL}\left(q \| \pi_{i}\right)+\gamma \mathbb{E}_{p\left(s^{\prime} \mid s, a\right)}\left[V_{\boldsymbol{\theta}_{i}}\left(s^{\prime}\right)\right]\right]
$$
对$Q_{\boldsymbol{\theta}_{i}}$进行拓展，再优化 "one-step" KL 正则化目标：
$$
\begin{aligned}
\max _{q} \overline{\mathcal{J}}_{s}\left(q, \theta_{i}\right) &=\max _{q} T^{\pi, q} Q_{\boldsymbol{\theta}_{i}}(s, a) \\
&=\max _{q} \mathbb{E}_{\mu(s)}\left[\mathbb{E}_{q(\cdot \mid s)}\left[Q_{\boldsymbol{\theta}_{i}}(s, a)\right]-\alpha \operatorname{KL}\left(q \| \pi_{i}\right)\right]
\end{aligned}
$$
其中，因$V_{\boldsymbol{\theta}_{i}}(s)=\mathbb{E}_{q(a \mid s)}\left[Q_{\boldsymbol{\theta}_{i}}(s, a)\right]$，故$Q_{\boldsymbol{\theta}_{i}}(s, a)=r(s, a)+\gamma V_{\boldsymbol{\theta}_{i}}(s)$。

最大化上式可得
$$
q_i=arg \ max \bar{J}(q,\theta_i)
$$
上式将$Q_{\theta_i}$视为常数。实践中，我们选择从 replay buffer 中采样的$\mu_q$作为平稳分布。

##### Constrained E-STEP

将正则项转化为硬约束：
$$
\begin{aligned}
&\max _{q} \mathbb{E}_{\mu(s)}\left[\mathbb{E}_{q(a \mid s)}\left[Q_{\theta_{i}}(s, a)\right]\right] \\
&\text { s.t. } \mathbb{E}_{\mu(s)}\left[\operatorname{KL}\left(q(a \mid s), \pi\left(a \mid s, \boldsymbol{\theta}_{i}\right)\right)\right]<\epsilon
\end{aligned}
$$
若选择显式参数化$q(a|s)$，则与TRPO/PPO类似。

本文选择一个非参数化表达的$q(a|s)$，来自于对s进行动作采样。为了能够在状态空间泛化，我们在 M-STEP 中拟合一个参数化策略（监督学习）。

##### Non Parametric Vatiational Distribution

优化上式可得闭式解：
$$
q_{i}(a \mid s) \propto \pi\left(a \mid s, \boldsymbol{\theta}_{i}\right) \exp \left(\frac{Q_{\theta_{i}}(s, a)}{\eta^{*}}\right)
$$
其中$η^*$可通做最小化下式获得：
$$
g(\eta)=\eta \epsilon+\eta \int \mu(s) \log \int \pi\left(a \mid s, \boldsymbol{\theta}_{i}\right) \exp \left(\frac{Q_{\theta_{i}}(s, a)}{\eta}\right) d a d s
$$


#### M-STEP

给定E-STEP中的$q_i$，我们可以优化关于$\theta$的J，即
$$
\max _{\boldsymbol{\theta}} \mathcal{J}\left(q_{i}, \theta\right)=\max _{\boldsymbol{\theta}} \mathbb{E}_{\mu_{q}(s)}\left[\mathbb{E}_{q(a \mid s)}[\log \pi(a \mid s, \boldsymbol{\theta})]\right]+\log p(\boldsymbol{\theta})
$$
这是一个加权的MAP问题，采样是由 E-STEP 中的变分分布加权。

由于这本质上是一个有监督的学习步骤，我们可以选择任何策略表示与任何先验结合进行正则化。

> In this paper we set $p(\theta)$ to a Gaussian prior around the current policy, i.e, $p(\boldsymbol{\theta}) \approx \mathcal{N}\left(\mu=\boldsymbol{\theta}_{i}, \Sigma=\frac{\underline{F}_{\boldsymbol{\theta}_{i}}}{\underline{\lambda}}\right)$ , where $\theta_i$ are the parameters of the current policy distribution, $F_{\theta_i}$ is the empirical Fisher information matrix and λ is a positive scalar.



则上式可以转化为
$$
\max _{\pi} \mathbb{E}_{\mu_{q}(s)}\left[\mathbb{E}_{q(a \mid s)}[\log \pi(a \mid s, \boldsymbol{\theta})]-\lambda \operatorname{KL}\left(\pi\left(a \mid s, \boldsymbol{\theta}_{i}\right), \pi(a \mid s, \boldsymbol{\theta})\right)\right]
$$
重写为硬约束
$$
\begin{array}{r}
\max _{\pi} \mathbb{E}_{\mu_{q}(s)}\left[\mathbb{E}_{q(a \mid s)}[\log \pi(a \mid s, \boldsymbol{\theta})]\right] \\
\text { s.t. } \mathbb{E}_{\mu_{q}(s)}\left[\operatorname{KL}\left(\pi\left(a \mid s, \boldsymbol{\theta}_{i}\right), \pi(a \mid s, \boldsymbol{\theta})\right)\right]<\epsilon
\end{array}
$$
该附加约束将过拟合风险最小化，使得策略的泛化性能更好。

在 E-STEP 中，我们使用 reverse, mode-seeking KL ，在 M-STEP 中，我们使用 forward, moment-matching KL，后者目的是减少参数化策略的熵崩溃趋势。



#### Policy Evaluation

> PE在RL中的作用是预测给定“状态实例”或“状态-动作实例”的“折扣累积回报的期望”。

我们需要通过一个稳定的policy evaluation operator来得到$Q_\theta(s,a)$的参数化表达，本文使用了Retrace算法。

本文用神经网络拟合Q函数$Q_{\theta_i}(s,a,\phi)$，方式是最小化平方项损失：
$$
\begin{aligned}
&\min _{\phi} L(\phi)=\min _{\phi} \mathbb{E}_{\mu_{b}(s), b(a \mid s)}\left[\left(Q_{\theta_{i}}\left(s_{t}, a_{t}, \phi\right)-Q_{t}^{\mathrm{ret}}\right)^{2}\right], \text { with } \\
&Q_{t}^{\mathrm{ret}}=Q_{\phi^{\prime}}\left(s_{t}, a_{t}\right)+\sum_{j=t}^{\infty} \gamma^{j-t}\left(\prod_{k=t+1}^{j} c_{k}\right)\left[r\left(s_{j}, a_{j}\right)+\mathbb{E}_{\pi\left(a \mid s_{j+1}\right)}\left[Q_{\phi^{\prime}}\left(s_{j+1}, a\right)\right]-Q_{\phi^{\prime}}\left(s_{j}, a_{j}\right)\right] \\
&c_{k}=\min \left(1, \frac{\pi\left(a_{k} \mid s_{k}\right)}{b\left(a_{k} \mid s_{k}\right)}\right)
\end{aligned}
$$
其中$Q_{\phi'}(s,a)$表示目标Q网络的输出，网络参数用$\phi'$表示，该参数在每次M-STEP后从$\phi$中复制。

**$b(a|s)$**表示一个任意行为策略的概率分布。b可以由replay buffer中的动作概率给出。

> We truncate the infinite sum after N steps by bootstrapping with $Q_{\phi'}$ (rather than considering a λ return). 



ref:[1]([4.1 —— 策略评估（Policy Evaluation） - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/537091925))

## Appendix

#### A Proof of Monotonic Improvement For the KL-Regularized Policy Optimization Procedure

> In this section we prove a monotonic improvement guarantee for KL-regularized policy optimization
> via alternating updates on π and q under the assumption that the prior on θ is uninformative.

##### A.1 Regularized Reinforcement Learning





##### A.2 Regularized Joint Policy Gradient





## Pseudocode

![image-20220703182940214](https://picbed-1310993658.cos.ap-guangzhou.myqcloud.com/picture/image-20220703182940214.png)

![image-20220703183216075](https://picbed-1310993658.cos.ap-guangzhou.myqcloud.com/picture/image-20220703183216075.png)

![image-20220703183228603](https://picbed-1310993658.cos.ap-guangzhou.myqcloud.com/picture/image-20220703183228603.png)

![image-20220703183247776](https://picbed-1310993658.cos.ap-guangzhou.myqcloud.com/picture/image-20220703183247776.png)





