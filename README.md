# What is this code about?

The code examines the run of binary mechanism and factorization based mechanism introduced in the paper (https://arxiv.org/abs/2202.11205). 
The two algorithms are executed $10^{6}$ times each and then the reported result is the average of all these executions. The reason to 
run the algorithm so many times is to ensure high confidence in the error bound because the stochasticity of samples (and the random 
bits used by python packages) might lead to erratic output. User can change the parameter repitition, but it is advisable to keep the 
parameter as high as possible.

# What are these algorithms?
The two algorithms at the center of this empirical evaluation are binary tree mechanism (henceforth, binary mechanism) and factorization mechanism.

## Binary mechanism
The binary mechanism is the privatized version of Bentley-Saxe transformation (https://www.sciencedirect.com/science/article/abs/pii/0196677480900152)
For a function $f$ of interest, in binary mechanism, we generate a binary tree. The leaf nodes of the tree are the evaluation of the function 
on every updates while every subnode contains the evaluation of the function on the leaves of the corresponding subtree. In case of binary counting, 
the function is just the sum, so the leaf nodes stores the value of the updates while any subnode computes the partial sum of the updates made by the 
leaves in the subtree corresponding to the node. 

The privatized version of the binary mecahnism adds appropriately scaled Gaussian noise (for approximate-DP) or Laplace noise (for pure-DP). 

## Factorization mechanism
The factorization mechanism proposed by Fichtenberger, Henzinger, and Upadhyay (https://arxiv.org/abs/2202.11205) takes the linear algebraic view of 
the continual counting. In particular, the computation can be seen as the matrix-vector product between the matrix $M_{\mathsf{count}}$ and the 
vector formed by the stream. The matrix $M_{\mathsf{count}}$ is a lower-triangular matrix with all lower-triangular entries being $1$. 

The factorization mechanism the proceeds in the following steps:
1. Compute two matrices $L$ and $R$ such that $M_{\mathsf{count}}= LR$.
2. At time $t$, sample a Gaussian random vector $z \sim \mathcal N(0, \sigma^2)$, where $\sigma^2$ is the variance required to preserve 
approximate-DP for sensitivity-$1$ function. 
3. On receiving the update $x_t$, compute $S_t = S_{t-1} + x_t$, where $S_0=0$. 
4. Running sum at time $t$ is computed by computing $S_t + L_t R(t) z$, where $L_t$ is the $t$-th row of $L$ and $R(t)$ is the $t \times t$ 
principal submatrix of $R$.

The factorization mechanism in Fichtenberger, Henzinger, and Upadhyay (https://arxiv.org/abs/2202.11205) first computes the evaluation of a function 
$f$ on $\{0,1, \cdots, T-1\}$. 
The function $f$ is then defined recursively: 
- $f(0)=1$, and 
- $f(k) =  \left(\frac{2k-1}{2k} \right) f(k-1)$  for $k\geq 1$. 

Then the lower-triangular matrices (or factors) $L=R$ are constructed with their $(i,j)$-th entry being $f(i-j)$ for $i \geq j$ and $0$, otherwise.

# How are the streams of updates generated?
The streams are generated using two different probability distribution depending on the use case.

## Continual counting
When performing the experiments for continual counting, we use Bernoulli distribution to generate the stream. A Bernoulli distribution $\mathsf{Ber}(p)$
is parameterized by a scalar $p \in [0,1]$ and the probability distribution function is defined as follows:
$\mathsf{Pr}[X = 1] =  p$ and  $\mathsf{Pr}[X = 1] =  1- p $. 

The stream is generated such that every update $x[i] \sim \mathsf{Ber}(p)$. 

### Choices of $p$
We run our experiments on different choices of $p$. We start with $p=0$, which corresponds to an all-zero stream. This stream gives the estimates of the 
additive error due to binary mechanism and the factorization mechanism. This is one the standard trick used in testing in industries before a 
large-scale industrial deployment. 

The other choices of $p$ used in this paper is the powers of $2$, in particular, $p = \{2^{-4}, 2^{-5}, 2^{-6}, 2^{-7}, 2^{-8}, 2^{-9}, 2^{-10} \}$. The 
higher value of $p$ would generate more dense stream while the smaller value of $p$ would generate sparse stream. These choices are made to ensure that
the empirical evidence is presented for a wide range of streams. 

## Histogram estimation
When performing the experiments of differentially private continual histogram estimation, we use the Zipf's distribution to generate the stream. This is 
in accordance with the most relevant work that studies differentially private continual histogram estimation (https://arxiv.org/abs/2103.16787). We generate
the stream according to the Zipf's distribution and estimates the frequency of most frequent element in the stream. 

# How is the comparison done?
The binary tree mechanism has been proposed that gives both pure-DP as well as approximate-DP guarantee. Since our algorithm gives approximate-DP guarantee,
 we compare our algorithm with the binary tree mechanism with approximate-DP guarantee. In particular, we use Gaussian mechanism as the underlying privacy mechanism 
 to instantiate the binary tree mechanism. We use the same value of privacy parameters $(\epsilon=0.5, \delta = 10^{-10})$ for both the binary mechanism as well
 as the factorization mechanism. 
 
 User can change the value of $\epsilon$ and $\delta$ to get these estimates for different privacy parameters. 
