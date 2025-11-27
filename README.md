# NumPy Transformer Encoder: Mathematical Derivation & Training Results

This page presents the **complete mathematical derivation and architecture** for a fully trainable Transformer encoder—implemented **only with NumPy**—where every backward pass has been manually derived and matches the formulas in the successful training code.

Author: [uesina15-max](https://github.com/uesina15-max)  
Date: **November 2025**  
**Result:** Trains successfully to MSE loss $\approx 0.004$ in 500 epochs.

---

## Transformer Encoder: Step-by-Step

### Forward Pass

#### 1. Input

- $\mathbf{X} \in \mathbb{R}^{B \times T \times d}$ 
- $B$: Batch size 
- $T$: Sequence length 
- $d$: Embedding dimension ($(d = d_{\text{model}})$)

#### 2. Positional Encoding (PE)

Sinusoidal PE injects order info into token embeddings:
$$
\begin{align*}
PE(pos, 2i) &= \sin\left(\frac{pos}{10000^{2i/d}}\right) \\
PE(pos, 2i+1) &= \cos\left(\frac{pos}{10000^{2i/d}}\right)
\end{align*}
$$

#### 3. Multi-Head Self-Attention (MHA)

Project input to $Q$, $K$, $V$:
$$
\begin{align*}
\mathbf{Q} &= \mathbf{X} W^Q + b^Q \\
\mathbf{K} &= \mathbf{X} W^K + b^K \\
\mathbf{V} &= \mathbf{X} W^V + b^V
\end{align*}
$$

Calculate scaled dot-product attention:
$$
A = \frac{\mathbf{Q} \mathbf{K}^\top}{\sqrt{d_k}} \qquad (B \times h \times T \times T)
$$
$$
\hat{A} ​​= \operatorname{softmax}(A)
$$
$$
\text{head}_i = \hat{A}_i \mathbf{V}_i
$$

Concatenate and linearly project the heads:
$$
\operatorname{MHA}(\mathbf{X}) = \operatorname{Concat}(\text{head}_1, ..., \text{head}_h)\, W^O + b^O
$$

#### 4. Add & LayerNorm + Feed-Forward Network (FFN)

-Residual + LayerNorm:
$$
\mathbf{Z} = \operatorname{LayerNorm}(\mathbf{X} + \operatorname{MHA}(\mathbf{X}))
$$
-feedforward: (with ReLU)
$$
\operatorname{FFN}(x) = \max(0, x W_1 + b_1) W_2 + b_2
$$
$$
\mathbf{O} = \operatorname{LayerNorm}(\mathbf{Z} + \operatorname{FFN}(\mathbf{Z}))
$$

---

### Backward Pass Highlights — Manual Derivations

#### Multi-Head Attention

Let $\delta Y = \frac{\partial \mathcal{L}}{\partial (\text{post-}W^O)}$:
$$
\delta V = \hat{A}^\top \delta Y, \quad
\delta \hat{A} ​​= \delta Y V^\top
$$
- Softmax derivative:
$$
\delta A = \hat{A} ​​\odot (\delta \hat{A} ​​- \hat{A} ​​(\delta \hat{A} ​​\cdot \mathbf{1}))
$$
$$
\delta Q = \delta A K / \sqrt{d_k}, \quad
\delta K = \delta A^\top Q / \sqrt{d_k}
$$

#### Layer Normalization (Bug-Free Version)

- Normalize each sample across features:
$$
\begin{align*}
\mu &= \frac{1}{d} \sum_j x_j \\
\sigma^2 &= \frac{1}{d} \sum_j (x_j - \mu)^2 \\
\hat{x} &= \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \\
y &= \gamma \hat{x} + \beta
\end{align*}
$$
Let $g_{\hat{x}} = \frac{\partial \mathcal{L}}{\partial y} \gamma$
$$
g_{\sigma^2} = \sum_j \left[ g_{\hat{x},j} (x_j - \mu) \left( -\frac{1}{2} \right) (\sigma^2 + \epsilon)^{-3/2} \right]
$$
$$
g_\mu = -\sum_j \frac{g_{\hat{x},j}}{\sqrt{\sigma^2 + \epsilon}} + g_{\sigma^2} \cdot \frac{\sum_j -2(x_j - \mu)}{d}
$$
$$
\frac{\partial \mathcal{L}}{\partial x_i} =
\frac{g_{\hat{x},i}}{\sqrt{\sigma^2 + \epsilon}} + \frac{2 g_{\sigma^2} (x_i - \mu)}{d} + \frac{g_\mu}{d}
$$

---

### Loss and Optimizer

#### Loss: Mean Squared Error (MSE)

Reconstruction task ($\hat{Y}$ is encoder output, $X$ is input):
$$
\mathcal{L} = \frac{1}{B T d} \|\hat{Y} - X\|_2^2
$$
Gradient:
$$
\frac{\partial \mathcal{L}}{\partial \hat{Y}} = \frac{2}{B T d} (\hat{Y} - X)
$$

#### Adam Optimizer

-First and second moment estimates:
$$
m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t
$$
$$
v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2
$$

- Bias correction and parameter update:
$$
\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \quad
\hat{v}_t = \frac{v_t}{1-\beta_2^t}
$$
$$
\theta_t = \theta_{t-1} - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
$$

---

## 🏆 Training Result

| Metric | Value |
|-----------------------|----------------------------------------|
| Architecture | 2 layers, $d_{\text{model}}=64$, 4 heads, $d_{\text{ff}}=256$ |
| Task | Input reconstruction (Auto-encoding) |
| Epochs | 500 |
| Final MSE Loss | 0.0043 |
| Result | Near-perfect reconstruction (per-token embedding error $\approx 0.02$) |

---

## License

MIT

---
For questions or to see the code: [GitHub Repo](https://github.com/uesina15-max/Transformer-algorithm-application-numpy-)  
Compiles well in the GitHub repository
