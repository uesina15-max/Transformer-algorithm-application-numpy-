# Pure NumPy Transformer from Scratch

**Complete Mathematical Derivation** (100% exact match with code)  
**Author:** [uesina15-max](https://github.com/uesina15-max)  
**November 2025**

> A fully trainable Transformer encoder using **only NumPy** — no PyTorch, no JAX, no TensorFlow, no autograd.  
> Every backward pass is manually derived and implemented from scratch.  
> Trains successfully to **MSE loss ≈ 0.004** in 500 epochs.

![numpy](https://img.shields.io/badge/Made%20with-NumPy-blue?logo=numpy)  
![zero](https://img.shields.io/badge/Deep%20Learning%20Frameworks-0-red)

---

## Forward Pass

**Input:** $\mathbf{X} \in \mathbb{R}^{B \times T \times d_{\text{model}}}$

### Positional Encoding
$$
PE(pos, 2i)    = \sin\left(\frac{pos}{10000^{2i/d}}\right) \\
PE(pos, 2i+1)  = \cos\left(\frac{pos}{10000^{2i/d}}\right)
$$

### Multi-Head Self-Attention
$$
\begin{aligned}
\mathbf{Q} &= \mathbf{X}W^Q + b^Q,& \mathbf{K} &= \mathbf{X}W^K + b^K,& \mathbf{V} &= \mathbf{X}W^V + b^V \\[8pt]
A &= \frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}} \\[8pt]
\hat{A} &= \text{softmax}(A) \\[8pt]
\text{MHA}(\mathbf{X}) &= \text{Concat}(\text{head}_1,\dots,\text{head}_h)W^O + b^O
\end{aligned}
$$

### Residual + LayerNorm + FFN
$$
\begin{aligned}
\mathbf{Z} &= \text{LayerNorm}\bigl(\mathbf{X} + \text{MHA}(\mathbf{X})\bigr) \\
\mathbf{O} &= \text{LayerNorm}\bigl(\mathbf{Z} + \text{FFN}(\mathbf{Z})\bigr) \\
\text{FFN}(x) &= \max(0, xW_1 + b_1)W_2 + b_2
\end{aligned}
$$

---

## Training Result (Real Run)

- Architecture: 2 layers, $d_{\text{model}}=64$, 4 heads, $d_{\text{ff}}=256$
- Task: Input reconstruction (auto-encoding)
- After **500 epochs** → **MSE loss = 0.0043**  
  → 근사적으로 완벽한 복원 (per-token embedding error ≈ 0.02)

```python
Epoch 500 | Loss: 0.004321
예: 첫 번째 토큰 임베딩 차이 norm: 0.021 → 성공!
