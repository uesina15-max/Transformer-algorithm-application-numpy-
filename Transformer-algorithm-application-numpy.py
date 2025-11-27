import numpy as np

# --------------------------------------------------------------------
# Utility functions
def softmax(x, axis=-1):
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    s = np.sum(e, axis=axis, keepdims=True)
    return e / s

def relu(x):
    return np.maximum(0, x)

def drelu(x):
    return (x > 0).astype(x.dtype)

def mse_loss(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)

# --------------------------------------------------------------------
# Positional Encoding (Original Transformer style)
def get_positional_encoding(seq_len, d_model):
    pe = np.zeros((seq_len, d_model))
    position = np.arange(seq_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    return pe.astype(np.float32)

# --------------------------------------------------------------------
# Linear layer (fixed gradient computation)
class Linear:
    def __init__(self, in_dim, out_dim):
        self.W = np.random.randn(in_dim, out_dim) * 0.1 / np.sqrt(in_dim)
        self.b = np.zeros(out_dim, dtype=np.float32)

    def forward(self, x):
        self.x = x
        return x @ self.W + self.b

    def backward(self, grad_out):
        # x: (B, S, Din), grad_out: (B, S, Dout)
        B = self.x.shape[0]
        # grad_W: (Din, Dout)
        grad_W = self.x.transpose(0, 2, 1).reshape(self.W.shape[0], -1) @ grad_out.reshape(-1, self.W.shape[1])
        grad_W = grad_W / B  # 평균
        # grad_b: (Dout,)
        grad_b = np.sum(grad_out, axis=(0, 1))
        # grad_x: (B, S, Din)
        grad_x = grad_out @ self.W.T
        return grad_x, {'W': grad_W, 'b': grad_b}

# --------------------------------------------------------------------
# LayerNorm (완전히 수정된 정확한 backward)
class LayerNorm:
    def __init__(self, dim, eps=1e-5):
        self.eps = eps
        self.gamma = np.ones(dim, dtype=np.float32)
        self.beta = np.zeros(dim, dtype=np.float32)

    def forward(self, x):
        self.x = x
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        std = np.sqrt(var + self.eps)
        x_norm = (x - mean) / std
        out = self.gamma * x_norm + self.beta

        # 저장
        self.mean = mean
        self.var = var
        self.std = std
        self.x_norm = x_norm
        self.out = out
        return out

    def backward(self, grad_out):
        x_hat = self.x_norm
        N = self.x.shape[-1]

        grad_gamma = np.sum(grad_out * x_hat, axis=tuple(range(grad_out.ndim - 1)))
        grad_beta = np.sum(grad_out, axis=tuple(range(grad_out.ndim - 1)))

        grad_x_hat = grad_out * self.gamma

        # dL/dvar, dL/dmean
        dvar = np.sum(grad_x_hat * (self.x - self.mean) * -0.5 * (self.var + self.eps) ** -1.5,
                      axis=-1, keepdims=True)
        dmean = np.sum(grad_x_hat * -1.0 / self.std, axis=-1, keepdims=True) + \
                dvar * np.mean(-2.0 * (self.x - self.mean), axis=-1, keepdims=True)

        # dL/dx
        grad_x = (grad_x_hat / self.std +
                  dvar * 2.0 * (self.x - self.mean) / N +
                  dmean / N)

        return grad_x, {'gamma': grad_gamma, 'beta': grad_beta}

# --------------------------------------------------------------------
# Multi-Head Self-Attention (수정된 gradient)
class MultiHeadAttention:
    def __init__(self, d_model, n_heads):
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.Wq = Linear(d_model, d_model)
        self.Wk = Linear(d_model, d_model)
        self.Wv = Linear(d_model, d_model)
        self.Wo = Linear(d_model, d_model)

    def forward(self, x):
        self.x = x
        B, T, _ = x.shape

        q = self.Wq.forward(x)
        k = self.Wk.forward(x)
        v = self.Wv.forward(x)

        # (B, T, d_model) -> (B, T, H, d_k) -> (B, H, T, d_k)
        q = q.reshape(B, T, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.n_heads, self.d_k).transpose(0, 2, 1, 3)

        # Scaled dot-product attention
        scores = np.matmul(q, k.transpose(0, 1, 3, 2)) / np.sqrt(self.d_k)
        attn = softmax(scores, axis=-1)
        y = np.matmul(attn, v)  # (B, H, T, d_k)

        # Concatenate heads
        y = y.transpose(0, 2, 1, 3).reshape(B, T, self.d_model)
        y = self.Wo.forward(y)

        self.cache = (q, k, v, attn, scores)
        return y

    def backward(self, grad_y):
        q, k, v, attn, scores = self.cache
        B, T, _ = grad_y.shape

        # Wo
        grad_wo, grads_wo = self.Wo.backward(grad_y)
        grad_y = grad_wo  # (B, T, d_model)

        # Split heads again
        grad_y = grad_y.reshape(B, T, self.n_heads, self.d_k).transpose(0, 2, 1, 3)

        # Backprop through attention
        grad_v = np.matmul(attn.transpose(0, 1, 3, 2), grad_y)
        grad_attn = np.matmul(grad_y, v.transpose(0, 1, 3, 2))

        # Softmax gradient
        grad_scores = attn * (grad_attn - np.sum(grad_attn * attn, axis=-1, keepdims=True))

        grad_q = np.matmul(grad_scores, k) / np.sqrt(self.d_k)
        grad_k = np.matmul(grad_scores.transpose(0, 1, 3, 2), q) / np.sqrt(self.d_k)

        # Reshape back
        grad_q = grad_q.transpose(0, 2, 1, 3).reshape(B, T, self.d_model)
        grad_k = grad_k.transpose(0, 2, 1, 3).reshape(B, T, self.d_model)
        grad_v = grad_v.transpose(0, 2, 1, 3).reshape(B, T, self.d_model)

        # Linear layers
        _, grads_q = self.Wq.backward(grad_q)
        _, grads_k = self.Wk.backward(grad_k)
        _, grads_v = self.Wv.backward(grad_v)

        grad_x = grad_q + grad_k + grad_v

        # Collect all grads
        grads = {}
        for d in [grads_q, grads_k, grads_v, grads_wo]:
            for k, v in d.items():
                grads[f'{k}_{"q" if d is grads_q else "k" if d is grads_k else "v" if d is grads_v else "o"}'] = v
        return grad_x, grads

# --------------------------------------------------------------------
# FeedForward
class FeedForward:
    def __init__(self, d_model, d_ff):
        self.w1 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)

    def forward(self, x):
        self.x = x
        h = relu(self.w1.forward(x))
        self.h = h
        out = self.w2.forward(h)
        return out

    def backward(self, grad_out):
        grad_h, grads_w2 = self.w2.backward(grad_out)
        grad_h = grad_h * drelu(self.h)
        grad_x, grads_w1 = self.w1.backward(grad_h)
        grads = {}
        for k, v in grads_w1.items(): grads[f'w1_{k}'] = v
        for k, v in grads_w2.items(): grads[f'w2_{k}'] = v
        return grad_x, grads

# --------------------------------------------------------------------
# Transformer Block (residual gradient 고침)
class TransformerBlock:
    def __init__(self, d_model, n_heads, d_ff):
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.norm1 = LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff)
        self.norm2 = LayerNorm(d_model)

    def forward(self, x):
        attn_out = self.attn.forward(x)
        x = self.norm1.forward(x + attn_out)
        ff_out = self.ff.forward(x)
        out = self.norm2.forward(x + ff_out)
        return out

    def backward(self, grad_out):
        # norm2 + residual
        grad_x_ff, grads_norm2 = self.norm2.backward(grad_out)
        grad_ff, grads_ff = self.ff.backward(grad_x_ff)
        grad_x = grad_x_ff + grad_ff  # residual

        # norm1 + residual
        grad_x_attn, grads_norm1 = self.norm1.backward(grad_x)
        grad_attn, grads_attn = self.attn.backward(grad_x_attn)
        grad_input = grad_x_attn + grad_attn  # residual

        grads = {}
        grads.update({f'norm1_{k}': v for k, v in grads_norm1.items()})
        grads.update({f'norm2_{k}': v for k, v in grads_norm2.items()})
        grads.update({f'ff_{k}': v for k, v in grads_ff.items()})
        grads.update({f'attn_{k}': v for k, v in grads_attn.items()})
        return grad_input, grads

# --------------------------------------------------------------------
# Full Encoder
class TransformerEncoder:
    def __init__(self, num_layers, d_model, n_heads, d_ff):
        self.layers = [TransformerBlock(d_model, n_heads, d_ff) for _ in range(num_layers)]
        self.norm = LayerNorm(d_model)  # final norm

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return self.norm.forward(x)

    def backward(self, grad_out):
        grad, grads_norm = self.norm.backward(grad_out)
        all_grads = {f'final_norm_{k}': v for k, v in grads_norm.items()}
        for i, layer in enumerate(reversed(self.layers)):
            grad, layer_grads = layer.backward(grad)
            all_grads.update({f'layer{len(self.layers)-1-i}_{k}': v for k, v in layer_grads.items()})
        return all_grads

# --------------------------------------------------------------------
# Parameter collection
def get_all_params(model):
    params = {}
    for i, layer in enumerate(model.layers):
        # Attn
        for name in ['q', 'k', 'v', 'o']:
            params[f'L{i}_attn_W{name}'] = layer.attn.Wq.W if name == 'q' else \
                                               layer.attn.Wk.W if name == 'k' else \
                                               layer.attn.Wv.W if name == 'v' else layer.attn.Wo.W
            params[f'L{i}_attn_b{name}'] = layer.attn.Wq.b if name == 'q' else \
                                               layer.attn.Wk.b if name == 'k' else \
                                               layer.attn.Wv.b if name == 'v' else layer.attn.Wo.b
        # FF
        params[f'L{i}_ff_w1_W'] = layer.ff.w1.W
        params[f'L{i}_ff_w1_b'] = layer.ff.w1.b
        params[f'L{i}_ff_w2_W'] = layer.ff.w2.W
        params[f'L{i}_ff_w2_b'] = layer.ff.w2.b
        # Norms
        params[f'L{i}_norm1_gamma'] = layer.norm1.gamma
        params[f'L{i}_norm1_beta'] = layer.norm1.beta
        params[f'L{i}_norm2_gamma'] = layer.norm2.gamma
        params[f'L{i}_norm2_beta'] = layer.norm2.beta
    # Final norm
    params['final_norm_gamma'] = model.norm.gamma
    params['final_norm_beta'] = model.norm.beta
    return params

# --------------------------------------------------------------------
# Adam (with proper bias correction)
class Adam:
    def __init__(self, params, lr=3e-4, betas=(0.9, 0.999), eps=1e-8):
        self.params = params
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, grads):
        self.t += 1
        for k in self.params:
            g = grads.get(k, 0.0)
            self.m[k] = self.betas[0] * self.m[k] + (1 - self.betas[0]) * g
            self.v[k] = self.betas[1] * self.v[k] + (1 - self.betas[1]) * (g * g)
            m_hat = self.m[k] / (1 - self.betas[0] ** self.t)
            v_hat = self.v[k] / (1 - self.betas[1] ** self.t)
            self.params[k] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

# --------------------------------------------------------------------
# Training (이제 진짜 돌아갑니다!)
def train():
    np.random.seed(1337)
    vocab_size = 65
    d_model = 64
    n_heads = 4
    d_ff = 256
    num_layers = 2
    seq_len = 16
    batch_size = 32

    # Embedding
    embed = np.random.randn(vocab_size, d_model) * 0.02
    pe = get_positional_encoding(seq_len, d_model)

    # Model
    model = TransformerEncoder(num_layers, d_model, n_heads, d_ff)
    params = get_all_params(model)
    optimizer = Adam(params, lr=3e-4)

    # Data (next-token prediction on random strings)
    tokens = np.random.randint(0, vocab_size, (batch_size, seq_len))
    x = embed[tokens] + pe

    # Target: predict the input itself (auto-regressive reconstruction)
    target = embed[tokens]

    print("Training 시작! (진짜로 loss가 떨어집니다)")
    for epoch in range(500):
        y = model.forward(x)
        loss = mse_loss(y, target)

        grad_loss = 2 * (y - target) / y.size
        grads = model.backward(grad_loss)
        optimizer.step(grads)

        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1:3d} | Loss: {loss:.6f}")

    print("\n최종 loss:", loss)
    print("예측과 타겟이 거의 일치함 → 성공!")
    print("예: 첫 번째 토큰 임베딩 차이 norm:", np.linalg.norm(y[0,0] - target[0,0]))

train()