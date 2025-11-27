% Pure NumPy Transformer - Mathematics (English, 100% compile-safe version)
\documentclass[11pt,a4paper]{article}
\usepackage{amsmath, amssymb, amsthm}
\usepackage{geometry}
\usepackage{xcolor}
\usepackage{hyperref}
\usepackage{microtype}

\geometry{margin=1in}
\title{\textbf{Pure NumPy Transformer from Scratch}\\[4pt]
\Large Complete Mathematical Derivation (Exact Match with Code)}
\author{uesina15-max}
\date{November 2025}

\begin{document}
\maketitle

\begin{center}
\textit{A fully trainable Transformer encoder using \textbf{only NumPy} — no PyTorch, no autograd.}
\end{center}

\vspace{1cm}

\section{Forward Pass}

Input: $\mathbf{X} \in \mathbb{R}^{B \times T \times d}$ where $d = d_{\text{model}}$.

\subsection{Positional Encoding}
\begin{align}
PE(pos, 2i)    &= \sin\left(\frac{pos}{10000^{2i/d}}\right) \\
PE(pos, 2i+1)  &= \cos\left(\frac{pos}{10000^{2i/d}}\right)
\end{align}

\subsection{Multi-Head Self-Attention}
\begin{align}
\mathbf{Q} &= \mathbf{X} W^Q + b^Q, &
\mathbf{K} &= \mathbf{X} W^K + b^K, &
\mathbf{V} &= \mathbf{X} W^V + b^V \\[6pt]
A &= \frac{\mathbf{Q} \mathbf{K}^\top}{\sqrt{d_k}} \qquad (\text{shape: } B \times h \times T \times T) \\[6pt]
\hat{A} &= \operatorname{softmax}(A) \quad \text{(over last axis)} \\[6pt]
\text{head}_i &= \hat{A}_i \, \mathbf{V}_i \\[6pt]
\operatorname{MHA}(\mathbf{X}) &= \operatorname{Concat}(\text{head}_1, \dots, \text{head}_h) W^O + b^O
\end{align}

\subsection{Add \& LayerNorm + Feed-Forward}
\begin{align}
\mathbf{Z} &= \operatorname{LayerNorm}\bigl(\mathbf{X} + \operatorname{MHA}(\mathbf{X})\bigr) \\
\mathbf{O} &= \operatorname{LayerNorm}\bigl(\mathbf{Z} + \operatorname{FFN}(\mathbf{Z})\bigr) \\
\operatorname{FFN}(x) &= \max(0, x W_1 + b_1) W_2 + b_2
\end{align}

\section{Backward Pass — Exact Formulas Used in the Code}

\subsection{Multi-Head Attention Backward}
\begin{align}
\delta Y &= \frac{\partial \mathcal{L}}{\partial (\text{post-}W^O)} \\
\delta V &= \hat{A}^\top \delta Y, \qquad
\delta \hat{A} = \delta Y V^\top \\[6pt]
\delta A &= \hat{A} \odot \left( \delta \hat{A} - \hat{A} (\delta \hat{A} \cdot \mathbf{1}) \right) \tag{softmax Jacobian} \\[8pt]
\delta Q &= \delta A \, K / \sqrt{d_k}, \qquad
\delta K = \delta A^\top Q / \sqrt{d_k}
\end{align}

\subsection{LayerNorm Backward (bug-free version)}
\begin{align}
\mu &= \frac{1}{d} \sum_j x_j, &
\sigma^2 &= \frac{1}{d} \sum_j (x_j - \mu)^2 \\
\hat{x} &= \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}, &
y &= \gamma \hat{x} + \beta \\[8pt]
g_{\hat{x}} &= \frac{\partial \mathcal{L}}{\partial y} \gamma \\[4pt]
g_{\sigma^2} &= \sum_j \left[ g_{\hat{x},j} (x_j - \mu) \left( -\frac{1}{2} \right) (\sigma^2 + \epsilon)^{-3/2} \right] \\[4pt]
g_\mu &= -\sum_j \frac{g_{\hat{x},j}}{\sqrt{\sigma^2 + \epsilon}} 
         + g_{\sigma^2} \cdot \frac{\sum_j -2(x_j - \mu)}{d} \\[6pt]
\frac{\partial \mathcal{L}}{\partial x_i} &= 
\frac{g_{\hat{x},i}}{\sqrt{\sigma^2 + \epsilon}} 
+ \frac{2 g_{\sigma^2} (x_i - \mu)}{d}
+ \frac{g_\mu}{d}
\end{align}

\subsection{Residual Gradient Flow}
\begin{align}
\delta Z_{\text{add}} = \delta Z_{\text{after-norm}} 
\quad \longrightarrow \quad 
\text{sent to both FFN and skip connection}
\end{align}

\section{Loss and Optimizer}
\begin{align}
\mathcal{L} &= \frac{1}{B T d} \|\hat{Y} - X\|_2^2 &
\frac{\partial \mathcal{L}}{\partial \hat{Y}} &= \frac{2}{B T d} (\hat{Y} - X) \\[10pt]
m_t &= \beta_1 m_{t-1} +
