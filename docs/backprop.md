# Backpropagation

$$
\begin{align*}
z_1 &= W_1 x + b_1 \qquad a_1 = g(z_1)
\\
z_2 &= W_2 a_1 + b_2 \qquad a_2 = g(z_2)
\\
z_3 &= W_3 a_2 + b_3 \qquad y = g(z_3)
\end{align*}
$$


### 1. $\frac {\partial L} {\partial W_3}$

$$
\begin{align*}
\frac {\partial L} {\partial W_3} &= 
\frac {\partial L} {\partial a_3} \cdot
\frac {\partial a_3} {\partial z_3} \cdot
\frac {\partial z_3} {\partial W_3} \\
&= ( - \frac y {a_3} + \frac {1 - y} {1 - a_3} ) \cdot
g(z_3) \lbrace 1 - g(z_3) \rbrace \cdot a_2
\end{align*}
$$

### 2. $\frac {\partial L} {\partial b_3}$

$$
\begin{align*}
\frac {\partial L} {\partial b_3} &=
\frac {\partial L} {\partial a_3} \cdot
\frac {\partial a_3} {\partial z_3} \cdot
\frac {\partial z_3} {\partial b_3} \\
&= ( - \frac y {a_3} + \frac {1 - y} {1 - a_3} ) \cdot
g(z_3) \lbrace 1 - g(z_3) \rbrace \cdot 1
\end{align*}
$$

### 3. $\frac {\partial L} {\partial W_1}$

$$
\begin{align*}
\frac {\partial L} {\partial W_1} &= 
\frac {\partial L} {\partial a_3} \cdot
\frac {\partial a_3} {\partial z_3} \cdot
\frac {\partial z_3} {\partial a_2} \cdot
\frac {\partial a_2} {\partial z_2} \cdot
\frac {\partial z_2} {\partial a_1} \cdot
\frac {\partial a_1} {\partial z_1} \cdot
\frac {\partial z_1} {\partial W_1} \\
&= ( - \frac y {a_3} + \frac {1 - y} {1 - a_3} ) \cdot
g(z_3) \lbrace 1 - g(z_3) \rbrace \cdot W_3 \\
& \cdot g(z_2) \lbrace 1 - g(z_2) \rbrace \times W_2 \times
g(z_1) \lbrace 1 - g(z_1) \rbrace \cdot x
\end{align*}
$$


## Functions
$$
\begin{align*}
L(a, y) &= - y \log {a} - (1 - y) \log {(1 - a)}
\\
g(z) &= \frac 1 {1 + e^{-z}}
\end{align*}
$$

## Partial derivatives
$$
\begin{align*}
\frac {\partial L} {\partial a} &= - \frac y a + \frac {1 - y} {1 - a}
\\
\frac {\partial a} {\partial z} &= g(z) \lbrace 1 - g(z) \rbrace
\\
\frac {\partial z} {\partial W} &= a_{pre}
\\
\frac {\partial z} {\partial b} &= 1
\end{align*}
$$

## Updates

$$
\begin{align*}
{W}^{'} &= W - \alpha \cdot \frac {\partial L} {\partial W}
\\
{b}^{'} &= b - \alpha \cdot \frac {\partial L} {\partial b}
\end{align*}
$$
