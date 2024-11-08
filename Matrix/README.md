# Matrix: SolvingMethod
This is a file that **stores** the code to calculate the **solution** of a system of equations

Basically, when solving systems with a **large number of equations**, the manual methods will be **time-consuming** or **memory-intensive**, so the **idea** is to begin with an initial guess at the solution (which **doesn't need** to be very accurate), and then **iteratively** refine this guess to get closer to the true solution. Once the approximation is **sufficiently precise**, it is considered the solution to the system

In this project, I use 4 methods: [Richardson Iteration](https://en.wikipedia.org/wiki/Modified_Richardson_iteration), [Jacobi Method](https://en.wikipedia.org/wiki/Jacobi_method), [Gauss-Seidel Method](https://en.wikipedia.org/wiki/Gauss%E2%80%93Seidel_method) and [SOR](https://en.wikipedia.org/wiki/Successive_over-relaxation)

# Detailed formula
Let $Ax=b$ be a square system of n linear equations, where:

$$\large
A = 
\begin{pmatrix}
a_{11} & a_{12} & \cdots & a_{1n} \\
a_{21} & a_{22} & \cdots & a_{2n} \\
\vdots & \vdots & \ddots & \vdots \\
a_{n1} & a_{n2} & \cdots & a_{nn} 
\end{pmatrix}\ \,\quad\quad x = 
\begin{pmatrix}
x_{1} \\
x_{2} \\
\vdots \\
x_{n}
\end{pmatrix}\ \,\quad\quad b = 
\begin{pmatrix}
b_{1} \\
b_{2} \\
\vdots \\
b_{n}
\end{pmatrix}
$$

$A$ and $b$ are **known**, $x$ is **unknown**, we can use a method to **approximate** $x$. The vector $x^\{(0)}$ denotes our **initial guess** for $x$ (often $x_{i}^\{(0)}=0$ for $i=1,2,...,n$). We denote $x^{(k)}$ as the $k$-th approximation or iteration of $x$, and $x^{(k+1)}$ as the next iteration of $x$

Note that when **entering data** of a system of equations, the computer will receive an **augmented matrix** $\tilde A = [A|b]$: 

$$\large
A = 
\begin{pmatrix}
a_{11} & a_{12} & \cdots & a_{1n} | b_{1}\\
a_{21} & a_{22} & \cdots & a_{2n} | b_{2}\\
\vdots & \vdots & \ddots & \vdots \ \ | \vdots\\
a_{n1} & a_{n2} & \cdots & a_{nn} | b_{n}
\end{pmatrix}
$$

So we need to **split** $\tilde A$ into $A$ and $b$, then $A$ can be **decomposed** into a diagonal component $D$, a lower triangular part $L$ and an upper triangular part $U$:

$$\large
D = 
\begin{pmatrix}
a_{11} & 0 & \cdots & 0 \\
0 & a_{22} & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & a_{nn} 
\end{pmatrix}\ \,\quad\quad L = 
\begin{pmatrix}
0 & 0 & \cdots & 0 \\
a_{21} & 0 & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
a_{1n} & a_{2n} & \cdots & 0 
\end{pmatrix}\ \,\quad\quad U = 
\begin{pmatrix}
0 & a_{12} & \cdots & a_{1n} \\
0 & 0 & \cdots & a_{2n} \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & 0 
\end{pmatrix}
$$

The solution is then obtained iteratively

## Richardson Iteration:
### Matrix-based formula: 

$$\Huge x^{(k+1)} = x^{(k)} + \omega(b - A x^{(k)})$$

where $\omega$ is a **scalar parameter** that has to be **chosen** such that the sequence $x^{(k)}$ **converges** (often  $\omega < 1$)
### Algorithm:
```
def richardsonIteration(self, A, x, k):
    A, b, _, _, _ = self.splittingMat(A)
    w = 1 / 100
    for _ in range(k):
        x_new = w * (b - np.dot(A, x)) + x
        if np.allclose(x, x_new):
            break
        x = x_new
    return x
```
## Jacobi Method:
### Matrix-based formula: 

$$\Huge x^{(k+1)} = D^{-1}(b - (L+U)x^{(k)})$$

### Algorithm:
```
def jacobiMethod(self, A, x, k):
    _, b, D, L, U = self.splittingMat(A)
    D = np.linalg.inv(D)
    for _ in range(k):
        x_new = np.dot(D, b - np.dot(L + U, x))
        if np.allclose(x, x_new):
            break
        x = x_new
    return x
```
## Gauss-Seidel Method:
### Matrix-based formula: 

$$\Huge x^{(k+1)} = (D+L)^{-1}(b - U x^{(k)})$$

### Algorithm:
```
def gaussSeidelMethod(self, A, x, k):
    _, b, D, L, U = self.splittingMat(A)
    S = np.linalg.inv(D + L)
    for _ in range(k):
        x_new = np.dot(S, b - np.dot(U, x))
        if np.allclose(x, x_new):
            break
        x = x_new
    return x
```
## SOR (Successive over-relaxation): 
### Matrix-based formula: 

$$\Huge x^{(k+1)} = (D + \omega L)^{-1}(\omega b - (\omega U + (\omega - 1)D) x^{(k)})$$

where $\omega$ is a **scalar parameter** or **relaxation factor** (often  $\omega > 1$)
### Algorithm:
```
def SOR(self, A, x, k):
    _, b, D, L, U = self.splittingMat(A)
    w = 3 / 2
    S = np.linalg.inv(D + w * L)
    for _ in range(k):
        x_new = np.dot(S, w * b - np.dot(w * U + (w - 1) * D, x))
        if np.allclose(x, x_new):
            break
        x = x_new
    return x
```
# Note

Not all 4 methods can always calculate **approximate** solutions for **every** system of equations, the accuracy can even be **0%**. For **Richardson Iteration** and **SOR** methods, which require $\omega$, **choosing** appropriate $\omega$ also **contributes** to increasing **accuracy**

Moreover, depending on the **arrangement of equations**, each method can produce **wildly** varying results, from **complete failure** to **near-perfect** accuracy, or even completely **incorrect** solutions
