# Chapter 1: Introduction to Quasi-Newton Methods
## [00:00] The Trade-off Between Newton's Method and Gradient Descent
In this lesson, we continue our analysis of Newton's Methods. As we observed in the previous lesson, Newton's Method is effective in handling complex functions due to its second-order convergence. However, it has a significant drawback: it requires calculating the Hessian matrix and solving a linear system involving it.
For comparison:
*   **Gradient Descent:** Each iteration has a computational cost on the order of O(n), as it is primarily based on dot products. It is an economically advantageous method in terms of operations, but its convergence is slow.
*   **Newton's Method:** It is theoretically much faster, with quadratic convergence. However, the cost per iteration is significantly higher, on the order of O(n³), due to the need to solve a linear system.
The main goal of **Quasi-Newton methods** is to find a middle ground: to maintain the excellent convergence properties of Newton's Method while reducing its computational cost.
## [00:50] The Fundamental Idea: Approximating the Hessian Matrix
Let's briefly recall Newton's Method. To minimize a function, the update is calculated as follows:
`Δx = -H⁻¹ * ∇f(x)`
where `H` is the Hessian matrix. It is assumed that `H` is a positive definite matrix, thus guaranteeing a unique minimum for the quadratic approximation `q` of the function.
The key idea of Quasi-Newton methods is to replace the true Hessian matrix `H` with an approximation, which we will call `B_k`. This matrix `B_k` will be our approximation of the Hessian at iteration `k`.
By replacing the Hessian with its approximation and maintaining the same logic, we obtain a new update rule. Assuming that `B_k` is also positive definite, the formula becomes:
`x_{k+1} = x_k - B_k⁻¹ * ∇f(x_k)`
This expression is very similar to that of Newton's Method, with the crucial difference that we use the approximation `B_k` instead of the true Hessian matrix.
## [01:44] The Search Direction and Step Length
Since we are using an approximation, the calculated term is no longer an exact update but rather a **search direction**, similar to what happens in the gradient method.
In the terminology of Quasi-Newton methods, we define the vector `d_k` as the **search direction**:
`d_k = -B_k⁻¹ * ∇f(x_k)`
Once the direction is defined, we must determine the step length to take along it. This problem can be solved, as we have seen in other contexts, by formulating a one-dimensional problem and using a **line search** algorithm. In practice, specific conditions can be used, such as the Wolfe conditions, which represent a smart implementation of the line search.
The complete update rule will therefore be:
`x_{k+1} = x_k + α_k * d_k`
where:
*   `d_k` is the search direction.
*   `α_k` is the step length (or *learning rate*, in machine learning terminology), calculated via a dedicated algorithm.
## [02:48] How to Calculate the Approximation `B_k`
The crucial point, for now excluding the calculation of `α_k`, is to understand how to update the approximation matrix `B_k` at each iteration. In other words, given the approximation `B_k`, how can we calculate `B_{k+1}`?
## [03:07] Structure of a Generic Quasi-Newton Algorithm
A typical Quasi-Newton algorithm follows an iterative procedure structured as follows:
1.  **Initialization:**
    *   Start with an initial estimate of the solution, `x_0`.
    *   Choose an initial estimate for the Hessian approximation, `B_0`. A common choice is the identity matrix.
    *   Define a tolerance `ε > 0` for the stopping criterion.
2.  **Stopping Criterion:**
    *   The algorithm stops when the norm of the gradient falls below the defined tolerance: `||∇f(x_k)|| < ε`.
3.  **Calculate the Search Direction:**
    *   Calculate the direction `d_k` using the current approximation `B_k` and the gradient `∇f(x_k)`.
4.  **Calculate the Step Length:**
    *   Determine the value of `α_k` via a line search algorithm.
5.  **Update the Solution:**
    *   Update the solution estimate: `x_{k+1} = x_k + α_k * d_k`.
6.  **Update the Hessian Approximation:**
    *   Calculate the new approximation `B_{k+1}`.
7.  **Iteration:**
    *   Set `k = k + 1` and return to the stopping criterion check (or to step 3, depending on the implementation).
# Chapter 2: Properties and Updates of the Approximation Matrix
## Principles and Properties of Quasi-Newton Methods
This section introduces Quasi-Newton methods as an evolution of Newton's method, with the goal of reducing computational complexity. The algorithm is based on an approximation of the Hessian matrix, denoted by `Bk`.
The key idea is to iteratively update this approximation `Bk` efficiently. The general scheme of the algorithm involves two critical steps:
1.  The calculation of the search direction `dk`.
2.  The update of the matrix `Bk` for the next iteration.
We will analyze in detail the properties that the `Bk` matrix must satisfy to ensure the effectiveness and stability of the algorithm.
### Fundamental Properties of the Approximation Matrix `Bk`
For the Quasi-Newton algorithm to work correctly, the `Bk` matrix must possess some essential properties at each iteration `k`.
1.  **Non-singularity**: `Bk` must be non-singular (i.e., invertible). This ensures that the search direction `dk = -Bk⁻¹ * ∇f(xk)` is well-defined and computable.
2.  **Descent direction**: The calculated direction `dk` must be a descent direction. This means that the dot product between the direction and the gradient must be negative (`∇f(xk)ᵀ * dk < 0`), ensuring that a small step in that direction reduces the value of the objective function.
3.  **Symmetry**: Since the Hessian matrix is symmetric, it is reasonable to require that its approximation `Bk` also be symmetric.
#### Ensuring Properties Through Positive Definiteness
An effective way to simultaneously satisfy all three properties listed above is to require that the matrix `Bk` be **symmetric and positive definite** at each iteration.
Let's see why:
*   **Non-singularity**: A positive definite matrix has all strictly positive eigenvalues. Since the determinant is the product of the eigenvalues, it will be non-zero, making the matrix non-singular.
*   **Symmetry**: It is satisfied by definition.
*   **Descent direction**: Let's check the condition `∇f(xk)ᵀ * dk < 0`. Substituting `dk` with its expression `-Bk⁻¹ * ∇f(xk)`, we get:
    `∇f(xk)ᵀ * (-Bk⁻¹ * ∇f(xk)) = -∇f(xk)ᵀ * Bk⁻¹ * ∇f(xk)`
    If `Bk` is positive definite, its inverse `Bk⁻¹` is also positive definite. The expression `yᵀ * A * y` (with `A` being positive definite and `y` a non-zero vector) is always positive. Therefore, `∇f(xk)ᵀ * Bk⁻¹ * ∇f(xk)` is a positive quantity. With the minus sign in front, the final result is negative, confirming that `dk` is a descent direction.
### Advanced Properties for Computational Efficiency
In addition to the basic properties, others are needed to make the method truly advantageous compared to the classic Newton's method.
4.  **Reduced computational cost**: The update from `Bk` to `Bk+1` must be "cheap," meaning it must require significantly fewer operations than the full calculation of the Hessian. The ideal is to be able to calculate `Bk+1` using information already available from the current iteration.
#### The Secant Condition
To guide the update of `Bk`, the **secant condition** is introduced. This condition is fundamental in Quasi-Newton methods. To define it, we introduce two important vectors:
*   `δk = xk+1 - xk`: the change in position.
*   `γk = ∇f(xk+1) - ∇f(xk)`: the change in the gradient.
Using a first-order Taylor expansion, we can approximate the change in the gradient as:
`γk ≈ H(xk) * δk`
where `H(xk)` is the true Hessian matrix.
The idea of the secant condition is to impose that the *new* approximation `Bk+1` satisfies this relationship exactly:
`Bk+1 * δk = γk`
This equation ensures that the new matrix `Bk+1` correctly "predicts" the observed change in the gradient, given the change in position.
**Geometric Interpretation (1D case)**
In the one-dimensional (1D) case, the vectors become scalars: `δk` is a scalar, `γk` is the change in the first derivative, and `Bk+1` is a scalar approximation of the second derivative. The secant condition becomes:
`bk+1 * (xk+1 - xk) = f'(xk+1) - f'(xk)`
From which we get:
`bk+1 = (f'(xk+1) - f'(xk)) / (xk+1 - xk)`
This is nothing more than an approximation of the second derivative using finite differences of the first derivative. The secant condition in the multidimensional case is therefore a generalization of this concept.
5.  **Closeness between successive iterations**: It is required that `Bk+1` be "close" to `Bk`. This avoids abrupt jumps in the Hessian approximation, making the algorithm more stable. From a theoretical point of view, this property helps to prove that if `xk` converges to the minimum `x*`, then `Bk` also converges to the true Hessian `H(x*)`. In practice, "closeness" translates to using **low-rank updates** (rank-1 or rank-2), which modify the matrix in a controlled way.
6.  **Complexity per iteration O(n²)**: The ultimate goal is for each complete iteration of the algorithm (including the calculation of the search direction `dk`) to have a computational cost of `O(n²)`, unlike the `O(n³)` cost of Newton's method (due to solving the linear system).
### Rank-One Updates
A first attempt to satisfy these properties is to use a rank-1 update. The update formula for `Bk` is:
`Bk+1 = Bk + u * uᵀ`
where `u` is a vector to be determined and `u * uᵀ` is a rank-1 matrix.
*   **Symmetry**: If `Bk` is symmetric, `Bk+1` will also be symmetric, since `u * uᵀ` is a symmetric matrix by construction.
To find the vector `u`, we impose the secant condition `Bk+1 * δk = γk`:
`(Bk + u * uᵀ) * δk = γk`
`Bk * δk + u * (uᵀ * δk) = γk`
Solving for `u`, and after some algebraic steps, we obtain the update formula known as **Symmetric Rank-One (SR1)**:
`Bk+1 = Bk + ((γk - Bk * δk) * (γk - Bk * δk)ᵀ) / ((γk - Bk * δk)ᵀ * δk)`
**Disadvantages of the SR1 update**:
1.  **No guarantee of positive definiteness**: Even if `Bk` is positive definite, there is no guarantee that `Bk+1` will be. This means the calculated direction might not be a descent direction.
2.  **Numerical instability**: The denominator `(γk - Bk * δk)ᵀ * δk` can become zero or very close to zero, making the update unstable.
#### Direct Update of the Inverse `Hk`
Calculating `Bk+1` has a cost of `O(n²)`, but to find the direction `dk` we still need to solve a linear system with `Bk+1`, which costs `O(n³)`. To overcome this obstacle, the idea is to directly update the inverse of the Hessian, `Hk = Bk⁻¹`.
Using the **Sherman-Morrison-Woodbury formula**, which allows for the calculation of the inverse of a matrix modified by a rank-one update, a formula for updating `Hk+1` from `Hk` can be derived.
The update formula for the inverse `Hk` in the SR1 case is:
`Hk+1 = Hk + ((δk - Hk * γk) * (δk - Hk * γk)ᵀ) / ((δk - Hk * γk)ᵀ * γk)`
**Advantages of updating the inverse**:
*   **Calculation of direction `dk`**: The search direction is now calculated with a simple matrix-vector multiplication:
    `dk = -Hk * ∇f(xk)`
    This operation has a cost of `O(n²)`.
*   **Total cost per iteration**: Updating `Hk` and calculating `dk` both require `O(n²)` operations. The entire algorithm thus has the desired complexity.
### Summary and Next Steps: The BFGS Method
The rank-1 update (SR1) satisfies the secant condition and achieves `O(n²)` complexity per iteration, but it fails to ensure that the matrices remain positive definite.
This limitation is critical. To overcome it, a more robust approach is needed. The next idea, which leads to one of the most widely used Quasi-Newton methods in practice, is the **BFGS method** (Broyden-Fletcher-Goldfarb-Shanno). This method uses a rank-2 update and is designed to satisfy all six desired properties, including the preservation of positive definiteness, making it much more reliable and stable.
# Chapter 3: The BFGS Method and Cholesky Updates
## [00:00] Key Idea: Rank-2 Update and Cholesky Factorization
The basic idea is to perform a low-rank update, but instead of a rank-1 update, a rank-2 update is performed. Starting from a Hessian approximation matrix `Bk`, two vectors, `u` and `v`, are sought to update it.
The resulting formula introduces two vectors, `delta_k` and `gamma_k`, already seen before. Although this update allows for the calculation of the new `Bk+1` with a computational cost of `n²` operations, the problem of calculating the descent direction `dk` by solving a linear system, which would require `n³` operations, remains. One could use the Sherman-Morrison-Woodbury formula, but there is a smarter and more stable method that also ensures the matrix remains positive definite.
### [00:30] The Approach with Cholesky Factorization
Instead of working directly on the Hessian approximation `Bk` or its inverse `Hk`, the idea is to update the **Cholesky factor** of `Bk`.
**Context: Cholesky Factorization**
Cholesky factorization is a specialization of LU factorization for positive definite matrices. Given a positive definite matrix `B`, this factorization finds a lower triangular matrix `L` such that:
`B = L * Lᵀ` (where `Lᵀ` is the transpose of `L`).
The diagonal elements of `L` are positive. This method is so reliable that in software like MATLAB and Python, the function `chol(A)` is used to check if a matrix `A` is positive definite: if the function returns an error, the matrix is not.
The goal is therefore to define an update rule that, starting from the factor `Lk`, produces a new factor `Lk+1` such that `Bk+1 = Lk+1 * (Lk+1)ᵀ`, ensuring that `Bk+1` is positive definite.
### [01:27] Calculating the Descent Direction `dk`
Assuming we have the factor `Lk` at step `k`, the matrix `Bk` can be written as `Lk * (Lk)ᵀ`. The equation to calculate the descent direction `dk` becomes:
`Lk * (Lk)ᵀ * dk = -∇f(xk)` (where `∇f(xk)` is the gradient).
Since `Lk` is a lower triangular matrix and `(Lk)ᵀ` is upper triangular, the solution of this system can be obtained by solving two simpler triangular systems:
1.  `Lk * y = -∇f(xk)` (lower triangular system, solvable with forward substitution).
2.  `(Lk)ᵀ * dk = y` (upper triangular system, solvable with backward substitution).
Solving triangular systems requires only `n²` operations, unlike the `n³` needed for a general linear system. In this way, the calculation of the descent direction `dk` becomes efficient.
The main question now is: how to move from `Lk` to `Lk+1` efficiently?
## [02:22] Derivation of the Update Rule
The goal is to find a new factor `L+` (which corresponds to `Lk+1`) starting from a given factor `L` (corresponding to `Lk`). The new factor must satisfy the secant condition and remain "close" to the previous factor so as not to lose the accumulated information.
The problem is reformulated more generally: starting from a factor `J` such that `B = J * Jᵀ` (where `J` is not necessarily triangular), a new factor `J+` is sought that minimizes the distance from `J` (measured by the Frobenius norm) and satisfies the secant condition.
The solution to this minimization problem leads to an update formula for `J` that has the form of a rank-1 update.
### [03:30] The BFGS Update Formula
By appropriately choosing the vectors involved in the minimization problem, we arrive at an update formula for the matrix `B` that is exactly the **BFGS update formula**:
`Bk+1 = Bk + (rank-1 update) + (rank-1 update)`
This is a **rank-2** update. A fundamental aspect is that this formula is independent of the initial choice of the factor `J`. This gives us the freedom to choose a convenient `J`, such as the Cholesky factor `Lk`.
### [04:20] The Triangularity Problem and the QR Solution
If we apply the rank-1 update directly to the factor `Lk`, we get a new factor `J+` that is **no longer triangular**. The matrix `J+` is "almost" triangular, being the sum of a triangular matrix and a rank-1 matrix (which can be a full matrix).
To restore the triangular structure, **QR factorization** is used.
**Context: QR Factorization**
Every square matrix `X` can be decomposed into the product `X = Q * R`, where:
*   `Q` is an orthogonal matrix (`Qᵀ * Q = I`, where `I` is the identity matrix).
*   `R` is an upper triangular matrix with non-negative diagonal elements.
The idea is as follows:
1.  We take the matrix `J+` (which is almost triangular).
2.  We calculate its QR factorization: `J+ = Q * R`. This calculation is efficient (cost `n²`) because `J+` is already close to a triangular form and **Givens rotations** can be used.
3.  The new Cholesky factor `Lk+1` will be the matrix `R`.
Since `Bk+1 = J+ * (J+)ᵀ = (Q * R) * (Q * R)ᵀ = Q * R * Rᵀ * Qᵀ`, and since `Q` is orthogonal, we get `Bk+1 = R * Rᵀ`. We have thus found a new factor `Lk+1 = R` that is triangular, ensuring that `Bk+1` is positive definite.
## [05:55] BFGS Algorithm with Cholesky Update
The complete algorithm integrates many concepts seen so far: rank updates, QR and Cholesky factorizations, the secant condition, and line search.
**Algorithm Steps:**
1.  **Initialization:** Choose a starting point `x0`, an initial Cholesky factor `L0` (e.g., the identity matrix), and a tolerance.
2.  **Iteration:** As long as the stopping condition is not met:
    a. Calculate the search direction `dk` by solving the two triangular systems based on `Lk`.
    b. Perform a **line search** to find a suitable step `alpha_k`.
    c. Update the solution: `xk+1 = xk + alpha_k * dk`.
    d. Calculate `delta_k` and `gamma_k`.
    e. **Update the factor `Lk`** to obtain `Lk+1` using the described method (rank-1 update followed by QR factorization).
### [06:48] Properties of the BFGS Method
*   **Computational cost:** `O(n²)` per iteration.
*   **Positive definiteness:** The update ensures that the matrix `Bk` remains positive definite.
*   **Convergence:** It has local **superlinear** convergence.
*   **Quadratic functions:** For quadratic objective functions and with an exact line search, it finds the exact minimum in at most `n` iterations (where `n` is the problem dimension).
### [07:25] Comparison of Optimization Methods
| Method | Cost per Iteration | Rate of Convergence |
| :--- | :--- | :--- |
| **Gradient Descent** | `O(n)` | Linear |
| **Quasi-Newton (BFGS)** | `O(n²)` | Superlinear |
| **Newton** | `O(n³)` | Quadratic |
Quasi-Newton methods represent an ideal compromise between the low cost per iteration of gradient descent and the rapid convergence of Newton's method.
## [08:15] Deep Dive into Line Search and the Wolfe Conditions
**Line search** is the process of determining the step length `alpha_k` along the descent direction `dk`. The goal is to minimize the function `φ(α) = f(xk + α * dk)`.
Except for simple functions (like quadratic ones), solving this 1D minimization problem exactly is difficult. Therefore, an **inexact line search** is used, which consists of an inner loop of iterations to find a "good" `alpha_k`.
A "good" `alpha_k` must avoid two extremes:
1.  Being too small, causing minimal progress.
2.  Being too large, moving too far from the solution.
The simple condition `f(xk+1) < f(xk)` is not sufficient to guarantee convergence. It is necessary to impose stricter conditions, known as the **Wolfe Conditions**.
### [09:20] The Wolfe Conditions
The Wolfe conditions are a set of two rules:
1.  **Armijo Rule (or sufficient decrease condition):**
    `f(xk + α * dk) ≤ f(xk) + c1 * α * ∇f(xk)ᵀ * dk`
    *   `c1` is a constant between 0 and 1 (usually small, e.g., `10⁻⁴`).
    *   **Meaning:** It ensures that the step is not too long and that the function decreases sufficiently.
2.  **Curvature condition:**
    `∇f(xk + α * dk)ᵀ * dk ≥ c2 * ∇f(xk)ᵀ * dk`
    *   `c2` is a constant between `c1` and 1.
    *   **Meaning:** It ensures that the slope at the new position is not too negative, avoiding steps that are too short.
In practice, the line search algorithm starts with a trial step (e.g., `α = 1`) and iteratively reduces it (e.g., by halving it) until the Armijo rule is satisfied. If the curvature condition is also to be implemented, the verification loop becomes more complex.
## [11:30] Disadvantages of the BFGS Method
Despite its numerous advantages, the main disadvantage of the BFGS method is related to **memory requirements**. For a problem with `n` variables, it is necessary to store and update an `n x n` matrix (the Hessian approximation or its Cholesky factor), which can become prohibitive for large-scale problems (with very large `n`).
# Chapter 4: The L-BFGS Method for Large-Scale Problems
## [00:00] The Practical Limits of the BFGS Method
To implement the BFGS (Broyden–Fletcher–Goldfarb–Shanno) method, it is necessary to store a matrix, whether it be `Bk` (Hessian approximation), `Hk` (inverse Hessian approximation), or a Cholesky factor `Lk`. In any case, the size of this matrix is on the order of `n²`, where `n` is the number of variables in the problem.
This memory requirement represents a significant obstacle for large-scale practical applications.
### Explanation: Memory Cost
-   **What does `n²` mean?** If a problem has `n` variables, the Hessian matrix (or its approximation) will have `n` rows and `n` columns, for a total of `n * n = n²` elements.
-   **Example 1:** With 100,000 variables (`n = 10^5`), the matrix will contain `(10^5)² = 10^10` elements. Using double precision, which requires 8 bytes per number, the memory needed to store this matrix would be about 80 gigabytes.
-   **Example 2:** With 1 million variables (`n = 10^6`), the required memory would rise to about 8 terabytes.
These amounts of memory are prohibitive for most hardware systems. Consequently, although the BFGS method is theoretically promising, its practical application is limited by the need to store the entire matrix.
## [00:36] The Solution: L-BFGS (Low-Memory BFGS)
To overcome the memory limitations of BFGS, a more practical version has been developed, called **L-BFGS**, where "L" stands for "Low Memory". This method is widely available in scientific computing libraries.
### [00:48] The Key Idea of L-BFGS
The fundamental idea of L-BFGS is to never build and store the entire `Hk` matrix (the inverse Hessian approximation). Instead, `Hk` is approximated on the fly using only a subset of the information collected during the most recent iterations.
In practice, the method keeps in memory only the `m` most recent pairs of vectors `(δ, γ)`, where:
-   `δ` represents the increment in position (`x_k - x_{k-1}`).
-   `γ` represents the change in the gradient (`∇f(x_k) - ∇f(x_{k-1})`).
The value of `m` is usually very small (typically between 10 and 20), and all older information is discarded.
## [01:18] How the Approximation Works
The L-BFGS method calculates the descent direction `dk` (given by the product `Hk * ∇f(x_k)`) without ever explicitly assembling the `Hk` matrix. It leverages a recursive formula that expresses `Hk` as a function of `H_{k-1}` and the `(δ, γ)` pairs.
By iterating this formula `m` times, `Hk` can be expressed as a function of an initial matrix `H0` (an initial estimate, often a scaled identity matrix) and the `m` most recent vector pairs. This calculation is performed through a "two-loop recursion" algorithm that iteratively updates the descent direction.
### [02:08] Advantages of the "Matrix-Free" Approach
This approach is considered "matrix-free" because it does not require the construction or storage of dense `n x n` matrices. The advantages are enormous:
1.  **Computational Cost:** The cost per iteration is reduced from `O(n²)` to `O(m*n)`.
    -   **Example:** With `n = 1 million` and `m = 10`, the cost drops from `10^12` operations to about `10 * 10^6 = 10 million` operations.
2.  **Memory Cost:** The required space is reduced from `O(n²)` to `O(m*n)`.
    -   **Example:** Instead of 8 terabytes, the necessary memory becomes manageable, as only `m` pairs of vectors of size `n` need to be stored.
In summary, the L-BFGS method solves the memory problem that plagues the standard BFGS, making it suitable for large-scale problems.
## [03:08] Comparison Between BFGS and L-BFGS
### Comparative Table
| Feature | Standard BFGS | L-BFGS (Low-Memory) |
| :--- | :--- | :--- |
| **Memory Usage** | `O(n²)` | `O(m*n)` |
| **Cost per Iteration** | `O(n²)` | `O(m*n)` |
| **Convergence** | Superlinear | Linear (but very fast) |
### [03:24] Convergence Analysis
The L-BFGS method sacrifices the superlinear convergence order of BFGS, returning to **linear** convergence. However, this convergence is significantly faster than that of the Gradient Descent method. Although the order of convergence is lower, the practical improvement in terms of speed is remarkable.
## [03:48] When to Use Each Method
The choice between BFGS and L-BFGS depends on the scale of the problem:
-   **BFGS:** Ideal for **prototyping** and for **small-scale** problems (e.g., `n` less than 1000). Its rapid superlinear convergence allows for quick verification of whether a model or approach works as expected.
-   **L-BFGS:** Indispensable for **large-scale** problems (`n` approximately greater than 1000). Using standard BFGS in these cases would almost certainly lead to memory exhaustion problems.
## [04:18] Conclusion
The lesson concludes here.