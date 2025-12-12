Here is the translation and reorganization of the text into a flowing, study-friendly format.
## Summary of Optimization Methods
### [00:00] Overview of Analyzed Methods
To summarize what we've seen so far, we started with the **gradient descent (steepest descent)** method, in both its classic and stochastic versions. For both, we observed linear convergence and a computational cost, in terms of the number of operations, on the order of O(n), where 'n' is the number of unknowns in the problem.
Next, we introduced the **Newton's method** as an example of a second-order method. We saw that, similar to the one-dimensional (1D) case where the proof is quite simple, this method guarantees **quadratic convergence**. However, this improvement in the order of convergence comes at a high cost in terms of operations per step: O(n³).
To find a compromise, we analyzed **quasi-Newton methods**. These methods approximate the action of the Hessian matrix. Consequently, we cannot expect the quadratic convergence of Newton's method, but we achieve **superlinear convergence**, which is a middle ground. The computational cost per iteration is reduced to O(n²).
### [00:48] The BFGS Method and its Low-Memory Variant (L-BFGS)
Within the quasi-Newton methods, we examined the **BFGS method** as a strategy for approximating the Hessian matrix. Although the standard BFGS reduces the computational cost to O(n²), it has a significant problem related to memory requirements: it is necessary to store the approximated Hessian matrix, which requires O(n²) space.
To overcome this limitation, the **L-BFGS (Low-memory BFGS)** variant was introduced. The key idea is not to store the entire Hessian matrix, but to keep only a subset of the history of vectors used to calculate it (e.g., those from the last 10 or 12 iterations). Using this "moving window" of vectors, the Hessian is reconstructed at each step.
This approach leads to a drastic reduction in memory requirements, which go from O(n²) to O(n * m), where 'm' is the number of past iterations stored (e.g., m=10).
*   **Practical Example:** If n is 1 million (10⁶), the required memory drops from 10¹² elements (BFGS) to about 10⁷ (L-BFGS with m=10), a huge saving.
The operational cost per step also benefits from this reduction. The trade-off, however, is a return to **linear convergence**, losing the superlinearity of the classic BFGS.
**When to use BFGS or L-BFGS?**
*   **L-BFGS:** It is the preferred choice when the number of unknowns 'n' is very large (large-scale or "in production" applications).
*   **BFGS:** It can be used when 'n' is small, for example, during prototyping or for testing new ideas.
## [01:58] The Levenberg-Marquardt Algorithm
Another very popular and widespread method in practical applications is the **Levenberg-Marquardt (LM) algorithm**. Let's analyze the ideas behind this algorithm, starting from the problem of **non-linear least squares**.
### [02:11] Context: The Non-Linear Least Squares Problem
As we have seen several times, the goal is to find a weight vector `w` that allows creating a non-linear model for a given set of points `(xi, yi)`.
1.  **Residual:** For each sample, we calculate the residual `r`, which represents the difference between the value predicted by the model and the actual value.
2.  **Cost Function:** The cost function `J(w)` is the sum of the squares of the residuals. In vector form, it is written as `J(w) = RᵀR`, which corresponds to the squared norm of the residual vector `||R||²`.
    *   *Note: A factor of 1/2 is often added in front of the cost function (`1/2 RᵀR`) to simplify derivative calculations, but it is not a conceptually fundamental element.*
The objective is to find the vector `w` that minimizes `J(w)`.
### [03:08] Comparison between Gradient Descent and Newton's Method
*   **Gradient Descent:** The weight update follows the direction opposite to the gradient: `w_k+1 = w_k - γ * ∇J`. This method is stable and converges, but it is very slow because it only uses first-order information.
*   **Newton's Method:** It uses the Hessian (second-order information). It is much faster but computationally expensive, due to the calculation and inversion of the Hessian matrix.
The **goal of the Levenberg-Marquardt algorithm** is to combine the advantages of both: to achieve the convergence speed of Newton's method while avoiding the explicit calculation of the Hessian.
### [04:00] Derivation and Gauss-Newton Approximation
To develop the algorithm, we recalculate the gradient and the Hessian of our cost function `J(w)`.
*   **Gradient:** `∇J = 2 * JᵀR`, where `J` is the Jacobian matrix of the residual function `R`.
*   **Hessian:** `H = 2 * JᵀJ + 2 * Σ(ri * H_ri)`, where `H_ri` is the Hessian of the single residual `ri`.
The Hessian is therefore composed of a first-order term (`2 * JᵀJ`) and a second-order term (`2 * Σ(ri * H_ri)`).
The fundamental idea of the **Gauss-Newton method** is to approximate the Hessian by discarding the second-order term:
`H ≈ 2 * JᵀJ`
**When is this approximation valid?**
1.  **Small residuals:** If the residuals `ri` are close to zero (i.e., the model fits the data well and we are close to the solution), the second term becomes negligible.
2.  **Quasi-linear model:** If the model is close to being linear, the second derivatives `H_ri` are close to zero.
### [05:40] The Gauss-Newton Method
Starting from the classic Newton update step, `H * Δw = -∇J`, we replace the gradient and the Hessian with their approximated expressions:
`(2 * JᵀJ) * Δw = -(2 * JᵀR)`
Simplifying, we obtain the fundamental equation of the Gauss-Newton method, analogous to the normal equations of linear least squares:
`(JᵀJ) * Δw = -JᵀR`
By solving this linear system for `Δw`, we get the update direction.
**Advantages:**
*   **Quadratic convergence:** If the method converges, the convergence is quadratic near the solution, where the Hessian approximation is valid.
*   **No Hessian calculation:** Calculating `JᵀJ` is much simpler than calculating the full Hessian. For example, using automatic differentiation (AD) in reverse mode (backward), each column of the Jacobian `J` can be calculated in a single pass.
**Disadvantages:**
*   **Can diverge:** If the initial guess is far from the solution, the Hessian approximation is not good and the algorithm can diverge.
*   **Ill-conditioned matrix:** The matrix `JᵀJ` could be singular or ill-conditioned, making the solution of the linear system unstable and unreliable.
### [07:20] The Key Idea of Levenberg-Marquardt: Adaptive Interpolation
The Levenberg-Marquardt (LM) algorithm aims to combine the stability of gradient descent with the speed of the Gauss-Newton method. The idea is to introduce a "damping" or regularization term into the Gauss-Newton equation:
`(JᵀJ + λI) * Δw = -JᵀR`
where `λ` (lambda) is a damping parameter and `I` is the identity matrix.
The parameter `λ` acts as a slider that interpolates between the two methods:
*   **If `λ → 0`:** The equation becomes `(JᵀJ) * Δw = -JᵀR`, which is the **Gauss-Newton method** (fast but potentially unstable).
*   **If `λ → ∞`:** The `λI` term dominates, and the equation simplifies to `λ * Δw ≈ -JᵀR`, which corresponds to a **gradient descent** step (stable but slow).
The real strength of the LM algorithm is that `λ` is not a constant, but is **dynamically updated** at each iteration. Furthermore, the addition of the `λI` term ensures that the matrix `(JᵀJ + λI)` is always invertible, solving the problem of ill-conditioning.
### [09:50] The Levenberg-Marquardt Algorithm in Practice
The algorithm proceeds as follows:
1.  **Initialization:** An initial weight vector `w₀` and an initial value for `λ₀` are chosen.
2.  **Iteration:** At each step `k`:
    a. The residual `R` and the Jacobian `J` are calculated for the current weights `w_k`.
    b. The cost function `J(w_k)` is calculated.
    c. The system `(JᵀJ + λ_k * I) * Δw_k = -JᵀR` is solved to find the increment `Δw_k`.
    d. A new candidate for the weights is calculated: `w_new = w_k + Δw_k`.
    e. The cost function is evaluated on the new candidate: `J(w_new)`.
3.  **`λ` Update Strategy:**
    *   **If `J(w_new) < J(w_k)` (the step improved the solution):**
        *   The step is **accepted**: `w_k+1 = w_new`.
        *   We are in a "good" region, so we can be more audacious. **`λ` is reduced** (e.g., `λ_k+1 = λ_k / μ`), moving towards the Gauss-Newton method to accelerate convergence.
    *   **If `J(w_new) ≥ J(w_k)` (the step did not improve or worsened the solution):**
        *   The step is **rejected**: `w_k+1 = w_k`.
        *   We are in a "difficult" region, so we must be more cautious. **`λ` is increased** (e.g., `λ_k+1 = λ_k * μ`), moving towards gradient descent to ensure stability. Step `c` is repeated with the new value of `λ`.
4.  **Stopping Criterion:** The algorithm stops when the norm of the increment `||Δw||` or the change in the cost function falls below a predefined tolerance threshold.
**Why is LM so effective?**
The LM algorithm adaptively automates a strategy that is often implemented manually: starting with a few steps of a stable method like gradient descent to get close to a good region and then switching to a faster method like BFGS or L-BFGS to refine the solution. LM does all this in a single dynamic framework.
### [12:20] Computational Cost and Limitations
*   **Cost:** A disadvantage is that at each iteration, an n×n linear system must be solved, which involves a cost of O(n³). There are various technical variants to reduce this cost, but the basic idea remains the one presented.
*   **Local Minima:** Like all the methods we've seen, LM does not guarantee finding the global minimum. It will find a local minimum, and which minimum is reached depends heavily on the choice of the starting point (initial guess).
*   **Hyperparameters:** Convergence is also influenced by other hyperparameters, such as the initial value of `λ` and the update factor `μ`. However, for standard applications, the literature provides reasonable and safe values or ranges of values.
In summary, for non-linear least squares problems, the Levenberg-Marquardt algorithm is often the method of choice.
### [13:50] Conclusions on Minimization Methods
We have introduced classic minimization methods, some improvements, and more recent developments used in machine learning. In the literature and in software libraries, there are many other variants, often optimized for specific applications. However, a solid understanding of the methods we have analyzed (gradient descent, Newton, quasi-Newton, LM) provides the foundation for understanding these more specialized variants as well.
# Introduction to Convolution
## [00:00:05] The Concept of Convolution
In mathematics, convolution is an operation that describes how the shape of one function is modified by another. Imagine we have a signal, which we will call `F`, and a filter, `G`. The convolution, indicated by the symbol `*`, shows us how the filter `G` acts on the signal `F`.
Convolution is fundamental in signal and image processing. Filters are tools designed to detect specific features (like corners or edges) or to modify the original image (for example, by applying a blur effect).
## [00:01:20] Discrete Convolution and Toeplitz Matrices
In practical applications and machine learning, we don't use convolution in its general sense, but its discrete counterpart. This is based on a specific class of matrices called **Toeplitz matrices**.
A Toeplitz matrix has a particular structure: once the first row is defined (e.g., A, B, C, D), the subsequent rows are obtained by shifting the previous row along the diagonal. The main characteristic is that the matrix is constant along each of its diagonals.
### [00:02:00] Circulant Matrices
An even more important subclass of Toeplitz matrices are **circulant matrices**, which are the true protagonists of discrete convolution. The fundamental difference is that when the first row is shifted to the right, the last element "reappears" at the beginning of the next row, creating a cyclic effect.
Consequently, a circulant matrix is completely defined by a single vector: the one that constitutes its first row. The size of the vector determines the size of the matrix.
**Properties of Circulant Matrices:**
1.  The product of two circulant matrices is still a circulant matrix.
2.  The product is commutative: `C * D = D * C`, an uncommon property for matrices.
## [00:03:08] Circulant Matrices and Polynomials
We can represent a circulant matrix in an alternative way. We introduce a **permutation** (or *shift*) **matrix**, denoted by `P`. This matrix, when multiplied by another, permutes its rows (if premultiplied) or columns (if postmultiplied).
For example, if `P` is a matrix that cyclically shifts the rows by one position, `P²` will shift them by two positions, and so on, until `Pⁿ` which, for an `n x n` matrix, will be equal to the identity matrix `I`.
Thanks to this property, every circulant matrix `C`, defined by the vector `c = (c₀, c₁, ..., cₙ₋₁)`, can be written as a polynomial in the matrix `P`:
`C = c₀ * I + c₁ * P + c₂ * P² + ... + cₙ₋₁ * Pⁿ⁻¹`
In practice, a circulant matrix `C` is the result of evaluating a polynomial (whose coefficients are the elements of the vector `c`) in the shift matrix `P`.
### [00:05:40] Cyclic Product of Polynomials
This polynomial representation simplifies the product between circulant matrices. The product of two circulant matrices `C` and `D` corresponds to the product of their respective polynomials `p(P)` and `q(P)`.
This is not a standard polynomial product, but a **cyclic product**. Since `Pⁿ = I`, every term of degree `n` or higher is "wrapped around" to lower degrees. For example, in a 3x3 context, `x³` behaves like `1`, so its coefficient is added to the degree-zero term.
In summary, the complex product between matrices is reduced to a cyclic product between the vectors that define their first rows.
## [00:07:25] Eigenvectors and Eigenvalues of Circulant Matrices
An extraordinary property of circulant matrices is that **all circulant matrices of a given size `n x n` share the same eigenvectors**.
These eigenvectors are the columns of the so-called **Fourier matrix**, a matrix constructed using the n-th roots of unity (complex numbers that, when raised to the power of `n`, give 1).
The **eigenvalues**, on the other hand, depend on the specific matrix. They are obtained by calculating the **Discrete Fourier Transform (DFT)** of the vector `c` that defines the first row of the matrix.
In short:
-   **Eigenvectors**: Fixed for a given size `n`, they are the columns of the Fourier matrix.
-   **Eigenvalues**: Specific to each matrix, they are calculated with the DFT of its generating vector.
## [00:09:48] The Convolution Rule and Computational Efficiency
The relationship between circulant matrices and the Fourier Transform leads to a fundamental rule: the **convolution rule**.
Since every circulant matrix `C` can be diagonalized using the Fourier matrix `F` (`C = F⁻¹ * Λ * F`, where `Λ` is the diagonal matrix of eigenvalues), the convolution product between two vectors `c` and `d` (which is equivalent to `C * d`) can be calculated much more efficiently.
**Standard procedure (direct product):**
-   Requires about `n²` operations.
**Procedure with the convolution rule:**
1.  Calculate the DFT of `d` (product `F * d`).
2.  Calculate the eigenvalues of `C` (the DFT of `c`).
3.  Multiply the two results element-wise (Hadamard product).
4.  Apply the inverse Fourier transform (`F⁻¹`) to get the final result.
Using the **Fast Fourier Transform (FFT)** algorithm, this process requires only `n * log(n)` operations. For large `n`, the efficiency gain is enormous.
**Key concept:** Convolution in the time (or space) domain is equivalent to a simple element-wise multiplication in the frequency domain.
## [00:12:10] 2D Convolution in Images
The same ideas extend from 1D sequences to 2D images. An image can be seen as a large matrix of pixels. A **filter** (or *kernel*) is a small matrix that is slid over the image.
For each position, the effect of the filter on a small portion of the image is calculated, producing a single pixel in the output image. This pixel summarizes the information of the analyzed region, such as the presence of an edge or a certain texture.
**Examples of filters:**
-   **Blur/Soften:** A kernel that calculates an average of the surrounding pixels, softening sharp transitions. For example, a value of 90 surrounded by low values can be "smoothed" to 19.
-   **Edge Detection:** A kernel designed to produce a high value when it detects a sharp change in pixel intensity in a specific direction (e.g., vertical).
## [00:14:40] Practical Example: Convolutional Neural Networks (CNNs)
Convolutional Neural Networks (CNNs) leverage these concepts for tasks like image recognition. Let's see an example of how a CNN recognizes a handwritten digit:
1.  **Input Layer:** The original image of the digit (e.g., a "3").
2.  **Convolutional Layer:** Several filters are applied to the input image. Each filter is specialized in detecting a different feature (curved lines, corners, etc.), producing several "feature maps".
3.  **Downsampling (Pooling) Layer:** The size of the feature maps is reduced to simplify the information and make the network more robust to small variations. A common technique is **Max Pooling**, which selects the maximum value from a small region (e.g., 2x2 or 3x3 pixels), keeping only the most prominent features.
4.  **Subsequent Layers:** Other convolutional and pooling layers are alternated to extract increasingly complex and abstract features.
5.  **Fully Connected Layer:** At the end, the two-dimensional feature maps are "flattened" into a single vector. This vector is processed by layers of a traditional neural network to classify the image.
6.  **Output Layer:** The final layer produces a probability vector. The position with the highest value indicates the predicted class (e.g., the highest value is at the position for the digit "3", indicating that the network has recognized that number).
This cascaded process allows the network to start from simple pixels, recognize edges and shapes, and finally identify complex objects like a digit, a face, or a pizza, even if their appearance is not perfectly standard.
# Universal Approximation of Neural Networks: A Visual Proof
## [00:00] Introduction: The Problem of Function Approximation
So far, in the context of neural networks, we have defined their topology (input layer, hidden layers, output layer) and their fundamental components, such as activation functions and cost functions.
We have seen that training a neural network means finding the set of weights and biases that minimize the cost function. To do this, first-order or second-order methods are used, which require the calculation of first derivatives (gradients) or second derivatives (Hessians). We have also learned how to calculate these derivatives through automatic differentiation.
Now, we want to address a more theoretical question: why did we choose a neural network to solve our problems?
### [00:40] Formalizing the Problem
Our goal is to find a function `f` that, given an input `x` (e.g., an image), returns an output `y` (e.g., the corresponding digit).
```
f(x) = y
```
The neural network, which we will denote as `fn`, is nothing more than a complex composite function, defined by its architecture, weights, biases, and activation functions.
The fundamental question is: **is this structure `fn` capable of representing any type of function `f`?** In other words, is the neural network a **universal approximator**?
To answer, we will follow two approaches:
1.  **Intuitive and Graphical Approach (Today):** A visual proof, starting from a continuous function `f` in one dimension (1D).
2.  **Formal Approach (Next Lesson):** A rigorous mathematical proof.
### [02:00] Limitations of the Universal Approximation Theorem
Before we begin, it is important to highlight two crucial aspects of the universal approximation theorem:
1.  **Existence Result:** The theorem guarantees that a neural network capable of approximating a given function **exists**, but it **does not provide a constructive method** to build it. It does not tell us how to find the optimal weights and biases.
2.  **Accuracy and Complexity:** The approximation depends on a desired tolerance `ε` (epsilon). The difference between the real function `f` and its approximation `f_tilde` must be less than this tolerance.
    ```
    || f - f_tilde || < ε
    ```
    Achieving very high accuracy (a very small `ε`) might require a neural network with such a large number of parameters (neurons, weights) that its construction becomes impractical. This introduces the concept of **neural network complexity**.
## [04:10] Visual Proof with the Sigmoid Function
The idea of the visual proof is to build, step by step, an approximation of any function using the basic building blocks of a neural network.
**Key steps:**
1.  Start with the **sigmoid function**.
2.  Transform it into a **step function** by manipulating its parameters.
3.  Combine two step functions to create a **rectangular pulse function**.
4.  Sum many rectangular pulse functions to approximate any curve.
### [04:55] From Sigmoid to Step Function
The sigmoid function `σ(z)` has the argument `z = wx + b`. Let's see how the parameters `w` (weight) and `b` (bias) modify its shape.
*   **Increasing the weight `w`:** If we increase `w` to very large values (e.g., 50, 100), the slope of the sigmoid becomes extremely steep, transforming it into an excellent approximation of a **step function**.
*   **Modifying the bias `b`:** By changing `b`, we can shift the position of the "jump" (the transition point). The position of the jump `x_jump` is given by the formula:
    ```
    x_jump = -b / w
    ```
So, by playing with `w` and `b`, we can create a step function and position it wherever we want on the x-axis.
### [07:48] From Step Function to Rectangular Pulse
To create a rectangular pulse, we need to combine two neurons (and therefore two step functions). The idea is to use two weights of opposite sign (e.g., `H` and `-H`).
Using an interactive implementation, we set:
*   Very large weights `w` (e.g., 500) to have sharp steps.
*   Two positions for the jumps, `S1` and `S2`, which determine the biases `b1` and `b2`.
*   Two output weights, `w01 = 1` and `w02 = -1`.
By combining these two functions, we obtain a rectangular pulse whose base is defined by the interval between `S1` and `S2`.
### [09:20] Approximating a Function with Rectangular Pulses
This approximation technique is very reminiscent of the definition of the **Riemann integral**, where the area under a curve is approximated by summing the areas of many rectangles.
Here we do something similar:
1.  We divide the domain of the function `f` into small intervals.
2.  In each interval, we build a rectangular pulse using two neurons.
3.  The height of each rectangle can be adjusted via an additional parameter (a weight in the output layer), so that it corresponds to the value of the function `f` at that point.
By increasing the number of rectangles (and thus the number of neurons), we can make the base of each rectangle smaller and smaller, obtaining an approximation of the function `f` with arbitrary accuracy.
*   **Example of Complexity:**
    *   5 rectangles require about 10 neurons.
    *   15 rectangles require about 30 neurons.
    *   50 rectangles require about 100 neurons.
This visual proof confirms the universal approximation property. The same concept extends to 2D functions (surfaces), where instead of rectangles, we use rectangular prisms, built by combining four functions.
## [12:30] Visual Proof with the ReLU Function
In practice, one of the most used activation functions is not the sigmoid, but **ReLU (Rectified Linear Unit)**. ReLU is defined as `max(0, z)` and is not a sigmoidal function, as it is unbounded on the right.
The universal approximation theorem holds for any non-linear "S-shaped" (sigmoidal) activation function. But does it also hold for ReLU? Let's see with a similar visual proof.
The idea is to use ReLU to build a **"hat function"**, similar to those used in the finite element method (FEM). A hat function is piecewise linear and non-zero only in a limited interval (compact support).
Once we have this basic function, we can combine many of them, each centered on a different node, to approximate any continuous function with a piecewise linear function.
### [15:30] From ReLU to "Hat Function"
For ReLU as well, the argument is `z = wx + b`.
*   **Modifying `b`:** Shifts the position of the "kink point".
*   **Modifying `w`:** Changes the slope of the non-zero part.
The position of the kink is always given by `x_kink = -b / w`.
To build the "hat function", we combine three ReLU functions:
1.  A ReLU that goes up.
2.  A second ReLU, with a larger, negative weight (e.g., -2), which reverses the slope and makes it go down.
3.  A third ReLU that brings the function back to zero, making it flat outside the interval of interest.
With an appropriate combination of weights and biases (e.g., weights `1`, `-2`, `1`), we get exactly the desired shape. The height of the "hat" can be scaled with an additional weight, just as we did for the rectangles.
### [17:50] Alternatives and Conclusions
It is interesting to note that by combining only two ReLU functions, one can obtain a sigmoid-like function. From there, one could follow the same path seen before (step -> rectangle). However, this would require four ReLUs for each rectangle. The "hat function" approach is more efficient (requires only three ReLUs) and produces a piecewise linear approximation, which is generally more accurate than a piecewise constant (step) approximation.
**In conclusion:**
We have visually demonstrated that neural networks, with both sigmoid and ReLU activation functions, can approximate any continuous function in 1D with arbitrary accuracy, provided we increase the number of neurons. This concept also extends to higher dimensions.
Next time, we will move from this empirical proof to a formal and rigorous proof of the theorem.