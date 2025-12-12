# Chapter 1: The Universal Approximation Theorem
## [00:00] Introduction and Context
The starting point is the intuitive explanation of the **universal approximation property** for a neural network, which we previously analyzed in the one-dimensional (1D) case and hinted at for the two-dimensional (2D) case. Now, we will use concepts from functional analysis to address a more formal result.
Let's define the variables:
*   **X**: Input vector (the *features*).
*   **Z**: Target, which is the output of the neural network.
The goal is to find a function `z = f(x)` that adequately represents a neural network. A neural network is, in fact, nothing more than a function built according to a specific architecture. Our aim is to find a function `f(x)` capable of approximating a series of input data and their corresponding outputs.
### [00:48] Connection to Approximation Theory
This problem falls within the field of **approximation theory**. In this domain, given a function `f` and some of its sampled points, the goal is to construct an approximating function that describes its behavior within a certain tolerance. Common techniques to achieve this include:
*   Lagrange interpolation (simple or composite).
*   Fourier series.
In our case, the question is: given any function, is it always possible to design a neural network that approximates it with an arbitrary, user-defined tolerance?
## [01:38] The Universal Approximator: Formal Definition
Consider a function belonging to a space `S`, in which a distance `d` is defined.
A neural network is a **universal approximator** for a space `S` with distance `d` if the space `U` of all possible network outputs (i.e., all possible neural networks that can be built) is **dense** in `S`.
### [02:00] The Concept of a Dense Space
Let's recall the definition of density: a space `U` is dense in a larger space `S` if, for every element `f` chosen in `S`, it is always possible to find an element `g` in `U` that approximates it with the desired precision.
Formally, for every function `f` in `S` and for every tolerance `ε > 0`, there exists a function `g` (representing our neural network) such that:
`d(f, g) < ε`
In other words, we can approximate any function `f` with arbitrary precision using a function `g` belonging to the space of neural networks. Our goal is to prove that this result holds under specific assumptions.
## [02:48] The Role of the Activation Function
A key ingredient in defining a neural network is the **activation function**. The proof we will see is based on the use of **sigmoidal** type functions.
A function `σ: ℝ → [0, 1]` is called sigmoidal if it satisfies the following conditions:
*   `lim (x → -∞) σ(x) = 0`
*   `lim (x → +∞) σ(x) = 1`
Examples of sigmoidal functions include the classic sigmoid function, the hyperbolic tangent, and other functions with an "S" shape.
### [03:40] The Discriminatory Property
A fundamental property for the proof is the **discriminatory property** of the activation function.
An activation function `f: ℝ → ℝ` is called **n-discriminatory** if the only measure `μ` for which the following relation holds is the null measure (`μ = 0`):
`∫ σ(wᵀy + b) dμ(y) = 0`
*   **Note on the integral**: The integral is understood in the Lebesgue sense, which is why `dμ` is used instead of `dx`. In the Lebesgue integral, the complexity shifts from the function to the ability to define a measure for the sets in the domain.
In general, an activation function is called **discriminatory** if the property holds for any dimension `m` of the vector `y`.
### [04:40] Intuitive Meaning of the Discriminatory Property
A discriminatory function is **non-destructive** with respect to the input. This means that, given an input, the result of the integral operation is almost always non-zero. In practice, the function is able to "preserve" the input information, without "flattening" or nullifying it. This ability to return information is crucial for proving the universal approximation property.
## [05:15] Mathematical Tools for the Proof
To proceed, let's recall some concepts from functional analysis.
*   **Compact Set `K`**: We consider a compact set in `ℝⁿ`, such as the unit hypercube `Iⁿ` (the interval `[0, 1]` in 1D, the unit square in 2D, the unit cube in 3D, etc.).
*   **C(K)**: This is the space of continuous functions defined on the compact set `K`.
*   **M(Iⁿ)**: This is the space of regular measures. A measure is regular when, despite potentially providing different numerical values, it captures the same idea of "how much space" a function occupies relative to the total space.
### [06:00] Riesz Representation Theorem
This theorem states that, given a linear functional `L`, it is always possible to find a measure `μ` such that the functional can be expressed as an integral:
`L(f) = ∫ f dμ`
The power of this theorem lies in guaranteeing the existence of a measure `μ` that allows any linear functional to be represented as an integral.
### [06:28] Hahn-Banach Theorem
Let's consider a reformulation of the theorem that is more useful for our purposes:
If `U` is a **non-dense** linear subspace of a normed space `X`, then there exists a linear functional `L` defined on `X` such that:
*   `L` is non-zero on `X`.
*   `L` is zero on the entire subspace `U`.
Visually, if we have a large space `X` and a non-dense subspace `U` within it, we can find a functional `L` that is null in `U` but non-null elsewhere in `X`.
### [07:30] Combined Application of the Theorems
Let's apply these two theorems to the space `C(Iⁿ)`.
1.  Assume that `U` is a non-dense linear subspace of `C(Iⁿ)`.
2.  By the **Hahn-Banach theorem**, there exists a linear functional `L` that is null on `U` and non-null on `C(Iⁿ)`.
3.  By the **representation theorem**, this functional `L` can be written as an integral.
Combining the two results, we can state that there exists a measure `μ` such that:
`∫ h dμ = 0` for every function `h` belonging to the subspace `U`.
This happens because the functional is null on `U`, and its integral representation must therefore also be null for every element of `U`.
## [08:30] The Universal Approximation Theorem for Single-Layer Networks
We are now ready to state the main result. Consider:
*   An activation function `σ` that is **discriminatory**.
*   A set of functions `g(x)` defined as:
    `g(x) = Σ (from j=1 to N) αⱼ σ(wⱼᵀx + θⱼ)`
This formula represents a **neural network with a single hidden layer** and `N` neurons, where:
*   `wⱼ` are the weight vectors.
*   `θⱼ` are the biases.
*   `αⱼ` are the weights of the output layer.
The idea is a linear combination of `σ` functions, where the coefficients `αⱼ` regulate their amplitude, replicating the intuitive approach seen earlier (where we combined "jumps" and "blocks" to approximate a function).
**Theorem**: The set of these functions `g(x)` is **dense** in `C(Iⁿ)`.
In practice, this means that any continuous function defined on the hypercube `Iⁿ` can be approximated with arbitrary precision by a neural network with a single hidden layer, provided a discriminatory activation function and a sufficient number of neurons are used.
### [10:15] Proof of the Theorem (by contradiction)
The proof is based on a contradiction.
1.  **Assumption for contradiction**: Assume that the set `U` of functions `g(x)` is **not dense** in `C(Iⁿ)`.
2.  **Consequence**: If `U` is not dense, by the combined result of Hahn-Banach and Riesz, there must exist a non-null measure `μ` such that `∫ h dμ = 0` for every `h` in `U`.
3.  **Substitution**: Since the functions in `U` have the form `g(x)`, we can substitute this expression into the integral:
    `∫ [Σ αⱼ σ(wⱼᵀx + θⱼ)] dμ(x) = 0`
    This relation must hold for any choice of parameters `αⱼ`, `wⱼ`, `θⱼ`.
4.  **Simplification**: Let's choose a very simple network with a single neuron (`N=1`) and a unit output weight (`α₁=1`). The equation becomes:
    `∫ σ(wᵀx + θ) dμ(x) = 0`
5.  **The Contradiction**: Here, the conflict arises. The theorem's initial hypothesis is that `σ` is **discriminatory**. A discriminatory function implies that the integral `∫ σ(...) dμ` is zero **only if the measure `μ` is null**. However, our reasoning, based on the assumption that `U` is not dense, led us to conclude that the integral is zero for a **non-null** measure `μ`.
6.  **Conclusion**: The initial assumption (that `U` is not dense) must be false. Therefore, `U` is dense in `C(Iⁿ)`.
We have thus proven that the set of single-layer neural networks can approximate any continuous function with arbitrary precision.
# Chapter 2: Discriminatory Property of Activation Functions
## [00:00] Are Sigmoidal Functions Discriminatory?
The proof of the Universal Approximation Theorem relies on the discriminatory property of the activation function. The next step is to verify if commonly used functions, like sigmoidal ones, actually possess this property.
**Goal**: To prove that every **sigmoidal** function is **discriminatory**.
To do this, we start from the assumption that the integral of a sigmoidal function `σ` with respect to a measure `μ` is zero, and we aim to show this implies `μ` must be the null measure.
`∫ σ(wᵀx + θ) dμ = 0`
### Geometric Preliminaries
To understand the proof, let's define some concepts. A hyperplane `P` in a space is defined by the equation `wᵀx + θ = 0`, where `w` is a weight vector and `θ` is a scalar (bias). This hyperplane divides the space into two half-spaces:
*   **H+**: The set of points `x` for which `wᵀx + θ > 0`.
*   **H-**: The set of points `x` for which `wᵀx + θ < 0`.
A fundamental lemma states that if a measure `μ` is null on all hyperplanes and on all half-spaces (both H+ and H-), then that measure is identically null.
### Proof of the Discriminatory Property
1.  **Constructing a Scaled Function:** We introduce an auxiliary function `σ_λ(x) = σ(λ * (wᵀx + θ) + φ)`, where `λ` is a scaling parameter and `φ` is a shift. We analyze the behavior of this function as `λ` tends to infinity.
2.  **Limit Analysis:** The limit of `σ_λ` as `λ → ∞` depends on the sign of the argument `wᵀx + θ`:
    *   If `x` is in the positive half-space (H+), the argument tends to `+∞`, and the limit of `σ_λ` is **1**.
    *   If `x` is in the negative half-space (H-), the argument tends to `-∞`, and the limit of `σ_λ` is **0**.
    *   If `x` is on the hyperplane (P), the argument is `φ`, and the limit of `σ_λ` is **σ(φ)**.
    Let's call this limit function `γ(x)`.
3.  **Applying the Dominated Convergence Theorem:** Thanks to Lebesgue's Dominated Convergence Theorem, we can swap the limit and the integral:
    `lim (λ→∞) ∫ σ_λ dμ = ∫ (lim (λ→∞) σ_λ) dμ = ∫ γ dμ`
    Substituting the values of `γ(x)` we found, the integral becomes:
    `1 * μ(H+) + 0 * μ(H-) + σ(φ) * μ(P) = μ(H+) + σ(φ) * μ(P)`
    Since the initial integral was zero by hypothesis, we get:
    `μ(H+) + σ(φ) * μ(P) = 0`
4.  **Analysis as `φ` Varies:** This relation holds for any value of `φ`. Let's now analyze the limit cases for `φ`:
    *   If `φ → +∞`, `σ(φ)` tends to 1. The equation becomes: `μ(H+) + μ(P) = 0`.
    *   If `φ → -∞`, `σ(φ)` tends to 0. The equation becomes: `μ(H+) = 0`.
5.  **Conclusions on the Measure:** Comparing the two results, if `μ(H+) = 0`, then `μ(P)` must also be zero. By symmetry, the same reasoning can be repeated to show that `μ(H-) = 0` as well. We have thus shown that the measure `μ` is null on the hyperplane and on both half-spaces. Based on the initial lemma, this implies that **`μ` is the null measure**.
This completes the proof: if the integral of a sigmoidal function with respect to `μ` is zero, then `μ` must be zero. Therefore, **sigmoidal functions are discriminatory**.
## [02:58] Consequences for "Shallow" Neural Networks
Since sigmoidal functions are discriminatory, the universal approximation theorem is valid for neural networks with a single hidden layer ("shallow" networks) that use these activation functions. This means a linear combination of sigmoidal functions can approximate any continuous function on a compact domain with arbitrary accuracy.
## [03:40] The Case of the ReLU Function
A natural question is whether other activation functions, like ReLU (Rectified Linear Unit), are also discriminatory. ReLU is not a sigmoidal function, as its limit for `x → +∞` is `+∞`.
### Proof of Discriminatory Property for ReLU (1D case)
Let's prove that ReLU is discriminatory in the one-dimensional (1D) case. The idea is to use a combination of ReLU functions to construct a function that behaves like a sigmoid.
1.  **Hypothesis:** We start, as before, with the assumption that the integral of the ReLU function with respect to a measure `μ` is null, and we want to prove that `μ = 0`.
2.  **Constructing a Sigmoid-like Function:** It is possible to construct a function `f(x)` with a sigmoidal shape (e.g., zero for `x < 0`, linearly increasing between 0 and 1, and constant for `x > 1`) using the difference of two appropriately shifted ReLU functions:
    `f(x) = ReLU(w*x + θ₁) - ReLU(w*x + θ₂)`
3.  **Integration and Deduction:** We integrate this function `f(x)` with respect to the measure `μ`:
    `∫ f(x) dμ = ∫ [ReLU(...) - ReLU(...)] dμ = ∫ ReLU(...) dμ - ∫ ReLU(...) dμ`
    By the initial hypothesis, both integrals on the right are null, so `∫ f(x) dμ = 0`.
4.  **Conclusion:** The function `f(x)` we constructed is a sigmoidal function. Since we have already proven that sigmoidal functions are discriminatory, the fact that its integral with respect to `μ` is null necessarily implies that **`μ = 0`**.
This proves that the ReLU function is also discriminatory in the 1D case. The result can be generalized to the multi-dimensional case, confirming that neural networks with ReLU activation also enjoy the universal approximation property.
### Handling Non-Differentiable Activation Functions (ReLU)
A common assumption in approximation theorems is that the activation function sigma (σ) is infinitely differentiable. This poses a problem for the **ReLU (Rectified Linear Unit)** function, which is not differentiable at the origin.
**Solution:** This obstacle can be overcome. The ReLU function can be "regularized" or "smoothed" around the origin. The idea is to select an increasingly smaller interval around the point of non-differentiability and create a smooth connection between the two branches of the function. By applying this technique, theoretical results can be extended even to non-smooth functions like ReLU.
# Chapter 3: Shallow vs. Deep Networks: Complexity and Efficiency
## [06:33] Limits of the Universal Approximation Theorem
The Universal Approximation Theorem is a **non-constructive existence theorem**. It guarantees that a neural network capable of approximating a given function *exists*, but it provides no guidance on how to build it:
*   It does not suggest the number of neurons needed.
*   It does not provide a method for finding the optimal weights and biases.
## [07:28] Network Complexity and the Curse of Dimensionality
Once existence is established, the next question concerns the **complexity** of the required network. How many neurons are needed to achieve a certain precision?
### "Shallow" Networks and the Curse of Dimensionality
For networks with a single hidden layer (shallow), it can be shown that the approximation error `E` depends on the number of neurons `N` according to the relation:
`E ≈ N^(-r/d)`
where:
*   `r` is the order of differentiability (regularity) of the function to be approximated.
*   `d` is the dimension of the input space (the number of features).
Inverting this formula to find the number of neurons `N` needed to achieve a tolerance `ε`, we get:
`N ≈ ε^(-d/r)`
This relationship highlights a critical problem: the number of neurons `N` grows **exponentially** with the input dimension `d`. This phenomenon is known as the **curse of dimensionality**. For problems with a high number of features, a "shallow" network might require a prohibitive number of neurons to achieve good accuracy.
### "Deep" Networks and Compositional Structure
Deep networks can mitigate this problem, especially if the function to be approximated has a **compositional** or hierarchical structure.
**Example of Compositional Structure:**
Imagine a function `f` with 8 input variables (`x₁`...`x₈`). If `f` is compositional, it can be decomposed into a hierarchy of simpler functions, each acting on a limited number of inputs.
*   **Level 1:** Functions `h₁` that operate on pairs of inputs (e.g., `h₁₁(x₁, x₂)`).
*   **Level 2:** Functions `h₂` that operate on the outputs of the previous level.
*   ... and so on, up to a single final function.
If a function has this structure and a deep network is used to approximate it, the number of neurons required for a given tolerance `ε` becomes:
`N ≈ (d-1) * ε^(-d̃/r)`
where:
*   `d` is the total input dimension.
*   `d̃` is the number of inputs for each sub-function in the hierarchy (in the example, `d̃ = 2`).
In this case, the exponential dependence is no longer tied to the total dimension `d`, but to the much smaller dimension `d̃`. This makes deep networks much more efficient than "shallow" networks for approximating complex functions that possess a hierarchical structure.
It has been shown that **any function can be approximated with arbitrary accuracy** using a binary representation (where `d̃` is 2). Therefore, considering this class of functions is not a significant constraint.
### Practical Example: Shallow vs. Deep
Let's consider a realistic case:
-   Input dimension (D): 1000
-   Function regularity (L): 10
-   Desired accuracy (ε): 10⁻²
-   Binary functions (`d̃`): 2
**Results:**
-   **Shallow Network:** Would require approximately **10²⁰⁰ neurons**, a computationally unfeasible number.
-   **Deep Network:** Would require approximately **2500 neurons**, a manageable number.
This enormous difference explains why, in practice, **deep neural networks are almost always the preferred choice** over shallow ones.
## The Power of Composition in Deep Networks
### Building Complexity with Composition
Recall how, starting from the ReLU function, it is possible to construct a "hat function" on the interval [0, 1].
If we compose this function `g(x)` with itself (`g(g(x))`), the number of "peaks" doubles.
-   **Depth 1 (g):** 1 peak (2⁰)
-   **Depth 2 (g o g):** 2 peaks (2¹)
-   **Depth 4:** 8 peaks
To approximate a function with `2^k` peaks:
-   A **shallow network** needs a number of neurons on the order of `2^k` (one neuron per peak).
-   A **deep network** needs only `k` neurons.
Here again, the complexity for shallow networks is exponential, while for deep networks it is linear (or logarithmic, depending on the perspective), confirming their superiority.
## Summary Comparison: Shallow vs. Deep
| Feature | Shallow Network (single-layer) | Deep Network |
| :--- | :--- | :--- |
| **Number of Neurons** | Very high, exponential growth with input dimensionality. | Manageable, linear (multiplicative) growth with dimensionality. |
| **Computation** | Parallel: all neurons process the input simultaneously. | Serial and parallel: parallel computation per layer, but serial between layers. |
| **Mathematical Structure** | Linear combination of functions. | Composition of functions. |
| **High-Frequency Functions** | Number of neurons on the order of `N`. | Number of neurons on the order of `log(N)`. |
| **Symmetry Handling** | Difficult and requires many neurons to represent symmetries in data. | Much more efficient at capturing and representing symmetries. |
| **Optimization** | The optimization problem (e.g., with Mean Square Error) is **convex**, easier to solve. | The problem is **non-convex**, with many local minima, making optimization more complex. |
### Key Conclusion
Although the existence of a solution is guaranteed for both architectures, **shallow networks suffer from exponential complexity** with respect to the input dimension. **Deep networks, on the other hand, have multiplicative complexity**, which makes them the practical choice for almost all modern applications.
# Chapter 4: Administrative and Organizational Discussion
## [Speaker 1] [Speaker 2] Appointment and Project Management
*   An appointment is set for the following Wednesday at 3:30 PM at the professor's office (Department of Mathematics, 7th floor) to discuss a project.
*   An office phone number (022-399-4517) is exchanged to facilitate contact.
*   **Project Presentations:** The presentation will take place about 10-12 days after the written exam. There will be formal sessions (January) and informal ones (e.g., April), but grade registration for informal sessions will only occur during official windows (e.g., June-July).
## [Speaker 1] [Speaker 2] Exam Overlap Issue
*   An issue of overlapping exams is raised for some students (e.g., with "Algorithms and Parallel Computing" and "Operations Research" on the 16th).
*   The professor expresses difficulty in moving the exam date, as it could create new overlaps and there are constraints on classroom availability.
*   A student has already contacted the dean's office to report the problem.
*   The professor commits to checking the possibility of moving the start time (e.g., to 4:00 PM), although this would mean finishing the exam very late (around 7:00 PM).
*   The possibility of creating two separate exam sessions for different degree programs (e.g., Mathematical Engineering, HPC, Computer Science) is discussed.
*   **Professor's Warning:** If exams are held on different dates, the tests will be different. The professor makes it clear from the outset that he will not accept post-exam complaints about alleged differences in the difficulty level between the two papers.