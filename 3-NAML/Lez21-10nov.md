# Beyond Stochastic Gradient Descent (SGD)
## [00:00] Limits of Stochastic Gradient Descent (SGD)
### [00:07] The Importance of Choosing the Learning Rate
If you have ever used machine learning or deep learning libraries, you will have realized that the choice of the **learning rate** is a hyperparameter of fundamental importance for the algorithm's performance.
The general rule, as previously emphasized, is as follows:
*   **Learning rate too small:** Oscillations are almost certainly avoided, but convergence can become extremely slow.
*   **Learning rate too large:** Convergence can be accelerated, but there is a risk of generating oscillations or, in some cases, even divergence, as we saw in the previous lesson.
A solution to this problem is the use of a **learning rate schedule**, which is a strategy that reduces its value as the number of iterations increases. However, this approach has a significant limitation: it is "blind," as it does not take into account the specific characteristics of the problem or the dataset being analyzed.
### [01:20] The Challenge of Sparse Datasets
In most practical applications (about 80% of cases), we deal with **sparse datasets**.
**What does "sparse dataset" mean?**
A typical example is movie ratings. A user might have rated only 20 movies out of the thousands available on a platform. This means that some *features* are much rarer than others. From the perspective of the movies, some famous titles receive many ratings, while niche films receive very few.
In the presence of sparse data, it is crucial for the optimization method to adapt the learning rate specifically for each parameter of the network, instead of using a single value for all. In particular:
*   Parameters associated with **rarer features** should have a **higher learning rate**, because they are "activated" and updated only a few times.
### [02:30] The Complexity of Real Cost Functions
The cost functions we encounter in practice are very different from ideal convex functions. Even in two-dimensional spaces, their visualization reveals a complex "landscape," characterized by:
*   A global minimum.
*   Many local minima.
*   Saddle points.
These features make optimization much more difficult than in the convex case.
### [03:10] Practical Example: Saddle Points and Local Minima
Let's consider a cost function with a global minimum, a local minimum, and a saddle point.
*   **Sensitivity to the starting point:** Starting from different points, the gradient descent algorithm can converge to a local minimum instead of the global one.
*   **Slowing down near saddle points:** A saddle point is characterized by a zero or near-zero gradient. When the algorithm approaches this area, its progress becomes extremely slow, as the gradient guiding the updates is almost zero. In some cases, the algorithm can get "stuck."
*   **Effect of the learning rate:** A higher learning rate can help overcome these areas, but if it is too high, it can cause oscillatory behavior, with the algorithm "jumping" from one side of the minimum region to the other.
This simple example clearly shows the limitations of classic gradient descent and its stochastic variant. The goal is therefore to develop more advanced optimization methods to overcome these problems.
## [05:10] Advanced Optimization Methods
In this section, we will present four methods, in order of historical appearance, designed to improve upon gradient descent.
### [05:35] Momentum Method
The first attempt to overcome the limitations of SGD is the **Momentum method**.
**Key Idea:** Instead of basing the update solely on the gradient calculated at the previous iteration, a "memory" of past updates is introduced.
The update rule uses a vector `v_t`, which determines the direction and magnitude of the step. This vector is not only composed of the current gradient but also includes a fraction of the vector `v` from the previous iteration.
`v_t = μ * v_{t-1} - γ * ∇J(w_{t-1})`
`w_t = w_{t-1} + v_t`
*   `v_t`: Update vector.
*   `μ`: Momentum term (usually set to 0.9).
*   `γ`: Learning rate.
The name "Momentum" comes from the fact that the algorithm's behavior resembles that of a ball rolling down a surface, accumulating inertia (or "momentum").
**Advantages:**
*   Reduces oscillations.
*   Often converges faster than standard gradient descent.
**Disadvantages:**
*   The algorithm is "blind" in the sense that, due to the accumulated inertia, it can "overshoot" the minimum and continue moving.
**Practical Demonstration:**
Returning to the example of the complex cost function, the Momentum method (in magenta) reaches the minimum much more quickly than standard gradient descent (in light blue). It is also more effective at avoiding getting stuck near saddle points. Increasing the momentum term (e.g., from 0.8 to 0.9) further improves performance.
### [08:38] Nesterov Accelerated Gradient (NAG)
An improvement on the Momentum method is the **Nesterov Accelerated Gradient (NAG)**.
**Key Idea:** Instead of calculating the gradient at the current position (`w_{t-1}`), it makes an estimate of the future position and calculates the gradient there. It is a "look-ahead" approach.
The formula is similar to that of Momentum, but the gradient is calculated at an "anticipated" point:
`∇J(w_{t-1} - μ * v_{t-1})`
In practice, you first calculate where the inertia (momentum) would take you (`w_{t-1} - μ * v_{t-1}`) and then calculate the gradient at that point to correct the trajectory.
**Advantages:**
*   It does not introduce new hyperparameters compared to Momentum.
*   It is more robust and generally offers better performance.
**Visualizing the Difference:**
*   **Momentum:** Calculates the gradient at the current position and then takes a large step in the direction of the update vector (sum of inertia and gradient).
*   **Nesterov (NAG):** First takes a step in the direction of inertia, calculates the gradient at that new position, and then corrects the trajectory. This allows it to "brake" in advance if it is approaching a minimum too quickly.
### [11:11] Adagrad (Adaptive Gradient)
The first method designed to address the problem of sparse datasets is **Adagrad**.
**Key Idea:** Adapt the learning rate for each individual parameter based on how frequently it is updated.
The update rule for a single parameter `w_i` is:
`w_{i, t+1} = w_{i, t} - (γ / sqrt(G_{ii, t} + ε)) * g_{i, t}`
*   `g_{i, t}`: Gradient with respect to parameter `i` at iteration `t`.
*   `G_t`: A diagonal matrix where each element on the diagonal `G_{ii, t}` is the **sum of the squares of past gradients** for parameter `i`.
*   `ε`: A small smoothing term to avoid division by zero.
**How it works:**
*   If a parameter is updated frequently (associated with a common feature), the sum of the squares of its gradients (`G_{ii, t}`) will grow rapidly. This will increase the denominator, effectively reducing its learning rate.
*   If a parameter is updated rarely (associated with a sparse feature), `G_{ii, t}` will grow slowly, keeping its learning rate relatively high.
**Advantages:**
*   Adapts the learning rate for each parameter, making it ideal for sparse data.
*   In theory, it reduces the need to manually tune the global learning rate `γ`. Once an initial value is set (e.g., 0.01), the algorithm adapts automatically.
**Main Disadvantage:**
*   The continuous accumulation of the squares of the gradients in the denominator causes this term to become very large. Consequently, the effective learning rate can become so small that it almost completely stops learning (an effect similar to the *vanishing gradient*), even when further updates would be necessary.
This limitation has driven research towards methods that avoid the indefinite accumulation of gradients.
**Practical Demonstration:**
In the visual example, Adagrad shows very slow performance. Even with an increased learning rate, its convergence is significantly lower than other methods, due to the rapid decay of the effective learning rate.
# Adaptive Optimization Algorithms
In this section, we will analyze some advanced optimization methods that overcome the limitations of stochastic gradient descent (SGD) and the momentum algorithm. Although the momentum approach shows good behavior, methods like AdaGrad, AdaDelta, RMSprop, and ADAM offer superior performance, especially in contexts with *sparse datasets*.
## AdaGrad: An Idea for Sparse Data
The AdaGrad (*Adaptive Gradient*) algorithm introduces a fundamental concept: adapting the *learning rate* for each individual parameter. This approach proves particularly effective when working with sparse data. However, in practice, it is not the most commonly used method.
## AdaDelta and RMSprop: The Evolution with Moving Averages
More modern and high-performing methods like ADAM and RMSprop are now the standard in neural network optimization. Both are available in major libraries like PyTorch and TensorFlow, allowing for easy experimentation.
### The Key Idea of AdaDelta
The AdaDelta method improves upon AdaGrad by introducing an **exponentially weighted moving average** (*decaying average*) of past squared gradients.
**Concept:** Instead of accumulating all past gradients from the beginning of training (as AdaGrad does), AdaDelta considers only a "window" of the most recent iterations. This prevents the learning rate from decreasing too quickly to the point of stalling.
A notable advantage of AdaDelta, in its theoretical formulation, is that it **does not require manually setting an initial learning rate**, as the weight update is determined entirely by the ratio of the moving averages of the gradients and the updates themselves.
### RMSprop: A Parallel Approach
Developed almost concurrently with AdaDelta, the RMSprop (*Root Mean Square Propagation*) algorithm is based on a very similar idea. The main difference is that RMSprop **explicitly keeps the learning rate** (η) in the numerator of the update formula, while using the moving average of the squared gradients in the denominator.
In summary, both AdaDelta and RMSprop share the idea of using a moving average to adapt the learning rate, solving the problem of AdaGrad's rapid learning rate decay.
## ADAM: The Synthesis of Momentum and Adaptation
The ADAM (*Adaptive Moment Estimation*) algorithm is now the most widespread optimization method, used in about 90% of cases. Its strength lies in combining two powerful concepts:
1.  **The Momentum Idea:** It accumulates a moving average of past gradients (the "first moment," `mt`) to accelerate convergence in the right direction.
2.  **The RMSprop Idea:** It uses a moving average of the squared gradients (the "second moment," `vt`) to adapt the learning rate for each parameter.
**Disadvantages of ADAM:**
*   **Greater complexity:** It introduces two additional hyperparameters, `beta1` and `beta2`, which control the decay of the moving averages of the first and second moments. Fortunately, the default values (e.g., `beta1=0.9`, `beta2=0.999`) work well in most cases.
*   **Higher memory requirement:** It requires storing two additional vectors (`mt` and `vt`), which can become very large in models with millions of parameters.
*   **Not always the best:** Although it is a *workhorse*, in some specific situations a simpler algorithm like stochastic gradient descent with momentum, paired with a well-calibrated learning rate schedule, can achieve better results.
### Visual Comparison of Performance
Analyzing the behavior of the algorithms on different cost surfaces, the following observations emerge:
*   **Complex surfaces:** In scenarios with multiple local minima, an algorithm's ability to find the global minimum strongly depends on the starting point. In some cases, ADAM demonstrates a greater ability to explore the surface and find the global minimum, while SGD and Momentum can get trapped in local minima.
*   **Plateau areas:** In nearly flat regions ("plateaus"), ADAM, thanks to its adaptive learning rate, proves to be more robust. With a sufficiently high learning rate, it manages to overcome the plateau and reach the minimum, whereas SGD can get stuck and Momentum can start to oscillate.
## Summary and Key Messages
The choice of optimization algorithm depends on the context:
1.  **Sparse Data:** If the dataset is sparse, it is advisable to use an adaptive method like AdaGrad, AdaDelta, RMSprop, or ADAM.
2.  **General Choice:** ADAM is generally the best and most robust choice for most applications.
3.  **Valid Alternatives:** There is no "golden rule." In some cases, a simpler method like SGD with momentum and a well-designed *learning rate scheduling* can outperform more complex algorithms.
The two fundamental concepts to take away from this lesson are:
*   The **idea of momentum**, to accelerate convergence.
*   The **idea of an adaptive learning rate**, to handle sparse data and complex cost surfaces.
The scientific literature offers many other variants, but these two principles form the foundation of modern optimization in deep learning.
# Second-Order Optimization Methods: Introduction to Newton's Method
## [00:00] Overview and Context
Up to this point, we have exclusively analyzed so-called **first-order methods**. These methods are defined as "first-order" because, to calculate the update at each iteration, they only use information derived from the gradient of the cost function.
However, there is another family of methods based on the use of higher-order information, specifically the **Hessian matrix**. Before delving into these specific methods, we will dedicate the next few minutes to a review of **Newton's method** in its general context.
## [00:35] Newton's Method for Root Finding
Many of you may have already encountered Newton's method in the context of finding the roots (or zeros) of a function. The goal is to find the value `x` for which a non-linear function `f(x)` becomes zero, i.e., `f(x) = 0`.
### Key Idea and Formulation
The fundamental idea of Newton's method is to locally approximate the function using its tangent lines. The process is as follows:
1.  Start from an initial point `x_k`.
2.  Calculate the tangent line to the function at that point.
3.  Find the point where this tangent line intersects the x-axis. This intersection point becomes the new estimate of the root, `x_{k+1}`.
4.  Project the new point onto the function and repeat the process.
Mathematically, the tangent line at point `x_k` can be expressed as the first-order Taylor expansion of the function `f` around `x_k`:
`f(x) ≈ f(x_k) + f'(x_k) * (x - x_k)`
To find the intersection with the x-axis, we set this expression to zero. Evaluating the equation for `x = x_{k+1}`, we get the update rule for Newton's method for root finding:
`x_{k+1} = x_k - f(x_k) / f'(x_k)`
### Geometric Interpretation
Starting from a point `x_0`, the tangent is drawn, its intersection with the x-axis is found to get `x_1`, a new tangent is drawn at `x_1` to find `x_2`, and so on, progressively approaching the root of the function.
## [02:30] Adapting Newton's Method for Optimization
Our main interest is not root finding, but the **minimization** of a function `f(x)`. The link between the two problems lies in a fundamental condition of optimization: at a minimum point, the slope (i.e., the first derivative) of the function is zero.
Therefore, minimizing `f(x)` is equivalent to finding the roots of its first derivative, `f'(x)`.
### Derivation of the Update Rule
We can apply Newton's method for root finding not to the function `f`, but to its derivative `f'`. Substituting `f` with `f'` in the previous formula, we get:
-   `f(x_k)` becomes `f'(x_k)` (the first derivative).
-   `f'(x_k)` becomes `f''(x_k)` (the second derivative).
The update rule for minimization thus becomes:
`x_{k+1} = x_k - f'(x_k) / f''(x_k)`
## [04:00] Geometric Interpretation of Optimization with Newton's Method
There is an alternative and very powerful geometric perspective for understanding this method in the context of optimization.
### Local Quadratic Approximation
Instead of approximating the function with a line (first-order Taylor expansion), we use a more faithful approximation: a **parabola** (second-order Taylor expansion).
The second-order Taylor expansion of `f` around `x_k` is a quadratic function `q(x)`:
`q(x) = f(x_k) + f'(x_k) * (x - x_k) + (1/2) * f''(x_k) * (x - x_k)^2`
This function `q(x)` represents a parabola that locally approximates the function `f`. To find the minimum of `f`, we look for the minimum of its quadratic approximation `q(x)`. We calculate the derivative of `q(x)` with respect to `x` and set it to zero:
`q'(x) = f'(x_k) + f''(x_k) * (x - x_k) = 0`
Evaluating this expression at `x = x_{k+1}`, we obtain exactly the same update rule seen before.
### Iterative Process
1.  Starting from `x_0`, the parabola that locally approximates the function is constructed.
2.  The vertex (minimum point) of this parabola is calculated, which becomes the new point `x_1`.
3.  The process is repeated at `x_1`, constructing a new parabola and finding its minimum, and so on.
**Important condition:** To avoid numerical problems, both in root finding (`f'(x_k) ≠ 0`) and in minimization (`f''(x_k) ≠ 0`), the denominators must not be zero.
## [06:45] Extension to the Multidimensional Case
Now let's extend these concepts to functions of multiple variables, `f(x)` where `x` is a vector in `R^n`.
### Notation and Prerequisites
-   We assume that the function `f` is twice differentiable. This is necessary because, analogous to the second derivative in the 1D case, here we will need the **Hessian matrix** (the matrix of second partial derivatives).
-   **Gradient (`∇f(x)`):** The vector of first partial derivatives.
-   **Hessian Matrix (`H`):** The `n x n` matrix of second partial derivatives.
The minimum condition becomes `∇f(x) = 0`. Minimizing `f` is therefore equivalent to finding the roots of its gradient.
### Derivation of the Multidimensional Formula
We apply Newton's method to the vector function `g(x) = ∇f(x)`. The multidimensional counterparts of the terms seen in the 1D case are:
-   `f'(x_k)` becomes the **gradient** `∇f(x_k)`.
-   `f''(x_k)` becomes the **Hessian matrix** `H(x_k)`.
-   Division by `f''` becomes multiplication by the **inverse of the Hessian matrix**, `H(x_k)⁻¹`.
The update rule for Newton's method in the multidimensional case is:
`x_{k+1} = x_k - H(x_k)⁻¹ * ∇f(x_k)`
### The Practical Approach: Solving a Linear System
In numerical analysis, directly calculating the inverse of a matrix (`H⁻¹`) is a costly and often numerically unstable operation. It is a "golden rule" to avoid it when possible.
By defining the update vector `Δx = -H(x_k)⁻¹ * ∇f(x_k)`, we can rewrite the relationship as a **system of linear equations**:
`H(x_k) * Δx = -∇f(x_k)`
At each step, instead of inverting the matrix, this linear system is solved to find the update vector `Δx`. The update rule then becomes:
`x_{k+1} = x_k + Δx`
Even in the multidimensional case, this formula can be derived by starting from the local quadratic approximation (second-order Taylor expansion) and finding its stationary point.
## [10:45] Challenges of Newton's Method in the Multidimensional Case
The transition from the 1D case to the n-dimensional case, although conceptually straightforward, introduces significant computational challenges.
1.  **Computational Cost:** The main challenge is solving the linear system `H * Δx = -g`. The cost of solving this system with classic methods is on the order of `O(n³)` floating-point operations. If `n` (the number of variables) is large, as is often the case in practical applications (e.g., deep learning), this cost becomes prohibitive.
2.  **Invertibility of the Hessian:** The Hessian matrix must be invertible. In practice, since we are looking for a minimum, we expect the function to be locally convex, which implies that the Hessian is positive definite and therefore invertible.
## [13:00] Towards Quasi-Newton Methods
The main difficulty, related to the computational cost of calculating and solving the system with the Hessian matrix, has led to the development of **Quasi-Newton methods**.
### Fundamental Idea
The idea behind these methods is not to use the true Hessian matrix `H`, but an **approximation** `B` that is easier to compute and manage.
A naive approach might be to approximate the Hessian using finite differences, but this method suffers from the same numerical instability problems related to floating-point operations, making it impractical.
### Next Steps: Practical Methods
Quasi-Newton methods use more sophisticated approaches to build effective approximations of the Hessian (or its inverse). In the next lesson, we will explore these methods, particularly one of the most well-known and used algorithms in deep learning libraries: **BFGS** (Broyden–Fletcher–Goldfarb–Shanno).
# Information on the Course Project
Now we will dedicate a few minutes to discussing the projects for the course.
## Structure and Types of Projects
A document with a list of project ideas will be shared. Some topics from past years have been removed because they were covered too frequently. The proposals are divided into two main categories:
1.  **Practical Project (Groups of 1-3 people):**
    *   **Objective:** Choose a topic or a scientific article and implement one of the proposed methods to solve a specific problem.
    *   **Deliverables:**
        *   Code (Jupyter Notebook or Python script).
        *   A report describing the dataset, the method used, implementation details, and a thorough discussion of the results.
    *   **Presentation:** You can use the report itself or prepare a presentation (e.g., PowerPoint, Beamer).
2.  **Theoretical Deep Dive (Individual):**
    *   **Objective:** To delve deeper into a theoretical topic mentioned only briefly during the course.
    *   **Deliverables:**
        *   A small report (10-15 pages) explaining the theory of the chosen topic.
    *   **Presentation:** A short lecture of about 20 minutes on the topic.
At the end of the projects document, there are additional ideas with references to datasets and scientific articles already included.
## Registration and Deadlines
**Registration Form:**
An online form will be published to be filled out with the following information:
*   Name and email of a contact person for the group.
*   Course type (8 or 10 ECTS). Students in the 8 ECTS course can also do the project to improve their grade.
*   Project type (practical or theoretical).
*   Indication of whether the project is one of those proposed or a personal idea.
*   Names and emails of other group members (if any). Only one member per group should fill out the form.
*   **Expected delivery date:** Indicate an approximate period (e.g., January/February, June/July, September). This information is purely for organizational purposes and is not binding.
*   **Need for materials:** Specify if you need articles, datasets, or other materials from the instructor.
*   **Personal project proposals:** If you are proposing a project not on the list, provide a detailed description with references (links to papers, etc.).
**Deadline for submission:**
Today's date is **November 10**. You are requested to fill out the form within a week or ten days, to get a general overview of the situation. Those who are not yet sure can fill it out later (e.g., in January).
## Support and Development
Once the project is chosen, you will begin by reading the provided material or what you already have. Throughout the development period, even after the course ends, the instructors are available for meetings (in person or online) to discuss any issues.
**Contacts:** For any problem, send an email to both instructors (the professor and Matteo) with the subject `[NANL 2025] <topic>`.
## Presentation and Evaluation Sessions
**Organization of Sessions:**
The project presentation sessions will be organized to coincide with the exam sessions:
*   **Winter Session:** Around mid or late February (after the second written and oral session).
*   **Summer Session:** Around late July.
*   **Autumn Session:** Around mid-September.
It is possible to arrange custom presentation dates (e.g., in April) for special needs (e.g., study abroad periods). In that case, the grade will be registered in the first available official session (June, July, or September).
**Joint Projects with Other Courses:**
It is possible to carry out projects that are valid for multiple courses (e.g., with Prof. Formaggia's course). In these cases, a more substantial work is expected. The presentations for the two courses might be separate, to focus on different aspects of the work.
## Questions and Answers
*   **Filling out the form for groups:** Only one person per group (the "leader") should fill out the form, entering the names and emails of the other members.
*   **Presenting at different times:** If members of a group are at different stages of their exam process, they can present the project together. The grade of those who have not yet completed the exam will be "frozen" and registered in the future.
*   **Programming languages:** The use of **Python** is recommended, as the instructors can provide support. Other languages like MATLAB, Fortran, and C are known, but support for C++ is not guaranteed.
*   **Final grade calculation (10 ECTS course):** The final grade will be a weighted average: 80% from the exam result (written and oral) and 20% from the project.
*   **Students of the 8 ECTS course:** The grade calculation is analogous. It is even more important for them to fill out the form, as the project is not mandatory.