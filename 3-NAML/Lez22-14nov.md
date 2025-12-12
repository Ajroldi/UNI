# Chapter 1: Correcting the Error from the Previous Lesson
## [00:00:00] Introduction and Today's Agenda
Ok, good afternoon everyone. What's the plan for today? I'll start by spending 10 minutes showing you a problem we encountered at the end of the last lab. During the live coding session, I couldn't get the same result as the solution and had to copy and paste a piece of code. Once I got home, I reviewed the code and realized the mistake I made is something that could happen to you as well. In fact, in past years I've seen students make the same mistake, and I myself struggled to find this bug. For this reason, I want to show you what was happening and why.
I believe it's very useful to understand the reason for this behavior and how to prevent it when writing code. After these five minutes, we will start the rest of the lab by implementing an artificial neural network from scratch. I think it will be a very interesting experience because we will see every single step, starting from zero up to the complete implementation of the network. The first application will be learning the XOR function, a very simple but non-trivial problem, as its structure cannot be learned by a simple linear model.
If we have time, I will also show you how to implement a neural network for a non-linear classification problem. Specifically, we will generate points belonging to two concentric circles, one small and one larger, and the goal will be to classify which of the two circles a point belongs to. It's a problem whose results can be visualized very well, but it is by no means trivial.
## Chapter 1: Error Correction from the Previous Lesson
### [00:01:28] Context of the Error: The Loss Function
Let's start with correcting the error from last time. I haven't uploaded the updated notebook; I'll show you the results directly, as it's a small modification to the previous one. The point where I got stuck was the implementation of the loss function. Specifically, I had implemented this function incorrectly, having to slightly modify a term to get the correct result.
The loss function is composed of:
1.  A regularization part: `λ * w²`
2.  The mean of the *hinge loss* function: `1/n * Σ(...)`
The problem was in this specific term: `1 - yᵢ * (wᵀxᵢ + b)`.
Written by itself, the problematic term was this: `yᵢ * (wᵀxᵢ + b)`.
Let's analyze the components:
-   `x`: a single vector of samples.
-   `w`: the model parameters, which are the concatenation of the weights `w` and the bias `b`.
-   `y`: a scalar value.
For each sample `i`, the operation to be performed is:
1.  Calculate the dot product between the sample `xᵢ` and the first part of the parameters (`w`).
2.  Add the bias `b` (a scalar).
3.  Multiply the whole thing by `yᵢ` (a scalar).
4.  The result is a scalar `ζᵢ` for each sample `i`.
### [00:02:38] Vectorized Implementation and the Broadcasting Problem
As usual, we perform this operation in *batch* mode, meaning with a single expression that operates on all samples at once. Instead of a single sample `xᵢ`, we use a matrix `X`, where each row is a sample. Instead of a single `yᵢ`, we use a vector `y`.
To isolate the problem, I extracted only the relevant part of the code. I imported NumPy, set a seed for reproducibility, and defined 10 samples with 2 features each. I then generated a matrix `X`, a vector of labels `y`, and the parameters `params` randomly.
-   **Matrix `X`**: Has 10 rows (samples) and 2 columns (features, `x1` and `x2`).
-   **Vector `y`**: A 1D vector with 10 elements.
-   **Parameters `params`**: A vector with 3 components, where the first two are the weights `W` and the third is the bias `b`.
The **correct implementation** is as follows:
1.  Take the matrix `X` and calculate the product with the first two parameters (`W`).
2.  Add the bias (`b`).
3.  Calculate the maximum between `1` and the previous result.
4.  Multiply this vector, element-wise, by the vector `y`.
The correct mean value of this operation is `0.44`.
The mistake I made last time was adding a `reshape(-1, 1)` to this intermediate term, which I called `decision`.
```python
# Error: transforming the vector into a column vector
decision = decision.reshape(-1, 1) 
```
At this point, `decision` is no longer a 1D vector, but a 2D **column vector**. From a mathematical point of view, this makes sense: we have a matrix `X` and we multiply each of its rows by a column vector `W`.
However, by doing this, the result becomes `0.3`. Why? The problem arises when we multiply `decision` (now a column vector with shape `(10, 1)`) by `y` (a 1D vector with shape `(10,)`).
NumPy, seeing two objects with different dimensions (one 2D and one 1D), applies **broadcasting**. Instead of performing an element-wise product, it performs an *outer product*. If we have a column vector `a` and a row vector `b`, broadcasting builds a matrix where the element `(i, j)` is the product `aᵢ * bⱼ`.
The problem is that the shapes of the two objects are different:
-   `decision.shape` is `(10, 1)` (2D object).
-   `y.shape` is `(10,)` (1D object).
NumPy interprets this operation as broadcasting, generating a `10x10` matrix instead of a vector of 10 elements. Since we ultimately calculate the mean (`mean()`), which operates on all elements, we still get a scalar. The function seems to work, but it's calculating a completely different operation.
**Moral of the story:** Be very careful with this type of operation. Always keep in mind whether you are working with 1D or 2D objects, because implicit broadcasting can occur and cause hard-to-find errors. Becoming good at quickly identifying these problems is crucial when using NumPy at an advanced level.
Any questions? Good, let's move on to the notebook.
# Chapter 2: Implementing a Neural Network from Scratch
## [00:07:28] Objective: Learning the XOR Function
The notebook is available on WēBIP. Open Google Colab and upload it. As I mentioned, the goal is to build a neural network from scratch that learns the XOR function.
To fully leverage the power of JAX, remember to change the runtime to "GPU". Let's start by connecting and importing the necessary libraries: NumPy, JAX, and Matplotlib.
The dataset we want to learn is very small, consisting of only four samples. For this reason, we will implement a *full gradient descent*, as it makes no sense to use mini-batches on such a small dataset. The inputs have two features (two bits) and the output is a single value (the result of the XOR function).
### [00:08:20] Network Architecture and Hyperparameters
We want to build the following neural network:
-   **Input**: 2 neurons.
-   **Hidden layers**:
    1.  First layer with 4 neurons.
    2.  Second layer with 3 neurons.
-   **Output**: 1 neuron.
The output represents the probability of the result being true or false, so we will constrain its value between 0 and 1 using the sigmoid function.
First, we define the **hyperparameters**, which are the parameters that define the network's architecture but are not the weights and biases that will be learned. In this case, we define the number and size of the layers: `[2, 4, 3, 1]`.
### [00:09:18] Initializing the Parameters
Our first task is to create the initial state of the parameters. The neural network is a function `f(x, θ)` that takes an input vector `x` and depends on the parameters `θ`. These parameters `θ` are a list of weights (`W`, matrices) and biases (`b`, vectors).
Our goal is to find the optimal function `f` that, given an input `x`, correctly predicts the output `y`. We want to minimize a **loss function** `L(f(x, θ), y)`, which measures the distance between the network's prediction and the true value.
To do this, we will use **gradient descent**, calculating the gradient of the loss function with respect to the parameters `θ`. Gradient descent is an iterative algorithm that, starting from an initial point `θ₀`, converges towards the point that minimizes the function. We must therefore provide this starting point.
We start by setting the seed for reproducibility. We will use NumPy for simplicity and then convert everything to JAX format.
We need to define the weights and biases for each layer:
1.  **First layer (from 2 to 4 neurons)**:
    -   `W1`: a weight matrix. Its size will be `(4, 2)`. The operation of a layer is `activation(W * x + b)`. If `x` is a column vector, `W` must have a number of columns equal to the dimension of `x` and a number of rows equal to the dimension of the layer's output.
    -   `b1`: a bias vector with a dimension equal to the layer's output, so `(4,)`.
    We will initialize the weights with random values drawn from a normal distribution and the biases to zero. More advanced initialization techniques exist that can speed up training, but for now, this choice is sufficient.
2.  **Second layer (from 4 to 3 neurons)**:
    -   `W2`: matrix of size `(3, 4)`.
    -   `b2`: vector of size `(3,)`.
3.  **Third layer (from 3 to 1 neuron)**:
    -   `W3`: matrix of size `(1, 3)`.
    -   `b3`: vector of size `(1,)`.
Finally, we group all the parameters into a single list `params = [W1, b1, W2, b2, W3, b3]` and convert it to a JAX-compatible format.
### [00:13:50] Exercise: Implement the Neural Network Function
Now it's your turn. The first task is to implement the neural network function. This function, which we will call `artificial_neural_network`, accepts two arguments:
-   `x`: the input matrix.
-   `params`: the list of weights and biases we just created.
The function must return the network's prediction.
-   For the hidden layers, use the **hyperbolic tangent** (`jmp.tanh` in JAX) as the activation function.
-   For the last layer (output), use the **sigmoid function**.
The sigmoid function can be defined from the hyperbolic tangent. Since `tanh` returns values between -1 and 1, the formula `(tanh(z) + 1) / 2` shifts the interval to [0, 2] and then scales it to [0, 1].
The idea is to pass the input through each layer, applying the transformation: `new_output = activation(W * previous_output + b)`. You need to repeat this process for the three layers of our network.
I'll give you five minutes to implement this function. I'll be here to help you.
# Chapter 3: Setting up the Forward Pass
## [00:14:35] Conventions and Data Preparation
By convention, when working with neural networks, the input matrix `x` is organized so that each column represents a sample and each row represents a feature. This allows the matrix-vector multiplication `W * x` to be performed in a standard way, as seen in the theoretical lessons.
However, our input data (`inputs`) is structured the other way around: each row is a sample. To align with the standard convention, the first operation to perform is to transpose the input matrix.
**Step-by-step procedure:**
1.  **Unpack Parameters:** Initially, the network parameters (weights `W` and biases `b`) are grouped into a single `params` structure. For clarity, we separate them into individual variables (`W1`, `b1`, `W2`, `b2`, etc.).
2.  **Transpose Input:** We call our input matrix `x` and transpose it (`x.T`) to conform to the "one sample per column" convention.
```python
# Example code for preparation
W1, b1, W2, b2, W3, b3 = params # Unpacking
x = inputs.T # Transposition
```
## [00:15:20] Handling Biases and Broadcasting
When we add the bias vector `b1` to the output of the `W1 * x` multiplication, we must ensure that the addition happens column-wise, as each column represents a different sample.
NumPy's **broadcasting** mechanism, if not handled correctly, might add the biases row-wise. To verify this, we can temporarily initialize `b1` with increasing values (e.g., `[0, 1, 2, 3]`) and observe the result. If the sum occurs by row, it is incorrect.
**Solution:**
To ensure correct column-wise addition, the bias vectors must be defined as **column vectors** (matrices of size `n x 1`). This is achieved by specifying their shape during initialization.
```python
# Correcting the shape of the biases
# Instead of a 1D vector, we use a 2D matrix (column vector)
b1 = np.zeros((n, 1))
```
This way, broadcasting will work as expected, adding the correct bias to each sample.
# Chapter 4: Implementing the Neural Network
## [00:16:23] Calculating the Layers (Forward Pass)
Once the input and parameters are set up correctly, we can calculate the output of each layer of the network. This process is known as the **forward pass**.
1.  **First Layer (Layer 2):**
    *   Calculate the matrix product between the first layer's weights (`W1`) and the transposed input (`x`).
    *   Add the bias vector (`b1`).
    *   Apply a non-linear activation function, in this case, the **hyperbolic tangent (`tanh`)**, to allow the network to learn complex relationships.
    ```python
    # Z1 = W1 @ x + b1
    # layer_2 = np.tanh(Z1)
    ```
2.  **Second Layer (Layer 3):**
    *   Repeat the same process using the output of the previous layer (`layer_2`) as input.
    ```python
    # Z2 = W2 @ layer_2 + b2
    # layer_3 = np.tanh(Z2)
    ```
3.  **Output Layer (Layer 4):**
    *   The last layer must produce a probability, so a value between 0 and 1. The `tanh` function produces values between -1 and 1.
    *   To transform the `tanh` output into a [0, 1] interval, a simple algebraic transformation that emulates the **sigmoid function** can be used: `(tanh(z) + 1) / 2`.
    *   First, calculate the linear output (`W3 @ layer_3 + b3`), then apply `tanh`, and finally scale the result.
    ```python
    # Z3 = W3 @ layer_3 + b3
    # A3 = np.tanh(Z3)
    # layer_4 = (A3 + 1) / 2
    ```
## [00:17:55] Finalizing the Output and Creating the Function
The final output (`layer_4`) will be a matrix where each column represents the prediction for a sample. To return to the initial convention of the input data (one sample per row), it is necessary to **transpose the final output matrix**.
At this point, the code developed in the notebook cell can be encapsulated into a reusable function, which we will call `ANN` (Artificial Neural Network).
**Function Structure:**
*   **Input:** The data matrix `X` (with samples on rows) and the parameters `params`.
*   **Internal Logic:** Executes all steps of the forward pass, including the initial transposition of the input.
*   **Output:** Returns the final prediction matrix, transposed to have samples on the rows.
```python
def ANN(X, params):
    # ... parameter unpacking and layer calculations ...
    # The transposition of X happens inside
    return layer_4.T # Final transposition
```
This approach, although it requires transpositions, keeps the `W * x` multiplication logic consistent with the theory and allows for step-by-step verification of the array shapes, leveraging the interactivity of notebooks.
# Chapter 5: Training the Network
## [00:20:15] Evaluating Initial Performance
Using randomly initialized parameters, the neural network produces incorrect predictions. For example, for the XOR problem:
*   Input `[0, 0]`: should give `0`, but might give `1` (incorrect).
*   Input `[0, 1]`: should give `1`, but might give `0` (incorrect).
The goal of training is to modify the parameters (`W1`, `b1`, etc.) so that the network's predictions become correct. This process is achieved through the **Gradient Descent** algorithm.
## [00:20:50] Defining the Loss Function
To guide the training, we need a **loss function**, which is a metric that quantifies how wrong the network's predictions are compared to the real values (targets). The goal of gradient descent is to minimize this function.
Two common loss functions are proposed:
1.  **Mean Squared Error (MSE):**
    *   **Formula:** It is the average of the squared difference between the prediction (`y_pred`) and the true value (`y_true`).
    *   **Use:** It is a generic loss function, widely used in regression problems. To implement it, you calculate the prediction using the `ANN` function and then compute the mean squared error with respect to the targets.
2.  **Cross-Entropy Loss:**
    *   **Formula:** `- (y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred))`
    *   **Use:** It is the standard choice for binary classification problems, where the output is a probability.
    *   **Practical Advantage:** This loss function heavily penalizes incorrect predictions made with high confidence. For example, if the correct value is `1` and the network predicts `0.01` (very sure of itself, but wrong), the `log(0.01)` term becomes a very large negative number. With the minus sign in front, the loss becomes a very large positive value, pushing the network to correct the error more decisively than MSE.
The next step is to implement these two loss functions to then use them in the network's training loop.
# Chapter 6: Implementing Cost Functions
## [00:21:35] Cost Function: Mean Squared Error (MSE)
Let's start by implementing the cost function based on Mean Squared Error (MSE). The approach is gradual: first, we write the code in a cell to test it, then we encapsulate it into a reusable function.
1.  **Calculate Prediction**: The first step is to get the prediction from the neural network. This is done by applying the `ANN` (Artificial Neural Network) function to the inputs `x` and the current model parameters.
    ```python
    # x corresponds to 'inputs'
    prediction = ANN(x, parameters)
    ```
2.  **Calculate Error**: Next, we calculate the error as the difference between the prediction and the target values (the real labels, `y`, which correspond to `outputs`). It's important to verify that `prediction` and `y` have the same shape to perform the subtraction without errors.
    ```python
    # The error is the difference between prediction and real value
    error = prediction - y
    ```
3.  **Calculate MSE**: The mean squared error is obtained by squaring each element of the error and calculating its mean. The result is a single scalar value, representing the model's "loss".
    ```python
    # We square and calculate the mean
    loss_mse = jnp.mean(error**2)
    ```
The final function, which we will call `quadratic_loss`, will accept `x`, `y`, and the model `parameters` as arguments, returning the loss value.
## [00:22:40] Cost Function: Cross-Entropy
For classification problems, cross-entropy is often a better choice than MSE. Let's implement it directly as a function.
1.  **Calculate Prediction**: As before, the starting point is the network's prediction.
    ```python
    prediction = ANN(x, parameters)
    ```
2.  **Implement the Formula**: The formula for binary cross-entropy is:
    `-[y * log(prediction) + (1 - y) * log(1 - prediction)]`
    This formula has an interesting property: since `y` can only be 0 or 1, only one of the two terms in the sum is active at a time.
    *   If `y = 1`, the second term becomes zero, and only `log(prediction)` remains.
    *   If `y = 0`, the first term becomes zero, and only `log(1 - prediction)` remains.
    We implement this logic using JAX functions:
    ```python
    # Calculate the two terms of the formula
    term1 = y * jnp.log(prediction)
    term2 = (1 - y) * jnp.log(1 - prediction)
    ```
3.  **Sum and Negative Sign**: We sum the contributions from all examples in the batch and add a negative sign. The loss function measures a discrepancy, so higher values indicate a greater error.
    ```python
    # Sum the results and invert the sign
    loss_cross_entropy = -jnp.sum(term1 + term2)
    ```
    **Note**: Instead of the sum (`jnp.sum`), the mean (`jnp.mean`) could be used. Using the mean makes the loss independent of the batch size, which is generally better practice.
# Chapter 7: Gradient Calculation and Training
## [00:24:03] Calculating Gradients with JAX
To train the network with the gradient descent method, we need to calculate the gradients of the cost functions with respect to the model parameters.
1.  **JIT Compilation**: To optimize performance, we use JAX's "Just-In-Time" (JIT) compilation on our cost functions.
    ```python
    loss_mse_jit = jax.jit(quadratic_loss)
    loss_cross_entropy_jit = jax.jit(cross_entropy_loss)
    ```
2.  **`jax.grad` Function**: JAX provides the `jax.grad` function to automatically calculate the gradient of a function. However, we need to specify with respect to which argument to calculate the gradient. Our cost functions (`quadratic_loss` and `cross_entropy_loss`) accept three arguments: `(x, y, params)`. We are interested in the gradient with respect to `params`, which is the third argument (index 2).
    ```python
    # We specify argnums=2 to calculate the gradient with respect to the third argument
    grad_mse_jit = jax.jit(jax.grad(quadratic_loss, argnums=2))
    grad_cross_entropy_jit = jax.jit(jax.grad(cross_entropy_loss, argnums=2))
    ```
    Omitting `argnums=2` would cause an error, as JAX would default to calculating the gradient with respect to the first argument (`x`), which is not what we want to optimize.
## [00:25:35] Implementing Gradient Descent
Now we have all the tools to implement the training algorithm. In this simplified example, we will use a "full-batch" approach, where the entire dataset is used in each iteration, without splitting it into mini-batches.
The pseudo-code for the training loop is as follows:
```python
# For a defined number of epochs (full iterations over the dataset)
for epoch in range(number_of_epochs):
    # 1. Calculate the gradient of the loss with respect to the parameters
    gradient = calculate_gradient(x_train, y_train, parameters)
    # 2. Update each parameter of the model
    # For each component (weight matrix W or bias vector b)
    for parameter, grad_parameter in zip(parameters, gradient):
        # Move in the opposite direction of the gradient
        parameter -= learning_rate * grad_parameter
    # 3. Save the current loss value to monitor training
    save_current_loss()
# 4. Visualize the loss trend over time
plot_loss_vs_epochs()
```
The parameter update is done by subtracting the gradient (multiplied by a step, the `learning_rate`), because the gradient points in the direction of the steepest ascent of the cost function, while we want to minimize it.
## [00:28:00] Practical Code for the Training Loop
Let's see how to translate the pseudo-code into working code.
1.  **Initialization**: We define the number of epochs, the `learning_rate`, and lists to store the history of the loss values.
    ```python
    learning_rate = 0.1
    number_of_epochs = 2000
    history_mse = []
    history_cross_entropy = []
    ```
2.  **Training Loop**:
    ```python
    # Choose the gradient function to use (e.g., MSE)
    gradient_function = grad_mse_jit
    for epoch in range(number_of_epochs):
        # Calculate the gradient on the entire dataset
        grads = gradient_function(inputs, outputs, params)
        # Update the parameters
        # grads has the same structure as params (a list of arrays)
        for i in range(len(params)):
            params[i] -= learning_rate * grads[i]
        # Calculate and save the losses (both MSE and Cross-Entropy)
        # We can monitor both metrics, even if we only use one for the update
        loss_m = loss_mse_jit(inputs, outputs, params)
        loss_c = loss_cross_entropy_jit(inputs, outputs, params)
        history_mse.append(loss_m)
        history_cross_entropy.append(loss_c)
    ```
## [00:30:50] Analysis of Results
After training, we visualize the trend of the cost functions.
*   **Loss Plot**: A plot of the loss versus epochs shows us if the training is converging. Ideally, the curve should drop rapidly at the beginning and then stabilize at a low value. Using a logarithmic scale on the Y-axis (`plt.yscale('log')`) can help to better visualize the descent when the values become very small.
*   **Comparison between MSE and Cross-Entropy**: An interesting experiment is to train the network using cross-entropy as the cost function and observe the trend of MSE. Often, optimizing cross-entropy leads to an even more marked reduction in MSE compared to optimizing MSE directly. This happens because cross-entropy severely penalizes incorrect predictions made with high confidence, pushing the model to learn more effectively. This is one of the main reasons why cross-entropy is the standard choice for classification problems.
*   **Verification of Final Predictions**: After training, we can use the model to make predictions on the input and verify that the returned probabilities are close to the expected values (e.g., >0.98 for class 1 and <0.02 for class 0).
# Chapter 8: Model Evaluation
## [00:34:03] Calculating Accuracy
Accuracy measures the percentage of correct predictions. To calculate it, we first need to convert the continuous probabilities produced by the model (e.g., 0.98) into discrete labels (0 or 1).
1.  **Decision Threshold**: A common threshold is 0.5. If the predicted probability is greater than 0.5, we classify it as 1; otherwise, as 0.
    ```python
    predictions_prob = ANN(inputs, params)
    predictions_classes = predictions_prob > 0.5  # Returns an array of True/False
    ```
2.  **Calculating Accuracy**: We compare the predicted classes with the real labels (`y`). Accuracy is the number of correct predictions divided by the total number of samples.
    ```python
    accuracy = jnp.mean(predictions_classes == y)
    # If all predictions are correct, the accuracy will be 1.0 (100%)
    ```
## [00:34:55] Confusion Matrix
For a more detailed evaluation, especially in problems with imbalanced classes, the **confusion matrix** is a fundamental tool. It shows not only how many predictions are correct, but also what kind of errors the model makes.
The matrix organizes predictions into:
*   **True Positives (TP)**: Correctly classified as positive.
*   **True Negatives (TN)**: Correctly classified as negative.
*   **False Positives (FP)**: Incorrectly classified as positive.
*   **False Negatives (FN)**: Incorrectly classified as negative.
We can easily calculate it using the `scikit-learn` library.
```python
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true=outputs, y_pred=predictions_classes)
print(cm)
```
An ideal confusion matrix for a two-class problem will have values only on the main diagonal (TN top-left, TP bottom-right) and zeros elsewhere, indicating that there were no classification errors.
# Chapter 9: Towards a More Realistic Use Case
## [00:36:00] Introduction to a New Dataset
Let's move on to a more complex problem. We will use the "make_circles" dataset from `scikit-learn`, which consists of two concentric circles. The goal is to classify points based on whether they belong to the inner or outer circle.
Features of this new scenario:
1.  **2-Dimensional Input**: The input `X` has two columns (the x and y coordinates of each point).
2.  **1-Dimensional Output**: The output `Y` is 0 or 1, depending on the circle of belonging.
3.  **Train/Test Split**: The dataset is split into a training set (80%) and a test set (20%). This is crucial for evaluating whether the model generalizes well to data it has never seen before.
4.  **Visualization**: The data is visualized with a scatter plot, using different colors for the two classes and different markers for the training and test points.
## [00:37:25] Generalizing the Code
To tackle more complex problems, it is useful to make our code more flexible and modular, instead of writing everything "by hand" for a specific architecture.
The goals are:
1.  **`initLayerParameters(key, n_in, n_out)`**: A function that initializes the weights and biases for a single layer, given the input and output dimensions.
2.  **`initializeMLPParameters(key, layer_sizes)`**: A function that, given a list of dimensions for each layer (e.g., `[2, 10, 10, 1]`), automatically creates the complete list of parameters for the entire neural network (a Multi-Layer Perceptron, MLP).
3.  **`forward(params, x)`**: A function that performs the forward pass through the network. Instead of manually writing the calculation for each layer, it will use a `for` loop to sequentially apply each layer of weights and activation function.
4.  **Mini-Batch Gradient Descent**: Instead of using the entire dataset at each step, we will implement mini-batch gradient descent. This requires an `update` function that calculates the gradient and updates the parameters using only a small subset of data (the mini-batch). This approach is more computationally efficient and often leads to faster and more stable convergence.
This generalization process consists of taking the concepts already implemented and encapsulating them into more abstract and reusable functions, preparing the ground for building more sophisticated deep learning models.
# Chapter 10: Neural Network Parameter Initialization
## [00:38:25] Introduction and Initial Setup
In this section, we will solve the exercise together step-by-step, introducing some tricks to make the code more general and flexible.
Let's start with the function that initializes the parameters of a single layer.
```python
def init_layer_params(key, in_dimension, out_dimension):
```
This function receives a JAX key for random number generation, the input dimension (`in_dimension`), and the output dimension (`out_dimension`).
## [00:38:45] Dimension Convention and Weight Initialization
Now, we will change the convention from before to show an alternative approach. Previously, samples were arranged in the columns of the matrices. Now, we will arrange them in the rows. Being flexible with these conventions is important because you never know which approach you will find in existing code.
With this new convention, the dimensions of the weight matrix `W` must be reversed.
```python
# Before: out_dimension, in_dimension
# Now: in_dimension, out_dimension
w = jax.random.normal(key, shape=(in_dimension, out_dimension))
```
This is because the number of columns of the input matrix `x` must match the number of rows of the weight matrix `w`.
The bias `b` is initialized to zero, and its dimension will be `out_dimension`.
```python
b = jnp.zeros((out_dimension,))
```
We don't use `(out_dimension, 1)` because broadcasting (the automatic adjustment of dimensions) now happens on the rows, no longer on the columns. Finally, the function returns the weights `w` and the bias `b`.
## [00:39:42] Initializing the Entire Network (MLP)
Now let's create the function to initialize the parameters of the entire neural network (Multi-Layer Perceptron, MLP).
```python
def init_mlp_params(key, layer_sizes):
```
This function needs a JAX key and a list `layer_sizes` that defines the dimensions of each layer (e.g., `[2, 4, 1]` for a network with a 2-neuron input, a 4-neuron hidden layer, and a 1-neuron output).
We need to generate a different random key for each layer to ensure that weights and biases are initialized independently.
```python
keys = jax.random.split(key, num=len(layer_sizes) - 1)
```
The number of keys needed is `len(layer_sizes) - 1`, because if we have `N` layers, we will have `N-1` pairs of weights and biases (one for each transition between layers).
Next, we iterate to create the parameters for each layer:
```python
params = []
for i in range(len(layer_sizes) - 1):
    params.append(
        init_layer_params(
            keys[i],
            layer_sizes[i],      # Input dimension of the current layer
            layer_sizes[i+1]     # Output dimension of the current layer
        )
    )
```
## [00:40:53] Practical Initialization Example
Let's verify the structure of the generated parameters with an example. Suppose we have `layer_sizes = [2, 4, 1]`.
- **First layer:** The weight matrix will have dimensions (2, 4) (input 2, output 4) and the bias will have dimension 4.
- **Second layer:** The weight matrix will have dimensions (4, 1) (input 4, output 1) and the bias will have dimension 1.
This confirms that our initialization logic is correct.
# Chapter 11: Forward and Cost Functions
## [00:41:45] Activation and Forward Propagation Function
Let's define the sigmoid activation function, which we will use for the output layer.
```python
def sigmoid(x):
    # Equivalent to jnp.tanh(x) / 2, but we use the standard definition
    return 1 / (1 + jnp.exp(-x))
```
Now let's implement the `forward` function, which calculates the network's output. This is the part where you need to pay close attention to the dimensions.
```python
def forward(params, x):
```
The function receives the network parameters and the input `x`. For each layer, except the last one, we perform the following operation:
1.  Multiply the input `x` by the weight matrix `w`.
2.  Add the bias `b`.
3.  Apply the `tanh` activation function.
We can use an elegant syntax to iterate over the parameters, directly unpacking the weights and biases from each tuple in the `params` list.
```python
# Loop over all layers except the last one
for w, b in params[:-1]:
    x = jnp.tanh(x @ w + b) # We use @ for matrix multiplication
```
For the last layer, we apply the sigmoid function, as we are dealing with a binary classification problem.
```python
# Last layer
final_w, final_b = params[-1]
output = sigmoid(x @ final_w + final_b)
return output
```
**Key points:**
- The broadcasting of the bias `b` (a vector) occurs correctly on the rows.
- The multiplication `x @ w` follows the convention where each row of `x` is a sample.
## [00:44:17] Cost Function (Binary Cross-Entropy)
Let's implement the cost function, "binary cross-entropy," which measures the network's error.
```python
def binary_cross_entropy(params, x, y):
    predictions = forward(params, x)
    loss = -jnp.mean(y * jnp.log(predictions) + (1 - y) * jnp.log(1 - predictions))
    return loss
```
The formula is the same as seen before and does not change with the new dimension convention.
# Chapter 12: Network Training
## [00:44:47] Updating Parameters with `tree_map`
The parameter update function (`update`) calculates the gradients and modifies the weights and biases to reduce the error.
```python
@jax.jit
def update(params, x, y, learning_rate):
```
We use the `@jax.jit` decorator to compile the function "just-in-time," optimizing its execution.
First, we calculate the gradients of the cost function with respect to the parameters.
```python
grad_fn = jax.grad(binary_cross_entropy)
grads = grad_fn(params, x, y)
```
Now, instead of using nested `for` loops to update every single weight and bias, we introduce a more powerful and general tool: `jax.tree_util.tree_map`.
```python
from jax import tree_util
updated_params = tree_util.tree_map(
    lambda p, g: p - learning_rate * g,
    params,
    grads
)
return updated_params
```
**How does `tree_map` work?**
- It applies a function (in this case, a `lambda` function) to every "leaf" of one or more tree-like data structures (like lists of tuples or dictionaries).
- `params` and `grads` are our tree structures. They have the same shape (a list of tuples).
- The `lambda p, g: p - learning_rate * g` function defines the update operation: it takes a parameter `p` and its corresponding gradient `g` and calculates the new parameter value.
- `tree_map` handles navigating the `params` and `grads` structures and applying this operation element by element, maintaining the correspondence.
This approach is much cleaner and more scalable than manual loops, especially for complex network architectures with parameters organized in nested dictionaries.
## [00:48:03] Training Loop
Now we can put everything together to train the network.
1.  **Setting Hyperparameters:**
    -   `layer_sizes`: `[2, 16, 1]` (2-dimensional input, 16-neuron hidden layer, 1-dimensional output).
    -   `learning_rate`: 0.01.
    -   `epochs`: 5000.
    -   `batch_size`: 64.
2.  **Main Loop (Epochs):**
    For each epoch, we shuffle the dataset to prevent the network from learning the order of the data.
    ```python
    # Generate a new key for the permutation
    key, subkey = jax.random.split(key)
    permutation = jax.random.permutation(subkey, x_train.shape[0])
    # Shuffle the data
    x_shuffled = x_train[permutation]
    y_shuffled = y_train[permutation]
    ```
3.  **Inner Loop (Mini-batch):**
    We iterate through the shuffled dataset, extracting small "mini-batches" of data.
    ```python
    num_batches = x_train.shape[0] // batch_size
    for i in range(num_batches):
        start = i * batch_size
        end = start + batch_size
        x_batch = x_shuffled[start:end]
        y_batch = y_shuffled[start:end]
        # Update parameters using the mini-batch
        params = update(params, x_batch, y_batch, learning_rate)
    ```
4.  **Monitoring Loss:**
    Periodically (e.g., every 100 epochs), we calculate and print the loss on the test dataset to monitor the training progress.
## [00:50:55] Model Evaluation
Once training is complete, we evaluate the model's performance on the test dataset.
1.  **Calculating Predictions:**
    The network's predictions are probabilities. We convert them to binary classes (True/False or 1/0) using a threshold of 0.5.
    ```python
    predictions_prob = forward(params, x_test)
    predictions_class = predictions_prob > 0.5
    ```
2.  **Calculating Accuracy:**
    Accuracy is the mean of the correct predictions.
    ```python
    accuracy = jnp.mean(predictions_class == y_test)
    print(f"Accuracy: {accuracy}")
    ```
    In our case, we achieve an accuracy of 98%, with only one classification error on the test set.
3.  **Confusion Matrix:**
    The confusion matrix gives us a more detailed view of the errors, showing how many samples of a class were classified correctly and how many were misclassified.
## [00:52:23] Visualizing the Results
Finally, we visualize the results.
- The training and test data are shown as points (circles and crosses).
- The colored background (pink and light blue) represents the **decision boundary** of the neural network. To obtain it, we create a very dense grid of points covering the entire space (`meshgrid`) and evaluate the network's output for each point. The color indicates the class predicted by the network in that region of space.
The visualization clearly shows how the network has learned to separate the two classes.
## [00:53:20] Conclusion
[Speaker 1] Ok, otherwise my time is up, so I wish you a good weekend and see you next week. Bye.
[Speaker 2] I'm about to stop the recording.