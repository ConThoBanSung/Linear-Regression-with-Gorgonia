# Linear Regression with Gorgonia

This application demonstrates how to use the Gorgonia library in Go to build a simple linear regression model.

## How It Works

1. **Create a Computation Graph:**  
   The application initializes a computation graph to perform the regression operations.

2. **Create Inputs:**  
   The input nodes include vector `x`, weight vector `w`, and scalar `b`.

3. **Perform Operations:**  
   The model performs the operation `y = w * x + b` and calculates the loss using Mean Squared Error.

4. **Compute Gradients:**  
   The model computes the gradients of the loss with respect to the parameters `w` and `b`.

## Flow of the Code

1. **Graph Initialization:**  
   A new computation graph is created to hold all operations and variables.

2. **Input Nodes Creation:**  
   Input nodes are created for `x`, `w`, and `b`, allowing the model to accept data and parameters.

3. **Forward Pass:**  
   The forward pass calculates the output `y` using the formula `y = w * x + b`, and the loss is computed by comparing `y` to a target value.

4. **Loss Calculation:**  
   The loss is defined as the Mean Squared Error between the predicted output `y` and the expected target value.

5. **Run the Virtual Machine (VM):**  
   The VM executes the graph to compute the outputs and the loss.

6. **Gradient Calculation:**  
   The gradients of the loss with respect to `w` and `b` are computed using backpropagation.

7. **Output Results:**  
   The application outputs the calculated values of `y`, loss, and the gradients.

## Expected Results

After running the application, you will receive:

- The output value `y` (the predicted value based on the input).
- The loss value (indicating how well the model performed).
- The gradients with respect to `w` (weights) and `b` (bias), which can be used for further optimization in training.

## Requirements

- Go
- Gorgonia

## Usage Instructions

1. Install Go and the necessary libraries.
2. Run the source code and adjust the input parameters as needed.

## References

- [Gorgonia Documentation](https://gorgonia.org/)
