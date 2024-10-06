package main

import (
	"fmt"
	"log"

	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

func main() {
	// Create a new computation graph
	g := gorgonia.NewGraph()

	// Create input nodes with appropriate types
	x := gorgonia.NewVector(g, tensor.Float64, gorgonia.WithShape(2), gorgonia.WithName("x"))
	w := gorgonia.NewVector(g, tensor.Float64, gorgonia.WithShape(2), gorgonia.WithName("w"))
	b := gorgonia.NewScalar(g, tensor.Float64, gorgonia.WithName("b"))

	// Create a simple operation: y = w * x + b
	wx := gorgonia.Must(gorgonia.Mul(w, x))                // w * x
	broadcastB := gorgonia.Must(gorgonia.Reshape(b, tensor.Shape{1})) // Reshape b for broadcasting
	y := gorgonia.Must(gorgonia.Add(wx, broadcastB))      // Add bias

	// Define a target tensor (for example, expected output)
	target := gorgonia.NewScalar(g, tensor.Float64, gorgonia.WithName("target"))

	// Define a loss function (Mean Squared Error)
	loss := gorgonia.Must(gorgonia.Mean(gorgonia.Must(gorgonia.Square(gorgonia.Must(gorgonia.Sub(y, target))))))

	// Create a VM (Virtual Machine) to run the graph
	vm := gorgonia.NewTapeMachine(g)

	// Set values for the inputs
	if err := gorgonia.Let(x, tensor.New(tensor.WithShape(2), tensor.WithBacking([]float64{1.0, 2.0}))); err != nil {
		log.Fatalf("Failed to set x: %v", err)
	}
	if err := gorgonia.Let(w, tensor.New(tensor.WithShape(2), tensor.WithBacking([]float64{0.5, -0.5}))); err != nil {
		log.Fatalf("Failed to set w: %v", err)
	}
	if err := gorgonia.Let(b, tensor.New(tensor.WithShape(), tensor.WithBacking([]float64{0.0}))); err != nil {
		log.Fatalf("Failed to set b: %v", err)
	}
	if err := gorgonia.Let(target, tensor.New(tensor.WithShape(), tensor.WithBacking([]float64{1.0}))); err != nil {
		log.Fatalf("Failed to set target: %v", err)
	}

	// Run the VM
	if err := vm.RunAll(); err != nil {
		log.Fatalf("Error during VM run: %v", err)
	}

	// Output the results after the forward pass
	fmt.Println("After Forward Pass:")
	fmt.Println("Output (y):", y.Value())
	fmt.Println("Loss:", loss.Value())

	// Reset the VM for backpropagation (no return value)
	vm.Reset()

	// Compute gradients with respect to w and b
	grads, err := gorgonia.Grad(loss, w, b)
	if err != nil {
		log.Fatalf("Error computing gradients: %v", err)
	}

	// Run the VM again to compute gradients
	if err := vm.RunAll(); err != nil {
		log.Fatalf("Error during VM run for gradient computation: %v", err)
	}

	// Display the results
	if grads[0] != nil {
		fmt.Println("Gradients (dy/dw):", grads[0].Value())
	} else {
		fmt.Println("Gradients (dy/dw): <nil>")
	}
	if grads[1] != nil {
		fmt.Println("Gradients (dy/db):", grads[1].Value())
	} else {
		fmt.Println("Gradients (dy/db): <nil>")
	}

	// Clean up the VM
	if err := vm.Close(); err != nil {
		log.Fatalf("Error closing VM: %v", err)
	}
}
