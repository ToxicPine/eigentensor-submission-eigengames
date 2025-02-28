# MNIST Demo with Eigentensor

**NOTE:** remember to run `pip install -e .` in the root directory of the `eigentensor` repo to install the dependencies for this demo. you will also need to install the `tinygrad` package using `pip install tinygrad`, as well as FastAPI, PIL, and numpy.

This is a demonstration of running MNIST digit classification using Eigentensor for graph storage and execution. MNIST is a classic dataset in machine learning consisting of handwritten digits (0-9) that serves as a standard benchmark for testing image classification models. This demo shows how Eigentensor can seamlessly work with existing TinyGrad models to classify these handwritten digits.

## Overview

The demo consists of two main parts:

1. A Next.js based frontend (anytensor-mnist-frontend) that provides a drawing canvas where users can draw digits and see the model's predictions in real-time
2. A backend server containing:
   - A TinyGrad MNIST model implementation
   - Code to compile and save the model graph using Eigentensor
   - A FastAPI server that can run inference using either:
     - Regular TinyGrad execution
     - Eigentensor graph execution from stored format

This code corresponds to the backend server.

## Key Points

- The model architecture is a standard CNN for MNIST classification
- The same model weights are used for both execution modes
- Predictions between TinyGrad and Eigentensor execution are identical, validating that:
  1. The graph compilation and storage works correctly
  2. The Eigentensor runtime executes the graph as expected
  3. Real models can be easily ported to run on Eigentensor

## Implementation Details

The main components are:

- Model definition and weights loading in TinyGrad
- Graph compilation and serialization using Eigentensor's `TensorContext`
- FastAPI endpoints that support both execution modes
- Image preprocessing to handle canvas input from the frontend
- Integration with a modern Next.js frontend featuring:
  - Real-time drawing capabilities
  - Immediate feedback on predictions
  - Responsive design for various screen sizes

This demonstrates that Eigentensor provides a viable path for taking existing TinyGrad models and running them through our graph execution system while preserving the exact same behavior.
