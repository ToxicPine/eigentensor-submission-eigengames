# EigenTensor MNIST Demo

This is a Next.js frontend demo showcasing EigenTensor's capabilities through a simple MNIST digit recognition application.

## Overview

The demo allows users to:
1. Draw a digit (0-9) on a canvas
2. Submit it to an EigenTensor-enabled GPU node for inference
3. See the model's prediction and confidence score

Behind the scenes, the drawn digit is:
1. Preprocessed into the correct tensor format
2. Sent to an EigenTensor node running the MNIST model
3. Processed with consensus verification between nodes
4. Results returned to display the prediction

## Getting Started
