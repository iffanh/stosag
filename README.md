# StoSAG: Stochastic Simplex Approximate Gradient Optimization

Python implementation of the Stochastic Simplex Approximate Gradient (StoSAG) algorithm for robust optimization under uncertainty.

## Overview

StoSAG is an ensemble-based stochastic optimization method designed for problems where objective functions are uncertain or noisy. This repository provides a flexible implementation and a test case using the Rosenbrock function.

## Contents

- `main.py`: Core optimizer class (`stosag`)
- `utilities.py`: Covariance and gradient calculation utilities
- `test.py`: Example script using the Rosenbrock function with uncertainty

## Installation

Clone the repository and install required packages:

```bash
pip install git+https://github.com/iffanh/stosag.git
```

## Usage

Run the provided test script:

```bash
python test_rosenbrock.py
```

You can modify `test_rosenbrock.py` to use your own objective functions and uncertainty models.

## How It Works

- Generates an ensemble of solutions around the current iterate
- Estimates gradients using ensemble covariances
- Updates iterates using stochastic gradient descent with backtracking line search

