# Parallel Collatz Sequence Analyzer

## Problem Description

This project implements a parallel solution to the **Collatz problem** using **plain C++17**.

The **Collatz sequence** for a number `n` is defined as:

- If `n` is even → next number is `n / 2`
- If `n` is odd → next number is `3n + 1`

This continues until `n` becomes `1`. The **Collatz sequence length** is the number of steps taken to reach `1`.

---

## Goal

Create a command-line program that:

- Accepts **multiple input ranges** in the form `start-end`
- Computes the **maximum Collatz sequence length** for any number in each range
- Uses **parallelism** to process ranges efficiently
- Implements **both static and dynamic parallel scheduling**, using only **plain C++17 threads**

---

## Build and run

```bash
# Build
make

# Run
./collatz_par range1 [range2] [range3] ...
```

---

## Report
A report with implementation details and results can be found [here](collatz-report.pdf).
