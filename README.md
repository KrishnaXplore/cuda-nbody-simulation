# Multicore Architecture Mini-Project: CUDA N-Body Simulation
**Team 19**  
**Team Members:** Kavana N, Kapale Krishna Shivaji, Anannya Mitra  

---

## 🌌 Project Overview

The **N-Body Simulation** calculates the dynamical physical evolution of a system of $N$ bodies/particles interacting via mutual gravitational forces over discrete time steps. Because every particle exerts a force on every other particle, the computational complexity of the all-pairs calculation is **$\mathcal{O}(N^2)$**.

Each pairwise interaction requires approximately 20 floating-point operations (FLOPs). For large datasets, this becomes a severe computational bottleneck on sequential CPUs. This project implements, benchmarks, and profiles progressive CUDA acceleration strategies on an **NVIDIA Tesla T4 GPU**.

---

## ⚙️ Mathematical Model

According to Newton's law of universal gravitation, the net force $\mathbf{F}_i$ on particle $i$ due to all other particles $j$ is:

$$\mathbf{F}_i = G \sum_{j \neq i} \frac{m_j \cdot (\mathbf{r}_j - \mathbf{r}_i)}{\left(\|\mathbf{r}_j - \mathbf{r}_i\|^2 + \epsilon^2\right)^{3/2}}$$

Where:
* $G = 6.67430 \times 10^{-11} \text{ m}^3\text{kg}^{-1}\text{s}^{-2}$ (Gravitational constant)
* $\epsilon = 10^{-9}$ (Softening factor, preventing infinite force and division-by-zero during close encounters)
* $\Delta t = 0.01 \text{ s}$ (Euler integration time-step)

---

## 🚀 Optimization Stages

1. **Sequential CPU Baseline (`nbody_cpu.cpp`)**:
   * Standard C++ implementation compiled with `-O3`.
   * Serves as the ground-truth reference baseline for calculating speedups.

2. **Task 1: Naive CUDA Kernel (`nbody_naive.cu`)**:
   * Parallelizes the outer loop across CUDA threads (1 thread per particle).
   * Reads particle positions directly from high-latency GPU global memory.

3. **Task 2: Shared-Memory Tiled Kernel (`nbody_shared.cu`)**:
   * Divides $N$ particles into tiles cached in on-chip **Shared Memory** (`__shared__`).
   * Eliminates redundant global memory round-trips via coalesced loads and `__syncthreads()`.
   * Evaluated across thread block sizes: **64, 128, and 256**.

4. **Task 3: Fully Optimized Kernel (`nbody_optimized.cu`)**:
   * **Loop Unrolling:** `#pragma unroll 16` to reduce branch overhead and maximize Instruction-Level Parallelism (ILP).
   * **Fast-Math Intrinsics:** Hardware-accelerated reciprocal square root `__frsqrt_rn()` with `-use_fast_math`.

5. **Numerical Accuracy Verification (`verify_accuracy.cu`)**:
   * Compares the output of the fast-math GPU kernel against double-precision CPU standards to measure floating-point error.

6. **Hardware Resource Profiling (`ptxas -v`)**:
   * Measures registers per thread, shared memory consumption, and theoretical occupancy on Turing architecture (`sm_75`).

---

## 📊 Benchmark Results

### Execution Time & Speedup ($N = 2048$ particles)

| Implementation | Execution Time (ms) | Speedup vs CPU | Throughput / Notes |
| :--- | :---: | :---: | :--- |
| **Sequential CPU (`-O3`)** | **22.371 ms** | $1.0\times$ | Reference Baseline |
| **Naive CUDA (Block: 256)** | **0.814 ms** | $27.5\times$ | Bound by Global Memory Bandwidth |
| **Shared Memory (Block: 64)** | **0.948 ms** | $23.6\times$ | Underutilizes Warp Schedulers |
| **Shared Memory (Block: 128)** | **0.804 ms** | $27.8\times$ | Optimal Shared Memory Tiling |
| **Shared Memory (Block: 256)** | **0.848 ms** | $26.4\times$ | Higher Register Pressure |
| **Fully Optimized CUDA** | **~0.396 ms** | **$\approx 56.5\times$** | Unrolled + `__frsqrt_rn` Fast Math |

### Compute Throughput vs Block Size ($N = 4096$)

$$\text{GFLOP/s} = \frac{20 \times N^2}{\text{Time (s)} \times 10^9}$$

* **Block Size 64:** 354.88 GFLOP/s (0.946 ms)
* **Block Size 128:** **423.67 GFLOP/s** (0.792 ms) *(Optimal)*
* **Block Size 256:** 422.51 GFLOP/s (0.794 ms)

---

## 🔍 Accuracy & Profiling Analysis

* **Numerical Precision:**
  $$\text{Mean Absolute Velocity Error} = 8.68954 \times 10^{-16} \text{ m/s}$$
  *Confirms that hardware fast-math intrinsics deliver over $2\times$ additional speedup with zero noticeable loss in physical accuracy.*

* **GPU Resource Consumption (Tesla T4, Compute Capability 7.5):**
  * **Registers per Thread:** 40 registers
  * **Stack Frame Spills:** 0 bytes (no register spilling to local memory)
  * **Barriers per Block:** 1 barrier

---

## 💻 How to Run

Open the interactive Jupyter Notebook directly in Google Colab:
* Notebook path: [`MAP_TEAM19.ipynb`](./MAP_TEAM19.ipynb)
* Ensure runtime is set to **GPU (T4 or higher)** via `Runtime -> Change runtime type -> T4 GPU`.
