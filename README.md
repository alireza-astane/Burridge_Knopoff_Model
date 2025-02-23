# Burridge-Knopoff Model Sandpile Mapping

This repository contains the implementation and analysis for the Sandpile Mapping project based on the Burridge-Knopoff (BK) model. The project was developed under the supervision of Dr. Saman Moghimi Araghi at Sharif University of Technology.

## Overview

The BK model provides a way to study the dynamics of fault systems and earthquake-like phenomena. In our approach, we map the dynamics of the BK model to those seen in sandpile models. This mapping aims to reveal statistical features such as avalanche distributions and criticality.

## Project Structure

- **BK_model/**  
  Contains Jupyter notebooks and scripts for simulating the BK model dynamics. Files include notebooks for numerical solutions (e.g. `BKMoldel.ipynb` and `Copy of BKMoldel.ipynb`) and code for data extraction and visualization.
  
- **sandpile_model/**  
  Hosts notebooks related to sandpile simulation and analysis.

## Key Techniques and Tools

- **GPU Programming with Torch**  
  The simulations leverage PyTorch for fast tensor computations. Models run on GPU (CUDA) to accelerate numerical simulations related to the BK model, such as time-stepping and event detection.

- **Numba**  
  The repo also utilizes Numba to accelerate certain numerical parts of the code, particularly for non-tensor operations. Numba’s Just-In-Time (JIT) compilation optimizes Python code for high performance.

- **Jupyter Notebooks**  
  Many experiments and result visualizations are documented in Jupyter Notebooks (`.ipynb`) to facilitate interactive exploration and reproduction of results.

## Methodology

1. **Burridge-Knopoff Model Simulation:**  
   - The BK model is simulated by considering the evolution of displacement and velocity fields.  
   - Differential equations of motion are solved using explicit time-stepping methods.  
   - Events (such as slips or avalanches) are determined by analyzing the differences in cumulative displacement over time.

2. **Sandpile Mapping:**  
   - Simulation data from the BK model is mapped onto a sandpile framework.  
   - Statistical analysis is conducted on event sizes, durations, and steps to investigate scaling laws and avalanche distributions.

3. **GPU and JIT Acceleration:**  
   - PyTorch is used on a CUDA-enabled GPU to perform high-speed numerical operations.  
   - Numba accelerates loop-intensive calculations where GPU cannot be directly employed.

## Results and Discussion

*Below is a placeholder for project figures and plots. Replace this section with your experimental results and explanations.*

**Figure 1:**  
_Description of plot showing avalanche size distributions mapped from the BK model._

**Figure 2:**  
_Description of comparison plots between theoretical predictions and simulation outputs._

## Future Work

- Refining the numerical methods for better stability and accuracy.
- Further investigations into the scaling behavior and universality classes.

## Acknowledgements

Special thanks to Dr. Saman Moghimi Araghi and Sharif University of Technology for guidance and support.
