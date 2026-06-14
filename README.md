# BBO Capstone Project 

## Section 1: Project Overview

This project is part of the Black‑Box Optimisation (BBO) capstone, where the goal is to optimise eight unknown functions without ever seeing their underlying equations. Instead of working with explicit models, I submit input vectors and receive a single numeric output for each function. The challenge is to use these outputs to understand the behaviour of each function and design better queries over time.

The purpose of the BBO capstone is to develop the ability to reason under uncertainty, iterate based on limited information and apply machine‑learning thinking to a system that behaves like a black box. This mirrors real‑world ML scenarios where models, APIs or physical systems cannot be inspected directly. Completing this project strengthens skills in experiment design, pattern recognition, iterative modelling and technical communication — all of which are valuable in data‑driven roles and future career paths.

## Section 2: Inputs and Outputs

Each of the eight functions accepts a vector of inputs within the range [0,1]. The dimensionality varies:

- Function 1 & 2: 2D  
- Function 3: 3D  
- Function 4 & 5: 4D  
- Function 6: 5D  
- Function 7: 6D  
- Function 8: 8D  

Queries must follow the required format:

`0.xxxxxx-0.xxxxxx-...`

Each coordinate must begin with `0.` and include six decimal places.

Example queries:

- Function 1: `0.900000-0.100000`  
- Function 3: `0.439031-0.547677-0.432788`

The portal returns a single numeric output for each query. This output acts as a performance signal and is the only information available about the hidden function. Some functions behave like maximisation problems (higher is better), while others behave like minimisation problems (less negative is better). These behaviours must be inferred from the data.

## Section 3: Challenge Objectives

The objective of the BBO capstone is to identify input regions that produce better outputs for each of the eight hidden functions. Because the structure of each function is unknown, the challenge is to make informed decisions based solely on observed outputs.

Key constraints include:

- A limited number of queries  
- No access to the true function  
- High‑dimensional input spaces  
- The need to justify each query based on previous results  

The aim is not only to find good points but to demonstrate a thoughtful, evolving strategy that adapts as more data becomes available.

## Section 4: Technical Approach

Across the first three rounds of queries, my approach has evolved as the dataset expanded from 10 to 12 points per function.

In the early rounds, I focused on exploration, spreading points across the domain to understand basic behaviour. As more data became available, I began identifying promising regions and adjusting my queries accordingly.

My strategy uses simple ML concepts and heuristics:

- Regression thinking helps me reason about trends and smooth surfaces.  
- SVM concepts help me imagine separating “high performance” vs “low performance” regions.  
  - Soft‑margin SVM ideas help when boundaries are uncertain.  
  - Kernel SVM ideas help when the surface appears non‑linear.  
- Iterative modelling guides how I refine queries based on previous outputs.  

Exploration is used for functions that appear flat or noisy, such as Function 1.  
Exploitation is used for functions with clear high‑performance regions, such as Functions 2, 5, 7 and 8.  
For functions with local optima, such as Function 3, I use small, controlled steps around the best point.

This approach is thoughtful because it treats each function differently based on observed behaviour, uses ML concepts to guide intuition and documents the reasoning behind each query.
