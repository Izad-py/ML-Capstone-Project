Datasheet for the BBO Capstone Project Dataset

1. Motivation
The purpose of this dataset is to support the Black‑Box Optimisation (BBO) capstone project, where the goal is to optimise eight unknown functions using sequential queries. The dataset captures all query points submitted across ten rounds and the corresponding outputs returned by the BBO system.
It enables analysis of optimisation behaviour, trend detection, exploration–exploitation balance, and the development of interpretable strategies for black‑box search.

2. Composition
The dataset contains:
10 rounds of queries
8 functions per round
19 total data points per function
Each query is a vector of floating‑point values in [0,1], formatted to six decimal places
Each output is a single numeric value (float)

The dataset includes:
Input vectors (queries)
Output values (function evaluations)
Metadata such as round number and timestamp

Gaps:
No information about the underlying functions
No gradients, structure, or noise model
No uncertainty estimates
Some regions of the search space remain unexplored due to sequential sampling

3. Collection Process
Queries were generated manually using an iterative optimisation strategy informed by:
Trend analysis
Local directional movement
Exploitation of high‑performing regions
Controlled exploration of uncertain areas
LLM‑assisted reasoning and prompt‑based decision support
The dataset was collected over ten weeks, with one query per function submitted each week through the BBO portal.

4. Preprocessing and Uses
No preprocessing was applied beyond formatting values to six decimal places.

Intended uses:
Trend analysis
Surrogate modelling
Exploration–exploitation strategy development
Reproducibility of optimisation decisions
Documentation of black‑box behaviour

Inappropriate uses:
Inferring the true underlying functions
Using the dataset as a benchmark for supervised learning
Treating outputs as noise‑free or deterministic without verification

5. Distribution and Maintenance
The dataset is publicly available in the GitHub repository for the BBO capstone project.
Terms of use follow standard academic fair‑use guidelines.
The dataset is maintained by the project author (ali), who updates it weekly as new queries are submitted.
