Model Card for the BBO Optimisation Approach

1. Overview
Name: Sequential Trend‑Guided Black‑Box Optimisation
Type: Manual + LLM‑assisted heuristic optimiser
Version: Stage 2, Round 10

2. Intended Use
This optimisation approach is suitable for:
Black‑box optimisation tasks with limited queries
Sequential decision‑making
Exploration–exploitation balancing
Interpretable optimisation strategies
Educational and research contexts

Avoid using this approach for:
High‑dimensional optimisation requiring thousands of samples
Noisy or discontinuous functions
Real‑time optimisation
Scenarios requiring formal convergence guarantees

3. Details of the Strategy
Across ten rounds, the strategy evolved through:
Early broad exploration
Identification of directional trends
Increasing exploitation in high‑performing regions (e.g., Function 5)
Controlled exploration in unstable or declining regions (e.g., Function 7)
LLM‑assisted reasoning using structured prompts, few‑shot patterns, and controlled decoding
Transparency improvements through documentation, reproducible logic, and explicit assumptions
The approach integrates:
Trend‑based heuristics
Local smoothness assumptions
Prompt‑driven interpretability
Incremental coordinate adjustments
Monitoring for emergent behaviours and diminishing returns

4. Performance Summary
Performance is evaluated by:
Output magnitude (higher is better for some functions, less negative is better for others)
Trend stability
Directional improvement
Reduction of uncertainty in the search space

Key observations:
Function 5 shows strong exponential improvement
Function 1 collapses to zero consistently
Function 8 shows smooth, stable behaviour
Functions 2, 3, 4, 6, 7 show mixed but interpretable trends

5. Assumptions and Limitations
Assumptions:
Local smoothness of functions
Directional trends are meaningful
Outputs are stable enough to guide exploitation
Small coordinate changes produce interpretable signals

Limitations:
Sampling bias toward early promising regions
No surrogate model or uncertainty quantification
One query per week limits exploration
Potential overfitting to local patterns
Black‑box nature prevents validation of underlying structure

6. Ethical Considerations
Transparency and interpretability support:
Reproducibility
Peer review
Responsible optimisation

Clear documentation of assumptions

Avoidance of misleading conclusions
