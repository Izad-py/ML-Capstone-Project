Initial codebase
my initial codebase for Week 1 (Module 13) was built entirely from scratch. at that stage, I had only completed Modules 1–12, so my thinking was shaped by foundational maths, probability, statistics, generalisation theory and basic model evaluation. I didn’t yet have clustering, PCA or RL tools — so the most natural starting point was a simple Python script that generated random queries, stored them and printed them for submission.

I chose to build the codebase myself rather than use a public library because the BBO challenge is intentionally open‑ended: the functions are unknown, the feedback is limited and the optimisation landscape is opaque. building from scratch gave me full control over how I stored data, tracked trends and evolved my strategy. it also made the repository more personal and reflective of my learning journey.

my GitHub repository includes:

weekly query files

datasheets and model cards

reflections for each module

the evolution of my strategy from Week 1 to Week 13

I structured it so peers can explore the logic behind each decision and see how the code matured over time.

Code modification
my code changed significantly week by week as I progressed through Modules 13–24.

Week 1–3 : basic exploration
I added:

random query generation

simple storage

basic trend printing

this reflected early ML concepts: variables, functions, logistic regression intuition and basic perceptron logic.

in the first three weeks, my code was extremely simple. I generated random queries, saved them and printed them. I added basic storage and tiny trend printing just so I could see what happened from one week to the next. at this stage, I didn’t have any optimisation tools or any real sense of structure. I was still thinking in terms of variables, functions and simple decision boundaries from logistic regression, and early perceptron intuition.

as I learned about SVM margins and neural network gradients, I started adding very small directional nudges. if a point improved slightly, I moved a tiny bit further in that direction. I also added basic comparisons between the last two outputs to see whether I should continue or reverse. it wasn’t real optimisation — more like early curiosity-driven exploration.

Week 4–6 (Modules 16–18): early structure
I introduced:

simple heuristics

small directional adjustments

basic variance checks

this came from neural network intuition (gradients, direction of change) and hyperparameter tuning ideas (grid vs random search).

As I learned more about neural networks, deeper architectures and hyperparameter tuning, my code became more structured. I introduced simple heuristics: if two rounds improved, increase the step size; if things worsened, shrink it. I also started adjusting multiple coordinates at once instead of just one, inspired by the idea of momentum.

I added basic variance checks to see which dimensions seemed to matter more. this came from thinking about gradients and direction of change, even though I couldn’t compute real gradients. hyperparameter tuning modules also influenced me — I experimented with small grid-like sweeps and random search logic.

these changes made my moves more deliberate and reduced the randomness of the early weeks.

Week 7–9 (Modules 19–21): pattern recognition
I added:

clustering‑style grouping

centroid tracking

early interpretability logic

Modules 19–21 emphasised structure, transparency and responsible modelling, which helped me refine how I interpreted the outputs.

after learning about LLMs, tokens, attention and interpretability, I started recognising patterns in the outputs. I added clustering-style grouping of past points, simple centroid tracking and early interpretability logic.

I began encoding trends like “up-up-down” to see whether a direction was stable or noisy. I also added a primitive form of attention: focusing more on dimensions that showed the biggest changes.

the interpretability module pushed me to document decisions more clearly, so I added justification strings and simple logging. this made the code easier to understand and helped me see why certain moves were chosen.

Week 10–11 (Modules 22–23): PCA‑inspired strategy
I implemented:

variance‑based dimension ranking

principal‑direction movement

noise filtering

PCA taught me how to identify the most informative directions — this became one of the most impactful changes in my optimisation.

once I reached PCA, my code changed dramatically. I implemented variance-based dimension ranking to identify which coordinates carried the most information. I added principal-direction movement, noise filtering and step-size scaling based on how much variance each dimension explained.

this was one of the most impactful changes. instead of guessing which dimension mattered, I used variance to guide movement. this made my optimisation far more efficient and helped me exploit strong trends in functions like F8 while avoiding wasted moves.

Week 12–13 (Module 24): reinforcement learning
I added:

exploration–exploitation balancing

reward‑based updates

bandit‑style reasoning

policy refinement

this was the biggest leap in sophistication. RL helped me treat each function as a separate decision environment and adjust my strategy based on reward history.

the reinforcement learning module transformed my strategy. I added exploration–exploitation balancing, reward-based updates and bandit-style reasoning. each function became its own “arm,” and I tracked which moves gave better rewards.

I added simple Q-value-like tracking, updating my expectations based on whether a move improved or worsened the output. I refined my policy each week, choosing when to explore uncertain regions and when to exploit strong trends.

this was the biggest leap in sophistication. my final queries were the most structured, deliberate and effective because they were guided by reward history rather than intuition alone.

Most impactful change
the shift from random exploration to variance‑driven and reward‑driven optimisation had the strongest impact. it allowed me to exploit strong trends (Functions 5 and 8) while exploring uncertain regions (Functions 4 and 6) more intelligently.

Final result
my scores evolved differently across functions:

Function 5: strong upward trajectory, reaching ~3947 in Week 13

Function 8: stable and high (~9.24)

Function 2: fluctuating but ended positively

Function 7: moderate positive behaviour

Function 3: small negative but structured

Function 4: consistently negative but predictable

Function 6: negative but stable

Function 1: flat at zero throughout

If I had more time
I would:

introduce RL earlier

use more sophisticated exploration strategies (epsilon‑greedy, softmax)

apply PCA‑style dimension ranking from Week 4 instead of Week 10

test boundary‑tightening sooner

a fresh start would allow me to build a more structured strategy earlier in the project.

Trade-offs and decisions
the biggest trade-off was balancing exploration and exploitation.

Exploration trade-offs
exploring too widely wastes rounds

exploring too narrowly risks missing better regions

noisy functions (4, 6) required more exploration

stable functions (5, 8) rewarded exploitation

Short-term vs long-term
some moves gave small immediate gains but didn’t align with long-term trends. other moves sacrificed short-term reward to explore new curvature. deciding when to trust a trend versus when to challenge it was a constant tension.

Noise vs structure
some outputs fluctuated unexpectedly. I had to decide whether:

the change was meaningful

or just noise

this shaped how aggressively I updated each coordinate.

Learning and application
the most important lesson I learned is that optimisation is fundamentally about learning structure from limited feedback. even without knowing the underlying functions, patterns emerge — clusters, basins, principal directions, reward trajectories — and these patterns can guide decisions meaningfully.

this mirrors real-world ML work:

uncertainty

limited data

noisy feedback

iterative improvement

balancing exploration and exploitation

documenting decisions clearly

I also learned the value of transparency. building datasheets, model cards and weekly reflections made my process more interpretable — a key theme from Module 21.

How I’ll apply this
in future ML projects, I will:

use variance‑based reasoning early

treat optimisation as iterative learning

document decisions clearly

balance exploration and exploitation deliberately

apply RL-inspired strategies to noisy environments

What surprised me
I was most surprised by:

how differently participants approached the same challenge

how much structure emerged from seemingly random outputs

how powerful simple heuristics became when applied consistently

how RL concepts naturally fit the BBO challenge

how much the early modules (maths, probability, statistics) shaped my thinking later

seeing peers’ repositories also highlighted how diverse optimisation strategies can be — some built complex models, others relied on intuition, and some used heavy exploration. this diversity made the challenge richer and more insightful.
