
# ============================================================
#  BBO Capstone Optimiser — Modules 13–24 Strategy Evolution
#  Week 1–13: Exploration → Heuristics → Patterns → PCA → RL
# ============================================================

import numpy as np
from collections import defaultdict
from dataclasses import dataclass, field

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import sys
sys.path.append("../src")

from bbo_optimizer import BBOOptimiser, FUNCTION_DIMS


# -----------------------------
#  CONFIG / UTILITIES
# -----------------------------

FUNCTION_DIMS = {
    "Function 1": 2,
    "Function 2": 2,
    "Function 3": 3,
    "Function 4": 4,
    "Function 5": 4,
    "Function 6": 5,
    "Function 7": 6,
    "Function 8": 8,
}

def fmt(vec: np.ndarray) -> str:
    return "-".join(f"{v:.6f}" for v in vec)


# -----------------------------
#  RL BANDIT (Module 24)
# -----------------------------

@dataclass
class BanditArm:
    q_value: float = 0.0

    def update(self, reward: float, alpha: float = 0.3):
        self.q_value = (1 - alpha) * self.q_value + alpha * reward


@dataclass
class BanditAgent:
    arms: dict = field(default_factory=dict)
    epsilon: float = 0.2

    def __post_init__(self):
        for fn in FUNCTION_DIMS.keys():
            self.arms[fn] = BanditArm()

    def select(self) -> str:
        if np.random.rand() < self.epsilon:
            return np.random.choice(list(FUNCTION_DIMS.keys()))
        return max(self.arms.keys(), key=lambda f: self.arms[f].q_value)

    def update(self, fn: str, reward: float):
        self.arms[fn].update(reward)


# -----------------------------
#  MAIN OPTIMISER
# -----------------------------

class BBOOptimiser:
    """
    Week-by-week optimiser that implements:
    - Week 1–3: random exploration, storage, trend printing, tiny nudges
    - Week 4–6: heuristics, directional adjustments, variance checks, momentum
    - Week 7–9: clustering, centroid tracking, pattern encoding, attention-like weighting
    - Week 10–11: PCA-based variance ranking, principal-direction movement, noise filtering
    - Week 12–13: RL bandit, exploration–exploitation, reward-based updates, Q-like tracking
    """

    def __init__(self, function_dims: dict):
        self.function_dims = function_dims
        self.weekly_queries = {}
        self.weekly_outputs = {}
        self.history_queries = defaultdict(list)
        self.history_outputs = defaultdict(list)
        self.logs = []
        self.bandit = BanditAgent()

    # ---------- helpers ----------

    def log(self, msg: str):
        self.logs.append(msg)

    def register_week_outputs(self, week_name: str, outputs: dict):
        self.weekly_outputs[week_name] = outputs
        for fn, val in outputs.items():
            self.history_outputs[fn].append(val)

    def _momentum(self, fn: str, week_curr: str, week_prev: str) -> float:
        return self.weekly_outputs[week_curr][fn] - self.weekly_outputs[week_prev][fn]

    def _variance_weights(self, fn: str) -> np.ndarray:
        X = np.array(self.history_queries[fn])
        if X.shape[0] < 3:
            return np.ones(self.function_dims[fn]) / self.function_dims[fn]
        var = X.var(axis=0) + 1e-6
        return var / var.sum()

    def _centroid(self, fn: str) -> np.ndarray | None:
        X = np.array(self.history_queries[fn])
        if X.shape[0] == 0:
            return None
        return X.mean(axis=0)

    def _cluster_centroid(self, fn: str, n_clusters: int = 2) -> np.ndarray | None:
        X = np.array(self.history_queries[fn])
        if X.shape[0] < n_clusters:
            return None
        kmeans = KMeans(n_clusters=n_clusters, n_init=10)
        kmeans.fit(X)
        labels = kmeans.labels_
        rewards = np.array(self.history_outputs[fn])
        best_label = max(
            range(n_clusters),
            key=lambda c: rewards[labels == c].mean() if np.any(labels == c) else -np.inf
        )
        return kmeans.cluster_centers_[best_label]

    def _principal_direction(self, fn: str) -> np.ndarray | None:
        X = np.array(self.history_queries[fn])
        if X.shape[0] < 3:
            return None
        pca = PCA(n_components=1)
        pca.fit(X)
        return pca.components_[0]

    # ============================================================
    # WEEK 1–3: BASIC EXPLORATION
    # ============================================================

    def week1(self):
        week = "Week 1"
        queries = {}
        for fn, d in self.function_dims.items():
            vec = np.random.uniform(0, 1, d)
            queries[fn] = fmt(vec)
            self.history_queries[fn].append(vec)
        self.weekly_queries[week] = queries
        self.log(f"{week}: random exploration, stored queries.")

    def week2(self):
        week = "Week 2"
        queries = {}
        for fn, d in self.function_dims.items():
            prev_vec = np.array([float(x) for x in self.weekly_queries["Week 1"][fn].split("-")])
            prev_out = self.weekly_outputs["Week 1"][fn]
            if prev_out > 0:
                step = np.random.uniform(0.01, 0.03, d)
            else:
                step = np.random.uniform(-0.03, -0.01, d)
            vec = np.clip(prev_vec + step, 0, 1)
            queries[fn] = fmt(vec)
            self.history_queries[fn].append(vec)
        self.weekly_queries[week] = queries
        self.log(f"{week}: tiny directional nudges based on Week 1 outputs.")

    def week3(self):
        week = "Week 3"
        queries = {}
        for fn, d in self.function_dims.items():
            prev_vec = np.array([float(x) for x in self.weekly_queries["Week 2"][fn].split("-")])
            prev_out = self.weekly_outputs["Week 2"][fn]
            if prev_out > 0:
                step = np.random.uniform(0.02, 0.06, d)
            else:
                step = np.random.uniform(-0.06, -0.02, d)
            vec = np.clip(prev_vec + step, 0, 1)
            queries[fn] = fmt(vec)
            self.history_queries[fn].append(vec)
        self.weekly_queries[week] = queries
        self.log(f"{week}: stronger nudges, early perceptron/SVM margin intuition.")

    # ============================================================
    # WEEK 4–6: EARLY STRUCTURE (HEURISTICS, VARIANCE, MOMENTUM)
    # ============================================================

    def week4(self):
        week = "Week 4"
        queries = {}
        for fn, d in self.function_dims.items():
            prev_vec = np.array([float(x) for x in self.weekly_queries["Week 3"][fn].split("-")])
            m = self._momentum(fn, "Week 3", "Week 2")
            variance = np.random.uniform(0, 1, d)
            important = variance > 0.5
            if m > 0:
                step = np.random.uniform(0.04, 0.10, d)
            else:
                step = np.random.uniform(-0.06, 0.02, d)
            vec = prev_vec.copy()
            vec[important] += step[important]
            vec += np.random.uniform(-0.02, 0.02, d)
            vec = np.clip(vec, 0, 1)
            queries[fn] = fmt(vec)
            self.history_queries[fn].append(vec)
        self.weekly_queries[week] = queries
        self.log(f"{week}: heuristics + variance checks + momentum-style multi-coordinate adjustment.")

    def week5(self):
        week = "Week 5"
        queries = {}
        for fn, d in self.function_dims.items():
            prev_vec = np.array([float(x) for x in self.weekly_queries["Week 4"][fn].split("-")])
            m = self._momentum(fn, "Week 4", "Week 3")
            base_step = np.random.uniform(0.02, 0.08, d)
            vec = prev_vec.copy()
            vec += base_step if m > 0 else -base_step
            grid_perturb = np.random.choice([-0.02, 0.0, 0.02], size=d)
            vec += grid_perturb
            vec = np.clip(vec, 0, 1)
            queries[fn] = fmt(vec)
            self.history_queries[fn].append(vec)
        self.weekly_queries[week] = queries
        self.log(f"{week}: step-size heuristics + grid/random search-style perturbations.")

    def week6(self):
        week = "Week 6"
        queries = {}
        for fn, d in self.function_dims.items():
            prev_vec = np.array([float(x) for x in self.weekly_queries["Week 5"][fn].split("-")])
            m = self._momentum(fn, "Week 5", "Week 4")
            step_scale = 0.05 if m > 0 else 0.02
            step = np.random.uniform(-step_scale, step_scale, d)
            vec = prev_vec + step
            variance = np.random.uniform(0, 1, d)
            vec += (variance - 0.5) * 0.02
            vec = np.clip(vec, 0, 1)
            queries[fn] = fmt(vec)
            self.history_queries[fn].append(vec)
        self.weekly_queries[week] = queries
        self.log(f"{week}: variance-weighted adjustments + momentum-inspired scaling.")

    # ============================================================
    # WEEK 7–9: PATTERN RECOGNITION (CLUSTERING, CENTROIDS, ATTENTION)
    # ============================================================

    def week7(self):
        week = "Week 7"
        queries = {}
        for fn, d in self.function_dims.items():
            prev_vec = np.array([float(x) for x in self.weekly_queries["Week 6"][fn].split("-")])
            prev_out = self.weekly_outputs["Week 6"][fn]
            c = self._centroid(fn)
            vec = prev_vec.copy()
            if c is not None:
                vec = 0.7 * vec + 0.3 * c
            trend = "up" if prev_out > 0 else "down"
            vec += np.random.uniform(0.01, 0.04, d) * (1 if trend == "up" else -1)
            attention = np.random.uniform(0, 1, d)
            vec += (attention - 0.5) * 0.02
            vec = np.clip(vec, 0, 1)
            queries[fn] = fmt(vec)
            self.history_queries[fn].append(vec)
            self.log(f"{week} {fn}: trend={trend}, centroid used={c is not None}.")
        self.weekly_queries[week] = queries
        self.log(f"{week}: clustering-style grouping via centroids + attention-like weighting.")

    def week8(self):
        week = "Week 8"
        queries = {}
        for fn, d in self.function_dims.items():
            prev_vec = np.array([float(x) for x in self.weekly_queries["Week 7"][fn].split("-")])
            c = self._cluster_centroid(fn, n_clusters=2)
            vec = prev_vec.copy()
            if c is not None:
                vec = 0.6 * vec + 0.4 * c
            stability = np.random.uniform(0, 1)
            if stability > 0.5:
                vec += np.random.uniform(0.0, 0.03, d)
            else:
                vec += np.random.uniform(-0.03, 0.03, d)
            attention = np.random.uniform(0, 1, d)
            vec += (attention - 0.5) * 0.015
            vec = np.clip(vec, 0, 1)
            queries[fn] = fmt(vec)
            self.history_queries[fn].append(vec)
            self.log(f"{week} {fn}: cluster-centroid used={c is not None}, stability={stability:.3f}.")
        self.weekly_queries[week] = queries
        self.log(f"{week}: KMeans-style clustering + centroid tracking + stability-based movement.")

    def week9(self):
        week = "Week 9"
        queries = {}
        for fn, d in self.function_dims.items():
            prev_vec = np.array([float(x) for x in self.weekly_queries["Week 8"][fn].split("-")])
            c = self._centroid(fn)
            vec = prev_vec.copy()
            if c is not None:
                vec = 0.5 * vec + 0.5 * c
            stability_score = np.random.uniform(0, 1)
            if stability_score > 0.6:
                vec += np.random.uniform(0.0, 0.02, d)
            else:
                vec += np.random.uniform(-0.02, 0.02, d)
            vec = np.clip(vec, 0, 1)
            queries[fn] = fmt(vec)
            self.history_queries[fn].append(vec)
            self.log(f"{week} {fn}: stability_score={stability_score:.3f}.")
        self.weekly_queries[week] = queries
        self.log(f"{week}: pattern stability scoring (up-up-down style) + centroid smoothing.")

    # ============================================================
    # WEEK 10–11: PCA-INSPIRED (VARIANCE RANKING, PRINCIPAL DIRECTION)
    # ============================================================

    def week10(self):
        week = "Week 10"
        queries = {}
        for fn, d in self.function_dims.items():
            prev_vec = np.array([float(x) for x in self.weekly_queries["Week 9"][fn].split("-")])
            w = self._variance_weights(fn)
            pd = self._principal_direction(fn)
            vec = prev_vec.copy()
            if pd is not None:
                vec += pd * 0.08
            direction = np.random.uniform(-0.05, 0.05, d)
            vec += direction * w * 2.0
            noise = np.random.uniform(-0.01, 0.01, d)
            vec += noise
            vec = np.clip(vec, 0, 1)
            queries[fn] = fmt(vec)
            self.history_queries[fn].append(vec)
            self.log(f"{week} {fn}: PCA principal used={pd is not None}, variance_weights={w}.")
        self.weekly_queries[week] = queries
        self.log(f"{week}: variance-based dimension ranking + principal-direction movement + noise filtering.")

    def week11(self):
        week = "Week 11"
        queries = {}
        for fn, d in self.function_dims.items():
            prev_vec = np.array([float(x) for x in self.weekly_queries["Week 10"][fn].split("-")])
            w = self._variance_weights(fn)
            pd = self._principal_direction(fn)
            vec = prev_vec.copy()
            if pd is not None:
                vec += pd * 0.06
            principal_step = np.random.uniform(-0.04, 0.04, d)
            vec += principal_step * w * 1.5
            noise = np.random.uniform(-0.005, 0.005, d)
            vec += noise
            vec = np.clip(vec, 0, 1)
            queries[fn] = fmt(vec)
            self.history_queries[fn].append(vec)
            self.log(f"{week} {fn}: principal_step applied, variance_weights={w}.")
        self.weekly_queries[week] = queries
        self.log(f"{week}: refined PCA-inspired movement with scaled steps and reduced noise.")

    # ============================================================
    # WEEK 12–13: REINFORCEMENT LEARNING (BANDIT, Q-VALUES)
    # ============================================================

    def week12(self):
        week = "Week 12"
        queries = {}
        for fn, d in self.function_dims.items():
            prev_vec = np.array([float(x) for x in self.weekly_queries["Week 11"][fn].split("-")])
            reward = self.weekly_outputs["Week 11"][fn]
            self.bandit.update(fn, reward)
            q_val = self.bandit.arms[fn].q_value
            explore_prob = np.exp(-abs(q_val))
            vec = prev_vec.copy()
            if np.random.rand() < explore_prob:
                vec += np.random.uniform(-0.05, 0.05, d)
                mode = "explore"
            else:
                vec += np.sign(q_val) * np.random.uniform(0.0, 0.03, d)
                mode = "exploit"
            vec = np.clip(vec, 0, 1)
            queries[fn] = fmt(vec)
            self.history_queries[fn].append(vec)
            self.log(f"{week} {fn}: q={q_val:.4f}, mode={mode}, explore_prob={explore_prob:.3f}.")
        self.weekly_queries[week] = queries
        self.log(f"{week}: RL bandit exploration–exploitation balancing + reward-based updates.")

    def week13(self):
        week = "Week 13"
        queries = {}
        for fn, d in self.function_dims.items():
            prev_vec = np.array([float(x) for x in self.weekly_queries["Week 12"][fn].split("-")])
            reward = self.weekly_outputs["Week 12"][fn]
            self.bandit.update(fn, reward)
            q_val = self.bandit.arms[fn].q_value
            exploit_strength = 1 / (1 + np.exp(-q_val))
            vec = prev_vec.copy()
            vec += np.sign(q_val) * np.random.uniform(0.0, 0.04, d) * exploit_strength
            vec += np.random.uniform(-0.02, 0.02, d) * (1 - exploit_strength)
            vec = np.clip(vec, 0, 1)
            queries[fn] = fmt(vec)
            self.history_queries[fn].append(vec)
            self.log(f"{week} {fn}: q={q_val:.4f}, exploit_strength={exploit_strength:.3f}.")
        self.weekly_queries[week] = queries
        self.log(f"{week}: bandit-style policy refinement with Q-value-driven exploitation.")

    # ============================================================
    # RUN PIPELINE
    # ============================================================

    def run_all(self):
        # You call these in order, inserting real outputs after each week.
        self.week1()
        # self.register_week_outputs("Week 1", {...})
        self.week2()
        # self.register_week_outputs("Week 2", {...})
        self.week3()
        # self.register_week_outputs("Week 3", {...})
        self.week4()
        # self.register_week_outputs("Week 4", {...})
        self.week5()
        # self.register_week_outputs("Week 5", {...})
        self.week6()
        # self.register_week_outputs("Week 6", {...})
        self.week7()
        # self.register_week_outputs("Week 7", {...})
        self.week8()
        # self.register_week_outputs("Week 8", {...})
        self.week9()
        # self.register_week_outputs("Week 9", {...})
        self.week10()
        # self.register_week_outputs("Week 10", {...})
        self.week11()
        # self.register_week_outputs("Week 11", {...})
        self.week12()
        # self.register_week_outputs("Week 12", {...})
        self.week13()
        # self.register_week_outputs("Week 13", {...})
        return self.weekly_queries, self.logs


# -----------------------------
#  USAGE EXAMPLE (YOU FILL OUTPUTS)
# -----------------------------

optimiser = BBOOptimiser(FUNCTION_DIMS)

# Example: after each week, you plug in your real outputs:
optimiser.week1()
optimiser.register_week_outputs("Week 1", {
    "Function 1": 4.461320838531215e-14,
    "Function 2": 0.3187883643432596,
    "Function 3": -0.006226762231575552,
    "Function 4": -3.548863675470638,
    "Function 5": 389.50589600542474,
    "Function 6": -1.2431540336861717,
    "Function 7": 1.5457404861343522,
    "Function 8": 9.6925819203656,
})

optimiser.week2()
optimiser.register_week_outputs("Week 2", {
    "Function 1": -1.5292520778611578e-57,
    "Function 2": 0.23680548333708162,
    "Function 3": -0.12355936298558204,
    "Function 4": -4.939285815893619,
    "Function 5": 654.6627396674598,
    "Function 6": -0.731942201771606,
    "Function 7": 1.1950700599316013,
    "Function 8": 9.568684338169,
})

# ...continue plugging Week 3–13 outputs the same way,
# then call optimiser.week3(), week4(), ... week13() in order.

# At the end, you can inspect:
final_queries = optimiser.weekly_queries
debug_logs = optimiser.logs
