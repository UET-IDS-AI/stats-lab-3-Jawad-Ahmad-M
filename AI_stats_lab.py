import numpy as np
import math


# =========================================================
# QUESTION 1 – Card Experiment
# =========================================================

def card_experiment():
    np.random.seed(42)

    # STEP 1
    deck = np.array([1]*4 + [0]*48)

    # STEP 2 – Analytical
    P_A = 4 / 52
    P_B = 4 / 52
    P_B_given_A = 3 / 51
    P_AB = P_A * P_B_given_A

    # STEP 4 – Simulation (without replacement)
    trials = 200_000
    count_A = 0
    count_A_and_B = 0

    for _ in range(trials):
        draw = np.random.choice(deck, size=2, replace=False)
        first, second = draw

        if first == 1:
            count_A += 1
            if second == 1:
                count_A_and_B += 1

    empirical_P_A = count_A / trials
    empirical_P_B_given_A = count_A_and_B / count_A

    # STEP 5 – Absolute Error
    absolute_error = abs(empirical_P_B_given_A - P_B_given_A)

    return (
        P_A,
        P_B,
        P_B_given_A,
        P_AB,
        empirical_P_A,
        empirical_P_B_given_A,
        absolute_error,
    )


# =========================================================
# QUESTION 2 – Bernoulli
# =========================================================

def bernoulli_lightbulb(p=0.05):
    np.random.seed(42)

    # STEP 2 – Theoretical
    theoretical_P_X_1 = p
    theoretical_P_X_0 = 1 - p

    # STEP 3 – Simulation
    trials = 100_000
    samples = np.random.rand(trials) < p

    empirical_P_X_1 = np.mean(samples)

    # STEP 5 – Absolute Error
    absolute_error = abs(empirical_P_X_1 - theoretical_P_X_1)

    return (
        theoretical_P_X_1,
        theoretical_P_X_0,
        empirical_P_X_1,
        absolute_error,
    )


# =========================================================
# QUESTION 3 – Binomial
# =========================================================

def binomial_bulbs(n=10, p=0.05):
    np.random.seed(42)

    # STEP 2 – Theoretical
    theoretical_P_0 = (1 - p) ** n
    theoretical_P_2 = math.comb(n, 2) * (p**2) * ((1 - p) ** (n - 2))
    theoretical_P_ge_1 = 1 - theoretical_P_0

    # STEP 3 – Simulation
    trials = 100_000
    samples = np.random.binomial(n, p, size=trials)

    empirical_P_ge_1 = np.mean(samples >= 1)

    # STEP 5 – Absolute Error
    absolute_error = abs(empirical_P_ge_1 - theoretical_P_ge_1)

    return (
        theoretical_P_0,
        theoretical_P_2,
        theoretical_P_ge_1,
        empirical_P_ge_1,
        absolute_error,
    )


# =========================================================
# QUESTION 4 – Geometric
# =========================================================

def geometric_die():
    np.random.seed(42)

    p = 1 / 6

    # STEP 3 – Theoretical
    theoretical_P_1 = p
    theoretical_P_3 = ((1 - p) ** 2) * p
    theoretical_P_gt_4 = (1 - p) ** 4

    # STEP 4 – Simulation
    trials = 200_000
    samples = np.random.geometric(p, size=trials)

    empirical_P_gt_4 = np.mean(samples > 4)

    # STEP 6 – Absolute Error
    absolute_error = abs(empirical_P_gt_4 - theoretical_P_gt_4)

    return (
        theoretical_P_1,
        theoretical_P_3,
        theoretical_P_gt_4,
        empirical_P_gt_4,
        absolute_error,
    )


# =========================================================
# QUESTION 5 – Poisson
# =========================================================

def poisson_customers(lam=12):
    np.random.seed(42)

    # STEP 2 – Theoretical
    theoretical_P_0 = math.exp(-lam)
    theoretical_P_15 = (math.exp(-lam) * lam**15) / math.factorial(15)

    theoretical_P_ge_18 = 1 - sum(
        (math.exp(-lam) * lam**k) / math.factorial(k)
        for k in range(18)
    )

    # STEP 3 – Simulation
    trials = 100_000
    samples = np.random.poisson(lam, size=trials)

    empirical_P_ge_18 = np.mean(samples >= 18)

    # STEP 5 – Absolute Error
    absolute_error = abs(empirical_P_ge_18 - theoretical_P_ge_18)

    return (
        theoretical_P_0,
        theoretical_P_15,
        theoretical_P_ge_18,
        empirical_P_ge_18,
        absolute_error,
    )
