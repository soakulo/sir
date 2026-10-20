import numpy as np
from scipy.integrate import solve_ivp

from config import (
    ALPHA,
    BETA,
    MU,
    K,
    PSI,
    STEPS_PER_YEAR,
)


def sir_rhs(t, state, R, alpha, beta, mu, k, psi):
    """
    Правая часть SIR-системы.

    state = [X, Y, Z]

    X — восприимчивые
    Y — инфицированные
    Z — выбывшие из инфекционного процесса

    R — контактный параметр на данном временном интервале
    """

    X, Y, Z = state

    infection = R * alpha * X * Y

    dX = (
        -infection
        + mu
        - mu * X
        + (k + psi) * Z
    )

    dY = (
        infection
        - beta * Y
        - mu * Y
    )

    dZ = (
        beta * Y
        - mu * Z
        - (k + psi) * Z
    )

    return [dX, dY, dZ]


def calculate_A(state, R, alpha=ALPHA):
    """
    Модельная заболеваемость:
    A(t) = R(t) * alpha * X(t) * Y(t)
    """

    X, Y, Z = state

    return R * alpha * X * Y


def simulate_year(
    initial_state,
    R,
    alpha=ALPHA,
    beta=BETA,
    mu=MU,
    k=K,
    psi=PSI,
    steps_per_year=STEPS_PER_YEAR,
):
    """
    Интегрирует систему на интервале длиной 1 год.

    Возвращает:
      state_next — состояние в конце года
      A_integral — интеграл модельной заболеваемости за год
    """

    t_eval = np.linspace(
        0.0,
        1.0,
        steps_per_year + 1,
    )

    sol = solve_ivp(
        fun=lambda t, y: sir_rhs(
            t,
            y,
            R,
            alpha,
            beta,
            mu,
            k,
            psi,
        ),
        t_span=(0.0, 1.0),
        y0=np.asarray(initial_state, dtype=float),
        t_eval=t_eval,
        method="RK45",
        rtol=1e-6,
        atol=1e-9,
    )

    if not sol.success:
        raise RuntimeError(
            f"Ошибка интегрирования: {sol.message}"
        )

    states = sol.y.T

    A_values = np.array([
        calculate_A(state, R, alpha)
        for state in states
    ])

    # Численное интегрирование A(t) за год
    A_integral = np.trapz(
        A_values,
        sol.t,
    )

    state_next = states[-1]

    return state_next, A_integral


def simulate_series(
    R_values,
    initial_state,
    alpha=ALPHA,
    beta=BETA,
    mu=MU,
    k=K,
    psi=PSI,
):
    """
    Симуляция всей последовательности лет.

    R_values[i] действует на интервале
    от года i до года i+1.
    """

    state = np.asarray(
        initial_state,
        dtype=float,
    )

    states = [state.copy()]
    A_model = []

    for R in R_values:
        state, A_year = simulate_year(
            initial_state=state,
            R=R,
            alpha=alpha,
            beta=beta,
            mu=mu,
            k=k,
            psi=psi,
        )

        states.append(state.copy())
        A_model.append(A_year)

    return (
        np.asarray(states),
        np.asarray(A_model),
    )