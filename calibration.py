import numpy as np
from scipy.optimize import minimize
from scipy.stats import linregress

from config import (
    ALPHA,
    BETA,
    MU,
    K,
    PSI,
    S0,
    R_MIN,
    R_MAX,
    SMOOTHNESS_WEIGHT,
    MIN_OBSERVATIONS,
)

from model import simulate_series


def prepare_observations(
    A_data,
    population_scale,
):
    """
    Переводит абсолютное число случаев
    в долю от масштаба населения модели.
    """

    A_data = np.asarray(
        A_data,
        dtype=float,
    )

    if population_scale <= 0:
        raise ValueError(
            "population_scale должен быть > 0"
        )

    if np.any(A_data < 0):
        raise ValueError(
            "В A_data обнаружены отрицательные значения"
        )

    return A_data / population_scale


def initial_conditions(
    A_data_fraction,
    s0=S0,
):
    """
    Начальное состояние модели.

    Здесь Y0 оценивается приближенно по первому
    наблюдению. Это стартовое приближение, а не
    точная формула из статьи.
    """

    if len(A_data_fraction) == 0:
        raise ValueError("Нет наблюдений")

    # Приближение для первого года:
    # A ≈ R * alpha * X * Y.
    # Берём умеренное стартовое R=1.
    y0 = (
        A_data_fraction[0]
        / max(ALPHA * s0, 1e-12)
    )

    y0 = float(
        np.clip(y0, 1e-8, 0.1)
    )

    z0 = max(
        0.0,
        1.0 - s0 - y0,
    )

    return np.array(
        [s0, y0, z0],
        dtype=float,
    )


def objective(
    R_values,
    A_data_fraction,
    initial_state,
    smoothness_weight=SMOOTHNESS_WEIGHT,
):
    _, A_model = simulate_series(
        R_values=R_values,
        initial_state=initial_state,
    )

    target = A_data_fraction[:len(A_model)]

    # Масштабируем ошибку, чтобы она не была
    # слишком маленькой для оптимизатора.
    scale = max(
        np.mean(np.abs(target)),
        1e-8,
    )

    residuals = (
        A_model - target
    ) / scale

    data_loss = np.sum(
        residuals ** 2
    )

    smoothness_loss = (
        np.sum(np.diff(R_values) ** 2)
        if len(R_values) > 1
        else 0.0
    )

    return (
        data_loss
        + smoothness_weight * smoothness_loss
    )


def fit_region(
    A_data,
    population_scale,
    years=None,
):
    """
    Калибровка модели для одного региона.

    A_data:
        абсолютное число новых случаев по годам

    population_scale:
        численность населения / масштаб модели

    years:
        годы наблюдений
    """

    A_data = np.asarray(
        A_data,
        dtype=float,
    )

    if len(A_data) < MIN_OBSERVATIONS:
        raise ValueError(
            f"Нужно не менее {MIN_OBSERVATIONS} наблюдений"
        )

    A_fraction = prepare_observations(
        A_data,
        population_scale,
    )

    initial_state = initial_conditions(
        A_fraction
    )

    # Для n годовых наблюдений моделируем n-1
    # годовых интервалов.
    n_intervals = len(A_data) - 1

    R_initial = np.ones(
        n_intervals,
        dtype=float,
    )

    bounds = [
        (R_MIN, R_MAX)
        for _ in range(n_intervals)
    ]

    result = minimize(
        fun=objective,
        x0=R_initial,
        args=(
            A_fraction[:-1],
            initial_state,
            SMOOTHNESS_WEIGHT,
        ),
        method="L-BFGS-B",
        bounds=bounds,
        options={
            "maxiter": 500,
            "ftol": 1e-10,
        },
    )

    if not result.success:
        print(
            "Предупреждение оптимизации:",
            result.message,
        )

    R_opt = result.x

    states, A_model_fraction = simulate_series(
        R_values=R_opt,
        initial_state=initial_state,
    )

    # Возвращаем модельные случаи в абсолютной шкале
    A_model_absolute = (
        A_model_fraction * population_scale
    )

    output_years = (
        years[:-1]
        if years is not None
        else np.arange(len(A_model_absolute))
    )
    print("A_data fraction:", A_fraction)
    print("A_model fraction:", A_model_fraction)
    print("R:", R_opt)
    print("Loss:", result.fun)
    print("Iterations:", result.nit)
    print("Function evaluations:", result.nfev)
    return {
        "success": bool(result.success),
        "message": str(result.message),
        "loss": float(result.fun),
        "R_values": R_opt,
        "A_data": A_data[:-1],
        "A_model": A_model_absolute,
        "years": np.asarray(output_years),
        "states": states,
        "initial_state": initial_state,
    }


def fit_nonlinearity(A_values, R_values):
    """Оценивает R(t) = R_0 * A(t)^p через логарифмическую регрессию

    Возвращает словарь с параметрами для всего периода, а также для фаз роста и
    спада.
    """
    A = np.asarray(A_values, dtype=float)
    R = np.asarray(R_values, dtype=float)

    def _calc_regression(a_subset, r_subset):
        if len(a_subset) < 2:
            return {"R0": np.nan, "p": np.nan, "r_squared": np.nan}
        # ln(R) = ln(R0) + p * ln(A)
        slope, intercept, r_value, _, _ = linregress(
            np.log(a_subset), np.log(r_subset)
        )
        return {
            "R0": float(np.exp(intercept)),
            "p": float(slope),
            "r_squared": float(r_value**2),
        }

    # 1. По всем точкам
    res_all = _calc_regression(A, R)

    # 2. Фазы роста и спада (по динамике A)
    diff_A = np.diff(A)
    # для первого элемента берем знак diff_A[0], для остальных - предыдущее изменение
    growth_mask = np.zeros(len(A), dtype=bool)
    growth_mask[1:] = diff_A > 0
    growth_mask[0] = (
        diff_A[0] > 0 if len(diff_A) > 0 else False
    )  # приближение для первой точки

    res_up = _calc_regression(A[growth_mask], R[growth_mask])
    res_down = _calc_regression(A[~growth_mask], R[~growth_mask])

    return {
        "all": res_all,
        "up": res_up,
        "down": res_down,
    }