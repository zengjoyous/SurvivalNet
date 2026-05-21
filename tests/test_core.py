import pandas as pd

from survivalnet import (
    CoxModel,
    c_index,
    fit_km,
    logrank_p_value,
    plot_grouped_km_with_pvalue,
    run_logrank_test,
    split_risk_group,
)


def make_data():
    return pd.DataFrame(
        {
            "time": [5, 6, 6, 2, 4, 3],
            "event": [1, 0, 1, 1, 0, 1],
            "group": ["A", "A", "B", "B", "A", "B"],
            "x1": [0.2, 0.1, 0.3, 0.8, 0.4, 0.6],
        }
    )


def test_fit_km():
    data = make_data()
    kmf = fit_km(data, "time", "event")
    assert hasattr(kmf, "survival_function_")


def test_logrank():
    data = make_data()
    result = run_logrank_test(data, "group", "time", "event", "A", "B")
    assert hasattr(result, "p_value")


def test_c_index():
    data = make_data()
    value = c_index(data["event"], data["time"], [0.1, 0.3, 0.2, 0.9, 0.4, 0.8])
    assert 0.0 <= value <= 1.0


def test_split_risk_group():
    groups = split_risk_group([0.1, 0.5, 0.9])
    assert list(groups) == ["low", "high", "high"]


def test_cox_model_fit():
    data = make_data()
    model = CoxModel().fit(data[["time", "event", "x1"]], "time", "event")
    assert not model.summary.empty
    assert not model.hazard_ratios.empty


def test_logrank_p_value():
    data = make_data()
    p_value = logrank_p_value(data, "group", "time", "event", "A", "B")
    assert 0.0 <= p_value <= 1.0


def test_plot_grouped_km_with_pvalue():
    import matplotlib

    matplotlib.use("Agg")

    data = make_data()
    ax = plot_grouped_km_with_pvalue(data, "time", "event", "group", "A", "B")
    assert ax is not None
