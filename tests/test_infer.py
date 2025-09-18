from churn_guard.infer import Predictor

def test_predict():
    p = Predictor()
    out = p.predict_one({
        "tenure_months": 10,
        "monthly_charges": 65.0,
        "total_charges": 650.0,
        "num_support_tickets": 1,
        "contract_type": "month_to_month",
        "payment_method": "credit_card",
        "has_addon_streaming": True,
    })
    assert set(out.keys()) == {"churn_probability", "churn_label"}
    assert 0.0 <= out["churn_probability"] <= 1.0
    assert out["churn_label"] in (0, 1)
