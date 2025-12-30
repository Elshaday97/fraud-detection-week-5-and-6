def test_case_selector_outputs():
    y_true = [1, 0, 1, 0]
    y_pred = [1, 1, 0, 0]

    from src.explainability.case_selector import select_cases

    cases = select_cases(y_true, y_pred)

    assert "tp" in cases
    assert "fp" in cases
    assert "fn" in cases
