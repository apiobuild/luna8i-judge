from src.providers.managed_model_registry import estimate_cost, get_model_providers


def test_gemini_flash_cost():
    cost = estimate_cost("gemini/gemini-2.5-flash", input_tokens=1_000_000, output_tokens=1_000_000)
    assert cost == 0.30 + 2.50


def test_gemini_pro_cost():
    cost = estimate_cost("gemini/gemini-2.5-pro", input_tokens=1_000_000, output_tokens=1_000_000)
    assert cost == 1.25 + 10.00


def test_unknown_model_raises():
    try:
        estimate_cost("unknown/model", input_tokens=100, output_tokens=100)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "unknown/model" in str(e)


def test_zero_tokens():
    cost = estimate_cost("gemini/gemini-2.5-flash", input_tokens=0, output_tokens=0)
    assert cost == 0.0


def test_get_model_providers_returns_all():
    providers = get_model_providers()
    ids = {p.id for p in providers}
    assert "gemini/gemini-2.5-flash" in ids
    assert "gemini/gemini-2.5-pro" in ids
    assert "openai/gpt-5.4" in ids
    assert "anthropic/claude-sonnet-4-6" in ids
    assert "deepseek/deepseek-v4-flash" in ids
    assert "xai/grok-3" in ids
    assert "xai/grok-3-mini" in ids
