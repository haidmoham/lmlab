import torch

from experiments.compare_position_encodings import BigramLogits, initialize_models
from src.language_models import BigramLanguageModel


def test_position_ablation_matches_non_position_weights_with_independent_parameters():
    models = initialize_models(vocab_size=16, seed=42)
    reference = models["transformer"].state_dict()
    parameter_ids = []
    for model in models.values():
        for name, value in model.state_dict().items():
            if not name.startswith("position_embedding_table."):
                torch.testing.assert_close(value, reference[name], rtol=0, atol=0)
        parameter_ids.extend(id(parameter) for parameter in model.parameters())
    assert len(parameter_ids) == len(set(parameter_ids)), (
        "treatments must own independent parameters"
    )
    no_positions = models["transformer"].position_embedding_table
    assert list(no_positions.parameters()) == []
    torch.testing.assert_close(no_positions(torch.arange(32)), torch.zeros(32, 32))
    no_encoding_count = sum(p.numel() for p in models["transformer"].parameters())
    sinusoidal_count = sum(p.numel() for p in models["transformer_sinusoidal"].parameters())
    assert no_encoding_count == sinusoidal_count


def test_bigram_adapter_preserves_predictions_and_only_uses_current_token():
    reference = BigramLanguageModel(vocab_size=16, n_embd=32)
    adapter = BigramLogits(vocab_size=16, n_embd=32)
    adapter.load_state_dict(reference.state_dict())
    token_ids = torch.arange(8).unsqueeze(0)
    expected_logits, _ = reference(token_ids)
    torch.testing.assert_close(adapter(token_ids), expected_logits)
    changed_history = token_ids.clone()
    changed_history[:, :-1] = 15
    torch.testing.assert_close(adapter(token_ids)[:, -1], adapter(changed_history)[:, -1])
