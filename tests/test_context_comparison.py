import torch

from experiments.compare_context_lengths import (
    CONTEXT_LENGTHS,
    evaluation_batches,
    initialize_models,
    training_batch,
)


def test_training_targets_match_across_contexts():
    stream = torch.arange(600)
    expected_targets = stream[11:267]
    for context in CONTEXT_LENGTHS:
        inputs, targets = training_batch(stream, start=10, context=context)
        assert inputs.shape == (256 // context, context)
        torch.testing.assert_close(targets.flatten(), expected_targets)
        torch.testing.assert_close(inputs.flatten() + 1, targets.flatten())


def test_evaluation_uses_same_targets_without_future_tokens():
    stream = torch.arange(600)
    target_positions = [256, 300, 500]
    for context in CONTEXT_LENGTHS:
        batches = evaluation_batches(stream, target_positions, context, batch_size=2)
        targets = torch.cat([batch_targets for _, batch_targets in batches])
        inputs = torch.cat([batch_inputs for batch_inputs, _ in batches])
        torch.testing.assert_close(targets, torch.tensor(target_positions))
        torch.testing.assert_close(inputs[:, -1], targets - 1)
        torch.testing.assert_close(inputs[:, 0], targets - context)


def test_initialization_shares_values_but_not_parameter_objects():
    models = initialize_models(vocab_size=32, seed=42)
    reference_weights = models[256].state_dict()
    all_parameter_ids = []
    for context, model in models.items():
        for name, weights in model.state_dict().items():
            expected_weights = reference_weights[name]
            if name == "position_embedding_table.weight":
                expected_weights = expected_weights[:context]
            torch.testing.assert_close(weights, expected_weights, rtol=0, atol=0)
        all_parameter_ids.extend(id(parameter) for parameter in model.parameters())
    assert len(set(all_parameter_ids)) == len(all_parameter_ids)
