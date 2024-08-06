import torch
import torch.nn as nn
import torch.nn.functional as F

# import all functions in math
from d2ac.utils.torchmath import (
    categorical_td_learning,
    soft_ce,
    two_hot,
    two_hot_inv,
)


def test_soft_ce():
    # Example inputs
    logits = torch.tensor([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])
    target = torch.tensor([0, 2]).unsqueeze(-1)
    cfg = type("test", (object,), {"num_bins": 3, "vmin": 0, "vmax": 2, "bin_size": 1})

    # Calculate the expected output
    pred = F.log_softmax(logits, dim=-1)
    soft_target = two_hot(target, cfg)
    expected_output = -(soft_target * pred).sum(-1, keepdim=True)

    # Calculate the actual output from the soft_ce function
    actual_output = soft_ce(logits, target, cfg)

    # Check if the actual output is close to the expected output
    assert torch.allclose(
        actual_output, expected_output
    ), f"Test failed: {actual_output} != {expected_output}"


def test_two_hot():
    # Example input and configuration
    input_values = torch.tensor([[0.0], [1.0], [2.0]])
    cfg = type("test", (object,), {"num_bins": 3, "vmin": 0, "vmax": 2, "bin_size": 1})

    # Actual output from two_hot function
    actual_output = two_hot(input_values, cfg, apply_symlog=False)

    # Expected output - adjust these based on your function's specific behavior
    expected_output = torch.tensor(
        [
            [1.0, 0.0, 0.0],  # Exact bin
            [0.0, 1.0, 0.0],  # Midway between two bins
            [0.0, 0.0, 1.0],  # Exact bin
        ]
    )

    # Check if the actual output is close to the expected output
    assert torch.allclose(
        actual_output, expected_output, atol=1e-5
    ), f"Test failed: {actual_output} != {expected_output}"


def test_two_hot_inv():
    # Example input: Soft two-hot encoded vectors (probabilities)
    input_vectors = torch.tensor(
        [
            [0.8, 0.2, 0.0],  # Represents a scalar value close to 0
            [0.0, 0.5, 0.5],  # Represents a scalar value between 1 and 2
            [0.0, 0.0, 1.0],  # Represents a scalar value close to 2
        ]
    )

    # Configuration
    cfg = type("test", (object,), {"num_bins": 3, "vmin": 0, "vmax": 2, "bin_size": 1})

    # Apply the two_hot_inv function
    actual_output = two_hot_inv(torch.log(input_vectors), cfg, apply_symexp=False)

    # Expected output: Scalars corresponding to the input vectors
    expected_output = torch.tensor([[0.2], [1.5], [2.0]])

    # Check if the actual output matches the expected output with a tolerance
    tolerance = 1e-5
    assert torch.allclose(
        actual_output, expected_output, atol=tolerance
    ), f"Test failed: {actual_output} != {expected_output}"


def test_categorical_td_learning():
    # Create a mock environment for testing
    batch_size = 5
    num_atoms = 10
    discount_t = 0.9

    # Generate random atoms for value distribution
    v_atoms = torch.linspace(-1, 1, steps=num_atoms)

    # Random rewards for a batch of examples
    r_t = torch.rand(batch_size, 1) * 2 - 1  # rewards between -1 and 1

    # Random logits for a batch of examples
    v_logits_t = torch.randn(batch_size, num_atoms)

    # Call the function
    td_target = categorical_td_learning(v_atoms, r_t, v_logits_t, discount_t)

    # Check if the shape of the output matches the expected shape
    assert td_target.shape == v_logits_t.shape, "Output shape mismatch"

    # Check if the output is a valid probability distribution
    assert torch.allclose(
        torch.sum(td_target, dim=-1), torch.tensor(1.0)
    ), "Sum of probabilities should be 1"


if __name__ == "__main__":
    test_soft_ce()
    test_two_hot()
    test_two_hot_inv()
    test_categorical_td_learning()
    print("All tests passed!")
