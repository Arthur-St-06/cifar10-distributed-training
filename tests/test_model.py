import torch
from model import SimpleModel

def test_model_forward():
    model = SimpleModel()
    dummy_input = torch.randn(4, 3, 32, 32)
    output = model(dummy_input)
    assert output.shape == (4, 10), "Output shape should be (batch, num_classes)"
