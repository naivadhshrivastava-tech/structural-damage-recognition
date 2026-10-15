"""
Unit tests for data preparation, transforms, and prediction logic
in the structural-damage-recognition pipeline.
"""

import tempfile
from pathlib import Path
import numpy as np
import pytest
import torch
from PIL import Image

from src.train import NpyImageDataset, build_model
from src.predict import CLASS_NAMES, predict


def test_npy_image_dataset_shapes_and_types():
    """Verify NpyImageDataset outputs correct tensor shape and label type."""
    num_samples = 4
    X_dummy = np.random.randint(0, 256, size=(num_samples, 224, 224, 3), dtype=np.uint8)
    y_dummy = np.array([0, 1, 0, 1], dtype=np.int64)
    indices = [0, 1, 2, 3]

    dataset = NpyImageDataset(X_dummy, y_dummy, indices, train=False, img_size=224)

    assert len(dataset) == num_samples
    img_tensor, label = dataset[0]

    assert isinstance(img_tensor, torch.Tensor)
    assert img_tensor.shape == (3, 224, 224)
    assert isinstance(label, int)
    assert label in (0, 1)


def test_build_model_supported_architectures():
    """Verify build_model builds valid models with 2-class classification heads."""
    supported_models = ["resnet50", "efficientnet_b0", "mobilenet_v2"]

    for name in supported_models:
        model, head = build_model(name)
        assert model is not None
        assert head is not None

        dummy_input = torch.randn(2, 3, 224, 224)
        output = model(dummy_input)
        assert output.shape == (2, 2)


def test_predict_pipeline_with_mock_model():
    """Test prediction helper function on an image."""
    device = torch.device("cpu")
    model, _ = build_model("efficientnet_b0")
    model.eval()

    with tempfile.TemporaryDirectory() as tmp_dir:
        img_path = Path(tmp_dir) / "test_sample.png"
        img = Image.new("RGB", (224, 224), color=(128, 128, 128))
        img.save(img_path)

        pred_label, confidence = predict(model, img_path, device)

        assert pred_label in CLASS_NAMES.values()
        assert 0.0 <= confidence <= 1.0
