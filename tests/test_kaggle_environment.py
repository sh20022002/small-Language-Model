"""
Kaggle environment simulation tests: mock Kaggle paths, GPU detection,
dataset loading, notebook compatibility checks.
"""
import pytest
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import torch

from my_slm.hybrid_tokeniztion import HybridTokenizer


class TestKagglePathSimulation:
    """Simulate Kaggle directory structure (/kaggle/working, /kaggle/input)."""

    def test_kaggle_working_directory_creation(self):
        """Simulate /kaggle/working directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            kaggle_working = os.path.join(tmpdir, "kaggle", "working")
            os.makedirs(kaggle_working, exist_ok=True)

            assert os.path.isdir(kaggle_working)
            assert "kaggle" in kaggle_working

    def test_kaggle_input_directory_creation(self):
        """Simulate /kaggle/input directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            kaggle_input = os.path.join(tmpdir, "kaggle", "input")
            os.makedirs(kaggle_input, exist_ok=True)

            assert os.path.isdir(kaggle_input)

    def test_save_model_to_kaggle_working(self):
        """Test saving model to simulated Kaggle working directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            working_dir = os.path.join(tmpdir, "kaggle", "working")
            os.makedirs(working_dir, exist_ok=True)

            # Create a small model
            model_path = os.path.join(working_dir, "model.pt")
            model = torch.nn.Linear(10, 10)
            torch.save(model.state_dict(), model_path)

            assert os.path.isfile(model_path)
            loaded = torch.load(model_path)
            assert "weight" in loaded

    def test_save_tokenizer_to_kaggle_working(self):
        """Test saving tokenizer to Kaggle working directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            working_dir = os.path.join(tmpdir, "kaggle", "working")
            os.makedirs(working_dir, exist_ok=True)

            # Create and save tokenizer
            tok = HybridTokenizer()
            tok.add_text("test data " * 100)
            tok.freeze_vocab(512)

            tok_path = os.path.join(working_dir, "tok.pkl.gz")
            tok.save(tok_path)

            assert os.path.isfile(tok_path)

            # Load it back
            tok2 = HybridTokenizer.load(tok_path)
            assert tok2.vocab_size == tok.vocab_size

    def test_load_from_kaggle_input(self):
        """Test loading dataset from simulated /kaggle/input."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = os.path.join(tmpdir, "kaggle", "input", "dataset")
            os.makedirs(input_dir, exist_ok=True)

            # Create a dummy dataset file
            dataset_file = os.path.join(input_dir, "data.txt")
            with open(dataset_file, "w") as f:
                f.write("Line 1\nLine 2\nLine 3\n")

            assert os.path.isfile(dataset_file)

            # Read it
            with open(dataset_file, "r") as f:
                lines = f.readlines()

            assert len(lines) == 3


class TestGPUDetection:
    """Test GPU availability detection."""

    def test_torch_cuda_available(self):
        """Check if CUDA is available."""
        cuda_available = torch.cuda.is_available()
        assert isinstance(cuda_available, bool)

    def test_device_selection_cpu_fallback(self):
        """Test selecting CPU device."""
        device = torch.device("cpu")
        assert device.type == "cpu"

    def test_device_selection_cuda_if_available(self):
        """Test selecting CUDA device if available."""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            assert device.type == "cuda"
        else:
            device = torch.device("cpu")
            assert device.type == "cpu"

    def test_model_to_device(self):
        """Test moving model to device."""
        model = torch.nn.Linear(10, 10)

        device = torch.device("cpu")
        model = model.to(device)

        # Check if model is on device
        for param in model.parameters():
            assert param.device.type == device.type

    def test_tensor_to_device(self):
        """Test moving tensor to device."""
        x = torch.randn(2, 10)

        device = torch.device("cpu")
        x = x.to(device)

        assert x.device.type == device.type

    def test_batch_to_device(self):
        """Test moving batch to device."""
        batch = {
            "input_ids": torch.randint(1, 256, (2, 8)),
            "attention_mask": torch.ones(2, 8, dtype=torch.long),
            "labels": torch.randint(1, 256, (2, 8)),
        }

        device = torch.device("cpu")

        # Move all tensors
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                assert v.device.type == device.type


class TestDatasetLoadingSimulation:
    """Simulate loading datasets in Kaggle environment."""

    def test_load_text_dataset_from_file(self):
        """Load text dataset from file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_file = os.path.join(tmpdir, "dataset.txt")

            with open(dataset_file, "w") as f:
                f.write("Hello world\n")
                f.write("Test data\n")
                f.write("More text\n")

            # Read it
            with open(dataset_file, "r") as f:
                lines = f.readlines()

            assert len(lines) == 3

    def test_tokenize_loaded_dataset(self):
        """Tokenize a loaded dataset."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_file = os.path.join(tmpdir, "dataset.txt")

            texts = ["Hello world test", "Another line here", "More data"] * 50
            with open(dataset_file, "w") as f:
                for text in texts:
                    f.write(text + "\n")

            # Create tokenizer
            tok = HybridTokenizer()

            # Add file to tokenizer
            tok.add_file(dataset_file)
            tok.freeze_vocab(512)

            # Tokenize a sample
            ids = tok.encode("Hello world")
            assert len(ids) > 0

    def test_json_dataset_loading(self):
        """Load JSON-formatted dataset."""
        import json

        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_file = os.path.join(tmpdir, "dataset.json")

            data = [
                {"question": "Q1", "answer": "A1"},
                {"question": "Q2", "answer": "A2"},
                {"question": "Q3", "answer": "A3"},
            ]

            with open(dataset_file, "w") as f:
                json.dump(data, f)

            # Load it
            with open(dataset_file, "r") as f:
                loaded = json.load(f)

            assert len(loaded) == 3
            assert loaded[0]["question"] == "Q1"

    def test_csv_dataset_loading(self):
        """Load CSV-formatted dataset."""
        import csv

        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_file = os.path.join(tmpdir, "dataset.csv")

            with open(dataset_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["question", "answer"])
                writer.writerow(["Q1", "A1"])
                writer.writerow(["Q2", "A2"])

            # Load it
            with open(dataset_file, "r") as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            assert len(rows) == 2
            assert rows[0]["question"] == "Q1"


class TestNotebookCompatibility:
    """Test compatibility with Jupyter notebooks."""

    def test_import_from_notebook(self):
        """Test imports work as they would in notebook."""
        try:
            from my_slm.hybrid_tokeniztion import HybridTokenizer
            from my_slm.transformer import Transformer
            from my_slm.train import train_model
            assert True
        except ImportError as e:
            pytest.fail(f"Import failed: {e}")

    def test_model_creation_in_notebook_context(self):
        """Test model creation like in notebook."""
        from my_slm.transformer import Transformer

        model = Transformer(
            vocab_size=256,
            dim=64,
            depth=2,
            heads=4,
            mlp_dim=128,
            window=16,
        )

        assert model is not None
        assert isinstance(model, torch.nn.Module)

    def test_tokenizer_creation_in_notebook_context(self):
        """Test tokenizer creation like in notebook."""
        tok = HybridTokenizer()
        tok.add_text("test data " * 100)
        tok.freeze_vocab(256)

        assert tok.vocab_size > 0

    def test_device_selection_notebook_pattern(self):
        """Test device selection pattern used in notebooks."""
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device_name)

        assert device.type in ("cuda", "cpu")

    def test_print_statements_work(self):
        """Test that print statements work (simple check)."""
        import io
        from contextlib import redirect_stdout

        f = io.StringIO()
        with redirect_stdout(f):
            print("Test output")

        output = f.getvalue()
        assert "Test output" in output

    def test_matplotlib_import_notebook_safe(self):
        """Test matplotlib import is safe for notebook."""
        try:
            import matplotlib.pyplot as plt
            # Don't actually show plots
            plt.ioff()
            assert True
        except Exception as e:
            pytest.skip(f"Matplotlib not available: {e}")

    def test_tqdm_import_notebook_safe(self):
        """Test tqdm import for progress bars."""
        try:
            from tqdm import tqdm
            # Create a simple progress bar
            items = list(tqdm([1, 2, 3], disable=True))
            assert len(items) == 3
        except ImportError:
            pytest.skip("tqdm not available")

    def test_numpy_integration(self):
        """Test numpy integration works."""
        import numpy as np

        arr = np.array([1, 2, 3])
        tensor = torch.from_numpy(arr)

        assert tensor.shape == (3,)

    def test_dict_unpacking_kwargs_pattern(self):
        """Test dictionary unpacking pattern used in notebooks."""
        config = {"vocab_size": 256, "dim": 64, "depth": 2, "heads": 4,
                 "mlp_dim": 128, "window": 16}

        model = torch.nn.Embedding(**config)
        # Should work (Embedding accepts vocab_size, embedding_dim)
        assert model is not None


class TestEnvironmentVariables:
    """Test environment variable handling."""

    def test_device_env_var(self):
        """Test device selection via environment variable."""
        with patch.dict(os.environ, {"DEVICE": "cpu"}):
            device_str = os.environ.get("DEVICE", "cpu")
            device = torch.device(device_str)
            assert device.type == "cpu"

    def test_model_dir_env_var(self):
        """Test model directory via environment variable."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"MODEL_DIR": tmpdir}):
                model_dir = os.environ.get("MODEL_DIR")
                assert os.path.isdir(model_dir)

    def test_data_dir_env_var(self):
        """Test data directory via environment variable."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"DATA_DIR": tmpdir}):
                data_dir = os.environ.get("DATA_DIR")
                assert os.path.isdir(data_dir)


class TestKagglePathIntegration:
    """Integration tests with Kaggle-like path structures."""

    def test_full_kaggle_workflow(self):
        """Simulate full Kaggle notebook workflow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create Kaggle directory structure
            working_dir = os.path.join(tmpdir, "kaggle", "working")
            input_dir = os.path.join(tmpdir, "kaggle", "input", "dataset")
            os.makedirs(working_dir, exist_ok=True)
            os.makedirs(input_dir, exist_ok=True)

            # Create input dataset
            dataset_file = os.path.join(input_dir, "data.txt")
            with open(dataset_file, "w") as f:
                for i in range(100):
                    f.write(f"Sample text {i}\n")

            # Load and tokenize
            tok = HybridTokenizer()
            tok.add_file(dataset_file)
            tok.freeze_vocab(512)

            # Save to working directory
            tok_path = os.path.join(working_dir, "tokenizer.pkl.gz")
            tok.save(tok_path)

            assert os.path.isfile(tok_path)

            # Load from working directory
            tok2 = HybridTokenizer.load(tok_path)
            assert tok2.vocab_size == tok.vocab_size

    def test_nested_dataset_loading(self):
        """Load datasets from nested Kaggle input directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create nested structure: /kaggle/input/datasets/
            base_dir = os.path.join(tmpdir, "kaggle", "input")
            for i in range(3):
                dataset_dir = os.path.join(base_dir, f"dataset_{i}")
                os.makedirs(dataset_dir, exist_ok=True)

                # Create files
                for j in range(2):
                    file_path = os.path.join(dataset_dir, f"data_{j}.txt")
                    with open(file_path, "w") as f:
                        f.write(f"Data from dataset {i} file {j}\n")

            # Verify all files exist
            for i in range(3):
                dataset_dir = os.path.join(base_dir, f"dataset_{i}")
                assert os.path.isdir(dataset_dir)
                files = os.listdir(dataset_dir)
                assert len(files) == 2
