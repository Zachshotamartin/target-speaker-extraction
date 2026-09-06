import torch

from tse.data import SpeechCorpus, build_cases
from tse.engine import load_model, train


def test_resume_matches_uninterrupted_training(corpus_files, tiny_config, tmp_path):
    root, manifest = corpus_files
    development = SpeechCorpus(root, manifest, "dev", 0.25, 0.25)
    dev_cases = tmp_path / "dev.json"
    build_cases(development, 2, 800, dev_cases)
    full = tmp_path / "full"
    resumed = tmp_path / "resumed"
    train(tiny_config, root, manifest, dev_cases, full, "cpu")
    first = tiny_config.model_copy(deep=True)
    first.training.max_optimizer_updates = 1
    train(first, root, manifest, dev_cases, resumed, "cpu")
    train(tiny_config, root, manifest, dev_cases, resumed, "cpu", resume=True)
    a, _ = load_model(full / "latest.pt")
    b, _ = load_model(resumed / "latest.pt")
    for left, right in zip(a.parameters(), b.parameters(), strict=True):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
