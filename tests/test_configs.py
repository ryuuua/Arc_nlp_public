from importlib.resources import files

from hydra import compose, initialize_config_module
from omegaconf import OmegaConf

DATASETS = {
    "ag_news",
    "dair-ai",
    "go_emotions_6labels",
    "twenty_newsgroups",
}
MODELS = {
    "deepseek_r1_distill_qwen_14b",
    "embeddinggemma_qat",
    "gemma2_27b_it",
    "gemma3_27b_it",
    "llama_embed_nemotron_8b",
    "qwen3_embedding",
    "roberta",
    "sentence_bert",
}


def test_condition_configs_are_exactly_the_4_by_8_grid() -> None:
    root = files("neureps_arc_nlp.configs.condition")
    configs = sorted(item for item in root.iterdir() if item.name.endswith(".yaml"))
    assert len(configs) == 32
    observed = set()
    for config in configs:
        payload = OmegaConf.load(config)
        observed.add((str(payload.dataset), str(payload.model)))
    assert observed == {(dataset, model) for dataset in DATASETS for model in MODELS}


def test_hydra_composes_a_condition_without_repo_relative_config_paths() -> None:
    with initialize_config_module(version_base="1.3", config_module="neureps_arc_nlp.configs"):
        config = compose(
            config_name="config",
            overrides=["condition=twenty_newsgroups__roberta", "seed=7"],
        )
    assert config.condition.dataset == "twenty_newsgroups"
    assert config.condition.model == "roberta"
    assert config.seed == 7
    assert str(config.input.path).endswith("twenty_newsgroups__roberta.npz")
