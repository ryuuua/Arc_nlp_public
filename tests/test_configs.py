from importlib.resources import files

import pytest
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


@pytest.mark.parametrize("profile,revision", [("paper_v2", "v2"), ("standard_v3", "v3"), ("standard_v4", "v4")])
def test_solver_profiles_select_matching_ids_and_distinct_outputs(profile, revision, monkeypatch) -> None:
    monkeypatch.delenv("AA_OUTPUT_ROOT", raising=False)
    with initialize_config_module(version_base="1.3", config_module="neureps_arc_nlp.configs"):
        config = compose(config_name="config", overrides=[f"solver={profile}", "k=3", "seed=7"])
    assert config.solver.id == config.solver.method_params.b_update == f"aa_projected_gradient_{revision}"
    if revision == "v2":
        assert config.solver.dtype == "float32"
        assert config.output.dir == "outputs/ag_news/deepseek_r1_distill_qwen_14b/seed_7"
    else:
        assert config.solver.backend == "numpy"
        assert config.solver.device == "cpu"
        assert config.solver.dtype == "float64"
        assert f"solver=aa_projected_gradient_{revision}/k=3/seed_7" in config.output.dir


def test_legacy_explicit_solver_id_override_remains_valid() -> None:
    with initialize_config_module(version_base="1.3", config_module="neureps_arc_nlp.configs"):
        config = compose(config_name="config", overrides=["solver.id=aa_projected_gradient_v2"])
    assert config.solver.id == config.solver.method_params.b_update == "aa_projected_gradient_v2"
