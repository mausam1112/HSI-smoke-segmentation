"""
Central model registry.

All models are registered here.  The config key ``model.name`` selects
which one is instantiated at training time.

Adding a new model
──────────────────
1. Implement it in ``src/models/your_model.py``.
2. Import it below and add an entry to ``MODEL_REGISTRY``.
3. Set ``model.name: ModelName`` in the experiment YAML — done.

Usage
-----
    from src.models.registry import build_model
    model = build_model(cfg)
"""

from __future__ import annotations

from src.models.unet import JustoUNetSimple
from src.models.mop import MixtureOfPrototypes

# Keys are the values used in config YAML → model.name
MODEL_REGISTRY: dict[str, type] = {
    "JustoUNetSimple": JustoUNetSimple,
    "MixtureOfPrototypes": MixtureOfPrototypes,
    # "NewModel": NewModel,
}


def build_model(cfg) -> "torch.nn.Module":
    """
    Instantiate the model described by *cfg*.

    The ``cfg.model`` block is forwarded as kwargs, minus the ``name`` key.
    This means every key in the ``model:`` section of the YAML maps 1-to-1
    to a constructor argument — no glue code needed per model.

    Parameters
    ----------
    cfg : DotDict
        Full merged experiment config.

    Returns
    -------
    torch.nn.Module
        Un-trained model instance.

    Raises
    ------
    ValueError
        If ``cfg.model.name`` is not registered.
    """
    mcfg = dict(cfg.model)           # shallow copy so we can pop safely
    name = mcfg.pop("name")

    if name not in MODEL_REGISTRY:
        available = ", ".join(MODEL_REGISTRY.keys())
        raise ValueError(
            f"Unknown model '{name}'. "
            f"Available models: {available}\n"
            f"Register new models in src/models/registry.py."
        )

    model_cls = MODEL_REGISTRY[name]
    return model_cls(**mcfg)
