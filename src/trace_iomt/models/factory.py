from .networks import MODELS


def build_model(name, n_features, hierarchy, cfg_model):
    if name not in MODELS:
        raise ValueError(f"Unknown model '{name}'. Choose from {sorted(MODELS)}")
    return MODELS[name](n_features, hierarchy, cfg_model)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
