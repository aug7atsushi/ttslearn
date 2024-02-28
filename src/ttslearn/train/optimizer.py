import torch


def get_optimizer(model, optimcfg):
    if optimcfg.name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=optimcfg.params.lr,
            weight_decay=optimcfg.params.weight_decay,
        )
        return optimizer
    elif optimcfg.name == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=optimcfg.params.lr,
            weight_decay=optimcfg.params.weight_decay,
        )
        return optimizer
    elif optimcfg.name == "rmsprop":
        optimizer = torch.optim.RMSprop(
            model.parameters(),
            lr=optimcfg.params.lr,
            weight_decay=optimcfg.params.weight_decay,
        )
        return optimizer
    else:
        raise ValueError(f"Not support optimizer {optimcfg.name}")
