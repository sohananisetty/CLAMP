from torch.optim import Adam, AdamW


def separate_weight_decayable_params(params):
    wd_params, no_wd_params = [], []
    for param in params:
        param_list = no_wd_params if param.ndim < 2 else wd_params
        param_list.append(param)
    return wd_params, no_wd_params


def get_optimizer(
    named_params,
    freeze_modules=[],
    lr=1e-4,
    wd=1e-2,
    betas=(0.9, 0.99),
    eps=1e-8,
    # filter_by_requires_grad=False,
    # group_wd_params=True,
):
    # has_wd = wd > 0

    exclude = (
        lambda n, p: p.ndim < 2
        or "bn" in n
        or "ln" in n
        or "bias" in n
        or "logit_scale" in n
    )
    include = lambda n, p: not exclude(n, p)

    named_parameters = list(named_params)

    # freeze text encoder

    # if args.freeze_text:
    print("Freeze Text and Audio!!!!")
    for n, p in named_parameters:
        if n.split(".")[0] in freeze_modules:
            p.requires_grad = False

    gain_or_bias_params = [
        p for n, p in named_parameters if exclude(n, p) and p.requires_grad
    ]
    rest_params = [p for n, p in named_parameters if include(n, p) and p.requires_grad]

    params = [
        {"params": gain_or_bias_params, "weight_decay": 0.0},
        {"params": rest_params, "weight_decay": wd},
    ]

    # if filter_by_requires_grad:
    #     params = list(filter(lambda t: t.requires_grad, params))

    # if group_wd_params and has_wd:
    #     wd_params, no_wd_params = separate_weight_decayable_params(params)

    #     params = [
    #         {"params": wd_params},
    #         {"params": no_wd_params, "weight_decay": 0},
    #     ]
    # if not has_wd:
    #     return Adam(params, lr=lr, betas=betas, eps=eps)

    return AdamW(params, lr=lr, betas=betas, eps=eps)
