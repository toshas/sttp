def spectral_penalty_d_optimal(svs, already_abs=False, already_normalized=False, eps=1e-7):
    """
    D-Optimal Regularizer as described in https://openreview.net/pdf?id=rJNH6sAqY7
    """
    svs = list(svs.values())
    if not already_abs:
        svs = [s.abs() for s in svs]
    if not already_normalized:
        svs = [s / s.max().clamp_min(eps) for s in svs]
    num_svs_total = sum(s.numel() for s in svs)
    reg = sum(-s.clamp_min(eps).log().sum() for s in svs)
    reg = reg / num_svs_total
    return reg


def spectral_penalty_divergence(
        svs, param_a=0.1, already_abs=False, already_normalized=False, already_sorted_descending=False, eps=1e-7
):
    """
    Divergence Regularizer as described in https://openreview.net/pdf?id=rJNH6sAqY7
    """
    svs = [s for s in svs.values() if s.numel() > 1]
    if not already_abs:
        svs = [s.abs() for s in svs]
    if not already_normalized:
        svs = [s / s.max().clamp_min(eps) for s in svs]
    if not already_sorted_descending:
        svs = [s.sort(descending=True).values for s in svs]
    var = param_a ** 2
    num_layers = len(svs)
    reg = [((1 - s[1:]) ** 2 / var - (s[:-1] - s[1:]).clamp_min(eps).log()).mean() for s in svs]
    reg = sum(reg) / num_layers
    return reg
