import hyperopt
from qibo.backends import _check_backend

from boostvqe.models.dbi.double_bracket import *
from boostvqe.models.dbi.utils import *


def gradient_numerical(
    loss_function,
    eo_d_type,
    params: list,
    loss_0,
    s_0,
    mode,
    delta: float = 1e-7,
):
    grad = np.zeros(len(params))
    for i in range(len(params)):
        params_new = deepcopy(params)
        params_new[i] += delta
        eo_d = eo_d_type.load(params_new)
        # find the increment of a very small step
        grad[i] = (loss_function(s_0, eo_d, mode) - loss_0) / delta

    # normalize gradient
    grad = grad / max(abs(grad))
    return grad


from boostvqe.models.dbi.utils_scheduling import adaptive_binary_search


def choose_gd_params(
    gci,
    eo_d_type,
    params,
    loss_0,
    s_0,
    mode,
    s_min=1e-4,
    s_max=2e-2,
    lr_min=1e-4,
    lr_max=1,
    threshold=1e-4,
    max_eval=30,
    please_be_adaptive=True,
    please_be_verbose=False,
):
    evaluated_points = {}
    max_eval = int(np.sqrt(max_eval))
    grad = gradient_numerical(gci.loss, eo_d_type, params, loss_0, s_0, mode=mode)

    def loss_func_lr(lr):
        filtered_entries = {k: v for k, v in evaluated_points.items() if k[0] == lr}
        if len(filtered_entries) > 0:
            return min(filtered_entries.values())
        elif lr < 0:
            return float("inf")
        else:
            params_eval = (1 - grad * lr) * deepcopy(params)
            eo_d = eo_d_type.load(params_eval)
            # given lr find best possible s_min and loss_min
            if please_be_adaptive:
                (
                    best_s,
                    best_loss,
                    evaluated_points_s,
                    exit_criterion,
                ) = adaptive_binary_search(
                    lambda s: gci.loss(s, eo_d, mode) if s > 0 else float("inf"),
                    threshold,
                    s_min,
                    s_max,
                    max_eval,
                )
            else:
                best_s, best_loss, evaluated_points_s = gci.choose_step(
                    d=eo_d, s_min=1e-4, s_max=2e-2, max_eval=max_eval
                )

            for s, l in evaluated_points_s.items():
                if l < 0:
                    evaluated_points[(lr, s)] = l
            if please_be_verbose:
                print(
                    f"For lr = {lr} found optimal s = {best_s} yielding loss = {best_loss} after terminating with {exit_criterion}"
                )
        return best_loss

    best_lr, best_loss, _, exit_criterion = adaptive_binary_search(
        loss_func_lr, threshold, lr_min, lr_max, max_eval
    )
    if please_be_verbose:
        print(
            f"For lr = {lr} found optimal s = {best_s} yielding loss = {best_loss} after terminating with {exit_criterion}"
        )
    best_params = (1 - grad * best_lr) * deepcopy(params)
    eo_d = eo_d_type.load(best_params)
    # find best_s
    for (lr, s), loss in evaluated_points.items():
        if lr == best_lr and loss == best_loss:
            best_s = s
    return eo_d, best_s, best_loss, evaluated_points, best_params, best_lr
