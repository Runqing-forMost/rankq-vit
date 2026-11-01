def prior_loss_schedule(iter, init=1., end=0.1, total=500):
    k = (init - end) / (-1 * total)
    return k * iter + init