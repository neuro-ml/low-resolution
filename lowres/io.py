from os.path import join as jp


from dpipe.io import load_pred


def load_pred_from_exp(_id, exp_path, n_exp=5):
    """Finds and loads the ``_id`` prediction in ``n_exp`` validation series of ``exp_path`` experiment."""
    pred = None

    for n in range(n_exp):
        try:
            pred = load_pred(_id, jp(exp_path, f'experiment_{n}/test_predictions'))
        except FileNotFoundError:
            pass

    if pred is None:
        raise FileNotFoundError(f'There is no such id `{_id}` over {n_exp} experiments.')

    return pred
