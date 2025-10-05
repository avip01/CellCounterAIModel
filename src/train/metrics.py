"""Training metrics helpers."""
def precision(tp, fp):
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0
