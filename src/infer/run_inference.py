def infer_single_image(image_path: str, model, cfg: dict):
    """
    1) read big image
    2) run candidate_extractor
    3) crop+pad and batch
    4) run model.predict
    5) apply threshold + nms
    6) return list of positives (bbox, score)
    """
    raise NotImplementedError
