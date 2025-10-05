"""Training callbacks (early stopping, checkpoints) - placeholders."""


class ModelCheckpoint:
    def __init__(self, path: str):
        self.path = path

    def save(self, model):
        pass
