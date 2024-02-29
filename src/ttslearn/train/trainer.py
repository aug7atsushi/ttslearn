from ttslearn.train.base_trainer import TrainerBase


class Trainer(TrainerBase):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
