from ttslearn.train.base_tester import TesterBase


class Tester(TesterBase):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
