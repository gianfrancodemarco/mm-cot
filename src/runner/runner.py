from abc import ABC, abstractmethod


class Runner(ABC):

    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def train(self) -> None:
        pass

    @abstractmethod
    def evaluate(self) -> dict:
        """ Generate the textual output for the dataset and returns the metrics """
        pass

    @abstractmethod
    def infer(self, sample) -> str:
        pass
