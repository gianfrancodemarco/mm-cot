from typing import Any

import mlflow

from src.args_parser import parse_args
args = parse_args()


class MLFlowLogging():
    def __init__(self, experiment_name: str = None) -> None:
        """
        :param experiment_name: The experiment name that is set by calling mlflow.set_experiment before running the wrapped method
        """

        self.experiment_name = experiment_name

    def __call__(self, func) -> Any:
        def wrapper(*args, **kwargs) -> dict:
            """ 
            Generate the textual output for the dataset and returns the metrics
            Logs the experiment on MLFlow
            """

            if self.experiment_name:
                mlflow.set_experiment(self.experiment_name)

            with mlflow.start_run():
                # model = mlflow.pytorch.load_model(self.args.model + '/pytorch_model.bin')
                # mlflow.pytorch.log_model(model, "model")

                metrics = func()

                log_params = {
                    **metrics,
                    "task": args.task,
                    "dataset": args.dataset,
                    "img_type": args.img_type,
                    # "output": filename,
                    "test_le": args.test_le
                }

                mlflow.log_params(log_params)
                mlflow.end_run()
        return wrapper
