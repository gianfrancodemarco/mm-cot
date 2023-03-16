from src.runner.chain_of_thought import ChainOfThought
import mlflow


def mlflow_logging(func):
    def wrapper(*args, **kwargs) -> dict:
        """ 
        Generate the textual output for the dataset and returns the metrics
        Logs the experiment on MLFlow
        """

        instance = args[0]
        with mlflow.start_run():
            # model = mlflow.pytorch.load_model(self.args.model + '/pytorch_model.bin')
            # mlflow.pytorch.log_model(model, "model")

            metrics = func(instance)

            log_params = {
                **metrics,
                "task": instance.args.task,
                "dataset": instance.args.dataset,
                "img_type": instance.args.img_type,
                "output": instance.filename,
                "test_le": instance.args.test_le
            }

            mlflow.log_params(log_params)
            mlflow.end_run()
    return wrapper


class MLFlowChainOfThought(ChainOfThought):

    @mlflow_logging
    def evaluate(self) -> dict:
        return super().evaluate()
