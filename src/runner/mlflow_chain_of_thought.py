from src.runner.chain_of_thought import ChainOfThought
import mlflow


class MLFlowChainOfThought(ChainOfThought):

    def evaluate(self) -> dict:
        """ 
        Generate the textual output for the dataset and returns the metrics
        Logs the experiment on MLFlow
        """

        run_name = f"{self.args.dataset} || {self.filename} || {self.args.img_type}".upper()
        with mlflow.start_run(run_name=run_name):
            # model = mlflow.pytorch.load_model(self.args.model + '/pytorch_model.bin')
            # mlflow.pytorch.log_model(model, "model")

            metrics = super().evaluate()

            log_params = {
                "metrics": metrics,
                "img_type": self.args.img_type,
                "output": self.filename,
            }

            mlflow.log_params(log_params)
            mlflow.end_run()
