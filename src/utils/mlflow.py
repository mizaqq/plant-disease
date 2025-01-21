import os

import mlflow


class MLFlowHandler:
    def __init__(self) -> None:
        self.login()
        self.run = mlflow.start_run()

    @staticmethod
    def login(uri: str = os.environ.get("MLFLOW_URI", "http://localhost:5000"), experiment: str = "default") -> None:
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(experiment)

    @staticmethod
    def close() -> None:
        mlflow.end_run()
