import os

import mlflow


class MLFlowRunManager:
    def __init__(self) -> None:
        self.manager = mlflow
        self.login()
        self.manager.start_run()

    def login(
        self, uri: str = os.environ.get("MLFLOW_URI", "http://localhost:5000"), experiment: str = "default"
    ) -> None:
        self.manager.set_tracking_uri(uri)
        self.manager.set_experiment(experiment)

    def close(self) -> None:
        self.manager.end_run()
