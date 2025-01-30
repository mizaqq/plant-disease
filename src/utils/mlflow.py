import os
from typing import Optional

import mlflow


class MLFlowRunManager:
    def __init__(
        self,
        run_id: Optional[str] = None,
        uri: str = os.environ.get("MLFLOW_URI", "http://localhost:5000"),
        experiment: str = "default",
    ) -> None:
        self.manager = mlflow
        self.login(uri, experiment)
        if run_id:
            self.manager.start_run(run_id)
        else:
            self.manager.start_run()

    def login(self, uri: str, experiment: str) -> None:
        self.manager.set_tracking_uri(uri)
        self.manager.set_experiment(experiment)

    def close(self) -> None:
        self.manager.end_run()
