import os

from label_studio_sdk import Client


class LabelStudioManager:
    def __init__(self, api_key: str = os.getenv('api_key'), url: str = 'http://localhost:8080') -> None:
        self.client = Client(url=url, api_key=api_key)

    def create_project(self, project_name: str, label_config: str) -> None:
        self.client.create_project(project_name, label_config)
