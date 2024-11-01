import json
import os
from dataclasses import asdict, dataclass, field


@dataclass
class BaseConfig:
    name: str = field(init=False)

    def __post_init__(self):
        self.name = self.__class__.__name__

    def to_dict(self) -> dict:
        """Transforms object into a Python dictionnary

        Returns:
            (dict): The dictionnary containing all the parameters"""
        return asdict(self)

    def to_json_string(self):
        """Transforms object into a JSON string

        Returns:
            (str): The JSON str containing all the parameters"""
        return json.dumps(self.to_dict())

    def save_json(self, dir_path, filename):
        """Saves a ``.json`` file from the dataclass

        Args:
            dir_path (str): path to the folder
            filename (str): the name of the file

        """
        with open(os.path.join(dir_path, f"{filename}.json"), "w", encoding="utf-8") as fp:
            fp.write(self.to_json_string())


class BasePipeline:
    def __init__(self):
        pass

    def __call__(self):
        """
        The method run the Pipeline process and must be implemented in a child class
        """
        raise NotImplementedError()
