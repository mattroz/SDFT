import os
import json

from typing import Union, Optional
from accelerate.tracking import GeneralTracker, on_main_process


class FileSystemTracker(GeneralTracker):
    """Tracker that logs scalars to a scalars.json file on the file system.
    """
    name = "file_system_tracker"
    requires_logging_directory = True

    @on_main_process
    def __init__(self, logging_dir: Union[str | os.PathLike]):
        """Initializes the FileSystemTracker.

        Args:
            logging_dir (Union[str | os.PathLike]): The directory where the scalars.json file will be stored.
        """
        self.path_to_log_file = os.path.join(logging_dir, "scalars.json")
        self.config = None
        self.run = []

    @property
    def tracker(self):
        return self.run

    @on_main_process
    def store_init_configuration(self, values: dict):
        self.config = values

    @on_main_process
    def log(self, values: dict, step: Optional[int] = None):
        log_entry = values.copy()
        log_entry["step"] = step
        self.run.append(log_entry)
        self._dump_to_file(log_entry)
        
    @on_main_process
    def _dump_to_file(self, entry):
        with open(self.path_to_log_file, "a") as f:
            json.dump(entry, f)
            f.write("\n")