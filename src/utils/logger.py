import sys
import os
import logging
import pathlib

from accelerate.logging import MultiProcessAdapter

def get_logger(name: str, log_level: str = None, path_to_log_file: pathlib.Path | None = None) -> logging.Logger:
    """
    Returns a `logging.Logger` for `name` that can handle multiprocessing.

    If a log should be called on all processes, pass `main_process_only=False` If a log should be called on all
    processes and in order, also pass `in_order=True`

    Args:
        name (`str`):
            The name for the logger, such as `__file__`
        log_level (`str`, *optional*):
            The log level to use. If not passed, will default to the `LOG_LEVEL` environment variable, or `INFO` if not
    """
    log_messages_format = "[%(asctime)-19s][%(name)-10s][%(levelname)-8s] %(message)s"
    log_date_format = "%Y-%m-%d %H:%M:%S"
    
    console_handler = logging.StreamHandler(sys.stdout)
    handlers = [console_handler]

    if path_to_log_file:
        file_handler = logging.FileHandler(filename=str(path_to_log_file))
        handlers.append(file_handler)

    if log_level is None:
        log_level = os.environ.get("ACCELERATE_LOG_LEVEL", None)

    logging.basicConfig(datefmt=log_date_format, 
                        format=log_messages_format, 
                        handlers=handlers)
    
    logger = logging.getLogger(name)
    
    if log_level is not None:
        logger.setLevel(log_level.upper())
        logger.root.setLevel(log_level.upper())

    return MultiProcessAdapter(logger, {})