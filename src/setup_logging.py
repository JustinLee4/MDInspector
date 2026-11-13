import logging
import logging.config
from pathlib import Path


def setup_logging(output_dir: Path, log_filename: str = "debug.log") -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file_path = output_dir / log_filename

    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        
        "formatters": {
            "detailed": {
                "format": "[%(asctime)s] [%(levelname)s] [%(name)s:%(lineno)d]: %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
            "clean": {
                "format": "%(message)s",
            },
        },
        
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": "INFO",
                "formatter": "clean",
                "stream": "ext://sys.stdout",
            },
            "file": {
                "class": "logging.FileHandler",
                "level": "DEBUG",
                "formatter": "detailed",
                "filename": str(log_file_path),
                "mode": "w",
            },
        },
        
        "loggers": {
            "matplotlib": {"level": "WARNING"},
        },
        
        "root": {
            "level": "DEBUG",
            "handlers": ["console", "file"],
        },
    }

    logging.config.dictConfig(logging_config)
    
    logging.captureWarnings(True)

    return logging.getLogger()