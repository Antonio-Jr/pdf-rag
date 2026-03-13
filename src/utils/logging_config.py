import logging
import logging.config
import sys


def setup_logging(default_level=logging.INFO):
    """
    Centralized logging configuration for the entire application.
    Ensures consistent formatting across Agents, Runners, and Tools.
    """
    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
            "detailed": {
                "format": "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d | %(message)s",
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": "DEBUG",
                "formatter": "standard",
                "stream": sys.stdout,
            },
        },
        "loggers": {
            # Root logger
            "": {
                "handlers": ["console"],
                "level": default_level,
                "propagate": True,
            },
            # Library-specific overrides (silencing noisy external libs)
            "httpx": {
                "level": "WARNING",
                "handlers": ["console"],
                "propagate": False,
            },
            "openai": {
                "level": "INFO",
                "handlers": ["console"],
                "propagate": False,
            },
        },
    }

    logging.config.dictConfig(logging_config)
    logging.info("Logging system initialized.")
