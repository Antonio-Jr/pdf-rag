"""Centralized logging configuration.

Sets up a consistent logging format for the entire application,
including timestamp, severity, and module name.  External libraries
like ``httpx`` and ``openai`` are configured with higher thresholds
to reduce noise.
"""

import logging
import logging.config
import sys


def setup_logging(default_level=logging.INFO):
    """Configure the application-wide logging system.

    Applies a ``dictConfig`` that installs a stdout console handler
    with a standard formatter.  Should be called once during
    application startup (typically inside the lifespan context).

    Args:
        default_level: Minimum log level for the root logger.
                       Defaults to ``logging.INFO``.
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
                "format": "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d | %(message)s", # noqa: E501
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
            "": {
                "handlers": ["console"],
                "level": default_level,
                "propagate": True,
            },
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
