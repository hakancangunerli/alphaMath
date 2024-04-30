"""
This module contains the logging configuration for the project.
"""

import logging
import sys


def setup_logging(logging_level=logging.DEBUG):

    # Set up the logging configuration
    logging.basicConfig(
        level=logging_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
