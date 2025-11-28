"""Core package for the vgs-chatbot application."""

import logging

# Instantiate module-level logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Set pattern for log messages
_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(format=_LOG_FORMAT)
