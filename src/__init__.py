"""
Task Generation Source Package

This package provides utilities for generating, validating, and managing
benchmark tasks for the tau2 benchmark system.
"""
import sys
import warnings

from loguru import logger

# Configure logging for the entire package.
# Disables tau2 debug logging to reduce noise during task generation.
logger.remove()
logger.add(sys.stderr, level="WARNING")

# Suppress Pydantic serialization warnings from tau2/litellm
warnings.filterwarnings("ignore", message="Pydantic serializer warnings")
