"""Kivra — small-footprint open-weight language models (scaffold).

Version 0.1.0. This is a runnable scaffold for fine-tuning Qwen checkpoints;
no model weights are trained, downloaded, or released by this package itself.
"""

__version__ = "0.1.0"

SPARK_PROFILE_ID = "kivra-spark-0.6b-instruct-v1"
VOLTAGE_PROFILE_ID = "kivra-voltage-3b-instruct-v1"

__all__ = ["SPARK_PROFILE_ID", "VOLTAGE_PROFILE_ID", "__version__"]
