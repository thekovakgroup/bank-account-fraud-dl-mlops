
# ============================================================
# MILESTONE 09 - PRIVACY-SAFE OPERATIONAL LOGGING
# Step 09.79 - Sanitized Inference Failure Logging
# ============================================================

import logging


# ------------------------------------------------------------
# 1. Create an application-specific logger.
# ------------------------------------------------------------

LOGGER = logging.getLogger(
    "fraud_detection.api"
)


# ------------------------------------------------------------
# 2. Record inference failures safely.
# ------------------------------------------------------------

def log_inference_failure(error: Exception) -> None:
    """
    Record a sanitized inference failure event.

    Never log:
        - Raw application records
        - Request bodies
        - Fraud probabilities
        - Exception messages
        - Exception tracebacks
        - Private model artifact paths

    Only record a fixed event identifier and the
    exception class name.
    """

    # Extract the exception category without exposing
    # the original exception message.
    exception_type = type(error).__name__

    # Record only approved diagnostic information.
    LOGGER.error(
        "event=fraud_inference_failed exception_type=%s",
        exception_type,
    )
