
# ============================================================
# MILESTONE 09 - PRIVACY-SAFE LOGGING TESTS
# Step 09.80 - Verify Sanitized Failure Logging
# ============================================================

import logging

from app.safe_logging import log_inference_failure


# ------------------------------------------------------------
# 1. Verify failure events exclude sensitive information.
# ------------------------------------------------------------

def test_inference_failure_logging_is_sanitized(caplog):
    """
    Verify that operational logs contain only approved
    diagnostic information.

    All sensitive test values are synthetic.
    """

    # --------------------------------------------------------
    # 2. Create synthetic sensitive information.
    # --------------------------------------------------------

    sensitive_marker = "SYNTHETIC_PRIVATE_APPLICATION"
    private_path = "SYNTHETIC_PRIVATE_MODEL_PATH"

    error = ValueError(
        f"{sensitive_marker} "
        f"{private_path} "
        "income=SYNTHETIC_FINANCIAL_VALUE"
    )

    # --------------------------------------------------------
    # 3. Capture error-level application logs.
    # --------------------------------------------------------

    with caplog.at_level(
        logging.ERROR,
        logger="fraud_detection.api",
    ):
        log_inference_failure(error)

    # --------------------------------------------------------
    # 4. Verify exactly one failure event was recorded.
    # --------------------------------------------------------

    assert len(caplog.records) == 1

    record = caplog.records[0]

    assert record.name == "fraud_detection.api"
    assert record.levelno == logging.ERROR

    # --------------------------------------------------------
    # 5. Verify the standardized log message.
    # --------------------------------------------------------

    assert record.getMessage() == (
        "event=fraud_inference_failed "
        "exception_type=ValueError"
    )

    # --------------------------------------------------------
    # 6. Verify sensitive information is excluded.
    # --------------------------------------------------------

    logged_output = caplog.text

    assert sensitive_marker not in logged_output
    assert private_path not in logged_output

    assert (
        "SYNTHETIC_FINANCIAL_VALUE"
        not in logged_output
    )

    # --------------------------------------------------------
    # 7. Verify exception details are not attached.
    # --------------------------------------------------------

    assert record.exc_info is None
    assert record.stack_info is None

    # --------------------------------------------------------
    # 8. Report successful verification.
    # --------------------------------------------------------

    print("Failure event: RECORDED")
    print("Exception category: VERIFIED")
    print("Synthetic financial data: EXCLUDED")
    print("Private artifact path: EXCLUDED")
    print("Exception traceback: EXCLUDED")
    print("Step 09.80: PASS")
