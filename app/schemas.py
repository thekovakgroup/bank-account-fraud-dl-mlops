
# ============================================================
# MILESTONE 09 — FASTAPI INPUT SCHEMAS
# Step 09.22 — Pydantic Request Validation
# ============================================================

from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    FiniteFloat,
    StrictStr,
)


# ------------------------------------------------------------
# 1. Define one bank application.
# ------------------------------------------------------------

class FraudApplication(BaseModel):
    """
    Validate the 29 raw features required by the frozen
    fraud detection model.

    Excludes:
        fraud_bool
        month
        device_fraud_count
    """

    model_config = ConfigDict(extra="forbid")

    # --------------------------------------------------------
    # 2. Numeric features (18).
    # --------------------------------------------------------

    income: FiniteFloat
    name_email_similarity: FiniteFloat
    prev_address_months_count: FiniteFloat
    current_address_months_count: FiniteFloat
    customer_age: FiniteFloat
    days_since_request: FiniteFloat
    intended_balcon_amount: FiniteFloat
    zip_count_4w: FiniteFloat
    velocity_6h: FiniteFloat
    velocity_24h: FiniteFloat
    velocity_4w: FiniteFloat
    bank_branch_count_8w: FiniteFloat
    date_of_birth_distinct_emails_4w: FiniteFloat
    credit_risk_score: FiniteFloat
    bank_months_count: FiniteFloat
    proposed_credit_limit: FiniteFloat
    session_length_in_minutes: FiniteFloat
    device_distinct_emails_8w: FiniteFloat

    # --------------------------------------------------------
    # 3. Categorical features (5).
    # --------------------------------------------------------

    payment_type: StrictStr = Field(min_length=1)
    employment_status: StrictStr = Field(min_length=1)
    housing_status: StrictStr = Field(min_length=1)
    source: StrictStr = Field(min_length=1)
    device_os: StrictStr = Field(min_length=1)

    # --------------------------------------------------------
    # 4. Binary features (6).
    # --------------------------------------------------------

    email_is_free: Literal[0, 1]
    phone_home_valid: Literal[0, 1]
    phone_mobile_valid: Literal[0, 1]
    has_other_cards: Literal[0, 1]
    foreign_request: Literal[0, 1]
    keep_alive_session: Literal[0, 1]


# ------------------------------------------------------------
# 5. Define the complete prediction request.
# ------------------------------------------------------------

class PredictionRequest(BaseModel):
    """
    Accept a nonempty collection of bank applications.

    The API will eventually return one calibrated fraud
    probability for each submitted application.
    """

    model_config = ConfigDict(extra="forbid")


    applications: list[FraudApplication] = Field(
        min_length=1,
        max_length=100,
        description=(
            "Submit between 1 and 100 bank applications "
            "per prediction request."
        ),
    )
