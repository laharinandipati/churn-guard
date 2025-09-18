from pydantic import BaseModel, Field
from typing import Literal

ContractType = Literal["month_to_month", "one_year", "two_year"]
PaymentMethod = Literal["credit_card", "debit_card", "bank_transfer", "paypal"]

class PredictRequest(BaseModel):
    tenure_months: int = Field(ge=0)
    monthly_charges: float = Field(ge=0)
    total_charges: float = Field(ge=0)
    num_support_tickets: int = Field(ge=0)
    contract_type: ContractType
    payment_method: PaymentMethod
    has_addon_streaming: bool

class PredictResponse(BaseModel):
    churn_probability: float = Field(ge=0, le=1)
    churn_label: int
