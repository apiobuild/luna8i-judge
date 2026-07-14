from pydantic import BaseModel


class ModelPricingResponse(BaseModel):
    input_per_1m: float
    output_per_1m: float


class ModelProviderResponse(BaseModel):
    id: str
    name: str
    provider: str
    tpm_ceiling: int
    pricing: ModelPricingResponse
