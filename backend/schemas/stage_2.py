from pydantic import BaseModel, Field

from backend.schemas.stage_1 import Stage1ExtractedFeatures, Stage1Output


class PredictionResponse(BaseModel):
	predicted_price: float = Field(gt=0)
	model_version: str = "ames_houses_v1"


class PredictionInterpretationRequest(BaseModel):
	features: Stage1ExtractedFeatures
	prediction_value: float | None = Field(default=None, gt=0)
	prediction: PredictionResponse | None = None


class PipelinePredictionResponse(BaseModel):
	query: str
	selected_prompt_version: str
	stage1_output: Stage1Output
	prediction: PredictionResponse



class PredictionInterpretationResponse(BaseModel):
	summary: str
	position_vs_market: str
	key_drivers: list[str] = Field(default_factory=list)
	caveats: list[str] = Field(default_factory=list)


class PipelineInterpretationResponse(BaseModel):
	query: str
	selected_prompt_version: str
	stage1_output: Stage1Output
	prediction: PredictionResponse
	interpretation: PredictionInterpretationResponse
