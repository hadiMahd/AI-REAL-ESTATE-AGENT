from pydantic import BaseModel, ConfigDict, Field


class Stage1Input(BaseModel):
	query: str


class Stage1ExtractedFeatures(BaseModel):
	model_config = ConfigDict(populate_by_name=True, extra="forbid")

	OverallQual: int | None = None
	GrLivArea: int | None = None
	GarageCars: int | None = None
	FullBath: int | None = None
	YearBuilt: int | None = None
	YearRemodAdd: int | None = None
	MasVnrArea: float | None = None
	Fireplaces: int | None = None
	BsmtFinSF1: int | None = None
	LotFrontage: float | None = None
	first_flr_sf: int | None = Field(default=None, alias="1stFlrSF")
	OpenPorchSF: int | None = None


class Stage1Output(BaseModel):
	features: Stage1ExtractedFeatures
	extracted_fields: list[str] = Field(default_factory=list)
	missing_fields: list[str] = Field(default_factory=list)
	completeness: float = Field(ge=0.0, le=1.0)
	ready_for_prediction: bool = False
