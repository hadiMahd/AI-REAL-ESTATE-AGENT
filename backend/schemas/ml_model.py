from pydantic import BaseModel, ConfigDict, Field


class MLModelInput(BaseModel):
	model_config = ConfigDict(populate_by_name=True, extra="forbid")

	OverallQual: int
	GrLivArea: int
	GarageCars: int
	FullBath: int
	YearBuilt: int
	YearRemodAdd: int
	MasVnrArea: float
	Fireplaces: int
	BsmtFinSF1: int
	LotFrontage: float
	first_flr_sf: int = Field(alias="1stFlrSF")
	OpenPorchSF: int


class MLModelOutput(BaseModel):
	prediction: int
