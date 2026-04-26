from pydantic import BaseModel, Field

class PredictionInput(BaseModel):
    pclass: int = Field(..., ge=1, le=3, description="Passenger Class (1, 2, or 3)")
    sex: str = Field(..., description="Sex ('male' or 'female')")
    age: float = Field(..., ge=0, le=120, description="Age in years")
    sibsp: int = Field(..., ge=0, description="Number of Siblings/Spouses Aboard")
    parch: int = Field(..., ge=0, description="Number of Parents/Children Aboard")
    fare: float = Field(..., ge=0, description="Passenger Fare")
    embarked: str = Field(..., description="Port of Embarkation ('C', 'Q', 'S')")

class PredictionOutput(BaseModel):
    prediction: int
    probability: float
    model: str
    confidence: str
