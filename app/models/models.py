from pydantic import BaseModel, Field, conint, confloat
from typing import List, Optional, Literal

class Passenger(BaseModel):
    """Representa um único passageiro para a predição."""
    passenger_id: Optional[int] = Field(None, alias="PassengerId", examples=[1])
    name: Optional[str] = Field(None, alias="Name", examples=["Braund, Mr. Owen Harris"])
    pclass: Literal[1, 2, 3] = Field(..., alias="Pclass", examples=[1], description="Classe do bilhete (1=1ª, 2=2ª, 3=3ª)")
    sex: Literal['male', 'female'] = Field(..., alias="Sex", examples=["female"], description="Sexo do passageiro")
    age: Optional[confloat(ge=0, le=150)] = Field(None, alias="Age", examples=[38.0], description="Idade em anos")
    sibsp: conint(ge=0) = Field(..., alias="SibSp", examples=[1], description="Número de irmãos/cônjuges a bordo")
    parch: conint(ge=0) = Field(..., alias="Parch", examples=[0], description="Número de pais/filhos a bordo")
    ticket: Optional[str] = Field(None, alias="Ticket", examples=["PC 17599"])
    fare: confloat(ge=0) = Field(..., alias="Fare", examples=[71.2833], description="Tarifa do passageiro")
    cabin: Optional[str] = Field(None, alias="Cabin", examples=["C85"])
    embarked: Optional[str] = Field(None, alias="Embarked", examples=["C"], description="Porto de Embarque (C=Cherbourg, Q=Queenstown, S=Southampton)")

    class Config:
        """
        Permite que o modelo seja populado usando os nomes de campo originais (ex: "PassengerId")
        em vez dos nomes de campo em Python (ex: "passenger_id").
        """
        allow_population_by_field_name = True

class PredictionRequest(BaseModel):
    """O corpo da requisição para o endpoint /predict."""
    passengers: List[Passenger]

class Prediction(BaseModel):
    """Representa um único resultado de predição."""
    prediction: Literal[0, 1] = Field(..., example=1, description="Predição de sobrevivência (0 = Não, 1 = Sim)")
    confidence: confloat(ge=0.0, le=1.0) = Field(..., example=0.95, description="Pontuação de confiança da predição")

class PredictionResponse(BaseModel):
    """O corpo da resposta para o endpoint /predict."""
    predictions: List[Prediction]

class HistoryItem(BaseModel):
    """Representa um único item no histórico de predições."""
    request: PredictionRequest
    response: PredictionResponse

