import os
import joblib
import pandas as pd
import asyncio
import logging
from contextlib import asynccontextmanager
from typing import List, Any
from collections import deque

from fastapi import FastAPI, HTTPException, status, Body
from fastapi.concurrency import run_in_threadpool
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic_settings import BaseSettings

# Make sure your models and custom transformers are correctly located
from .models.models import PredictionRequest, PredictionResponse, HistoryItem
from .custom_transformers import AloneFeatureCreator, AgeBinner

# --- Configuração e Inicialização ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AppSettings(BaseSettings):
    """
    Manages application settings using Pydantic for better configuration.
    """
    models_dir: str = "/app/models"
    default_model_name: str = "XGBoostClassifier.joblib"
    history_size: int = 100
    request_timeout: float = 10.0

# Create an instance of the settings
settings = AppSettings()

# --- Variáveis Globais ---
# Esta variável irá armazenar o modelo de machine learning atualmente ativo.
model: Any = None

# Use a deque for capped, memory-safe history
prediction_history: deque = deque(maxlen=settings.history_size)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gerencia o ciclo de vida da aplicação. Carrega o modelo padrão na inicialização.
    """
    global model
    default_model_path = os.path.join(settings.models_dir, settings.default_model_name)
    
    logger.info(f"Tentando carregar o modelo padrão: {settings.default_model_name}")
    try:
        model = await run_in_threadpool(joblib.load, default_model_path)
        logger.info(f"Modelo padrão '{settings.default_model_name}' carregado com sucesso.")
    except Exception as e:
        logger.error(f"Não foi possível carregar o modelo padrão: {e}")
        model = None
    
    yield  
    
    logger.info("Encerrando a aplicação...")
    model = None

# --- Aplicação FastAPI ---
app = FastAPI(
    title="API de Predição de Sobrevivência no Titanic",
    description="API para prever a sobrevivência no Titanic. Carrega modelos dinamicamente.",
    version="1.2.0",
    lifespan=lifespan
)

# Expõe as métricas para o Prometheus
Instrumentator().instrument(app).expose(app)

# --- Endpoints da API ---
@app.get("/v1/health", status_code=status.HTTP_200_OK, tags=["Saúde"])
async def health_check():
    """Verifica o status da API e se um modelo está carregado."""
    model_status = f"carregado: {model.__class__.__name__}" if model is not None else "não carregado"
    return {"status": "API está em execução", "model_status": model_status}

@app.post("/v1/load", status_code=status.HTTP_200_OK, tags=["Modelo"])
async def load_new_model(model_name: str = Body(..., embed=True, example="LogisticRegression.joblib")):
    """
    Carrega um novo modelo. Inclui validação de segurança contra Path Traversal.
    """
    global model
    
    try:
        
        models_dir_abs = os.path.abspath(settings.models_dir)
        model_path_abs = os.path.abspath(os.path.join(models_dir_abs, model_name))

        if not model_path_abs.startswith(models_dir_abs) or not os.path.exists(model_path_abs):
            logger.error(f"Tentativa de carregar modelo inválido ou não existente: {model_name}")
            raise FileNotFoundError(f"Modelo '{model_name}' não encontrado ou inválido.")
            
        new_model = await run_in_threadpool(joblib.load, model_path_abs)
        model = new_model #        
        logger.info(f" Modelo '{model_name}' carregado e selecionado com sucesso.")
        return {"message": f"Modelo '{model_name}' carregado com sucesso."}

    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Modelo '{model_name}' não encontrado."
        )
    except Exception as e:
        logger.error(f"Falha ao carregar o modelo '{model_name}': {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Falha ao carregar o modelo: {str(e)}"
        )

@app.post("/v1/predict", response_model=PredictionResponse, tags=["Predição"])
async def predict(request: PredictionRequest):
    """Realiza uma predição de sobrevivência com timeout e execução em thread separada."""
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Nenhum modelo carregado."
        )

    try:
        passenger_data = [p.model_dump(by_alias=True) for p in request.passengers]
        df_to_predict = pd.DataFrame(passenger_data)
        
        # Roda predições numa thread e aplica timeout
        predictions_future = run_in_threadpool(model.predict, df_to_predict)
        
        if hasattr(model, "predict_proba"):
            confidences_future = run_in_threadpool(model.predict_proba, df_to_predict)
            predictions, confidences = await asyncio.gather(
                asyncio.wait_for(predictions_future, timeout=settings.request_timeout),
                asyncio.wait_for(confidences_future, timeout=settings.request_timeout)
            )
        else:
            predictions = await asyncio.wait_for(predictions_future, timeout=settings.request_timeout)
            confidences = [[0.0, 0.0] for _ in predictions]

        response_data = [
            {"prediction": int(pred), "confidence": float(max(conf))}
            for pred, conf in zip(predictions, confidences)
        ]
        response = PredictionResponse(predictions=response_data)
        prediction_history.append(HistoryItem(request=request, response=response))
        return response
        
    except asyncio.TimeoutError:
        logger.error("A requisição de predição excedeu o tempo limite.")
        raise HTTPException(
            status_code=status.HTTP_408_REQUEST_TIMEOUT,
            detail="A requisição de predição demorou muito para ser processada."
        )
    except Exception as e:
        logger.error(f"Falha na predição: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ocorreu um erro durante a predição: {str(e)}"
        )

@app.get("/v1/history", response_model=List[HistoryItem], tags=["Histórico"])
async def get_history():
    """Retorna o histórico das últimas predições (tamanho configurável)."""
    return list(prediction_history)