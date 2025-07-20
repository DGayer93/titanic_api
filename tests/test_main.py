import pytest
from fastapi.testclient import TestClient
import os
import joblib
import pandas as pd


from app.main import app, AppSettings

client = TestClient(app)


# --- Pytest Fixtures ---

class FakePipeline:
    """Uma classe simples que simula um pipeline do scikit-learn."""
    def predict(self, X):
        return [1] * len(X)

    def predict_proba(self, X):
        return [[0.1, 0.9]] * len(X)


@pytest.fixture(scope="module")
def mock_model_files(tmpdir_factory):
    """
    Cria um diretório temporário e um arquivo de modelo falso para os testes.
    """
    temp_dir = tmpdir_factory.mktemp("models")
    model_path = os.path.join(temp_dir, "fake_model.joblib")
    fake_model = FakePipeline()
    joblib.dump(fake_model, model_path)
    yield str(temp_dir)


# --- Funções de Teste ---

def test_health_check():
    """
    Testa o endpoint /v1/health para garantir que a API está em execução.
    """
    response = client.get("/v1/health")
    assert response.status_code == 200
    json_response = response.json()
    assert "status" in json_response
    assert "model_status" in json_response

def test_load_model_successfully(mock_model_files, monkeypatch):
    """
    Testa o endpoint /v1/load, substituindo o objeto de configurações.
    """
    
    test_settings = AppSettings(models_dir=mock_model_files)
    monkeypatch.setattr("app.main.settings", test_settings)
    
    response = client.post("/v1/load", json={"model_name": "fake_model.joblib"})
    assert response.status_code == 200
    assert response.json() == {"message": "Modelo 'fake_model.joblib' carregado com sucesso."}

def test_load_nonexistent_model(monkeypatch):
    """
    Testa se a API retorna um erro 404 ao tentar carregar um modelo que não existe.
    """
    
    test_settings = AppSettings(models_dir="/tmp")
    monkeypatch.setattr("app.main.settings", test_settings)

    response = client.post("/v1/load", json={"model_name": "nonexistent_model.joblib"})
    assert response.status_code == 404
    assert "não encontrado" in response.json()["detail"]

def test_predict_with_loaded_model(mock_model_files, monkeypatch):
    """
    Testa o endpoint /v1/predict depois que um modelo foi carregado com sucesso.
    """
    
    test_settings = AppSettings(models_dir=mock_model_files)
    monkeypatch.setattr("app.main.settings", test_settings)

    # Load the model first
    client.post("/v1/load", json={"model_name": "fake_model.joblib"})

    payload = {
      "passengers": [
        {
          "Pclass": 1, "Sex": "female", "Age": 38, "SibSp": 1,
          "Parch": 0, "Fare": 71.2833, "Embarked": "C"
        }
      ]
    }
    response = client.post("/v1/predict", json=payload)
    
    assert response.status_code == 200
    json_response = response.json()
    assert len(json_response["predictions"]) == 1

def test_predict_without_model_loaded(monkeypatch):
    """
    Testa se a API retorna um erro 400 se o /v1/predict for chamado antes de um modelo ser carregado.
    """
    monkeypatch.setattr("app.main.model", None)
    
    payload = {"passengers": [{"Pclass": 1, "Sex": "female", "Age": 38, "SibSp": 1, "Parch": 0, "Fare": 71.2833, "Embarked": "C"}]}
    
    response = client.post("/v1/predict", json=payload)
    
    assert response.status_code == 400
    assert "Nenhum modelo carregado" in response.json()["detail"]