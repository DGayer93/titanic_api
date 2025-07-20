

### Para execução desse projeto, siga os seguintes passos


## Passo 1

Crie a venv  
`python3 -m venv venv`

Ative a venv

No Linux ou Mac:

`source venv/bin/active`

No Windows:

`venv/Scripts/activate`


instale os requerimentos

`pip install -r requirements.txt`


## Passo 2

Execute o `model_development.ipynb` 

Após executá-lo, a pasta saved_models irá ser criada e com os seguintes arquivos 

1. `LogisticRegression.joblib`

2. `RadomForestClassifier.joblib`

3. `XGBoostClassifier.joblib`

Eles serão os modelos possiveis para carregamento no endpoint `v1/load`



## Passo 3

Suba a aplicação com o comando

`docker compose up`


A aplicação estará disponivel no `localhost:8000` e a documentação no `localhost:8000/docs`

O prometheus disponivel no `localhost:9090` 

Grafana disponivel no `localhost:3000`








