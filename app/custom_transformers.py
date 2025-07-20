import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# Definimos os transformadores do sklearn aqui para evitar bug de contexto no joblib
class AloneFeatureCreator(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_ = X.copy()
        X_['Alone'] = ((X_['SibSp'] + X_['Parch']) == 0).astype(int)
        return X_
    
class AgeBinner(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return (X <= 18).astype(int)