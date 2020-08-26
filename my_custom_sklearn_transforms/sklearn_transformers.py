from sklearn.base import BaseEstimator, TransformerMixin
import imblearn
from imblearn.over_sampling import SMOTE
import pandas as pd

# All sklearn Transforms must have the `transform` and `fit` methods
class DropColumns(BaseEstimator, TransformerMixin):
    def _init_(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Primeiro realizamos a cópia do dataframe 'X' de entrada
        data = X.copy()
        # Retornamos um novo dataframe sem as colunas indesejadas
        return data.drop(labels=self.columns, axis='columns')
    
class SetIndex(BaseEstimator, TransformerMixin):
    def _init_(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Primeiro realizamos a cópia do dataframe 'X' de entrada
        data = X.copy()
        # Retornamos um novo dataframe sem as colunas indesejadas
        return data.set_index(self.columns, inplace=True)
    
   
class SmoteResample(object):
    def _init_(self):
        pass

    def fit(self, X, y):
        X_resampled, y_resampled = SMOTE().fit_resample(X, y)

        X_resampled = pd.DataFrame(X_resampled, columns=X.columns)
        
        return X_resampled, y_resampled
