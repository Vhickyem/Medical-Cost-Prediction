"""
This file contains two custome transformers, built to preprocess the data loaded
into the models used.

* ExtraAttributesAdder which adds extra attributes to the data based on user's discretion

"""

# create a custom transformer to add extra attributes
from sklearn.base import BaseEstimator, TransformerMixin

# define class and __init__() method
class ExtraAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self,
                 add_bmi_smoker=True,
                 add_age_bmi=True,
                 add_children_smoker=True,
                 add_age_smoker=True):
        self.add_bmi_smoker = add_bmi_smoker
        self.add_age_bmi = add_age_bmi
        self.add_children_smoker = add_children_smoker
        self.add_age_smoker = add_age_smoker

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        # Add BMI-Smoker
        if self.add_bmi_smoker and 'bmi' in X.columns and 'smoker' in X.columns:
            X['bmi_smoker'] = X["bmi"] * X["smoker"].map({'yes': 1, 'no': 0})

        # Add Age-BMI
        if self.add_age_bmi and 'age' in X.columns and 'bmi' in X.columns:
            X['age_bmi'] = X["age"] * X["bmi"]

        # Add Children_Smoker
        if self.add_children_smoker and 'children' in X.columns and 'smoker' in X.columns:
            X['children_smoker'] = X["children"] * X["smoker"].map({'yes': 1, 'no': 0})

        # Add Age_Smoker
        if self.add_age_smoker and 'age' in X.columns and 'smoker' in X.columns:
            X['age_smoker'] = X["age"] * X["smoker"].map({'yes': 1, 'no': 0})

        return X
    

"""

* DropLowImportanceFeatures based on their importance

"""

# based on the results from grid search cv feature importance, I want to try dropping some models

from sklearn.base import BaseEstimator, TransformerMixin

class DropLowImportanceFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
    
    def fit(self, X, y=None):
        if isinstance(self.columns, str):
            self.columns_ = [self.columns]
        else:
            self.columns_ = list(self.columns)  # ensure it's a list
        return self
    
    def transform(self, X):
        X_transformed = X.copy()
        cols_to_drop = [col for col in self.columns_ if col in X_transformed.columns]
        X_transformed = X_transformed.drop(columns=cols_to_drop)
        return X_transformed
    
    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            raise ValueError("input features must be provided")
        return [f for f in input_features if f not in self.columns_]