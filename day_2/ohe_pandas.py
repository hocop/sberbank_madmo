import pandas as pd
from sklearn.preprocessing import OneHotEncoder

class OneHotEncoderPandas:
    def __init__(self, categorical_columns, *args, **kwargs):
        self.categorical_columns = categorical_columns
        kwargs['sparse'] = False
        self.ohe = OneHotEncoder(*args, **kwargs)
    
    def fit(self, X):
        self.ohe.fit(X[self.categorical_columns])
    
    def get_feature_names(self):
        return self.ohe.get_feature_names(self.categorical_columns)

    def transform(self, X):
        cat_names = self.get_feature_names()

        X_cat = X[self.categorical_columns]
        X_cat = self.ohe.transform(X_cat)
        X_cat = pd.DataFrame(X_cat, columns=cat_names, index=X.index)

        X_num = X.drop(self.categorical_columns, axis=1)

        return pd.concat([X_num, X_cat], axis=1)
