from typing import Optional

import pandas as pd
from sklearn.externals import joblib

from forecaster.exceptions import ModelNotFittedException
from forecaster.features.processor import FeatureProcessor
from forecaster.label_encoder import MultiColumnLabelEncoder


class PriceModel:
    def __init__(self):
        self.le = MultiColumnLabelEncoder()
        self.features = FeatureProcessor()
        self.estimator = None

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        prepared_X = self.features.add_features(X)

        if self.estimator is None:
            raise ModelNotFittedException("Model not fitted yet!")

        transformed_X = self.le.transform(prepared_X)

        return self.estimator.preidct(transformed_X)

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        raise NotImplementedError("Method not implemented yet")

    def save(self):
        """
        method for saving serialized estimator
        :return:
        """
        joblib.dump(self.estimator, 'lgb.pkl')
        joblib.dump(self.le, 'label_encoder.pkl')

    def load(self,
             model_path: str,
             label_encoder_path: Optional[str] = None) -> None:
        self.estimator = joblib.load(model_path)
        if label_encoder_path is not None:
            self.le = joblib.load(label_encoder_path)
