import os

import pandas as pd
import numpy as np
from sklearn.externals import joblib

from forecaster.exceptions import (ModelNotFittedException,
                                   ModelNotFoundException,
                                   EmptyDatasetException)

from forecaster.features.processor import FeatureProcessor
from forecaster.resources import MODELS_PATH


class PriceModel(object):
    def __init__(self, models_path: str = MODELS_PATH):
        self.features_processor = FeatureProcessor()
        self.estimator = None
        self.models_path = models_path

    def predict(self, test_dataset: pd.DataFrame) -> pd.DataFrame:

        prepared_data = self.features_processor.add_features(test_dataset)

        if self.estimator is None:
            raise ModelNotFittedException("Model not fitted yet!")

        prediction = pd.DataFrame()
        prediction['pred'] = np.expm1(self.estimator.predict(prepared_data))
        return prediction.to_dict()

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        raise NotImplementedError("Method not implemented yet")

    def save(self):
        """
        method for saving serialized estimator
        :return:
        """
        if self.estimator:
            joblib.dump(self.estimator, 'lgbm.pkl')
        else:
            raise ModelNotFoundException("Model not found")

    def load(self) -> None:
        files = os.listdir(self.models_path)
        if 'lgbm.pkl' in files:
            self.estimator = joblib.load(os.path.join(MODELS_PATH, 'lgbm.pkl'))
        else:
            raise ModelNotFoundException(f'Model not found'
                                         f' in {self.models_path}')
