import numpy as np
import os

import pandas as pd

from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler

from forecaster.features.exception import DataIsMissingException
from forecaster.resources import MODELS_PATH


class FeatureProcessor(object):
    """
    class for processing data and add some features for that
    """

    def __init__(self, scaler: StandardScaler = None):
        self.to_normalize = ['dayofyear', 'month', 'weekofyear', 'flat_area']
        if not scaler:
            self.__init_serilaized_scaler()

    def add_features(self, data: pd.DataFrame) -> pd.DataFrame:
        if data is None:
            raise DataIsMissingException("Empty dataset!")

        self._add_date_features(data)
        data.drop(['date', 'flat_thermal_characteristic', 'geo_city'],
                  axis=1, inplace=True)
        data['flat_area'] = data['flat_area'].astype(int)

        X = data[[col for col in data.columns if
                  col not in ['rent_total', 'log_rent_total']]].copy()

        X['log_rent_base'] = np.log1p(X['rent_base'])
        X.drop(columns=['rent_base'], axis=1, inplace=True)

        X[self.to_normalize] = self.scaler.transform(X[self.to_normalize]).copy()
        prepared = self._add_one_hot_features(X)
        return prepared

    def _add_one_hot_features(self, X):
        for col in X.select_dtypes(['object']).columns:
            X[col] = X[col].astype('category')
        X['cnt_rooms'] = X['cnt_rooms'].astype('category')
        return pd.get_dummies(X)

    def _add_date_features(self, data):
        print(data.columns)
        data['flat_type'].replace("appartment", "apartment", inplace=True)
        data['date'] = pd.to_datetime(data.date)
        data['weekday'] = data.date.dt.weekday
        data['weekofyear'] = data.date.dt.weekofyear
        data['month'] = data.date.dt.month
        data['season'] = data.month.apply(lambda dt: (dt % 12 + 3) // 3)
        data['dayofyear'] = data.date.dt.dayofyear

    def __init_serilaized_scaler(self):
        with open(os.path.join(MODELS_PATH, 'scaler.pkl'), 'rb') as f:
            self.scaler = joblib.load(f)
