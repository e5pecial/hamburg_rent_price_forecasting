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

    def __init__(self, scaler=None):
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

        X[self.to_normalize] = self.scaler.transform(
            X[self.to_normalize]).copy()

        return self._add_one_hot_features(X)

    def _add_one_hot_features(self, X):
        for col in X.select_dtypes(['object']).columns:
            X[col] = X[col].astype('category')
        X['cnt_rooms'] = X['cnt_rooms'].astype('category')
        X['weekday'] = X['weekday'].astype('category')
        X_with_dummies = pd.get_dummies(X)
        for col in self.__get_columns_from_train():
            if col not in X_with_dummies.columns:
                X_with_dummies[col] = 0
        return X_with_dummies

    def _add_date_features(self, data):
        data['flat_type'].replace("appartment", "apartment", inplace=True)
        data['date'] = pd.to_datetime(data.date)
        data['weekday'] = data.date.dt.weekday
        data['weekofyear'] = data.date.dt.weekofyear
        data['month'] = data.date.dt.month
        # data['season'] = data.month.apply(lambda dt: (dt % 12 + 3) // 3)
        data['dayofyear'] = data.date.dt.dayofyear

    def __init_serilaized_scaler(self):
        with open(os.path.join(MODELS_PATH, 'scaler.pkl'), 'rb') as f:
            scaler = joblib.load(f)
        self.scaler = scaler

    def __get_columns_from_train(self):
        return ['flat_area', 'weekofyear', 'month', 'dayofyear',
                'log_rent_base',
                'cnt_rooms_1', 'cnt_rooms_2', 'cnt_rooms_3', 'cnt_rooms_4',
                'flat_type_apartment', 'flat_type_ground_floor',
                'flat_type_half_basement', 'flat_type_loft',
                'flat_type_maisonette',
                'flat_type_penthouse', 'flat_type_raised_ground_floor',
                'flat_type_roof_storey', 'flat_type_terraced_flat',
                'flat_interior_quality_average',
                'flat_interior_quality_luxury',
                'flat_interior_quality_normal', 'flat_interior_quality_simple',
                'flat_interior_quality_sophisticated',
                'flat_condition_first_time_use',
                'flat_condition_first_time_use_after_refurbishment',
                'flat_condition_good', 'flat_condition_mediocre',
                'flat_condition_mint_condition', 'flat_condition_renovated',
                'flat_age_60+', 'flat_age_<1', 'flat_age_<10', 'flat_age_<20',
                'flat_age_<30', 'flat_age_<40', 'flat_age_<5', 'flat_age_<50',
                'flat_age_<60', 'has_elevator_f', 'has_elevator_t',
                'has_balcony_f',
                'has_balcony_t', 'has_garden_f', 'has_garden_t',
                'has_kitchen_f',
                'has_kitchen_t', 'has_guesttoilet_f', 'has_guesttoilet_t',
                'geo_city_part_altona', 'geo_city_part_bergedorf',
                'geo_city_part_eimsbuettel', 'geo_city_part_hamburg-nord',
                'geo_city_part_harburg', 'geo_city_part_mitte',
                'geo_city_part_wandsbek', 'weekday_0', 'weekday_1',
                'weekday_2',
                'weekday_3', 'weekday_4', 'weekday_5', 'weekday_6']
