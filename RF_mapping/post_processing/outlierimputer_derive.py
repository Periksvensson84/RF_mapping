import pandas as pd
import numpy as np
import json
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.linear_model import BayesianRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import GridSearchCV
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.metrics import mean_squared_error
from validation import Validation as Val

class OutlierImputerDeriveSTD:
    models = {
        "RFR": RandomForestRegressor(n_estimators=100, random_state=42),
        "HGBR": HistGradientBoostingRegressor(),
        "KNR": KNeighborsRegressor(n_neighbors=10),
        "XGB": XGBRegressor(n_estimators=100, learning_rate=0.1),
        "SVR": SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1),
        "BR": BayesianRidge(),
        "Poly": make_pipeline(PolynomialFeatures(degree=2), BayesianRidge())
    }

    param_grids = {
        "RFR": {"n_estimators": [50, 100, 200], "max_depth": [None, 10, 20]},
        "HGBR": {"max_iter": [100, 200], "learning_rate": [0.05, 0.1, 0.2]},
        "KNR": {"n_neighbors": [5, 10, 15], "weights": ["uniform", "distance"]},
        "XGB": {"n_estimators": [50, 100, 200], "learning_rate": [0.05, 0.1, 0.2]},
        "SVR": {"C": [1, 10, 100], "gamma": ["scale", 0.1, 1], "epsilon": [0.01, 0.1, 0.2]},
        "BR": {"alpha_1": [1e-6, 1e-5, 1e-4], "lambda_1": [1e-6, 1e-5, 1e-4]},
        "Poly": {"bayesianridge__alpha_1": [1e-6, 1e-5], "polynomialfeatures__degree": [2]}
    }

    def __init__(self, model=None, hybrid_model_selection=True, log_file="model_performance.json"):
        self.model = model
        self.hybrid_model_selection = hybrid_model_selection
        self.best_models = {}
        self.log_file = log_file

    @staticmethod
    def detect_outliers_velocity(df: pd.DataFrame, threshold: float = 2.0):
        df_velocity = df.diff().abs()
        df_velocity.iloc[0, :] = 0  # Avoid NaNs in the first row

        mean, std = df_velocity.mean(), df_velocity.std()
        outlier_mask = (df_velocity < (mean - threshold * std)) | (df_velocity > (mean + threshold * std))
        outlier_mask.iloc[0, :] = False  # First row should not be an outlier

        # Additionally, mark outliers directly by too high a velocity, e.g. 100
        outlier_mask |= (df_velocity > 50)

        df[outlier_mask] = np.nan  # Mark outliers as NaN
        return df

    def impute_with_ml(self, df: pd.DataFrame, target_col: str):
        df_copy = df.copy()

        # Split into train (non-NaN) and test (NaN)
        train_df = df_copy.dropna(subset=[target_col])
        train_df = train_df.dropna(how="any")
        test_df = df_copy[df_copy[target_col].isna()]
        if test_df.empty:
            return df_copy  # Nothing to impute

        feature_cols = [col for col in df.columns if col != target_col]
        X_train, y_train = train_df[feature_cols], train_df[target_col]

        if self.hybrid_model_selection:
            best_model, best_score = None, float("inf")

            for model_name, model in self.models.items():
                param_grid = self.param_grids.get(model_name, {})
                grid = GridSearchCV(model, param_grid, scoring="neg_mean_squared_error", cv=3)
                grid.fit(X_train, y_train)
                mse = -grid.best_score_

                if mse < best_score:
                    best_score, best_model = mse, grid.best_estimator_

            self.best_models[target_col] = best_model
            ml_model = best_model
        else:
            if self.model not in self.models:
                raise ValueError(f"Invalid model: {self.model}, use one of {list(self.models.keys())}")
            ml_model = self.models[self.model]
            ml_model.fit(X_train, y_train)

        df_copy.loc[test_df.index, target_col] = ml_model.predict(test_df[feature_cols])
        return df_copy

    def iterative_imputation(self, df: pd.DataFrame, max_iter=100):
        df_copy = df.copy()

        # Use best models if available, otherwise fallback to BayesianRidge
        estimator = self.best_models.get("global", BayesianRidge())

        imputer = IterativeImputer(estimator=estimator, max_iter=max_iter, random_state=42)
        imputed_array = imputer.fit_transform(df_copy)

        df_imputed = pd.DataFrame(imputed_array, columns=df.columns, index=df.index)
        return df_imputed

    def impute_outliers(self, df: pd.DataFrame, std_threshold: float = 2.0, method="iterative"):
        df_copy = self.detect_outliers_velocity(df.copy(), std_threshold)

        if method == "iterative":
            df_copy = self.iterative_imputation(df_copy)
        elif method == "ml":
            for col in df.columns:
                df_copy = self.impute_with_ml(df_copy, col)

        with open(self.log_file, "w") as f:
            json.dump({col: str(model) for col, model in self.best_models.items()}, f, indent=4)

        return df_copy
