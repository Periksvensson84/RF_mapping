import numpy as np
import pandas as pd
from sklearn.linear_model import RANSACRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from validation import Validation as Val
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

class OutlierImputer:
    # Define the pairs of points that form the square and filament
    # for later angle calculations, easier to modify
    square_pairs = [
        [0, 1, 2, 3],
        [2, 3, 6, 7],
        [6, 7, 4, 5],
        [4, 5, 0, 1]
    ]

    filament_pairs = [
        [0, 1, 2, 3],
        [2, 3, 4, 5],
        [4, 5, 6, 7],
        [6, 7, 8, 9],
        [8, 9, 10, 11]
    ]

    models = {
            "RFR": RandomForestRegressor(n_estimators=100, random_state=42),
            "HGBR": HistGradientBoostingRegressor(),
            "KNR": KNeighborsRegressor(n_neighbors=10)
        }
    methods = ["square_std", "square_ransac", "filament"]

    def __init__(self,
                 model: str = "RFR",
                 std_threshold: int|float = 1.0,
                 min_samples: int = 10,
                 residual_threshold: float = 0.01,
                 max_trials: int = 300):

        self.model = model
        self.std_threshold = std_threshold
        self.min_samples = min_samples
        self.residual_threshold = residual_threshold
        self.max_trials = max_trials

    @staticmethod
    def compute_angles(df: pd.DataFrame,
                       pairs: list):

        angles = []
        for pair in pairs:
            x1, y1, x2, y2 = df.iloc[:, pair[0]], \
                             df.iloc[:, pair[1]], \
                             df.iloc[:, pair[2]], \
                             df.iloc[:, pair[3]]
            # Compute the angle between the two points relative to the x-axis
            angles.append(np.arctan2(y2 - y1, x2 - x1))
        return np.column_stack(angles)

    def detect_outliers_square_std(self,
                                   df: pd.DataFrame,
                                   pairs: list):

        # Compute the angles between the square points
        angles = self.compute_angles(df, pairs)
        angle_df = pd.DataFrame(angles, columns=[
            f"angle_{i+1}" for i in range(angles.shape[1])
            ], index=df.index)
        # Compute the mean and standard deviation of the angles
        angle_means, angle_stds = angle_df.mean(), angle_df.std()

        # Detect outliers based on the standard deviation threshold
        for i, pair in enumerate(pairs):
            outlier_mask = \
                (angle_df.iloc[:, i] - angle_means.iloc[i]).abs() > \
                self.std_threshold * angle_stds.iloc[i]
            df.loc[outlier_mask, [df.columns[pair[2]],
                                  df.columns[pair[3]]]] = np.nan

        return df

    def detect_outliers_ransac(self,
                               df: pd.DataFrame,
                               pairs: list):
        # Compute the angles between the square points
        angles = self.compute_angles(df, pairs)
        angle_df = pd.DataFrame(angles,
                                columns=[f"angle_{i+1}" for i in range(angles.shape[1])],
                                index=df.index)
        # Initialize the inlier mask
        inlier_mask = np.ones(len(df), dtype=bool)

        # Fit RANSAC regressor to each angle column
        for col in angle_df.columns:
            ransac = RANSACRegressor(min_samples=self.min_samples,
                                     residual_threshold=self.residual_threshold,
                                     max_trials=self.max_trials)
            ransac.fit(np.arange(len(df)).reshape(-1, 1), angle_df[col])
            # Update the inlier mask
            inlier_mask &= ransac.inlier_mask_

        # Detect outliers based on the inlier mask
        df.loc[~inlier_mask, df.columns[:8]] = np.nan
        return df

    def impute_with_ml(self,
                       df: pd.DataFrame,
                       target_col: str):
        # Split the DataFrame into training and testing sets based on
        # the target column
        df_copy, train_df, test_df = df.copy(), \
                                     df.dropna(), \
                                     df[df[target_col].isna()]
        if test_df.empty:
            return df_copy

        # Define the feature columns and target column
        feature_cols = [col for col in df.columns if col != target_col]
        # Extract the training data
        X_train, y_train = train_df[feature_cols], train_df[target_col]

        # Select regression model
        if self.model not in self.models:
            raise ValueError(f"Invalid model name: {self.model}, use RFR, HGBR, or KNR")
        ml_model = self.models[self.model]
        # Fit the model
        ml_model.fit(X_train, y_train)
    
        # Impute the missing values
        df_copy.loc[df_copy[target_col].isna(), target_col] = \
            ml_model.predict(test_df[feature_cols])
        return df_copy

    def impute_outliers(self,
                        df: pd.DataFrame,
                        method:str ="square_ransac"):
        # Make a copy of the DataFrame and drop rows with all NaN values
        # to avoid errors in the outlier detection
        df_copy, valid_rows = df.copy(), df.dropna(how="all")
        if valid_rows.empty:
            return df_copy

        if method == "square_std":
            # Detect outliers based on the standard deviation threshold
            valid_rows = self.detect_outliers_square_std(valid_rows,
                                                         self.square_pairs)
        elif method == "square_ransac":
            # Detect outliers based on RANSAC regression square points
            valid_rows = self.detect_outliers_ransac(valid_rows,
                                                     self.square_pairs)
        elif method == "filament":
            # Detect outliers based on RANSAC regression for filament points
            valid_rows = self.detect_outliers_ransac(valid_rows,
                                                     self.filament_pairs)

        for col in df.columns:
            # Impute missing values using machine learning
            valid_rows = self.impute_with_ml(valid_rows, col)

        # Update the DataFrame with the imputed values
        df_copy.update(valid_rows.astype(np.float32))
        return df_copy
