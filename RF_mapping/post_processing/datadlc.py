import pandas as pd
import numpy as np
import cv2
from outlierimputer import OutlierImputer

class DataDLC:
    homography_points = dst_points = np.array([[100, 100],
                                               [200, 100],
                                               [200, 200],
                                               [100, 200]], dtype=np.float32)

    # This is the initial class, reading the h5 file for DLC data, then
    # getting bending coefficient, homography, plotting, and saving the homography video
    def __init__(self,
                 h5_path: str):
        #! validate h5_path, excel_path, and their types
        df = pd.read_hdf(h5_path)
        # Flatten the MultiIndex
        df.columns = [f"{bodypart}_{coord}" for bodypart, coord in zip(df.columns.get_level_values(1), df.columns.get_level_values(2))]
        self.df_monofil = df.loc[:, df.columns.str.startswith(('FR', 'FG', 'FB')) & ~df.columns.str.endswith('likelihood')]
        self.df_square = df.loc[:, df.columns.str.startswith(('Top_left', 'Top_right', 'Bottom_left', 'Bottom_right')) & ~df.columns.str.endswith('likelihood')]
        self.df_likelihoods = df.loc[:, df.columns.str.endswith('likelihood')]

    def get_likelihoods(self):
        overall_average = self.df_likelihoods.mean().mean()
        bodypart_average = self.df_likelihoods.mean()
        return f"Overall average likelihood: \n{overall_average}\n" + \
               f"Bodypart average likelihoods: \n{bodypart_average}"

    def impute_outliers(self,
                        model: str="HGBR",
                        std_threshold: int|float=1.0,
                        min_samples: int=10,
                        residual_threshold: int|float=0.01,
                        max_trials: int=300,
                        method: str="square_ransac",
                        filament: bool=False):
        #! validate model, std_threshold, min_samples, residual_threshold, max_trials
        # Initialize the OutlierImputer object
        outlier_imputer = OutlierImputer(model=model,
                                         std_threshold=std_threshold,
                                         min_samples=min_samples,
                                         residual_threshold=residual_threshold,
                                         max_trials=max_trials)
        
        # Impute outliers for the square and monofilament points
        self.df_square = outlier_imputer.impute_outliers(self.df_square, method=method)
        if filament:
            self.df_monofil = outlier_imputer.impute_outliers(self.df_monofil, method=method)

    def get_bending_coefficients(self):
        # Initialize a list to store bending coefficients
        bending_coefficients = []

        # Process each row of df_monofil
        for index, row in self.df_monofil.iterrows():
            # Step 1: Extract x and y coordinates for the current row
            x_coords = row.filter(like="_x").values
            y_coords = row.filter(like="_y").values

            # Step 2: Center coordinates around their mean
            x_centered = x_coords - np.mean(x_coords)
            y_centered = y_coords - np.mean(y_coords)

            # Step 3: Fit a polynomial (degree 2)
            degree = 2
            coefficients = np.polyfit(x_centered, y_centered, degree)
            bending_coeff = coefficients[0]  # Coefficient of the quadratic term

            # Store the bending coefficient for this frame
            bending_coefficients.append(abs(bending_coeff))

        # Step 4: Add the bending coefficients as a new series attribute
        self.df_bending_coefficients = pd.Series(bending_coefficients, name='Bending_Coefficient')
        return self.df_bending_coefficients

    def apply_homography(self):
        transformed_monofil_points = []

        for i in range(len(self.df_square)):
            # Find the homography matrix
            h_matrix = self._get_homography_matrix(i)

            monofil_points = np.array([
                [self.df_monofil.iloc[i]['FR1_x'], self.df_monofil.iloc[i]['FR1_y']],
                [self.df_monofil.iloc[i]['FR2_x'], self.df_monofil.iloc[i]['FR2_y']],
                [self.df_monofil.iloc[i]['FG1_x'], self.df_monofil.iloc[i]['FG1_y']],
                [self.df_monofil.iloc[i]['FG2_x'], self.df_monofil.iloc[i]['FG2_y']],
                [self.df_monofil.iloc[i]['FB1_x'], self.df_monofil.iloc[i]['FB1_y']],
                [self.df_monofil.iloc[i]['FB2_x'], self.df_monofil.iloc[i]['FB2_y']]
            ], dtype=np.float32)

            # Apply homography to the monofilament points
            monofil_points_transformed = cv2.perspectiveTransform(
                monofil_points.reshape(-1, 1, 2), h_matrix
                ).reshape(-1, 2)
            # Store the transformed points
            transformed_monofil_points.append(monofil_points_transformed.flatten())

        columns = ['tf_FR1_x', 'tf_FR1_y', 'tf_FR2_x', 'tf_FR2_y', 'tf_FG1_x', 'tf_FG1_y', 'tf_FG2_x', 'tf_FG2_y', 'tf_FB1_x', 'tf_FB1_y', 'tf_FB2_x', 'tf_FB2_y']
        self.df_transformed_monofil = pd.DataFrame(transformed_monofil_points, columns=columns)

        return self.df_transformed_monofil

    def _get_homography_matrix(self, index: int):
        src_points = np.array([
            [self.df_square.iloc[index]['Top_left_x'], self.df_square.iloc[index]['Top_left_y']],
            [self.df_square.iloc[index]['Top_right_x'], self.df_square.iloc[index]['Top_right_y']],
            [self.df_square.iloc[index]['Bottom_right_x'], self.df_square.iloc[index]['Bottom_right_y']],
            [self.df_square.iloc[index]['Bottom_left_x'], self.df_square.iloc[index]['Bottom_left_y']]
        ], dtype=np.float32)

        # Find the homography matrix
        h_matrix, _ = cv2.findHomography(src_points, self.homography_points)
        return h_matrix

    def _merge_data(self):
        # Merge the DataFrames:
        # firstly, merge square and monofil,
        # then merge the transformed monofil
        # then the bending coefficient

        self.df_merged = pd.concat([self.df_square,
                                    self.df_monofil,
                                    self.df_transformed_monofil,
                                    self.df_bending_coefficients], axis=1)
        
        return self.df_merged
        