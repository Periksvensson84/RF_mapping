import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import cv2

class DLCPostProcessing:
    def __init__(self, h5_path, excel_path = None):
        #! validate h5_path, excel_path, and their types
        self._read_hdf(h5_path)
        #! self._read_excel(excel_path)

    def _read_hdf(self, h5_path):
        #! validate path and types
        df = pd.read_hdf(h5_path)
        # Flatten the MultiIndex
        df.columns = [f"{bodypart}_{coord}" for bodypart, coord in zip(df.columns.get_level_values(1), df.columns.get_level_values(2))]
        self.df_monofil = df.loc[:, df.columns.str.startswith(('FR', 'FG', 'FB')) & ~df.columns.str.endswith('likelihood')]
        self.df_square = df.loc[:, df.columns.str.startswith(('Top_left', 'Top_right', 'Bottom_left', 'Bottom_right')) & ~df.columns.str.endswith('likelihood')]
        self.df_likelihoods = df.loc[:, df.columns.str.endswith('likelihood')]

    def _read_excel(self, excel_path):
        #! validate path and types
        df = pd.read_excel(excel_path)
        self.df_neuron = df.loc[:, df.columns.str.startswith('Neuron')] #! check excel for col name
        self.df_accelerometer = df.loc[:, df.columns.str.startswith('Accel')] #! check excel for col name

    def get_likelihoods(self):
        overall_average = self.df_likelihoods.mean().mean()
        bodypart_average = self.df_likelihoods.mean()
        return f"Overall average likelihood: \n{overall_average}\n" + \
               f"Bodypart average likelihoods: \n{bodypart_average}"

    def _get_bending_coefficients(self):
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

        # Step 4: Add the bending coefficients as a new column in df_monofil
        self.df_monofil["Bending_Coefficient"] = bending_coefficients

    def plot_bending_angles(self, figsize=(10, 5)):
        self._get_bending_coefficients()
        plt.figure(figsize=figsize)
        plt.plot(self.df_monofil['Bending_Coefficient'], label='Bending Angle')
        plt.xlabel('Frame')
        plt.ylabel('Bending Angle (degrees)')
        plt.title('Bending Angle Over Time')
        plt.legend()
        plt.show()

    def get_bending_peaks(self, threshold=0.14, distance=20):
        self._get_bending_coefficients()
        max_val = self.df_monofil['Bending_Coefficient'].max()
        threshold = threshold * max_val
        peaks, _ = find_peaks(self.df_monofil['Bending_Coefficient'], height=threshold, distance=distance)
        return peaks

    def plot_bending_w_peaks(self, threshold=0.14, distance=20, figsize=(10, 5)):
        peaks = self.get_bending_peaks(threshold, distance)
        plt.figure(figsize=figsize)
        plt.plot(self.df_monofil['Bending_Coefficient'], label='Bending Angle')
        plt.plot(peaks, self.df_monofil['Bending_Coefficient'][peaks], 'x', label='Peaks')
        plt.xlabel('Frame')
        plt.ylabel('Bending Angle (degrees)')
        plt.title('Bending Angle with Peaks')
        plt.legend()
        plt.show()

    def apply_homography(self):
        frames = []
        transformed_monofil_points = []

        for i in range(len(self.df_square)):
            src_points = np.array([
                [self.df_square.iloc[i]['Top_left_x'], self.df_square.iloc[i]['Top_left_y']],
                [self.df_square.iloc[i]['Top_right_x'], self.df_square.iloc[i]['Top_right_y']],
                [self.df_square.iloc[i]['Bottom_right_x'], self.df_square.iloc[i]['Bottom_right_y']],
                [self.df_square.iloc[i]['Bottom_left_x'], self.df_square.iloc[i]['Bottom_left_y']]
            ], dtype=np.float32)

            dst_points = np.array([
                [100, 100],
                [200, 100],
                [200, 200],
                [100, 200]
            ], dtype=np.float32)

            # Find the homography matrix
            H, _ = cv2.findHomography(src_points, dst_points)
            # Create a blank image with white background
            frame = np.ones((300, 300, 3), dtype=np.uint8) * 255
            # Draw the square
            cv2.polylines(frame, [np.int32(dst_points)], isClosed=True, color=(0, 255, 0), thickness=1)

            monofil_points = np.array([
                [self.df_monofil.iloc[i]['FR1_x'], self.df_monofil.iloc[i]['FR1_y']],
                [self.df_monofil.iloc[i]['FR2_x'], self.df_monofil.iloc[i]['FR2_y']],
                [self.df_monofil.iloc[i]['FG1_x'], self.df_monofil.iloc[i]['FG1_y']],
                [self.df_monofil.iloc[i]['FG2_x'], self.df_monofil.iloc[i]['FG2_y']],
                [self.df_monofil.iloc[i]['FB1_x'], self.df_monofil.iloc[i]['FB1_y']],
                [self.df_monofil.iloc[i]['FB2_x'], self.df_monofil.iloc[i]['FB2_y']]
            ], dtype=np.float32)

            # Apply homography to the monofilament points
            monofil_points_transformed = cv2.perspectiveTransform(monofil_points.reshape(-1, 1, 2), H).reshape(-1, 2)
            # Store the transformed points
            transformed_monofil_points.append(monofil_points_transformed.flatten())

            # Draw the transformed monofilament points
            for point in monofil_points_transformed:
                cv2.circle(frame, (int(point[0]), int(point[1])), 2, (255, 0, 0), -1)
            
            # Append the frame to the list
            frames.append(frame)

        columns = ['tf_FR1_x', 'tf_FR1_y', 'tf_FR2_x', 'tf_FR2_y', 'tf_FG1_x', 'tf_FG1_y', 'tf_FG2_x', 'tf_FG2_y', 'tf_FB1_x', 'tf_FB1_y', 'tf_FB2_x', 'tf_FB2_y']
        df_transformed_monofil = pd.DataFrame(transformed_monofil_points, columns=columns)

        return frames, df_transformed_monofil

    def save_video(self, frames, output_path, fps=30):
        height, width, _ = frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        for frame in frames:
            out.write(frame)

        out.release()
        
    #! Next add neuron and accelerometer processing methods, especially plotting
    #! all three together with bending angles.