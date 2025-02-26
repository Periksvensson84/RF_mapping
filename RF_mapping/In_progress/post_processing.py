import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D
import seaborn as sns
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
        #! DRY, try collapsing the simple plotting methods
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
        #! DRY, try collapsing the simple plotting methods
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
        self.df_transformed_monofil = pd.DataFrame(transformed_monofil_points, columns=columns)

        return frames, self.df_transformed_monofil

    def save_video(self, frames, output_path, fps=30):
        height, width, _ = frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        for frame in frames:
            out.write(frame)

        out.release()



    def _xlsx_processing(self, xlsx_file, original_freq = 1000, target_freq = 30):
        # Read the Excel file
        df_nerve = pd.read_excel(xlsx_file)
        
        # Extract the relevant columns #! subject to change
        df_nerve = df_nerve.loc[:, ["Spikes nw-1-01", "5 Accelerometer", "Time"]]
        df_nerve.columns = ["Nerve", "Accelerometer", "Time"]
        df_nerve.dropna(inplace=True, subset=["Nerve", "Accelerometer", "Time"])

        # Calculate the downsampling factor
        downsample_factor = int(original_freq / target_freq)
        # Apply a rolling window with a maximum function to preserve binary components
        downsampled_df = df_nerve.rolling(window=downsample_factor, min_periods=1).max()
        # Downsample the DataFrame by selecting every downsample_factor-th row
        downsampled_df = downsampled_df.iloc[::downsample_factor]

        self.df_external = downsampled_df
        return downsampled_df

    def plot_accelerometer_and_neuron_w_peaks(self, figsize=(10, 5), show_peaks=False):
        fig, ax1 = plt.subplots(figsize=figsize)

        # Plot accelerometer data
        color = 'tab:blue'
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Acceleration', color=color)
        ax1.plot(self.df_external['Time'],
                 self.df_external['Accelerometer'],
                 color=color, label='Accelerometer')
        ax1.tick_params(axis='y', labelcolor=color)

        if show_peaks:
            # Find peaks in the accelerometer data
            peaks, _ = find_peaks(-self.df_external['Accelerometer'],
                                prominence=0.14,
                                distance=20)
            ax1.plot(self.df_external['Time'].iloc[peaks],
                    self.df_external['Accelerometer'].iloc[peaks],
                    'x', color='green', label='contact')

        # Create a second y-axis to plot neuron activity
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('Neuron Activity', color=color)
        ax2.plot(self.df_external['Time'],
                 self.df_external['Nerve'],
                 color=color, label='Neuron')
        ax2.tick_params(axis='y', labelcolor=color)

        # Add legends
        fig.tight_layout()
        fig.legend(loc='upper right',
                   bbox_to_anchor=(1, 1),
                   bbox_transform=ax1.transAxes)

        plt.title('Accelerometer Data and Neuron Activity Over Time')
        plt.show()

    def _synchronize(self):
        #! Implement this method such that it syncs bending coefficient with
        #! neuron and accelerometer data. This should involve using the 5
        #! starting stops in the neuron data to align the data.
        
        #! In the end, return df with neuron and accelerometer data aligned with
        #! bending coefficient data in addition to the transformed monofilament
        #! tip point.
        df_rf_mapping.columns = ['tf_FB2_x',
                                 'tf_FB2_y',
                                 'bending_coefficient',
                                 'neuron',
                                 'accelerometer']
        return df_rf_mapping

    def _plot_rf_mapping(self,
                         ax,
                         high_bending_neuron_on,
                         high_bending_neuron_off,
                         low_bending_neuron_on):
        sns.scatterplot(x='tf_FB2_x', y='tf_FB2_y',
                        size='bending_coefficient',
                        sizes=(50, 200), alpha=0.5,
                        edgecolor=None,
                        data=high_bending_neuron_on,
                        color='red', ax=ax)
        sns.scatterplot(x='tf_FB2_x', y='tf_FB2_y',
                        size='bending_coefficient',
                        sizes=(50, 200), alpha=0.5,
                        edgecolor=None,
                        data=high_bending_neuron_off,
                        color='blue', ax=ax)
        sns.scatterplot(x='tf_FB2_x', y='tf_FB2_y',
                        size='bending_coefficient',
                        sizes=(50, 200), alpha=0.5,
                        edgecolor=None,
                        data=low_bending_neuron_on,
                        color='green', ax=ax)
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
                   markersize=10, label='High Bending, Neuron On'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='blue',
                   markersize=10, label='High Bending, Neuron Off'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='green',
                   markersize=10, label='Low Bending, Neuron On')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        ax.set_title('RF Mapping of Neuron Activity')
        ax.set_xlabel('tf_FB2_x')
        ax.set_ylabel('tf_FB2_y')
        ax.invert_yaxis()

    def plot_rf_mapping(self, threshold=0.14, figsize=(15, 10)):
        threshold = threshold * self.df_rf_mapping['bending_coefficient'].max()
        self.df_rf_mapping = self._synchronize()
        high_bending_neuron_on = self.df_rf_mapping[
            (self.df_rf_mapping['bending_coefficient'] > threshold) &
            (self.df_rf_mapping['neuron'] == 1)
        ]
        high_bending_neuron_off = self.df_rf_mapping[
            (self.df_rf_mapping['bending_coefficient'] > threshold) &
            (self.df_rf_mapping['neuron'] == 0)
        ]
        low_bending_neuron_on = self.df_rf_mapping[
            (self.df_rf_mapping['bending_coefficient'] <= threshold) &
            (self.df_rf_mapping['neuron'] == 1)
        ]

        fig, ax = plt.subplots(figsize=figsize)
        self._plot_rf_mapping(ax,
                              high_bending_neuron_on,
                              high_bending_neuron_off,
                              low_bending_neuron_on)
        plt.show()

    def plot_realtime_rf_mapping(self,
                                 threshold=0.14,
                                 output_path='animation.gif',
                                 fps=30,
                                 figsize=(10, 10)):
        threshold = threshold * self.df_rf_mapping['bending_coefficient'].max()
        self.df_rf_mapping = self._synchronize()
        high_bending_neuron_on = self.df_rf_mapping[
            (self.df_rf_mapping['bending_coefficient'] > threshold) &
            (self.df_rf_mapping['neuron'] == 1)
        ]
        high_bending_neuron_off = self.df_rf_mapping[
            (self.df_rf_mapping['bending_coefficient'] > threshold) &
            (self.df_rf_mapping['neuron'] == 0)
        ]
        low_bending_neuron_on = self.df_rf_mapping[
            (self.df_rf_mapping['bending_coefficient'] <= threshold) &
            (self.df_rf_mapping['neuron'] == 1)
        ]

        fig, ax = plt.subplots(figsize=figsize)

        def init():
            ax.clear()
            self._plot_rf_mapping(ax,
                                  high_bending_neuron_on.iloc[:1],
                                  high_bending_neuron_off.iloc[:1],
                                  low_bending_neuron_on.iloc[:1])
            return ax

        def update(frame):
            ax.clear()
            self._plot_rf_mapping(ax,
                                  high_bending_neuron_on.iloc[:frame],
                                  high_bending_neuron_off.iloc[:frame],
                                  low_bending_neuron_on.iloc[:frame])
            return ax

        ani = FuncAnimation(fig,
                            update,
                            frames=len(self.df_rf_mapping),
                            init_func=init, blit=False)
        ani.save(output_path, writer=PillowWriter(fps=fps))
        plt.show()
