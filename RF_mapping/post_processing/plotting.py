import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
from sklearn.preprocessing import MinMaxScaler
import cv2
from mergeddata import MergedData
from validation import Validation as Val

class Plotting:
    @staticmethod
    def _get_lim():
        return -100, 300

    @staticmethod
    def plot_line(series: pd.Series,
                  xlabel: str,
                  ylabel: str,
                  title: str,
                  figsize: tuple[int] = (12,6)):
        Val.validate_type(series, pd.Series, "Series")
        Val.validate_strings(xlabel=xlabel, ylabel=ylabel, title=title)

        try:
            fig, ax = plt.subplots(figsize=figsize)
            ax.plot(series, marker='o', linestyle='-', color='b')
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            plt.show()
        except Exception as e:
            raise Exception(f"Error plotting line: {e}")

    @staticmethod
    def plot_lines(df: pd.DataFrame,
                   columns: list[str],
                   xlabel: str,
                   ylabel_1: str,
                   ylabel_2: str,
                   title: str,
                   figsize: tuple[int] = (12,6)):
        Val.validate_dataframe(df, required_columns=columns, name="DataFrame")
        Val.validate_type(df, pd.DataFrame, "DataFrame")
        Val.validate_strings(xlabel=xlabel, ylabel_1=ylabel_1,
                             ylabel_2=ylabel_2, title=title)

        try:
            # plot two axes on the same plot
            fig, ax1 = plt.subplots(figsize=figsize)
            ax2 = ax1.twinx()

            ax1.set_xlabel(xlabel)
            ax1.set_ylabel(ylabel_1, color='b')
            ax1.tick_params(axis='y', labelcolor='b')
            ax2.set_ylabel(ylabel_2, color='r')
            ax2.tick_params(axis='y', labelcolor='r')
            ax1.set_title(title)

            ax1.plot(df[columns[0]], linestyle='-', color='tab:blue')
            ax2.plot(df[columns[1]], linestyle='-', color='tab:red')

            fig.tight_layout()
            fig.legend(loc='upper right', bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)

            plt.title(title)
            plt.show()
        except Exception as e:
            raise Exception(f"Error plotting lines: {e}")

    @staticmethod
    def plot_homography_animated(homography_points: np.ndarray,
                                 df_transformed_monofil: pd.DataFrame,
                                 filepath: str,
                                 fps: int=30,
                                 figsize: tuple[int] = (12,12)):
        Val.validate_array(homography_points, shape=(4,2), name="Homography Points")
        Val.validate_type(df_transformed_monofil, pd.DataFrame, "DataFrame")
        Val.validate_path(filepath, file_types=[".mp4", ".avi"])
        Val.validate_type(fps, int, "FPS")
        Val.validate_positive(fps, "FPS")

        try:
            fig, ax = plt.subplots(figsize=figsize)
            ax.set_xlim(Plotting._get_lim())
            ax.set_ylim(Plotting._get_lim())
            ax.set_xlabel('homography x (mm)')
            ax.set_ylabel('homography y (mm)')

            # Plot destination points
            for point in homography_points:
                ax.axhline(y=point[1], color='gray', linestyle='--', alpha=0.5)
                ax.axvline(x=point[0], color='gray', linestyle='--', alpha=0.5)

            line, = ax.plot([], [], 'bo-')

            def init():
                line.set_data([], [])
                return line,

            def update(frame):
                points = df_transformed_monofil.iloc[frame].values.reshape(-1, 2)
                line.set_data(points[:, 0], points[:, 1])
                return line,

            anim = FuncAnimation(fig, update, frames=len(df_transformed_monofil), init_func=init, blit=True, interval=1000/fps)
            plt.show()
            anim.save(filepath, fps=fps, extra_args=['-vcodec', 'libx264'])
        except Exception as e:
            raise Exception(f"Error creating animation: {e}")


    # For synchronized data
    @staticmethod
    def plot_rf_mapping(merged_data: MergedData,
                        x_col: str, y_col: str,
                        bending_col: str,
                        spikes_col: str,
                        homography_points: np.ndarray,
                        threshold: float = 0.14,
                        title: str = 'RF Mapping of Neuron Activity',
                        xlabel: str = "homography x",
                        ylabel: str = "homography y",
                        figsize: tuple[int] = (12, 12)):
        Val.validate_type(merged_data, MergedData, "MergedData")
        Val.validate_strings(x_col=x_col, y_col=y_col,
                             bending_col=bending_col, spikes_col=spikes_col,
                             xlabel=xlabel, ylabel=ylabel, title=title)
        Val.validate_array(homography_points, shape=(4,2), name="Homography Points")
        Val.validate_type(threshold, float, "Threshold")
        Val.validate_positive(threshold, "Threshold")

        try:
            fig, ax = plt.subplots(figsize=figsize)
            df = merged_data.threshold_data(threshold)
            ax.set_xlim(Plotting._get_lim())
            ax.set_ylim(Plotting._get_lim())
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)

            for point in homography_points:
                ax.axhline(y=point[1], color='gray', linestyle='--', alpha=0.5)
                ax.axvline(x=point[0], color='gray', linestyle='--', alpha=0.5)

            # Normalize the bending_col values to a range of sizes
            norm = plt.Normalize(df[bending_col].min(),
                                df[bending_col].max())
            sizes = norm(df[bending_col]) * 200  # Scale to a range of sizes

            # Define colors for spikes (0 and 1)
            colors = df[spikes_col].apply(lambda x: 'red' if x > 0 else 'blue')

            ax.scatter(df[x_col], df[y_col],
                    c=colors, s=sizes,
                    alpha=0.5,
                    edgecolors=None,
                    linewidth=0.5)

            # Add custom legend for colors
            from matplotlib.lines import Line2D
            custom_lines = [Line2D([0], [0],
                                marker='o', color='w',
                                markerfacecolor='blue',
                                markersize=10,
                                label='No Spike'),
                            Line2D([0], [0],
                                marker='o', color='w',
                                markerfacecolor='red',
                                markersize=10,
                                label='Spike')]
            ax.legend(handles=custom_lines,
                    loc="upper left",
                    title="Neuron Spike Status\n(Circle Size ∝ Bending Coefficient)")

            plt.show()
        except Exception as e:
            raise Exception(f"Error plotting RF mapping: {e}")

    @staticmethod
    def plot_rf_mapping_animated(merged_data: MergedData,
                                 x_col: str, y_col: str,
                                 bending_col: str,
                                 spikes_col: str,
                                 homography_points: np.ndarray,
                                 filepath: str,
                                 xlabel: str = "homography x",
                                 ylabel: str = "homography y",
                                 fps: int = 30,
                                 figsize: tuple[int] = (12, 12)):
        Val.validate_type(merged_data, MergedData, "MergedData")
        Val.validate_strings(x_col=x_col, y_col=y_col,
                            bending_col=bending_col, spikes_col=spikes_col,
                            xlabel=xlabel, ylabel=ylabel)
        Val.validate_array(homography_points, shape=(4,2), name="Homography Points")
        Val.validate_path(filepath, file_types=[".mp4", ".avi"])
        Val.validate_type(fps, int, "FPS")
        Val.validate_positive(fps, "FPS")

        try:
            fig, ax = plt.subplots(figsize=figsize)
            df = merged_data.threshold_data(0.14)
            ax.set_xlim(Plotting._get_lim())
            ax.set_ylim(Plotting._get_lim())
            ax.set_title('RF Mapping Animation')
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)

            # Plot reference grid (homography points)
            for point in homography_points:
                ax.axhline(y=point[1], color='gray', linestyle='--', alpha=0.5)
                ax.axvline(x=point[0], color='gray', linestyle='--', alpha=0.5)

            # Normalize bending coefficient for size scaling
            scaler = MinMaxScaler(feature_range=(5, 30))  # Ensures reasonable dot sizes
            df["scaled_size"] = scaler.fit_transform(df[[bending_col]])

            # Map spike values (0 = blue, 1 = red)
            color_map = df[spikes_col].apply(lambda x: 'red' if x > 0 else 'blue')

            # Store drawn circles
            circles = []

            def update(frame):
                """Adds a new circle to the plot each frame."""
                current_row = df.iloc[frame]  # Get data for current frame

                x, y = current_row[x_col], current_row[y_col]
                size = current_row["scaled_size"] * 0.15  # Scale down for better visibility
                color = color_map.iloc[frame]

                # Create and add circle
                circle = plt.Circle((x, y), size, color=color, alpha=0.5, edgecolor="k", linewidth=0.5)
                ax.add_patch(circle)
                circles.append(circle)

                return circle,

            # Create animation
            anim = FuncAnimation(fig, update, frames=len(df), interval=1000/fps, blit=False)

            # Static legend for spike colors with a note about size
            color_legend = [Line2D([0], [0], marker='o', color='w',
                                    markerfacecolor='blue', markersize=10, label='No Spike'),
                            Line2D([0], [0], marker='o', color='w',
                                    markerfacecolor='red', markersize=10, label='Spike')]
            spike_legend = ax.legend(handles=color_legend, loc="upper left", title="Spike Status\n(Circle Size ∝ Bending Coefficient)")

            plt.show()
            anim.save(filepath, fps=fps, extra_args=['-vcodec', 'libx264'])
        except Exception as e:
            raise Exception(f"Error creating animation: {e}")

    # Used by the below methods
    @staticmethod
    def scatter_ax(ax: plt.Axes,
                   dfs: tuple,
                   x_col: str,
                   y_col: str,
                   size_col: str):
        Val.validate_type(ax, plt.Axes, "Axes")
        Val.validate_type(dfs, tuple, "DataFrame Columns")
        Val.validate_strings(x_col=x_col, y_col=y_col, size_col=size_col)

        sns.scatterplot(x=x_col, y=y_col, 
                        size=size_col, sizes=(25, 50), # Enlarge the sizes
                        alpha=0.3, edgecolor=None,
                        data=dfs[0], color='blue', ax=ax)
        sns.scatterplot(x=x_col, y=y_col,
                        size=size_col, sizes=(25, 50), # Enlarge the sizes
                        alpha=0.3, edgecolor=None,
                        data=dfs[1], color='red', ax=ax)
        sns.scatterplot(x=x_col, y=y_col,
                        size=size_col, sizes=(25, 50), # Enlarge the sizes
                        alpha=0.3, edgecolor=None,
                        data=dfs[2], color='green', ax=ax)
        
        # color legend, blue = high bend w neuron, red = high bed w/o neuron, green = low bend w neuron
        legend_elements = [Line2D([0], [0], marker='o', color='w',
                                    markerfacecolor='blue',
                                    markersize=10,
                                    label='High Bend & Neuron Spike'),
                            Line2D([0], [0], marker='o', color='w',
                                    markerfacecolor='red',
                                    markersize=10,
                                    label='High Bend & no Neuron Spike'),
                            Line2D([0], [0], marker='o', color='w',
                                    markerfacecolor='green',
                                    markersize=10,
                                    label='Low Bend & Neuron Spike')]
        
        legend = ax.legend(handles=legend_elements,
                        title=f'Bending Coefficient and Neuron Cases\n(Circle Size ∝ {size_col})',
                        loc='upper left')
        return legend

    @staticmethod
    def plot_kde_scatter(merged_data: MergedData,
                        x_col: str, y_col: str,
                        homography_points: np.ndarray,
                        kde: bool = True,
                        scatter: bool = True,
                        size: str = "IFF",
                        title: str = 'KDE and Scatter Plot',
                        xlabel: str = 'X', ylabel: str = 'Y',
                        figsize: tuple[int] = (12, 12),
                        # Required inputs for frame overlay:
                        frame: bool = False,
                        video_path: str = None,
                        index: int = None):
        Val.validate_type(merged_data, MergedData, "MergedData")
        Val.validate_array(homography_points, shape=(4,2), name="Homography Points")
        Val.validate_strings(x_col=x_col, y_col=y_col,
                             size=size, title=title,
                             xlabel=xlabel, ylabel=ylabel)
        Val.validate_type(kde, bool, "KDE")
        Val.validate_type(scatter, bool, "Scatter")
        Val.validate_type(frame, bool, "Frame")
        if frame:
            Val.validate_path(video_path, file_types=[".mp4", ".avi"])
            Val.validate_type(index, int, "Index")
            Val.validate_positive(index, "Index", zero_allowed=True)

        fig, ax = plt.subplots(figsize=figsize)

        if frame:
            # get h_matrix for the frame
            dst_min, dst_max = 300, 500
            dst_points = np.array([[dst_min, dst_max],
                                [dst_max, dst_max],
                                [dst_max, dst_min],
                                [dst_min, dst_min]])
            h_matrix = merged_data.dlc._get_homography_matrix(index, dst_points)

            if video_path is None or index is None:
                raise ValueError("video_path, and index must be provided when frame is True")

            # Load the video frame
            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ret, frame = cap.read()
            cap.release()

            if not ret:
                print("Error: Could not read video frame.")
                return

            # Convert BGR (OpenCV) to RGB (Matplotlib)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Get original frame dimensions
            h, w, _ = frame.shape

            # Warp the frame using the homography matrix
            frame_transformed = cv2.warpPerspective(frame, h_matrix, (w, h))

            # Plot the transformed frame
            change = -300
            # Offset to center the frame in the plot due to different h_matrix dst_points
            ax.imshow(frame_transformed, extent=[change, w+change, h+change, change])

        # Set proper plot limits
        ax.set_xlim(Plotting._get_lim())
        ax.set_ylim(Plotting._get_lim())
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        for point in homography_points:
            ax.axhline(y=point[1], color='gray', linestyle='--', alpha=0.5)
            ax.axvline(x=point[0], color='gray', linestyle='--', alpha=0.5)

        if kde:
            df = merged_data.threshold_data(0.14)
            sns.kdeplot(x=df[x_col], y=df[y_col],
                        fill=True,
                        cmap='vlag',
                        bw_adjust=0.3,
                        ax=ax,
                        alpha=0.5)
        if scatter:
            legend = Plotting.scatter_ax(ax,
                                            merged_data.plotting_split(0.14),
                                            x_col, y_col,
                                            size)
            ax.add_artist(legend)

        plt.show()
        return fig, ax

    @staticmethod
    def plot_scroll_over_video(merged_data: pd.DataFrame,
                               columns: list[str],
                               video_path: str,
                               output_path: str):
        Val.validate_type(merged_data, pd.DataFrame, "Merged Data")
        Val.validate_type_in_list(columns, str, "Columns")
        Val.validate_path(video_path, file_types=[".mp4", ".avi"])
        Val.validate_path(output_path, file_types=[".mp4", ".avi"])
        
        df_merged = merged_data.df_merged.copy()

        # Load the video
        cap = cv2.VideoCapture(video_path)
        frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Set scrolling plot height to 1/5th of video height
        scroll_height = frame_height // 5  
        figsize = (frame_width / 100, scroll_height / 100)

        # Video writer setup
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, frame_rate, (frame_width, frame_height + scroll_height))

        window_size = 100  # Number of data points visible in scroll window

        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Create figure for scrolling plot
            fig_scroll, ax_scroll = plt.subplots(len(columns), 1, figsize=figsize, sharex=True)
            if len(columns) == 1:
                ax_scroll = [ax_scroll]

            start_idx = max(0, frame_idx - window_size // 2)
            end_idx = min(len(df_merged), start_idx + window_size)
            data_window = df_merged.iloc[start_idx:end_idx]


            if end_idx == len(df_merged):
                end_xlim = frame_idx + 50
            else:
                end_xlim = end_idx
            if start_idx == 0:
                start_xlim = -50 + frame_idx
                end_xlim = start_xlim + 100
            else:
                start_xlim = start_idx

            for i, col in enumerate(columns):
                ax = ax_scroll[i]
                ax.clear()
                ax.plot(data_window.index, data_window[col], label=col, color=f'C{i}')
                ax.set_ylim(0, df_merged[col].max())
                ax.set_xlim(start_xlim, end_xlim)
                ax.set_ylabel(col, color=f'C{i}')
                ax.axvline(min(start_idx, start_xlim) + window_size // 2, color='black', linestyle='--')
                ax.xaxis.set_visible(False)  # Hide x-axis numbers

            # Convert Matplotlib figure to an image
            fig_scroll.canvas.draw()
            scroll_img = np.array(fig_scroll.canvas.renderer.buffer_rgba())[:, :, :3]  # Convert to RGB
            scroll_img = cv2.cvtColor(scroll_img, cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV
            plt.close(fig_scroll)

            # Resize and combine with video frame
            combined_frame = np.vstack((scroll_img, frame))
            #combined_frame = cv2.cvtColor(combined_frame, cv2.COLOR_BGR2RGB)  # Convert to RGB for display
            out.write(combined_frame)

            frame_idx += 1

        cap.release()
        out.release()
