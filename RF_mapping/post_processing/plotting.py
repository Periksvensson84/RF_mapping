import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
from sklearn.preprocessing import MinMaxScaler
import cv2
from mergeddata import MergedData

class Plotting:
    @staticmethod
    def plot_line(series: pd.Series,
                  xlabel: str, ylabel: str,
                  title: str,
                  figsize: tuple=(12,6)):
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(series, marker='o', linestyle='-', color='b')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        plt.show()

    @staticmethod
    def plot_lines(df: pd.DataFrame,
                   columns: list[str],
                   xlabel: str,
                   ylabel_1: str,
                   ylabel_2: str,
                   title: str,
                   figsize=(12,6)):
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

    @staticmethod
    def plot_homography_animated(homography_points: np.ndarray,
                                 df_transformed_monofil: pd.DataFrame,
                                 filepath: str,
                                 fps: int=30,
                                 figsize = (12,12)):
        #! validation of homography_points and df_transformed_monofil
        #! validation of filepath, specifically ending in .mp4
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_xlim(0, 300)
        ax.set_ylim(0, 300)

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

        # Save the animation as a video file
        try:
            anim.save(filepath, fps=fps, extra_args=['-vcodec', 'libx264'])
        except Exception as e:
            print(f"Error saving animation: {e}")


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
                        figsize=(12, 12)):
        fig, ax = plt.subplots(figsize=figsize)
        df = merged_data.threshold_data(threshold)
        ax.set_xlim(0, 300)
        ax.set_ylim(0, 300)
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

        scatter = ax.scatter(df[x_col], df[y_col],
                            c=colors, s=sizes,
                            alpha=0.5,
                            edgecolors=None,
                            linewidth=0.5)


        # Create a legend for sizes
        handles, labels = scatter.legend_elements(prop="sizes",
                                                  alpha=0.6)
        legend2 = ax.legend(handles,
                            labels,
                            loc="upper right",
                            title="Bending Coefficient Sizes")

        # Add the legend to the plot
        ax.add_artist(legend2)

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
                  title="Spike Status")

        plt.show()

    @staticmethod
    def plot_rf_mapping_animated(merged_data: MergedData,
                                 x_col: str, y_col: str,
                                 bending_col: str, spikes_col: str,
                                 homography_points: np.ndarray,
                                 filepath: str,
                                 xlabel: str = "homography x",
                                 ylabel: str = "homography y",
                                 fps: int = 30, figsize=(12, 12)):
        #! this method should work, but it does take forever to run
        fig, ax = plt.subplots(figsize=figsize)
        df = merged_data.df_merged
        ax.set_xlim(0, 300)
        ax.set_ylim(0, 300)
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
        
        # Save animation
        try:
            anim.save(filepath, fps=fps, extra_args=['-vcodec', 'libx264'])
        except Exception as e:
            print(f"Error saving animation: {e}")


    @staticmethod
    def scatter_ax(ax,
                    dfs: tuple,
                    x_col,
                    y_col,
                    size_col, sizes):
        sns.scatterplot(x=x_col, y=y_col, 
                        size=size_col, sizes=sizes,
                        alpha=0.3, edgecolor=None,
                        data=dfs[0], color='blue', ax=ax)
        sns.scatterplot(x=x_col, y=y_col,
                        size=size_col, sizes=sizes,
                        alpha=0.3, edgecolor=None,
                        data=dfs[1], color='red', ax=ax)
        sns.scatterplot(x=x_col, y=y_col,
                        size=size_col, sizes=sizes,
                        alpha=0.3, edgecolor=None,
                        data=dfs[2], color='green', ax=ax)
        
        # color legend, blue = high bend w neuron, red = high bed w/o neuron, green = low bend w neuron
        legend_elements = [Line2D([0], [0], marker='o', color='w',
                                    markerfacecolor='blue',
                                    markersize=10,
                                    label='High Bend & Neuron'),
                            Line2D([0], [0], marker='o', color='w',
                                    markerfacecolor='red',
                                    markersize=10,
                                    label='High Bend & no Neuron'),
                            Line2D([0], [0], marker='o', color='w',
                                    markerfacecolor='green',
                                    markersize=10,
                                    label='Low Bend & Neuron')]
        
        ax.legend(handles=legend_elements,
                title='Bending Coefficient and Neuron Cases',
                loc='upper left')

    @staticmethod
    def plot_kde_scatter(merged_data: MergedData,
                         x_col: str, y_col: str,
                         homography_points: np.ndarray,
                         kde: bool = True,
                         scatter: bool = True,
                         size = "IFF",
                         title: str = 'KDE and Scatter Plot',
                         xlabel: str = 'X', ylabel: str = 'Y', figsize=(12, 12)):
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_xlim(0, 300)
        ax.set_ylim(0, 300)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        for point in homography_points:
                ax.axhline(y=point[1], color='gray', linestyle='--', alpha=0.5)
                ax.axvline(x=point[0], color='gray', linestyle='--', alpha=0.5)

        if kde:
            sns.kdeplot(x=df[x_col], y=df[y_col],
                        fill=True,
                        cmap='vlag',
                        bw_adjust=0.3,
                        ax=ax)
        if scatter:
            ax.scatter(x_col, y_col,
                       s=df[size], # size based on IFF
                       sizes=np.arange(5, 35, 5), # size range
                       c='gray',
                       alpha=0.35,
                       data=df,
                       edgecolors=None)
            
        # Create a legend for the scatter plot sizes
        size_legend = np.arange(5, 35, 5)  # Example sizes
        legend_elements = [Line2D([0], [0], marker='o', color='w',
                                markerfacecolor='gray',
                                markersize=np.sqrt(s), label=f'{s}')
                        for s in size_legend]
        ax.legend(handles=legend_elements, title=size, loc='upper right')

        plt.show()
        return fig, ax

    @staticmethod
    def plot_kde_scatter_over_frame(merged_data: MergedData,
                                    h_matrix: np.ndarray,
                                    video_path: str,
                                    index: int,
                                    x_col: str, y_col: str,
                                    homography_points: np.ndarray,
                                    kde: bool = True,
                                    scatter: bool = True,
                                    size: str = "IFF",
                                    title: str = 'KDE and Scatter Plot',
                                    xlabel: str = 'X', ylabel: str = 'Y',
                                    figsize=(12, 12)):
        
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

        # Warp the frame using the homography matrix
        h, w, _ = frame.shape
        frame_transformed = cv2.warpPerspective(frame, h_matrix, (w, h))

        # Create figure and axis
        fig, ax = plt.subplots(figsize=figsize)
        
        # Show the transformed frame in the background
        ax.imshow(frame_transformed)

        # Overlay KDE + scatter plot
        if kde:
            sns.kdeplot(x=df[x_col], y=df[y_col],
                        fill=True,
                        cmap='vlag',
                        bw_adjust=0.3,
                        ax=ax,
                        alpha=0.5)
        if scatter:
            sizes = np.arange(5, 35, 5)
            ax.scatter(df[x_col], df[y_col],
                        s=df[size],  # Size based on IFF
                        sizes=sizes, # Size range
                        color='gray',
                        alpha=0.5,
                        label="Scatter Data",
                        edgecolors=None)
            
            # Create a legend for the scatter plot sizes
            legend_elements = [Line2D([0], [0], marker='o', color='w',
                                      markerfacecolor='gray',
                                      markersize=np.sqrt(s),
                                      label=f'{s}')
                               for s in sizes]
            ax.legend(handles=legend_elements, title=size, loc='upper right')

        ax.set_xlim(0, 300)
        ax.set_ylim(0, 300)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        for point in homography_points:
                    ax.axhline(y=point[1], color='gray', linestyle='--', alpha=0.5)
                    ax.axvline(x=point[0], color='gray', linestyle='--', alpha=0.5)

        plt.show()
        return fig, ax
