import pandas as pd
import numpy as np
import cv2
import re

class DataNeuron:
    def __init__(self,
                 xclc_path: str,
                 original_freq: int):

        # Validate path and types
        if not isinstance(xclc_path, str):
            raise ValueError("xclc_path must be a string")
        if not isinstance(original_freq, int):
            raise ValueError("original_freq must be a number")

        self.df = pd.read_excel(xclc_path)
        self.original_freq = original_freq

        # Check for 2 required columns: Time and Spikes
        for column in self.df.columns:
            if re.search(r"Time", column, re.IGNORECASE):
                self.df.rename(columns={column: "Time"}, inplace=True)
            elif re.search(r"Spikes", column, re.IGNORECASE):
                self.df.rename(columns={column: "Spikes"}, inplace=True)
            elif re.search(r"IFF|Freq", column, re.IGNORECASE):
                self.df.rename(columns={column: "IFF"}, inplace=True)
            else:
                pass
        if not all(col in self.df.columns for col in ['Time', 'Spikes']):
            raise ValueError("Required Time and Spikes columns not found in the xclc file")

        # if data is not at a consistent frequency, fill the missing samples
        if self._get_frequency() != original_freq:
            self.fill_samples()
        # Check for optional column: IFF/Freq, if not found, calculate it
        if not 'IFF' in self.df.columns:
            self.calculate_iff()

    def calculate_iff(self):

        # Create Instantaneous Frequency Firing (IFF):
        # 1 divided by the difference between the current time and last spike time
        spikes_loc = self.df[self.df['Spikes'] == 1].index
        self.df["IFF"] = np.nan
        
        for i in range(1, len(spikes_loc)):
            diff = self.df.loc[spikes_loc[i], 'Time'] - \
                   self.df.loc[spikes_loc[i-1], 'Time']
            # calculate the reciprocal of the difference
            self.df.loc[spikes_loc[i], "IFF"] = 1 / diff

        # fill the NaN values with the previous non-NaN value
        self.df["IFF"].fillna(method='ffill', inplace=True)
        # fill the remaining NaN values with 0
        self.df["IFF"].fillna(0, inplace=True)

    def _get_frequency(self):

        time_diffs = np.diff(self.df['Time'])
        # Calculate the frequency as the reciprocal of the mean time difference
        current_freq = 1 / np.mean(time_diffs).round()
        return current_freq

    def fill_samples(self):

        interval = 1 / self.original_freq  # Compute time step based on frequency
        min_time, max_time = 0, self.df['Time'].max()

        # Generate complete range of timestamps
        full_time_range = np.arange(min_time, max_time + interval, interval)

        # Create a new DataFrame with 0s
        full_df = pd.DataFrame({'Time': full_time_range, 'Spikes': 0})

        # Merge with original data, ensuring 1s are preserved
        filled_df = full_df.merge(self.df, on='Time', how="outer", suffixes=('', '_original'))
        filled_df['Spikes'] = filled_df['Spikes' + '_original'].fillna(0).astype(int)
        filled_df = filled_df.drop(columns=['Spikes' + '_original'])

        # Fill IFF column if it exists
        if 'IFF' in filled_df.columns:
            filled_df['IFF'].fillna(method='ffill', inplace=True)
            filled_df['IFF'].fillna(0, inplace=True)

        # Update the DataFrame
        self.df = filled_df

    def downsample(self,
                   target_freq: int):

        # Calculate the downsampling factor
        downsample_factor = int(self.original_freq / target_freq)
        print(downsample_factor)

        # Apply a rolling window with a maximum function to preserve binary components
        downsampled_df = pd.DataFrame()

        # Downsample the time column by selecting the first value in the window
        #downsampled_df['Time'] = \
        #    self.df['Time'].rolling(window=downsample_factor,
        #                                      min_periods=1).min()

        # Downsample the iff/freq column by picking the maximum value in the window
        downsampled_df['IFF'] = \
            self.df['IFF'].rolling(window=downsample_factor,
                                             min_periods=1).max()

        # Downsample the spikes column by summing the values in the window
        downsampled_df['Spikes'] = \
            self.df['Spikes'].rolling(window=downsample_factor,
                                                min_periods=1).sum()

        # Downsample the DataFrame by selecting every downsample_factor-th row
        downsampled_df = downsampled_df.iloc[::downsample_factor]

        # Reset the index to ensure it is sequential
        downsampled_df.reset_index(drop=True, inplace=True)

        # Update the DataFrame
        self.downsampled_df = downsampled_df
        return self.downsampled_df

    def _fill_downsample_length(self,
                                target_length: int):

        # Fill the data up to a target length by forward filling the data
        self.downsampled_df = self.downsampled_df.reindex(range(target_length))
        self.downsampled_df['Spikes'].fillna(0, inplace=True)
        self.downsampled_df['IFF'].ffill(inplace=True)

        return self.downsampled_df
