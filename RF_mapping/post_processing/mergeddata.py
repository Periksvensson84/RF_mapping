# this class is meant to merge the data from DataDLC and DataNeuron objects
# with respect to the time index or peaks
# Or if they're already the same length of time, just merge them directly
from datadlc import DataDLC
from dataneuron import DataNeuron
import pandas as pd
from validation import Validation as Val
from scipy.signal import correlate

#! implement proper synchonrizing of the data
#! use the first 10 neuron spikes as sync points

class MergedData:
    def __init__(self,
                 dlc: DataDLC,
                 neuron: DataNeuron,
                 max_gap_fill: int = 10):

        Val.validate_type(dlc, DataDLC, "DLC Object")
        Val.validate_type(neuron, DataNeuron, "Neuron Object")
        Val.validate_type(max_gap_fill, int, "Max Gap Fill")

        self.dlc = dlc
        self.neuron = neuron
        self.max_gap_fill = max_gap_fill
        self.df_merged = None
        self._merge()

    def _merge(self) -> pd.DataFrame:
        # Merge the DataFrames
        df_dlc = self.dlc._merge_data()
        df_neuron = self._fill_neuron_up()
        
        # Create binary threshold columns
        threshold = 0.15 * df_dlc['Bending_Coefficient'].max()
        df_dlc['Bending_Binary'] = (df_dlc['Bending_Coefficient'] > threshold).astype(int)
        
        # Fill gaps in neuron Spikes column with dynamic width
        df_neuron['Spikes_Filled'] = df_neuron['Spikes'].copy()
        gap_start = None
        for i in range(len(df_neuron)):
            if df_neuron['Spikes'][i] == 1:
                if gap_start is not None and (i - gap_start) <= self.max_gap_fill:
                    df_neuron['Spikes_Filled'][gap_start:i] = 1
                gap_start = i + 1
            elif df_neuron['Spikes'][i] == 0 and gap_start is None:
                gap_start = i
        
        # Perform sequence alignment using cross-correlation
        correlation = correlate(df_dlc['Bending_Binary'], df_neuron['Spikes_Filled'], mode='full')
        best_shift = correlation.argmax() - (len(df_neuron) - 1)
        
        # Shift df_neuron index accordingly
        df_neuron = df_neuron.shift(periods=best_shift).reset_index(drop=True)
        
        # Merge the DataFrames
        self.df_merged = pd.concat([df_dlc, df_neuron], axis=1)
        return self.df_merged

    def _fill_neuron_up(self) -> pd.DataFrame:
        # Fill the neuron data up to the length of the dlc data
        target_length = len(self.dlc.df_square)
        self.neuron._fill_downsample_length(target_length)
        return self.neuron.downsampled_df

    def _get_threshold(self,
                       threshold: float) -> float:
        Val.validate_type(threshold, float, "Threshold")
        Val.validate_positive(threshold, "Threshold")

        return threshold * self.df_merged['Bending_Coefficient'].max()

    def plotting_split(self,
                       threshold: float) -> tuple:
        #! validation of threshold done within _get_threshold()

        # split the data into 3 parts:
        # high bending_coefficient with neuron firing
        # high bending_coefficient without neuron firing
        # low bending_coefficient with neuron firing
        # low bending_coefficient without neuron firing will get sorted out
        threshold = self._get_threshold(threshold)
        
        # high bending_coefficient with neuron firing
        high_bend_w_neuron = \
            self.df_merged[(self.df_merged['Bending_Coefficient'] > threshold) \
                & (self.df_merged['Spikes'] >= 1)]
        high_bend_wo_neuron = \
            self.df_merged[(self.df_merged['Bending_Coefficient'] > threshold) \
                & (self.df_merged['Spikes'] == 0)]
        low_bend_w_neuron = \
            self.df_merged[(self.df_merged['Bending_Coefficient'] <= threshold) \
                & (self.df_merged['Spikes'] >= 1)]
        
        return high_bend_w_neuron, high_bend_wo_neuron, low_bend_w_neuron

    def threshold_data(self,
                       threshold: float) -> pd.DataFrame:
        #! validation of threshold done within _get_threshold()

        # Cleaned data where the bending coefficient is above the threshold
        # or the Spikes column is not 0
        threshold = self._get_threshold(threshold)
        self.df_merged_cleaned = \
            self.df_merged[(self.df_merged['Bending_Coefficient'] > threshold) \
                | (self.df_merged['Spikes'] != 0)]
        return self.df_merged_cleaned

    def _save_data(self,
                   df: pd.DataFrame,
                   path: str,
                   file_format: str):
        Val.validate_type(df, pd.DataFrame, "DataFrame")
        Val.validate_type(path, str, "Path")
        Val.validate_type(file_format, str, "File Format")
        Val.validate_path(path, file_format)

        try:
            if file_format == 'csv':
                df.to_csv(path, index=False)
            elif file_format == 'excel':
                df.to_excel(path, index=False)
            else:
                raise ValueError(f"Unsupported file format: {file_format}")
            print(f"Data saved to {path}")
        except Exception as e:
            raise Exception(f'Error saving data to {path}: {e}')

    def save_full_data(self,
                       path: str,
                       file_format: str = 'csv'):
        self._save_data(self.df_merged, path, file_format)

    def save_cleaned_data(self,
                          path: str,
                          file_format: str = 'csv'):
        self._save_data(self.df_merged_cleaned, path, file_format)
