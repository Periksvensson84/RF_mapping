# this class is meant to merge the data from DataDLC and DataNeuron objects
# with respect to the time index or peaks
# Or if they're already the same length of time, just merge them directly
from datadlc import DataDLC
from dataneuron import DataNeuron
import pandas as pd

class MergedData:
    def __init__(self,
                 dlc: DataDLC,
                 neuron: DataNeuron):
        self.dlc = dlc
        self.neuron = neuron
        self._merge()

    def _merge(self) -> pd.DataFrame:
        # Merge the DataFrames
        df_dlc = self.dlc._merge_data()
        # Fill the neuron data up to the length of the dlc data
        df_neuron = self._fill_neuron_up()
        
        # Merge the DataFrames
        self.df_merged = pd.concat([df_dlc, df_neuron], axis=1)
        return self.df_merged
        
    def _fill_neuron_up(self) -> pd.DataFrame:
        # Fill the neuron data up to the length of the dlc data
        target_length = len(self.dlc.df_square)
        self.neuron._fill_downsample_length(target_length)
        return self.neuron.downsampled_df


    def plotting_split(self,
                       threshold: float) -> tuple:
        # split the data into 3 parts:
        # high bending_coefficient with neuron firing
        # high bending_coefficient without neuron firing
        # low bending_coefficient with neuron firing
        # low bending_coefficient without neuron firing will get sorted out
        threshold = threshold * self.df_merged['Bending_Coefficient'].max()
        
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
        # Cleaned data where the bending coefficient is above the threshold
        # or the Spikes column is not 0
        threshold = threshold * self.df_merged['Bending_Coefficient'].max()
        self.df_merged_cleaned = \
            self.df_merged[(self.df_merged['Bending_Coefficient'] > threshold) \
                | (self.df_merged['Spikes'] != 0)]
        return self.df_merged_cleaned


    def _save_data(self,
                   df: pd.DataFrame,
                   path: str,
                   file_format: str):
        try:
            if file_format == 'csv':
                df.to_csv(path, index=False)
            elif file_format == 'excel':
                df.to_excel(path, index=False)
            print(f"Data saved to {path}")
        except Exception as e:
            raise Exception(f'Error saving data to {path}: {e}')

    def save_full_data(self, path: str, file_format: str = 'csv'):
        self._save_data(self.df_merged, path, file_format)

    def save_cleaned_data(self, path: str, file_format: str = 'csv'):
        self._save_data(self.df_merged_cleaned, path, file_format)
