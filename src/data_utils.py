import os
import torch
import numpy as np
import pandas as pd


class MultivariateTimeSeriesFromCSV(torch.utils.data.Dataset):
    def __init__(self, path:str, feature_cols:str | list, label_col: str, window_size=5, prediction_window = 1, prediction_offset=1, standardize=True):
        """Dataset class for windowing multivariate time series loaded directly from csv files. 

        Args:
            path (str): _description_
            feature_cols (str | list): col name or list of col names to be taken as features
            label_col (list): col name to be taken as label 
            window_size (int, optional): Lookback window. Defaults to 5.
            prediction_window (int, optional): Prediction window. Defaults to 1.
            prediction_offset (int, optional): Offset of prediction window. Defaults to 1, which corresponds to the next sample after the lookback window.
        """


        self.path = path
        self.standardize = standardize

        self.feature_cols = feature_cols
        self.label_col = label_col

        self.window_size = window_size
        self.prediction_window = prediction_window
        self.prediction_offset = prediction_offset

        self.dataframes = self._get_dataframe(self.path)

        assert isinstance(self.dataframes, list)

        self.windows, self.targets = self._prepare_windows()

        if self.standardize:
            self._standardize()

    def _get_dataframe(self, path:str) -> list:
            
        """Gets the CSV dataframes from the specified path and returns them as a list 

        Returns:
            path (str): Path to the data as string
        """
        assert isinstance(path, str), "Path must be a string"
        
        # If the path is a single file:
        if os.path.isfile(self.path):
            return [pd.read_csv(path, index_col=[0])]
            
        # If the path is a directory
        else:
            file_list = os.listdir(path)
            csv_file_list = [file for file in file_list if file.endswith('.csv')]

            return_list = []
            
            # Iterate through the files
            for file in csv_file_list:
                return_list.append(pd.read_csv(os.path.join(path, file), index_col=[0]))

            return return_list
        
    def _prepare_windows(self):

        # Predeclare lists
        windows = []
        targets = []
        n_samples = []

        # Iterate through the dataframes
        for df in self.dataframes:
            
            # Get cols inputs and labals
            features = df[self.feature_cols].values
            labels = df[self.label_col].values   

            # Skips df if empty (e.g. broken recording)
            if len(features) < self.window_size + self.prediction_offset + self.prediction_window:
                continue

            # Iteration range
            samples = len(df) - (self.window_size + self.prediction_offset + self.prediction_window - 1)

            assert samples > 0, f'The number of samples cannot be negative, while for df the shape is: {df.shape} and samples are :{samples}'

            for i in range(samples):
                
                # Get X, y
                X = features[i: i + self.window_size]
                y = labels[i + self.window_size + self.prediction_offset: i + self.window_size + self.prediction_offset + self.prediction_window]

                # Store as list
                windows.append(X)
                targets.append(y)
            
            n_samples.append(samples)

        # Increases speed of transformation
        windows = np.array(windows)
        targets = np.array(targets)

        assert windows.shape[1] == self.window_size
        assert targets.shape[1] == self.prediction_window
        assert windows.shape[0] == sum(n_samples)

        return torch.tensor(windows, dtype=torch.float32), torch.tensor(targets, dtype=torch.float32)
    
    def _standardize(self):
        self.windows = (self.windows - self.windows.mean((0, 1))) / self.windows.std((0, 1))

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return self.windows[idx], self.targets[idx]
    

def split_data_train_val_test(dataset, splits=(0.7, 0.1, 0.2)) -> tuple:
    """Splits a time series dataset along the time variable

    Args:
        dataset (torch.utils.data.Dataset): Torch Dataset object
        splits (tuple, optional): Splits. Defaults to (0.7, 0.1, 0.2).

    Returns:
        tuple: tuple of three Torch Dataset objects
    """
    n_windows = len(dataset)

    assert sum(splits) == 1, 'The sum of splits has to be equal to 1'

    train_idx   = range(0, int(splits[0]*n_windows))
    val_idx     = range(int(splits[0] * n_windows), int((splits[0]+splits[1]) * n_windows))
    test_idx    = range(int((splits[0]+splits[1]) * n_windows), n_windows)

    train_dataset   = torch.utils.data.Subset(dataset, train_idx)
    val_dataset     = torch.utils.data.Subset(dataset, val_idx)
    test_dataset    = torch.utils.data.Subset(dataset, test_idx)

    return train_dataset, val_dataset, test_dataset


def prepare_data(path: str, out_path: str):
    """Transforms the dataframe to the desired shape. Divides the frames by day and saves them as csv.

    Args:
        path (str): In data path
        out_path (str): Out data path
    """
    
    df_loaded_data = pd.read_parquet(path)
    df_loaded_data.drop(columns=['Irr', 'Image', 'Hum', 'Temp', 'img', 'Target_GHIr', 'Target_GHICS', 'Target_CSI', 'Target'], inplace=True)
    df_loaded_data.rename(columns={'ghi1':'ghi_measured', 'ghi':'ghi_clearsky'}, inplace=True)
    df_loaded_data['residual'] = df_loaded_data['y_test'] - df_loaded_data['y_true']

    date_range = pd.date_range(df_loaded_data.index.min().date(), df_loaded_data.index.max().date(), freq='1D')

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    for day in date_range:
        extracted_day = df_loaded_data.loc[str(day.date())]
        extracted_day.to_csv(os.path.join(out_path, fr'input_data_{str(day.date())}.csv'), sep=',')
