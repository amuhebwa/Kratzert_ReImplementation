import code
import glob
import argparse
from kratzert_lstm import LSTM
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import MinMaxScaler
from generated_kfold_sets import sets_kfoldsets
from helper_functions import calculate_NRMSE, calculate_KGE, calculate_NSE, calculate_RBIAS

np.random.seed(1234)

better_columns_order = [
    'Albedo', 'Avg_Skin_Temp','PlantCanopyWater', 'CanopyWaterEvpn', 'DirectEvonBareSoil',
    'Evapotranspn', 'LngWaveRadFlux', 'NetRadFlux', 'PotEvpnRate','Pressure', 'SpecHmd', 'HeatFlux',
    'Sen.HtFlux', 'LtHeat','StmSurfRunoff', 'BsGndWtrRunoff', 'SnowMelt', 'TotalPcpRate','RainPcpRate',
    'RootZoneSoilMstr', 'SnowDepthWtrEq','DwdShtWvRadFlux', 'SnowDepth', 'SnowPcpRate', 'SoilMst10',
    'SoilMst40', 'SoilMst100', 'SoilMst200', 'SoilTmp10', 'SoilTmp40','SoilTmp100', 'SoilTmp200', 'NetShtWvRadFlux',
    'AirTemp', 'Tspn', 'WindSpd',

    'basin_Albedo', 'basin_Avg_Skin_Temp','basin_PlantCanopyWater', 'basin_CanopyWaterEvpn',
    'basin_DirectEvonBareSoil', 'basin_Evapotranspn','basin_LngWaveRadFlux', 'basin_NetRadFlux', 'basin_PotEvpnRate',
    'basin_Pressure', 'basin_SpecHmd', 'basin_HeatFlux','basin_Sen.HtFlux', 'basin_LtHeat', 'basin_StmSurfRunoff',
    'basin_BsGndWtrRunoff', 'basin_SnowMelt', 'basin_TotalPcpRate','basin_RainPcpRate', 'basin_RootZoneSoilMstr',
    'basin_SnowDepthWtrEq', 'basin_DwdShtWvRadFlux', 'basin_SnowDepth','basin_SnowPcpRate', 'basin_SoilMst10',
    'basin_SoilMst200','basin_SoilMst40', 'basin_SoilMst100', 'basin_SoilTmp10','basin_SoilTmp200', 'basin_SoilTmp40',
    'basin_SoilTmp100', 'basin_NetShtWvRadFlux', 'basin_AirTemp', 'basin_Tspn', 'basin_WindSpd',

    'length', 'sinuosity', 'slope', 'uparea', 'lengthdir', 'strmDrop',
    'width_mean', 'max_width',

    'width',

    'NDVI','Q', 'discharge'

]

columns_to_scale = [
    'basin_Albedo', 'basin_Avg_Skin_Temp','basin_PlantCanopyWater', 'basin_CanopyWaterEvpn',
    'basin_DirectEvonBareSoil', 'basin_Evapotranspn','basin_LngWaveRadFlux', 'basin_NetRadFlux', 'basin_PotEvpnRate',
    'basin_Pressure', 'basin_SpecHmd', 'basin_HeatFlux','basin_Sen.HtFlux', 'basin_LtHeat', 'basin_StmSurfRunoff',
    'basin_BsGndWtrRunoff', 'basin_SnowMelt', 'basin_TotalPcpRate','basin_RainPcpRate', 'basin_RootZoneSoilMstr',
    'basin_SnowDepthWtrEq', 'basin_DwdShtWvRadFlux', 'basin_SnowDepth','basin_SnowPcpRate', 'basin_SoilMst10',
    'basin_SoilMst200','basin_SoilMst40', 'basin_SoilMst100', 'basin_SoilTmp10','basin_SoilTmp200', 'basin_SoilTmp40',
    'basin_SoilTmp100', 'basin_NetShtWvRadFlux', 'basin_AirTemp', 'basin_Tspn', 'basin_WindSpd','width',
    'Albedo', 'Avg_Skin_Temp','PlantCanopyWater', 'CanopyWaterEvpn', 'DirectEvonBareSoil',
    'Evapotranspn', 'LngWaveRadFlux', 'NetRadFlux', 'PotEvpnRate','Pressure', 'SpecHmd', 'HeatFlux',
    'Sen.HtFlux', 'LtHeat','StmSurfRunoff', 'BsGndWtrRunoff', 'SnowMelt', 'TotalPcpRate','RainPcpRate',
    'RootZoneSoilMstr', 'SnowDepthWtrEq','DwdShtWvRadFlux', 'SnowDepth', 'SnowPcpRate', 'SoilMst10',
    'SoilMst40', 'SoilMst100', 'SoilMst200', 'SoilTmp10', 'SoilTmp40','SoilTmp100', 'SoilTmp200', 'NetShtWvRadFlux',
    'AirTemp', 'Tspn', 'WindSpd', 'NDVI','Q', 'discharge',
]

columns_to_decompose = [
    'Albedo', 'Avg_Skin_Temp','PlantCanopyWater', 'CanopyWaterEvpn', 'DirectEvonBareSoil',
    'Evapotranspn', 'LngWaveRadFlux', 'NetRadFlux', 'PotEvpnRate','Pressure', 'SpecHmd', 'HeatFlux',
    'Sen.HtFlux', 'LtHeat','StmSurfRunoff', 'BsGndWtrRunoff', 'SnowMelt', 'TotalPcpRate','RainPcpRate',
    'RootZoneSoilMstr', 'SnowDepthWtrEq','DwdShtWvRadFlux', 'SnowDepth', 'SnowPcpRate', 'SoilMst10',
    'SoilMst40', 'SoilMst100', 'SoilMst200', 'SoilTmp10', 'SoilTmp40','SoilTmp100', 'SoilTmp200', 'NetShtWvRadFlux',
    'AirTemp', 'Tspn', 'WindSpd','NDVI','Q',

    'basin_Albedo', 'basin_Avg_Skin_Temp','basin_PlantCanopyWater', 'basin_CanopyWaterEvpn',
    'basin_DirectEvonBareSoil', 'basin_Evapotranspn','basin_LngWaveRadFlux', 'basin_NetRadFlux', 'basin_PotEvpnRate',
    'basin_Pressure', 'basin_SpecHmd', 'basin_HeatFlux','basin_Sen.HtFlux', 'basin_LtHeat', 'basin_StmSurfRunoff',
    'basin_BsGndWtrRunoff', 'basin_SnowMelt', 'basin_TotalPcpRate','basin_RainPcpRate', 'basin_RootZoneSoilMstr',
    'basin_SnowDepthWtrEq', 'basin_DwdShtWvRadFlux', 'basin_SnowDepth','basin_SnowPcpRate', 'basin_SoilMst10',
    'basin_SoilMst200','basin_SoilMst40', 'basin_SoilMst100', 'basin_SoilTmp10','basin_SoilTmp200', 'basin_SoilTmp40',
    'basin_SoilTmp100', 'basin_NetShtWvRadFlux', 'basin_AirTemp', 'basin_Tspn', 'basin_WindSpd',

    'width',
]

static_cols = ['length', 'sinuosity', 'uparea', 'max_width', 'lengthdir', 'strmDrop', 'width_mean',]

class KratzertModel(nn.Module):
    """
    A model that wraps around LSTM/EA-LSTM with a fully connected layer.
    """

    def __init__(self, input_size_dyn: int, hidden_size: int, initial_forget_bias: int = 5,
                 dropout: float = 0.0, concat_static: bool = False, no_static: bool = False):
        """
        Initialize the model.

        Parameters:
        - input_size_dyn (int): Number of dynamic input features.
        - hidden_size (int): Number of LSTM cells/hidden units.
        - initial_forget_bias (int, optional): Initial forget gate bias. Defaults to 5.
        - dropout (float, optional): Dropout probability in range [0, 1]. Defaults to 0.0.
        - concat_static (bool, optional): If True, uses standard LSTM, else uses EA-LSTM. Defaults to False.
        - no_static (bool, optional): If True, runs standard LSTM. Defaults to False.
        """
        super(KratzertModel, self).__init__()

        # Model attributes
        self.input_size_dyn = input_size_dyn
        self.hidden_size = hidden_size
        self.initial_forget_bias = initial_forget_bias
        self.dropout_rate = dropout
        self.concat_static = concat_static
        self.no_static = no_static

        # Model layers
        self.lstm = LSTM(
            input_size=input_size_dyn,
            hidden_size=hidden_size,
            initial_forget_bias=initial_forget_bias
        )
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x_d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the model.

        Parameters:
        - x_d (torch.Tensor): Tensor with dynamic input features of shape [batch, seq_length, n_features]

        Returns:
        - out (torch.Tensor): Network predictions
        - h_n (torch.Tensor): Hidden states of each time step
        - c_n (torch.Tensor): Cell states of each time step
        """
        h_n, c_n = self.lstm(x_d)
        last_h = self.dropout(h_n[:, -1, :])
        out = self.fc(last_h)

        return out, h_n, c_n


def train_one_epoch(model, train_loader, loss_fn, optimizer, device, clip_norm=True, clip_value=1.0):
    model.train()
    total_loss = 0.0
    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad() # delete the old gradients
        _predictions, _, _ = model(x_batch)
        loss = loss_fn(_predictions.squeeze(), y_batch.squeeze())  # Squeeze both predictions and y_batch
        loss.backward()
        if clip_norm:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
        optimizer.step()
        total_loss += loss.item() * len(y_batch)
    return total_loss / len(train_loader.dataset)

"""
def validate(model, val_loader, loss_fn):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            predictions, _, _ = model(x_batch)
            loss = loss_fn(predictions.squeeze(), y_batch.squeeze())  # Squeeze both predictions and y_batch
            total_loss += loss.item() * len(y_batch)
    return total_loss / len(val_loader.dataset)
"""

def create_lookup_dict(f):
    files = glob.glob(f)
    data_dict = {}
    for file in files:
        comid = file.split('/').pop().split('_').pop().split('.')[0]
        data_dict.update({comid: file})
    return data_dict

def create_dataset_forecast(_dataset, n_steps_in, n_steps_out):
    X, y = list(), list()
    for i in range(len(_dataset)):
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out - 1
        if out_end_ix > len(_dataset):
            break
        seq_x, seq_y = _dataset[i:end_ix, :-1], _dataset[end_ix - 1:out_end_ix, -1]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

def process_station_dataset(filepath):
    station_dataset = pd.read_csv(filepath)
    static_df = station_dataset[static_cols]
    static_df = np.log10(static_df + epsilon)
    station_dataset[static_cols] = static_df[static_cols]
    station_dataset = station_dataset[~station_dataset['discharge'].isna()]
    # Fill nans with interpolation
    station_dataset = station_dataset.interpolate(method='linear', limit_direction='both')
    scalers = {}
    for i, current_column in enumerate(columns_to_scale):
        current_scaler = MinMaxScaler(feature_range=(0, 1))
        scalers['scaler_' + str(current_column)] = current_scaler
        station_dataset[current_column] = (current_scaler.fit_transform(station_dataset[current_column].values.reshape(-1, 1))).ravel()
        del current_scaler
    station_dataset = station_dataset[better_columns_order]
    return station_dataset, scalers

def parse_args():
    parser = argparse.ArgumentParser(description='File Parameters')
    parser.add_argument('--set_index', type=int, required=True)
    return parser.parse_args()
if __name__=="__main__":
    args = parse_args()
    set_index = int(args.set_index)
    base_dir = '/gypsum/eguide/projects/amuhebwa/rivers_ML/kratzert'
    discharge_lookup = create_lookup_dict(f'{base_dir}/lumped_complete_dataset/StationId_*.csv')
    stationsIds = list(discharge_lookup.keys())
    """
    'batch_size': 2000,
    'clip_norm': True,
    'clip_value': 1,
    'dropout': 0.4,
    'epochs': 30,
    'hidden_size': 256,
    'initial_forget_gate_bias': 5,
    'log_interval': 50,
    'learning_rate': 1e-3,
    'seq_length': 270,
    
    """
    batch_size = 64 # 2000
    sequence_length = 270
    forecast_days = 1
    hidden_size = 256
    initial_forget_gate_bias = 5
    learning_rate = 1e-3
    clip_norm = True
    clip_value = 1
    dropout = 0.4
    num_epochs = 100
    epsilon = 1e-6
    no_input_features = len(better_columns_order)-1
    experimentType = 'dimLessConcatDf'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    training_set = sets_kfoldsets[set_index]
    unique_id = '_'.join([str(s) for s in training_set])

    # Instantiate the model
    kratzert_model = KratzertModel(no_input_features, hidden_size, initial_forget_gate_bias, dropout)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        kratzert_model = nn.DataParallel(kratzert_model)

    kratzert_model = kratzert_model.to(device)
    optimizer_kratzert = torch.optim.Adam(kratzert_model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()


    """
    For Testing, let us use only 3 stations
    """

    # training_set = training_set[:2]

    dataset_list = []
    for idx, station_id in enumerate(training_set):
        file_path = discharge_lookup[station_id]
        current_dataset, current_scalers = process_station_dataset(file_path)
        dataset_list.append(current_dataset)
        del current_dataset

    complete_dataset = pd.concat(dataset_list)
    x_train, y_train = create_dataset_forecast(complete_dataset.to_numpy(), sequence_length, forecast_days)
    # x_train_tensor = torch.FloatTensor(x_train).to(device)
    # y_train_tensor = torch.FloatTensor(y_train).to(device)
    x_train_tensor = torch.FloatTensor(x_train)
    y_train_tensor = torch.FloatTensor(y_train)

    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    # We want to free space by deleting some of the variables
    del complete_dataset, x_train, y_train, x_train_tensor, y_train_tensor, train_dataset

    #print(f"Training on station {idx+1}/{len(training_set)}: {station_id}")

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(kratzert_model, train_dataloader, loss_fn, optimizer_kratzert, device)
        # val_loss = validate(kratzert_model, val_loader_reshaped, loss_fn)
        # print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}")
        # Release cached GPU memory
        torch.cuda.empty_cache()

    # del current_dataset, x_train, y_train, x_train_tensor, y_train_tensor, train_dataset, train_dataloader

    # Save the model
    torch.save(kratzert_model.state_dict(), f'{base_dir}/trained_models/{experimentType}_kratzert_model_{unique_id}.pth')

    # perform inference
    # First set the model to evaluation mode
    kratzert_model.eval()
    test_stations = [s for s in stationsIds if s not in training_set]

    # create arrays to store the results
    kge_arr, nse_arr, rbias_arr, nrmse_arr = [], [], [], []
    with torch.no_grad():
        for idx, station_id in enumerate(test_stations):
            test_file_path = discharge_lookup[station_id]
            test_dataset, test_scalers = process_station_dataset(test_file_path)
            x_test, y_test = create_dataset_forecast(test_dataset.to_numpy(), sequence_length, forecast_days)
            x_test_tensor = torch.FloatTensor(x_test).to(device)
            kratzert_model = kratzert_model.to(device)
            predictions = kratzert_model(x_test_tensor)
            predictions = predictions[0].cpu().numpy().ravel()
            y_test = y_test.ravel()
            nse = calculate_NSE(y_test, predictions)
            kge = calculate_KGE(y_test, predictions)
            rbias = calculate_RBIAS(y_test, predictions)
            nrmse = calculate_NRMSE(y_test, predictions)
            # print KGE, NSE, RBIAS, NRMSE
            print(f"Testing on station {idx+1}/{len(test_stations)}: {station_id} - NSE: {nse:.4f} - KGE: {kge:.4f} - RBIAS: {rbias:.4f} - NRMSE: {nrmse:.4f}")
            kge_arr.append(kge)
            nse_arr.append(nse)
            rbias_arr.append(rbias)
            nrmse_arr.append(nrmse)
            del test_dataset, x_test, y_test, x_test_tensor, predictions
            # Release cached GPU memory
            torch.cuda.empty_cache()

    # create a dataframe to store the results
    results_df = pd.DataFrame({'station_id': test_stations, 'KGE': kge_arr, 'NSE': nse_arr, 'RBIAS': rbias_arr, 'NRMSE': nrmse_arr})
    results_df.to_csv(f'{base_dir}/kratzert_prediction_results/{experimentType}_kratzert_model_{unique_id}.csv', index=False)
