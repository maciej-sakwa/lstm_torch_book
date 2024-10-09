import os

import torch
from torch.utils.data import DataLoader

from src.data_utils import MultivariateTimeSeriesFromCSV, split_data_train_val_test, prepare_data
from src.models import LSTM
from src.model_utils import train_epoch, val_epoch



def train(epochs, model, optimizer, loss_fn, train_generator, val_generator):

    es_patience = 5
    es_val_loss = 0
    best_val_loss = 1_000_000
    early_stopping_list = []

    for ep in range(epochs):
        print(f'Epoch {ep+1}')

        model.train(True)
        avg_loss = train_epoch(model, train_generator, optimizer, loss_fn)

        model.eval()
        avg_val_loss = val_epoch(model, val_generator, loss_fn)

        print(f'LOSS train {avg_loss:2f} valid {avg_val_loss:2f}')

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model_path = f'model_test'
            torch.save(model.state_dict(), model_path)
        
        early_stopping_list.append(avg_val_loss)
        if len(early_stopping_list) >= es_patience:
            es_val_loss = min(early_stopping_list[-es_patience:])
        
        if es_val_loss > best_val_loss:
            print('Early Stopping...')
            break


def main(data_inpath, data_outpath):
    
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    torch.manual_seed(1)

    EPOCHS = 100
    PREDICTION_WINDOW = 15
    PREDICTION_OFFSET = 1
    PREDICTION_LOOKBACK = 60
    FEATURE_COLS = ['ghi_measured', 'humidity', 'temperature' , 'ghi_clearsky', 'elevation']
    PRED_COLS = 'residual'

    print(f"Using {device} device")

    if not os.path.exists(data_outpath):
        print('Preparing data...')
        prepare_data(data_inpath, data_outpath)

    with torch.device(device):

        dataset = MultivariateTimeSeriesFromCSV(
            path=data_outpath, 
            feature_cols=FEATURE_COLS, 
            label_col=PRED_COLS, 
            window_size=PREDICTION_LOOKBACK, 
            prediction_window=PREDICTION_WINDOW, 
            prediction_offset=PREDICTION_OFFSET, 
            standardize=False)
        
        train_dataset, val_dataset, test_dataset = split_data_train_val_test(dataset)

        train_dataloader    = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_dataloader      = DataLoader(val_dataset, batch_size=32, shuffle=True)
        test_dataloader     = DataLoader(test_dataset, batch_size=1, shuffle=False)

        learning_rate = 0.001
        input_size = 5
        hidden_size = 128
        output_size = PREDICTION_WINDOW
        num_layers = 1

        lstm = LSTM(input_size=input_size, output_size=output_size, hidden_size=hidden_size, num_layers=num_layers)
        loss_fn = torch.nn.L1Loss()
        optimizer = torch.optim.Adam(lstm.parameters(), lr = learning_rate)

        train(EPOCHS, model=lstm, optimizer=optimizer, loss_fn=loss_fn, train_generator=train_dataloader, val_generator=val_dataloader)


if __name__ == '__main__':
    main()