import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import streamlit as st
from datetime import date
import torch
import torch.nn as nn
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_stacked_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_stacked_layers = num_stacked_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_stacked_layers,
                            batch_first=True)

        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out



model_file_path = 'model.pt'
model = torch.load(model_file_path)

# Set the device (GPU or CPU) based on availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Create a Streamlit app
def main():
    st.title("LSTM Model Deployment")
    st.write("This Streamlit app deploys a pre-trained LSTM model for prediction.")
    
    
    df = pd.read_csv("df.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    last_date = df['Date'].max().date()
    #start_date = st.date_input('Start date:', value=date(2023, 10, 22), key='start_date')
    end_date = st.date_input('End date:', value=date(2023, 10, 23), key='end_date')
    start_date = last_date+timedelta(days=1)
    forecast_dates = pd.date_range(start=start_date, end=end_date, freq='D')

    # Create a DataFrame with the date column
    forecast_set = pd.DataFrame({'Date': forecast_dates})
    forecast_set['Date'] = pd.to_datetime(forecast_set['Date'])
    stacked_np = np.load('stacked_np.npy')
    shifted_df = pd.read_csv('shifted_df.csv', dtype='float64')
    
    shifted_df_as_np_flipped = np.fliplr(stacked_np)
    
    shifted_df_as_tensor = torch.tensor(shifted_df_as_np_flipped.copy())
    torch.set_printoptions(precision=6)
    last = shifted_df_as_tensor[-1][1:]
    last = last.unsqueeze(0).unsqueeze(-1)
    
    shifted_df_no_date = shifted_df.drop(['Date'], axis=1)
    
    prices_scaler = joblib.load('min_max_scaler.pkl')
    
    for i in range(forecast_dates.shape[0]):
        last = shifted_df_as_tensor[-1][1:]
        last = last.unsqueeze(0).unsqueeze(-1).float()
        with torch.no_grad():
            predicted = model(last.to(device)).to('cpu').numpy()
        row = torch.cat([shifted_df_as_tensor[-1]   [1:],torch.tensor(predicted).reshape(1)],dim=0).unsqueeze(0)
        quick_reverse = prices_scaler.inverse_transform(row[:,4:].to('cpu').numpy())
        new_row = {'Price':quick_reverse[0][7],
                   'Price(t-1)':quick_reverse[0][6],
                   'Price(t-2)':quick_reverse[0][5],
                   'Price(t-3)':quick_reverse[0][4],
                   'Price(t-4)':quick_reverse[0][3],
                   'Price(t-5)':quick_reverse[0][2],
                   'Price(t-6)':quick_reverse[0][1],
                   'Price(t-7)':quick_reverse[0][0],
                   'Real_GDP':df.iloc[-1]["Real_GDP"],
                   'CPI':df.iloc[-1]["CPI"],
                   'inflation_rate':df.iloc[-1]["inflation_rate"],
                   'PALLFNFINDEXM':df.iloc[-1]["PALLFNFINDEXM"]
                   }
        shifted_df_as_tensor = torch.cat((shifted_df_as_tensor, row),dim=0)
        shifted_df_no_date = shifted_df_no_date._append(new_row,ignore_index=True)
    
    final_set = prices_scaler.inverse_transform(shifted_df_as_tensor[:,4:].to('cpu').numpy())
    display_df = pd.DataFrame(shifted_df_no_date["Price"])
    display_df = display_df.iloc[6049:]
    display_df = display_df.reset_index(drop=True)
    forecast_set = pd.concat([forecast_set, display_df],axis=1)
    

    # Display the date input to the user.
    st.write('Start date:', start_date)
    st.write('End date:', end_date)

    # Button to make predictions
    if st.button("Generate Prediction"):


        # Display the prediction
        st.write(f"Prediction: {forecast_set}")

if __name__ == "__main__":
    main()
