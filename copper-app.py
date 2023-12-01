import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import streamlit as st
from datetime import date
import torch
import torch.nn as nn
from PIL import Image
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
    image = Image.open('qara_logo.png')
    st.image(image, caption='QARA Data Science', use_column_width=True)
    #st.image('qara_logo.png', caption='QARA Data Science', use_column_width=True)
    st.title("Copper AI Forecasting App")
    st.write("Forecasting of Copper Prices using LSTM Neural Network")

    st.image('price_prediction_plot.jpg', caption='Model Results', use_column_width=True)
    df = pd.read_csv("final_merged_copper_dataset.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    last_date = df['Date'].max().date()

    end_date = st.date_input('End date:', value=date(2023, 11, 30), key='end_date')
    start_date = last_date+timedelta(days=1)
    forecast_dates = pd.date_range(start=start_date, end=end_date, freq='B')  # 'B' excludes weekends

    # Create a DataFrame with the date column
    forecast_set = pd.DataFrame({'Date': forecast_dates})
    forecast_set['Date'] = pd.to_datetime(forecast_set['Date'])

    stacked_np = np.load('stacked_np.npy')
    shifted_df = pd.read_csv('shifted_df.csv')
    shifted_df = shifted_df.drop(['Date'], axis=1)
    shifted_df = shifted_df.astype(float)
    
    shifted_df_as_np_flipped = np.fliplr(stacked_np)
    
    shifted_df_as_tensor = torch.tensor(shifted_df_as_np_flipped.copy())
    torch.set_printoptions(precision=6)
    last = shifted_df_as_tensor[-1][1:]
    last = last.unsqueeze(0).unsqueeze(-1)
    
    shifted_df_no_date = shifted_df.copy()
    
    prices_scaler = joblib.load('prices_scaler.pkl')
    LOOKBACK = 15
    for i in range(forecast_dates.shape[0]):
        last = shifted_df_as_tensor[-1][1:]
        last = last.unsqueeze(0).unsqueeze(-1).float()
        with torch.no_grad():
            predicted = model(last.to(device)).to('cpu').numpy()
        row = torch.cat([shifted_df_as_tensor[-1][1:], torch.tensor(predicted).reshape(1)], dim=0).unsqueeze(0)
        quick_reverse = prices_scaler.inverse_transform(row[:, 5:].to('cpu').numpy())
        new_row = {'Price': quick_reverse[0][LOOKBACK],
                   'Price(t-1)': quick_reverse[0][LOOKBACK - 1],
                   'Price(t-2)': quick_reverse[0][LOOKBACK - 2],
                   'Price(t-3)': quick_reverse[0][LOOKBACK - 3],
                   'Price(t-4)': quick_reverse[0][LOOKBACK - 4],
                   'Price(t-5)': quick_reverse[0][LOOKBACK - 5],
                   'Price(t-6)': quick_reverse[0][LOOKBACK - 6],
                   'Price(t-7)': quick_reverse[0][LOOKBACK - 7],
                   'Price(t-8)': quick_reverse[0][LOOKBACK - 8],
                   'Price(t-9)': quick_reverse[0][LOOKBACK - 9],
                   'Price(t-10)': quick_reverse[0][LOOKBACK - 10],
                   'Price(t-11)': quick_reverse[0][LOOKBACK - 11],
                   'Price(t-12)': quick_reverse[0][LOOKBACK - 12],
                   'Price(t-13)': quick_reverse[0][LOOKBACK - 13],
                   'Price(t-14)': quick_reverse[0][LOOKBACK - 14],
                   'Price(t-15)': quick_reverse[0][LOOKBACK - 15],
                   'Real_GDP': df.iloc[-1]["Real_GDP"],
                   'CPI': df.iloc[-1]["CPI"],
                   'inflation_rate': df.iloc[-1]["inflation_rate"],
                   'PALLFNFINDEXM': df.iloc[-1]["PALLFNFINDEXM"],
                   'DEXCHUS': df.iloc[-1]["DEXCHUS"]
                   }
        shifted_df_as_tensor = torch.cat((shifted_df_as_tensor, row), dim=0)
        shifted_df_no_date = shifted_df_no_date._append(new_row, ignore_index=True)
    
    final_set = prices_scaler.inverse_transform(shifted_df_as_tensor[:,5:].to('cpu').numpy())
    display_df = pd.DataFrame(shifted_df_no_date["Price"])
    display_df = display_df.iloc[6037:]
    display_df = display_df.reset_index(drop=True)
    forecast_set = pd.concat([forecast_set, display_df],axis=1)
    

    # Display the date input to the user.
    st.write('Start date:', start_date)
    st.write('End date:', end_date)



    # Button to make predictions
    if st.button("Generate Prediction"):


        st.dataframe(forecast_set)  # Same as st.write(df)

if __name__ == "__main__":
    main()
