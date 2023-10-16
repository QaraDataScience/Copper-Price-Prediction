import streamlit as st
import torch
import pickle

# Define your LSTM model class (replace with your actual model class)
class LSTMModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_stacked_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_stacked_layers = num_stacked_layers

        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_stacked_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, 1)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# Load your pre-trained LSTM model from a pickle file
with open('LSTM.pkl', 'rb') as file:
    model = pickle.load(file)

# Set the device (GPU or CPU) based on availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Create a Streamlit app
st.title("Forecasting Copper Prices")
st.write("This Streamlit app deploys a pre-trained LSTM model for forecasting copper prices.")

# User input
user_input = st.number_input("Enter a value:", min_value=0.0, max_value=100.0, value=0.0)

# Button to make predictions
if st.button("Generate Prediction"):
    # Prepare the input tensor
    input_data = torch.tensor([[user_input]], dtype=torch.float32).to(device)

    # Make predictions
    with torch.no_grad():
        prediction = model(input_data)

    # Display the prediction
    st.write(f"Prediction: {prediction.item()}")
