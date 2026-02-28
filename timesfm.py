import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import TimesFmModelForPrediction, TimesFmConfig

df = pd.read_csv("Month_Value_1.csv")

#Converting Period column to datetime
df['Period'].info()
df['Period'] = pd.to_datetime(df['Period'])
df['Period'].info()

print(df.isna().sum())

# Drop rows with missing values
df = df.dropna()

#Removed data after 2019-12-01 because I want to evaluate the performance of the model in 12 months forecast
df = df[df['Period'] < '2020-01-01'] 


df = df[['Period', 'Sales_quantity']].copy()

df_infer = df[df['Period'] < '2019-01-01']  # For inference
df_test = df[df['Period'] >= '2019-01-01'] # For evaluation


# Define the configuration for the TimesFM model
config = TimesFmConfig(
    patch_length=32,
    context_length=512,      # Length of input context
    horizon_length=128,     # Length of prediction horizon
    freq_size=3,            # 0: High, 1: Medium, 2: Low
    num_hidden_layers=50,   # Number of Transformer layers
    hidden_size=1280,       # Size of hidden layers
    intermediate_size=1280,
    num_attention_heads=16,
    head_dim=80,
    quantiles=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
)


model = TimesFmModelForPrediction.from_pretrained(
    "google/timesfm-2.0-500m-pytorch",
    config=config,           # Used the config defined above
    dtype=torch.bfloat16,
    attn_implementation="sdpa",
    device_map="auto"
)


forecast_input = [df_infer["Sales_quantity"].values.astype(np.float32)]
frequency_input = [2]  #0 for daily data, 1 for weekly, 2 for monthly

forecast_input_tensor = [
    torch.tensor(ts, dtype=torch.bfloat16).to(model.device)
    for ts in forecast_input
]
frequency_input_tensor = torch.tensor(frequency_input, dtype=torch.long).to(model.device)

with torch.no_grad():
    outputs = model(
        past_values=forecast_input_tensor,
        freq=frequency_input_tensor,
        return_dict=True
    )
    point_forecast = outputs.mean_predictions.float().cpu().numpy()

print("Point forecast shape:", point_forecast.shape)

print(point_forecast[:, :12])

# Get actual values
actual = df_test['Sales_quantity'].values

# Get predicted values
predicted = point_forecast[:, :12].flatten()

# Create x-axis (days)
days = np.arange(1, 13)

#plot
plt.figure(figsize=(14, 6))
plt.plot(days, actual, label='Actual', marker='o', linewidth=2, markersize=5)
plt.plot(days, predicted, label='Predicted', marker='s', linewidth=2, markersize=5)

plt.xlabel('Months', fontsize=12)
plt.ylabel('Sales_quantity', fontsize=12)
plt.title('Predicted vs Sales_quantity (12 Months)', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("sales_quantity.png", dpi=300, bbox_inches="tight")
plt.show()