import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

df_last = pd.read_csv('../data/signal_sin_stand_model_response_v0.csv')

plt.style.use('seaborn-whitegrid')

folder = Path('../data/Ball_in_tube_validation_datasets/')

x = df_last['Unnamed: 0'].tolist()

plt.plot(x, df_last['model_response'], label='Model response')
plt.plot(x, df_last['stand_response'], label='Stand response')
plt.legend()
plt.savefig('signal_sin_stand_model_response_v0.jpg')
plt.show()

plt.plot(x, df_last['model_angular_velocities'], label='Model response')
plt.plot(x, df_last['angular_velocity'], label='Stand response')
plt.legend()
plt.savefig('signal_sin_stand_model_angular_velocity_response_v0.jpg')
plt.show()
