import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
# import seaborn as sns

df_last = pd.read_csv('../data/signal_sin_stand_model_response_v0.csv')
# df = pd.read_csv('../data/signal_stand_stand_response.csv')
# df_v2 = pd.read_csv('../data/signal_stand_stand_response_v2.csv')
# df_v3 = pd.read_csv('../data/signal_stand_stand_response_v3.csv')
# df_v4 = pd.read_csv('../data/signal_stand_stand_response_v4.csv')
# df_v5 = pd.read_csv('../data/stand_model_response_v2.csv')
# df_v6 = pd.read_csv('../data/stand_model_response_v3.csv')

plt.style.use('seaborn-whitegrid')

folder = Path('../data/Ball_in_tube_validation_datasets/')

# clear_df = df.loc[df['y_target'] > 0.9]
# clear_df2 = df.loc[df['episode_reward'] > 50]
# print(clear_df2)
x = df_last['Unnamed: 0'].tolist()
# for file in folder.glob('*.csv'):
#     df = pd.read_csv(file)
#     x = [x for x in range(df['position'].shape[0])]
#     plt.plot(x, df['position'], label=file.name.replace('.csv', ''))
#     plt.ylabel('position')
# plt.plot(x, df['stand_response'], label='Control')
# plt.plot(x, df_v2['stand_response'], label='Control_v2')
# plt.plot(x, df_v3['stand_response'], label='Control_v3')
# plt.plot(x, df_v4['stand_response'], label='Control_v4')
# plt.plot(x, df_v5['model_response'], label='Model_control')
# plt.plot(x, df_v6['model_response'], label='Model_control_v3')
# plt.plot(x, df['response'], label='Stand response')
plt.plot(x, df_last['model_response'], label='Model response')
plt.plot(x, df_last['stand_response'], label='Stand response')
plt.legend()
# plt.savefig('stand_response_all_versions.jpg')
plt.savefig('signal_sin_stand_model_response_v0.jpg')
plt.show()

plt.plot(x, df_last['model_angular_velocities'], label='Model response')
plt.plot(x, df_last['angular_velocity'], label='Stand response')
plt.legend()
plt.savefig('signal_sin_stand_model_angular_velocity_response_v0.jpg')
plt.show()

# for file in folder.glob('*.csv'):
#     df = pd.read_csv(file)
#     x = [x for x in range(df['current'].shape[0])]
#     plt.plot(x, df['current'], label=file.name.replace('.csv', ''))
#     plt.ylabel('current')

# plt.legend()
# # plt.savefig('stand_response_all_versions.jpg')
# plt.savefig('stand_response_current.jpg')
# plt.show()

# for file in folder.glob('*.csv'):
#     df = pd.read_csv(file)
#     x = [x for x in range(df['angular_velocity'].shape[0])]
#     plt.plot(x, df['angular_velocity'], label=file.name.replace('.csv', ''))
#     plt.ylabel('angular_velocity')
#
#
# plt.legend()
# # plt.savefig('stand_response_all_versions.jpg')
# plt.savefig('stand_response_angular_velocity.jpg')
# plt.show()
#
# for file in folder.glob('*.csv'):
#     df = pd.read_csv(file)
#     x = [x for x in range(df['object_velocity'].shape[0])]
#     plt.plot(x, df['object_velocity'], label=file.name.replace('.csv', ''))
#     plt.ylabel('object_velocity')
#
# plt.legend()
# # plt.savefig('stand_response_all_versions.jpg')
# plt.savefig('stand_response_object_velocity.jpg')
# plt.show()
