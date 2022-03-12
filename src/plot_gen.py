import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

df = pd.read_csv('../data/models/model_v25_(fixed_target)_dynamic_his.csv')


clear_df = df.loc[df['y_target'] > 0.9]
clear_df2 = df.loc[df['episode_reward'] > 50]
print(clear_df2)
plt.plot([x for x in range(len(df['episode']))], df['episode_reward'].rolling(window=100).mean(), label='pandas')

#plt.plot(df['episode'], df['moving_average'], label='manual')
plt.legend()
plt.savefig('fg_kde_v24.jpg')
plt.show()
plt.close()


