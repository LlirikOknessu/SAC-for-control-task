import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

df = pd.read_csv('../data/learning_on_stand_new_reward_dynamic_his.csv')


clear_df = df.loc[df['y_target'] > 0.9]
clear_df2 = df.loc[df['episode_reward'] > 50]
print(clear_df2)
plt.plot(df['episode'], df['episode_reward'].rolling(window=100).mean())
plt.savefig('fg_kde.jpg')
plt.show()
