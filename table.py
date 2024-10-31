#make table with npy files
import numpy as np
rwd_dyna0 = np.load('dyna_0_escape_room_reward.npy')
rwd_dyna1 = np.load('dyna_1_escape_room_reward.npy')
rwd_dyna10 = np.load('dyna_10_escape_room_reward.npy')
rwd_dyna100 = np.load('dyna_100_escape_room_reward.npy')
rwd_dyna1000 = np.load('dyna_1000_escape_room_reward.npy')
rwd_rmax = np.load('rmax_escape_room_reward.npy')
#make table
import pandas as pd
df = pd.DataFrame({'Dyna (0)': rwd_dyna0, 'Dyna (1)': rwd_dyna1, 'Dyna (10)': rwd_dyna10, 'Dyna (100)': rwd_dyna100, 'Dyna (1000)': rwd_dyna1000, 'RMax': rwd_rmax})

df.index = np.arange(1, len(df)+1)
print('\n\n')
print(df)
print('\n\n')