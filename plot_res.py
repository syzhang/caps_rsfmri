"""
plot results csv
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# df = pd.read_csv('./result/static_corr_cv.csv')
# at = np.unique(df['atlas'])
# for n, c in enumerate(at):
#     df_tmp = df[df['atlas']==c]
#     sns.catplot(data=df_tmp, x='pca_num', y='accuracy', hue='combination', col='correlation', kind='point')
#     plt.savefig(f'./figs/static_corr_cv_{c}.png')

# plt.subplots()
# sns.catplot(data=df, x='pca_num', y='accuracy', hue='atlas', col='correlation', kind='point')
# plt.savefig(f'./figs/static_corr_cv.png')


###############
df = pd.read_csv('./result/dynamic_corr_cv.csv')
# at = np.unique(df['atlas'])
sns.catplot(data=df, x='pca_num', y='accuracy', hue='combination', col='atlas', kind='point')
plt.savefig(f'./figs/dynamic_corr_cv.png')

# plt.subplots()
# sns.catplot(data=df, x='pca_num', y='accuracy', col='atlas', kind='point')
# plt.savefig(f'./figs/dynamic_corr_cv_mean.png')