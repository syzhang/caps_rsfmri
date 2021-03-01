"""
plot results csv
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('./result/static_corr_cv.csv')
combs = np.unique(df['combination'])
for n, c in enumerate(combs):
    df_tmp = df[df['combination']==c]
    plt.subplots()
    sns.lineplot(data=df_tmp, x='pca_num', y='accuracy', hue='correlation')
    plt.title(c)
    plt.savefig('./figs/static_corr_cv'+str(n)+'.png')