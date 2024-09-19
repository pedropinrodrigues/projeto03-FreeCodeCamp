import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1
df = pd.read_csv('medical_examination.csv')

# 2
df['overweight'] = np.where((df['weight'] / (df['height'] / 100) ** 2) > 25, 1, 0)

# 3
df['cholesterol'] = np.where(df['cholesterol'] == 1, 0, 1)
df['gluc'] = np.where(df['gluc'] == 1, 0, 1)

# 4
def draw_cat_plot():
    # 5
    df_cat = pd.melt(df, id_vars=['cardio'], value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'], 
                 var_name='variable', value_name='value')

    # 6
    df_cat_0 = df_cat[df_cat['cardio'] == 0].value_counts()
    df_cat_1 = df_cat[df_cat['cardio'] == 1].value_counts()

    # 7
    sns.catplot(x='variable', hue='value', col='cardio', kind='count', data=df_cat)
    


    # 8
    fig = plt.show()


    # 9
    fig.savefig('catplot.png')
    return fig


# 10
def draw_heat_map():
    # 11
    df_heat = df[
    (df['height'] >= df['height'].quantile(0.025)) &
    (df['height'] <= df['height'].quantile(0.975)) &
    (df['weight'] >= df['weight'].quantile(0.025)) &
    (df['weight'] <= df['weight'].quantile(0.975))
]

    # 12
    corr = df_heat.corr()

    # 13
    mask = pd.DataFrame(np.triu(np.ones_like(corr, dtype=bool)))



    # 14
    fig, ax = plt.subplots(figsize=(12, 12))

    # 15
    sns.heatmap(corr, annot=True, fmt='.1f', mask=mask, vmax=.3, center=0, square=True, linewidths=.5, cbar_kws={'shrink': .5})


    # 16
    fig.savefig('heatmap.png')
    return fig
