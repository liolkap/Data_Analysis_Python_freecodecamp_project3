import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1
df = pd.read_csv('medical_examination.csv')

# 2
df['overweight'] = df['weight'] / ((df['height'] / 100) ** 2) > 25
df['overweight'] = df['overweight'].replace({True: 1, False: 0})

# 3
df.loc[df['cholesterol'] == 1, 'cholesterol'] = 0
df.loc[df['cholesterol'] > 1, 'cholesterol'] = 1
df.loc[df['gluc'] == 1, 'gluc'] = 0
df.loc[df['gluc'] > 1, 'gluc'] = 1


# 4
def draw_cat_plot():
    # 5
    df_cat = pd.melt(df, id_vars=['cardio'],
                     value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])

    # 6
    df_cat1 = pd.DataFrame(df_cat.loc[df_cat['cardio'] == 1].value_counts()).reset_index().rename(
        columns={'count': 'total'})
    df_cat2 = pd.DataFrame(df_cat.loc[df_cat['cardio'] == 0].value_counts()).reset_index().rename(
        columns={'count': 'total'})
    df_cat = pd.concat([df_cat1, df_cat2], ignore_index=True)
    df_cat = df_cat.sort_values(by='variable', ascending=True)

    # 7
    p = sns.catplot(data=df_cat, kind='bar', x='variable', y='total', hue='cardio', aspect=2.2, legend=False)

    # 8
    fig = p.fig

    # 9
    fig.savefig('catplot.png')
    return fig


# 10
def draw_heat_map():
    # 11
    df_heat = df.loc[df['ap_lo'] <= df['ap_hi']]
    df_heat = df_heat.loc[(df['height'] >= df['height'].quantile(0.025)) |
                          (df['height'] <= df['height'].quantile(0.975)) |
                          (df['weight'] >= df['weight'].quantile(0.025)) |
                          (df['weight'] <= df['weight'].quantile(0.975))]

    # 12
    corr = df_heat.corr().apply(round, ndigits=1).astype(float)

    # 13
    mask = np.tril(np.ones_like(corr, dtype=bool))

    # 14
    fig, ax = plt.subplots(figsize=(13, 6))

    # 15
    annot = np.where(~mask, corr, np.nan)
    sns.heatmap(data=corr, ax=ax, annot=annot)

    # 16
    fig.savefig('heatmap.png')
    return fig
