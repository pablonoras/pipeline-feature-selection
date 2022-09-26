from sklearn.model_selection import train_test_split

def split_df(df, test_size=0.2, target='is_chargeback_ap_efec', seed=0):
    y = df[target].fillna(0)
    X = df.drop([target], axis=1).fillna(0)

    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)