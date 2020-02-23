import numpy as np

def remove_outliers(df, cols):
  for col in df.columns:
    if col in cols:
      upper_lim = df[col].quantile(.95)
      lower_lim = df[col].quantile(.05)

      df = df[(df[col] < upper_lim) & (df[col] > lower_lim)]

  return df

def standardize(df, cols):
  for col in df.columns:
    if col in cols:
      df[col] = (df[col] - df[col].mean()) / df[col].std()

  return df

def log_transform(df, cols):
  for col in df.columns:
    if col in cols:
      df[col] = (df[col] - df[col].min() + 1).transform(np.log)

  return df

def drop_cols(df, cols):
  return df.drop(cols, axis=1)

def drop_missing_value_rows(df, threshold):
  return df.loc[df.isnull().mean(axis=1) < threshold]