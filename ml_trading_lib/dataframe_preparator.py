import talib as ta
import pandas as pd
import numpy as np
import os
import pdb

from price_normalizator import PriceNormalizator
from dataframe_info import DataframeInfo

class DataframePreparator:

  @staticmethod
  def sma_indic(df, tp, min_p, max_p):
    indic_name = f'sma_{tp}'

    t = ta.SMA(PriceNormalizator.normalize_price(df['close'], min_p, max_p).to_numpy(), timeperiod=tp)
    df[indic_name] = pd.Series(t)
    df[indic_name] = df[indic_name].fillna(PriceNormalizator.normalize_price(df['close'], min_p, max_p))

    df = DataframePreparator.normalize_column(df, indic_name)

    return df

  @staticmethod
  def ema_indic(df, tp, min_p, max_p):
    indic_name = f'ema_{tp}'

    t = ta.EMA(PriceNormalizator.normalize_price(df['close'], min_p, max_p).to_numpy(), timeperiod=tp)
    df[indic_name] = pd.Series(t)
    df[indic_name] = df[indic_name].fillna(PriceNormalizator.normalize_price(df['close'], min_p, max_p))

    df = DataframePreparator.normalize_column(df, indic_name)

    return df

  @staticmethod
  def rsi_indic(df, tp, min_p, max_p):
    indic_name = f'rsi_{tp}'

    t = ta.RSI(df['close'], timeperiod=tp)
    df[indic_name] = pd.Series(t)
    df = DataframePreparator.normalize_column(df, indic_name)

    df[indic_name] = df[indic_name].fillna(-1)

    return df

  @staticmethod
  def adx_indic(df, tp, min_p, max_p):
    indic_name = f'adx_{tp}'

    t = ta.ADX(df['high'], df['low'], df['close'], timeperiod=tp)
    df[indic_name] = pd.Series(t)
    df = DataframePreparator.normalize_column(df, indic_name)

    df[indic_name] = df[indic_name].fillna(-1)
    return df

  @staticmethod
  def natr_indic(df, tp, min_p, max_p):
    indic_name = f'natr_{tp}'

    t = ta.NATR(df['high'], df['low'], df['close'], timeperiod=tp)
    df[indic_name] = pd.Series(t)
    df = DataframePreparator.normalize_column(df, indic_name)

    df[indic_name] = df[indic_name].fillna(-1)
    return df


  @staticmethod
  def normalize_column(df, column_name):
    min_p = df[column_name].min()
    max_p = df[column_name].max()

    df[column_name] = PriceNormalizator.normalize_price(df[column_name], min_p, max_p)
    return df

  @staticmethod
  def add_indicators(df, min_p, max_p):
    df = DataframePreparator.sma_indic(df, 25, min_p, max_p)
    df = DataframePreparator.sma_indic(df, 50, min_p, max_p)
    df = DataframePreparator.sma_indic(df, 100, min_p, max_p)

    df = DataframePreparator.ema_indic(df, 25, min_p, max_p)
    df = DataframePreparator.ema_indic(df, 50, min_p, max_p)
    df = DataframePreparator.ema_indic(df, 100, min_p, max_p)
    
    df = DataframePreparator.rsi_indic(df, 25, min_p, max_p)
    df = DataframePreparator.rsi_indic(df, 50, min_p, max_p)
    df = DataframePreparator.rsi_indic(df, 100, min_p, max_p)
    
    df = DataframePreparator.adx_indic(df, 25, min_p, max_p)
    df = DataframePreparator.adx_indic(df, 50, min_p, max_p)
    df = DataframePreparator.adx_indic(df, 100, min_p, max_p)

    df = DataframePreparator.natr_indic(df, 25, min_p, max_p)
    df = DataframePreparator.natr_indic(df, 50, min_p, max_p)
    df = DataframePreparator.natr_indic(df, 100, min_p, max_p)

    df = DataframePreparator.normalize_column(df, 'volume')
    df = DataframePreparator.normalize_column(df, 'num_trades')

    return df

  @staticmethod
  def prepare_dataframe(df, min_p=None, max_p=None):

    if type(df) == str:
      df = pd.read_csv(df)

    # df = df.reindex(index=df.index[::-1]).reset_index(drop=True)
    if min_p == None:
      min_p = df[['open', 'high',	'low', 'close']].min().min()
    if max_p == None:
      max_p = df[['open', 'high',	'low', 'close']].max().max()
    # df[['open', 'high',	'low', 'close']] = (df[['open', 'high',	'low', 'close']]-min_p)/(max_p-min_p)

    df = DataframePreparator.add_indicators(df.copy(), min_p, max_p) 

    return df, min_p, max_p

  @staticmethod
  def get_cols():
    inidcs = ['sma_25', 'sma_50', 'sma_100',
          'rsi_25', 'rsi_50', 'rsi_100',
          'ema_25', 'ema_50', 'ema_100',
          'adx_25', 'adx_50', 'adx_100',
          'natr_25', 'natr_50', 'natr_100',
         ]
    cols = ['open', 'high', 'low', 'close']
    cols.extend(inidcs)
    return cols

  @staticmethod
  def prepare_one_pair(path, time_units='ms'):
    df_list = []
    df, min_p, max_p = DataframePreparator.prepare_dataframe(path)
    df['open_time_unix'] = df['open_time']
    df['open_time'] = pd.to_datetime(df['open_time'], unit=time_units)
    df_list.append(DataframeInfo(df, path, min_p, max_p, len(df)))
    return df_list

  @staticmethod
  def prepare_multiple_pairs(path, time_units='ms'):
    df_list = []
    for root, dirs, files in os.walk(path):
      for file in files:
          if file.endswith('.csv'):
            df, min_p, max_p = DataframePreparator.prepare_dataframe(f'{path}/{file}')
            df['open_time_unix'] = df['open_time']
            df['open_time'] = pd.to_datetime(df['open_time'], unit=time_units)
            df_list.append(DataframeInfo(df, file, min_p, max_p, len(df)))

    df_size = min(list(map(lambda x: len(x.df), df_list)))
    for i in range(len(df_list)-1):
      df_list[i].df = df_list[i].df[:df_size]
      print(len(df_list[i].df))
    return df_list

  @staticmethod
  def remove_nulls_in_f2(f, f2, t):
    t = t.copy()
    nulls = list(f2[f2.isna().any(axis=1)].index)
    nulls.reverse()

    for i in nulls:
      f = f.drop([i]).reset_index(drop=True)
      f2 = f2.drop([i]).reset_index(drop=True)
      del t[i]
    return f, f2, t

  @staticmethod
  def remove_nulls_in_f2_np(f, f2, t):
    nulls = np.where(np.isnan(f2))[0]
    nulls = np.flipud(nulls)

    for i in nulls:
      f = np.delete(f, i, 0)
      f2 = np.delete(f2, i, 0)
      t = np.delete(t, i, 0)

    return f, f2, t