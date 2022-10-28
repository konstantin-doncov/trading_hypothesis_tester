import tqdm
import pandas as pd
import numpy as np
import pathlib
from price_normalizator import PriceNormalizator
from dataframe_preparator import DataframePreparator

from keras.utils.np_utils import to_categorical  
import pdb

class DataframeSplitter:
  @staticmethod
  def create_columns(columns, size):
    extended_columns = []
    for i in range(size):
        for c in columns:
            extended_columns.append(f'{c}_{i}')
            
    return extended_columns

  @staticmethod
  def dataframe_to_row(df):
    df_out = df.reset_index(drop=True).stack()
    df_out.index = df_out.index.map('{0[1]}_{0[0]}'.format)
    return df_out.to_frame().T
  
  @staticmethod
  def long_calculate_amount(buy_price, take_profit_sell_price, profit, taker_fee, maker_fee):
    profit_for_one_amount = take_profit_sell_price - (take_profit_sell_price * maker_fee) - buy_price - (buy_price * taker_fee)
    amount = profit / profit_for_one_amount
    return amount

  @staticmethod
  def long_calculate_stop_loss(buy_price, amount, loss, taker_fee, maker_fee):
    money_spent_on_buy = (buy_price * amount) + (buy_price * amount * taker_fee) 
    money_received_on_sell = money_spent_on_buy - loss
    left_side = amount - (amount * maker_fee)
    stop_loss = money_received_on_sell / left_side
    return stop_loss

  @staticmethod
  def short_calculate_amount(sell_price, take_profit_buy_price, profit, taker_fee, maker_fee):
    profit_for_one_amount = sell_price - (sell_price * maker_fee) - (take_profit_buy_price + (take_profit_buy_price * taker_fee))
    amount = profit / profit_for_one_amount
    return amount

  @staticmethod
  def short_calculate_stop_loss(sell_price, amount, loss, taker_fee, maker_fee):
    money_received_on_sell = (sell_price * amount) - (sell_price * amount * taker_fee) 
    money_spent_on_buy = money_received_on_sell + loss
    left_side = amount + (amount * maker_fee)
    stop_loss = money_spent_on_buy / left_side
    return stop_loss

  @staticmethod
  def get_gain(df, current_bar, last_x_bars):
    last_5_bars = df.iloc[current_bar-last_x_bars:current_bar, :]
    return (last_5_bars['high'] - last_5_bars['low']).max()

  @staticmethod
  def get_last_up_level(df, current_bar, last_x_bars_for_level, close, min_p, max_p):
    last_bars = df.iloc[current_bar-last_x_bars_for_level:current_bar, :]
    return DataframeSplitter.get_normalized_delta(last_bars['high'].max(), close, min_p, max_p)

  @staticmethod
  def get_last_down_level(df, current_bar, last_x_bars_for_level, close, min_p, max_p):
    last_bars = df.iloc[current_bar-last_x_bars_for_level:current_bar, :]
    return DataframeSplitter.get_normalized_delta(close, last_bars['low'].min(), min_p, max_p)

  @staticmethod
  def get_normalized_delta(first, second, min_p, max_p):
    first = PriceNormalizator.normalize_price(first, min_p, max_p)
    second = PriceNormalizator.normalize_price(second, min_p, max_p)
    return first - second
  
  @staticmethod
  def features_target_split(df, min_p, max_p, pair_number, config):
        
    features = []
    target = []

    features2 = []
    sup = []

    start_index = config['bars']
    end_index = len(df) - 1

    f2_columns=[
      'take_profit_buy_price', 
      'take_profit_sell_price', 
      'stop_loss_buy_price', 
      'stop_loss_sell_price', 
      'last_up_level1',
      'last_down_level1',
      'last_up_level2',
      'last_down_level2',
      'pair_number'
      ]

    dates = []

    current_seq = []
    longest_seq = []

    number_of_trades = 0

    number_of_trades_per_month = {y: {x: 0 for x in range(1, 13)} for y in range(2017, 2023)}

    max_buy = 0

    test_info = []


    for current_bar in tqdm.tqdm(range(start_index, end_index)):
      gain = DataframeSplitter.get_gain(df, current_bar, config['last_x_bars'])

      take_profit_buy_price = df.at[current_bar, 'close'] + gain
      take_profit_sell_price = df.at[current_bar, 'close'] - gain
      
      buy_amount = DataframeSplitter.long_calculate_amount(df.at[current_bar, 'close'], take_profit_buy_price, config['profit'], config['taker_fee'], config['maker_fee'])
      stop_loss_buy_price = DataframeSplitter.long_calculate_stop_loss(df.at[current_bar, 'close'], buy_amount, config['profit'], config['taker_fee'], config['maker_fee'])
      sell_amount = DataframeSplitter.short_calculate_amount(df.at[current_bar, 'close'], take_profit_sell_price, config['profit'], config['taker_fee'], config['maker_fee'])
      stop_loss_sell_price = DataframeSplitter.short_calculate_stop_loss(df.at[current_bar, 'close'], sell_amount, config['profit'], config['taker_fee'], config['maker_fee'])

      action = 0 # 0 - nothing, 1 - down, 2 - up
      
      range_start = min(current_bar+1, len(df) - 1)
      range_finish = min(current_bar+21, len(df) - 1)
    
      stop_loss_buy_already_was = False
      stop_loss_sell_already_was = False

      if buy_amount <= 0:
        stop_loss_buy_already_was = True

      if sell_amount <= 0:
        stop_loss_sell_already_was = True

      for i in range(range_start, range_finish):
          take_profit_sell = df.at[i, 'low'] <= take_profit_sell_price
          take_profit_buy = df.at[i, 'high'] >= take_profit_buy_price

          stop_loss_buy = df.at[i, 'low'] <= stop_loss_buy_price
          stop_loss_sell = df.at[i, 'high'] >= stop_loss_sell_price

          if stop_loss_buy:
            stop_loss_buy_already_was = True

          if stop_loss_sell:
            stop_loss_sell_already_was = True

          both_take_profits = take_profit_buy and take_profit_sell

          if both_take_profits or (stop_loss_buy_already_was and stop_loss_sell_already_was):
              action = 0
              break

          if take_profit_sell and not stop_loss_sell_already_was:
              action = 1
              break

          if take_profit_buy and not stop_loss_buy_already_was:
              action = 2
              break

      # if action == 1 or action == 2:
      indices = range(current_bar-config['bars'], current_bar)   
      di = df.loc[indices, config['columns']]
      di[['open', 'high',	'low', 'close']] = (di[['open', 'high',	'low', 'close']]-min_p)/(max_p-min_p)
      f = DataframeSplitter.dataframe_to_row(di)
      features.append(f)
      target.append(action)

      close = df.at[current_bar, 'close']

      features2.append([
        DataframeSplitter.get_normalized_delta(take_profit_buy_price, close, min_p, max_p),
        DataframeSplitter.get_normalized_delta(take_profit_sell_price, close, min_p, max_p), 
        DataframeSplitter.get_normalized_delta(stop_loss_buy_price, close, min_p, max_p), 
        DataframeSplitter.get_normalized_delta(stop_loss_sell_price, close, min_p, max_p),
        # PriceNormalizator.normalize_price(take_profit_buy_price, min_p, max_p), 
        # PriceNormalizator.normalize_price(take_profit_sell_price, min_p, max_p), 
        # PriceNormalizator.normalize_price(stop_loss_buy_price, min_p, max_p), 
        # PriceNormalizator.normalize_price(stop_loss_sell_price, min_p, max_p),
        DataframeSplitter.get_last_up_level(df, current_bar, config['last_x_bars_for_level1'], df.at[current_bar, 'close'], min_p, max_p),
        DataframeSplitter.get_last_down_level(df, current_bar, config['last_x_bars_for_level1'], df.at[current_bar, 'close'], min_p, max_p),
        DataframeSplitter.get_last_up_level(df, current_bar, config['last_x_bars_for_level2'], df.at[current_bar, 'close'], min_p, max_p),
        DataframeSplitter.get_last_down_level(df, current_bar, config['last_x_bars_for_level2'], df.at[current_bar, 'close'], min_p, max_p),
        pair_number
      ])
      # pdb.set_trace()
              
      test_take_profit = None
      test_stop_loss = None
      close_date = None 

      if action == 1:
        test_take_profit = take_profit_sell_price
        test_stop_loss = stop_loss_sell_price
        close_date = df.at[i, 'open_time']
      elif action == 2:
        test_take_profit = take_profit_buy_price
        test_stop_loss = stop_loss_buy_price
        close_date = df.at[i, 'open_time']


      test_info.append({
          'open_date': df.at[current_bar, 'open_time'],
          'action': action,
          'take_profit': test_take_profit,
          'stop_loss': test_stop_loss,
          'close_date': close_date
      })



      # if df.at[current_bar, 'open_time'].strftime('%Y-%m-%d %H:%M:%S') == '2022-05-05 17:00:00':
      #   pdb.set_trace()

      # sup.append(split_debug(stop_loss_buy_already_was, take_profit_buy_price, stop_loss_buy_price, stop_loss_sell_already_was, take_profit_sell_price, stop_loss_sell_price,
                              # df.at[current_bar, 'open_time'], df.at[i, 'open_time'],  df.at[current_bar, 'close'], action, pair))
      
      if action == 1 or action == 2:

        if action == 1:
          amount = buy_amount
        elif action == 2:
            amount = sell_amount

        buy = amount * df.at[current_bar, 'close']

        if buy > max_buy:
          max_buy = buy

      dates.append(df.at[current_bar, 'open_time'])

      # if len(current_seq) > len(longest_seq):
      #   longest_seq = current_seq
      # current_seq = []

      # number_of_trades += 1
      # number_of_trades_per_month[df.at[current_bar, 'open_time'].year][df.at[current_bar, 'open_time'].month] += 1

      # else:
      #   # current_seq.append(df.at[current_bar, 'open_time'])
      #   pass

    print(max_buy)
      
    return pd.concat(features).reset_index(drop=True), pd.DataFrame.from_records(features2, columns=f2_columns), target, test_info
    # return number_of_trades

  @staticmethod
  def prepare_train_test_sets(df_info, idf, config):
    df_train = df_info.df.iloc[:config['train_size'],:].reset_index(drop=True)
    o = DataframeSplitter.features_target_split(df_train, df_info.min_p, df_info.max_p, idf, config)

    f = o[0]
    f2 = o[1]
    t = o[2]

    df_valid = df_info.df.iloc[config['train_size']+1:config['train_size']+config['test_size']+1,:].reset_index(drop=True)
    o_valid = DataframeSplitter.features_target_split(df_valid, df_info.min_p, df_info.max_p, idf, config)

    f_valid = o_valid[0]
    f2_valid = o_valid[1]
    t_valid = o_valid[2]
    ti_valid = o_valid[3]

    return f, f2, t, f_valid, f2_valid, t_valid, ti_valid

  @staticmethod
  def insert_row(idx, df, df_insert):
    return df.iloc[:idx, ].append(df_insert).append(df.iloc[idx:, ]).reset_index(drop = True)
  
  @staticmethod
  def prepare_multiple_train_test_sets(df_list, config):
    f = None
    f2 = None
    t = []

    f_valid = None
    f2_valid = None
    t_valid = []
    test_info_valid = []

    for idf in range(len(df_list)):
      print('prepare_train_test_sets ', idf, ':')
      res = DataframeSplitter.prepare_train_test_sets(df_list[idf], idf, config)

      f = res[0] if f is None else f.append(res[0]).reset_index(drop=True)
      f2 = res[1] if f2 is None else f2.append(res[1]).reset_index(drop=True)
      t.extend(res[2])

      f_valid = res[3] if f_valid is None else f_valid.append(res[3]).reset_index(drop=True)
      f2_valid = res[4] if f2_valid is None else f2_valid.append(res[4]).reset_index(drop=True)
      t_valid.extend(res[5])
      test_info_valid.extend(res[6])

    return f, f2, t, f_valid, f2_valid, t_valid, test_info_valid
  
  @staticmethod
  def prepare_multiple_train_test_sets_and_save(df_list, path_to_save, offset, config):

    for idf in range(len(df_list)):
      print('prepare_train_test_sets ', idf, ':')
      f, f2, t, f_valid, f2_valid, t_valid, _ = DataframeSplitter.prepare_train_test_sets(df_list[idf], idf, config)

      ohe_ind = f2.shape[1] - 1

      f = f.to_numpy()
      f2 = f2.to_numpy()
      t = np.array(t)

      f_valid = f_valid.to_numpy()
      f2_valid = f2_valid.to_numpy()
      t_valid = np.array(t_valid)

      f, f2, t = DataframePreparator.remove_nulls_in_f2_np(f, f2, t)
      f_valid, f2_valid, t_valid = DataframePreparator.remove_nulls_in_f2_np(f_valid, f2_valid, t_valid)

      f2 = np.append(np.delete(f2, ohe_ind, axis=1), to_categorical(f2[:, ohe_ind] + offset, num_classes=config['pairs']), axis=1)
      f2_valid = np.append(np.delete(f2_valid, ohe_ind, axis=1), to_categorical(f2_valid[:, ohe_ind] + offset, num_classes=config['pairs']), axis=1)

      save_dir_path = f'{path_to_save}/{df_list[idf].name}/'
      
      pathlib.Path(save_dir_path).mkdir(parents=True, exist_ok=True) 

      f_path = f'{save_dir_path}f.npy'
      f2_path = f'{save_dir_path}f2.npy'
      t_path = f'{save_dir_path}t.npy'

      f_valid_path = f'{save_dir_path}f_valid.npy'
      f2_valid_path = f'{save_dir_path}f2_valid.npy'
      t_valid_path = f'{save_dir_path}t_valid.npy'

      np.save(f_path, f)
      np.save(f2_path, f2)
      np.save(t_path, t)

      np.save(f_valid_path, f_valid)
      np.save(f2_valid_path, f2_valid)
      np.save(t_valid_path, t_valid)


  @staticmethod
  def backtrader_features_split(df, min_p, max_p, config):
    df_orig = df
    df = df[-51:][config['columns']].reset_index(drop=True)

    features = []
    features2 = []


    last_5_bars = df.iloc[-(config['last_x_bars']+1):-1, :]
    gain = (last_5_bars['high'] - last_5_bars['low']).max()

    take_profit_buy_price = df.at[len(df)-1, 'close'] + gain
    take_profit_sell_price = df.at[len(df)-1, 'close'] - gain

    buy_amount = DataframeSplitter.long_calculate_amount(df.at[len(df)-1, 'close'], take_profit_buy_price, config['profit'], config['taker_fee'], config['maker_fee'])
    stop_loss_buy_price = DataframeSplitter.long_calculate_stop_loss(df.at[len(df)-1, 'close'], buy_amount, config['profit'], config['taker_fee'], config['maker_fee'])

    sell_amount = DataframeSplitter.short_calculate_amount(df.at[len(df)-1, 'close'], take_profit_sell_price, config['profit'], config['taker_fee'], config['maker_fee'])
    stop_loss_sell_price = DataframeSplitter.short_calculate_stop_loss(df.at[len(df)-1, 'close'], sell_amount, config['profit'], config['taker_fee'], config['maker_fee'])

    df[['open', 'high',	'low', 'close']] = (df[['open', 'high',	'low', 'close']] - min_p)/(max_p - min_p)
    f = DataframeSplitter.dataframe_to_row(df[:-1])
    features.append(f)
    features2.append([PriceNormalizator.normalize_price(take_profit_buy_price, min_p, max_p), PriceNormalizator.normalize_price(take_profit_sell_price, min_p, max_p), PriceNormalizator.normalize_price(stop_loss_buy_price, min_p, max_p), PriceNormalizator.normalize_price(stop_loss_sell_price, min_p, max_p)])
  
    # if is_debug:
    #   pdb.set_trace()

    return pd.concat(features).reset_index(drop=True), \
            pd.DataFrame.from_records(features2,
          columns=['take_profit_buy_price', 'take_profit_sell_price', 'stop_loss_buy_price','stop_loss_sell_price'])

