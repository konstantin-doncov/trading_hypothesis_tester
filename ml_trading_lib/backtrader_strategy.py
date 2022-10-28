import backtrader as bt
import pandas as pd
import numpy as np
from dataframe_preparator import DataframePreparator
from dataframe_splitter import DataframeSplitter

class MyStrategy1(bt.Strategy):
  def __init__(self, model, initial_df, trades_entries, min_p, max_p, config) -> None:
      self.buyed = False
      self.df = (initial_df[['Date', 'Open', 'High', 'Low', 'Close']]
                .rename({'Date': 'datetime', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close'}, axis=1))
      self.iter = len(self.df)
      self.model = model #= tf.keras.models.load_model("/content/drive/MyDrive/Colab Notebooks/ML trading/model_save/model", compile=False, custom_objects={'ImprovedCCE': ImprovedCCE})
      self.config = config
      self.trades_entries = trades_entries
      self.min_p = min_p
      self.max_p = max_p

      self.brackets1 = None

  def get_current_date(self):
    return self.data.datetime.datetime()

  def get_current_pd_timestamp(self):
    return pd.Timestamp(self.get_current_date())

  def get_current_date_string(self):
    return self.get_current_date().strftime('%m.%d.%Y, %H:%M:%S')

  def log(self, txt):
    date = self.get_current_date_string()

    print(f'{date}, {txt}')

  def next(self):
    self.df.loc[self.iter] = [self.get_current_date(), self.datas[0].open[0], self.datas[0].high[0], self.datas[0].low[0], self.datas[0].close[0]]

    self.iter += 1
    current_date = self.get_current_date_string()

    # pdb.set_trace()

    if (self.iter >= 51 and not self.buyed):
      df, _, _ = DataframePreparator.prepare_dataframe(self.df, self.min_p, self.max_p)

      # is_debug = False
      # if current_date == '05.05.2022, 17:20:00':
      #   is_debug = True

      f, f2 = DataframeSplitter.backtrader_features_split(df, self.min_p, self.max_p, self.config)
      f_lstm = f.to_numpy().reshape(len(f), self.config['bars'], len(self.config['columns']))
      preds = self.model.predict((f_lstm, f2))

      last_5_bars = df.iloc[-(self.config['last_x_bars']+1):-1, :]
      gain = (last_5_bars['high'] - last_5_bars['low']).max()

      # if (current_date == '05.05.2022, 16:55:00' or -V
      #    current_date == '05.05.2022, 17:00:00' or V
      #    current_date == '05.05.2022, 17:05:00' or V
      #    current_date == '05.05.2022, 17:10:00' or V
      #    current_date == '05.05.2022, 17:15:00' or V
      #    current_date == '05.05.2022, 17:20:00'): V
      #   preds[0][1] = 0.81
      #   # pdb.set_trace()
      # else:
      #   preds[0][1] = 0


      self.log(f'{self.df.loc[self.iter-1:self.iter-1]}\n')

      argmax = np.argmax(preds[0])

      # if current_date == '05.05.2022, 17:05:00':
      #   pdb.set_trace()


      if preds[0][argmax] < self.config['threshold']:
        return

      # print(preds[0][argmax])

      # next_open = orig_df.at[self.iter + 1, 'Open']

      if argmax == 1:
        close = self.df.at[len(self.df) - 1, 'close']
        take_profit = close - gain
        amount = DataframeSplitter.short_calculate_amount(close, take_profit, self.config['profit'], self.config['taker_fee'], self.config['maker_fee'])
        stop_loss = DataframeSplitter.short_calculate_stop_loss(close, amount,self.config['profit'], self.config['taker_fee'], self.config['maker_fee'])

        # pdb.set_trace()

        # if next_open < stop_loss and next_open > take_profit:
        self.brackets = self.sell_bracket(price=close, size=amount, exectype=bt.Order.Market, limitprice=take_profit, stopprice=stop_loss, plimit=stop_loss, stopexec=bt.Order.StopLimit, tradeid=self.iter)
        self.trades_entries[self.iter] = {
            'open_date': self.df.at[len(self.df) - 1, 'datetime'],
            'action': 1,
            'take_profit': take_profit,
            'stop_loss': stop_loss
        }
        # mainside = self.sell(price=close, size=amount, exectype=bt.Order.Market, transmit=False)
        # self.buy(price=stop_loss, size=amount, exectype=bt.Order.Stop, transmit=False, parent=mainside)
        # self.buy(price=take_profit, size=amount, exectype=bt.Order.Limit, transmit=True, parent=mainside)
        
        self.log(f'Opened trade, last close: {close}, take profit: {take_profit}, stop loss: {stop_loss}')
          # self.buyed = True

      if argmax == 2:
        close = self.df.at[len(self.df) - 1, 'close']
        take_profit = close + gain
        amount = DataframeSplitter.long_calculate_amount(close, take_profit, self.config['profit'], self.config['taker_fee'], self.config['maker_fee'])
        stop_loss = DataframeSplitter.long_calculate_stop_loss(close, amount, self.config['profit'], self.config['taker_fee'], self.config['maker_fee'])

        # pdb.set_trace()

        # if next_open > stop_loss and next_open < take_profit:
        self.brackets = self.buy_bracket(size=amount, exectype=bt.Order.Market, limitprice=take_profit, stopprice=stop_loss,)
          # print('buyed')
          # self.buyed = True
        self.trades_entries[self.iter] = {
            'open_date': self.df.at[len(self.df) - 1, 'datetime'],
            'action': 2,
            'take_profit': take_profit,
            'stop_loss': stop_loss
        }

      # self.buy_order = self.buy(size=amount)
      # buy_price = self.buy_order.created.price

      # self.sell_order = self.sell(size=amount, exectype=bt.Order.Limit, price=take_profit)
      # self.sell_order = self.sell(size=amount, exectype=bt.Order.Limit, price=take_profit)

      
  def notify_trade(self, trade):
    if trade.isclosed:
      open_price = trade.history[0].event.price
      close_price = trade.history[1].event.price

      self.trades_entries[trade.tradeid]['close_date'] = self.get_current_pd_timestamp()
      self.trades_entries[trade.tradeid]['win'] = trade.pnlcomm > 0

      # is_long = trade.history[0].event.size > 0

      # is_stop_loss = ''

      # min_price_during_trade = self.df[self.df.datetime >= trade.open_datetime()][['open', 'high', 'low', 'close']].min().min()
      # max_price_during_trade = self.df[self.df.datetime >= trade.open_datetime()][['open', 'high', 'low', 'close']].max().max()

      self.log(f'Closed order which opened at {trade.open_datetime().strftime("%m.%d.%Y, %H:%M:%S")}: open price: {open_price} -> close price: {close_price} = pnl: {trade.pnlcomm}')
      # if (not is_long and close_price < min_price_during_trade) or (is_long and close_price > max_price_during_trade):
      #   self.log('PROBLEM TRADE DETECTED!')
      # pdb.set_trace()

  # def notify_order(self, order):
  #   # 1. If order is submitted/accepted, do nothing 
  #   if order.status in [order.Submitted, order.Accepted]:
  #       return
  #   # 2. If order is buy/sell executed, report price executed
  #   if order.status in [order.Completed]: 

  #       if order.isbuy():
  #           self.log('BUY ORDER PLACED,    Price:{0:8.2f}, Cost:{1:8.2f}, Comm:{2:8.2f}'.format(order.created.price,order.created.value,order.created.comm))
  #           self.log('BUY ORDER EXECUTED,  Price:{0:8.2f}, Cost:{1:8.2f}, Comm:{2:8.2f}'.format(order.executed.price,order.executed.value,order.executed.comm))
  #           self.buyprice = order.executed.price
  #           self.buycomm = order.executed.comm


  #       if order.issell():
  #           self.log('SELL ORDER PLACED,   Price:{0:8.2f}, Cost:{1:8.2f}, Comm:{2:8.2f}'.format(order.created.price,order.created.value,order.created.comm))
  #           self.log('SELL ORDER EXECUTED, Price:{0:8.2f}, Cost:{1:8.2f}, Comm:{2:8.2f}'.format(order.executed.price, order.executed.value,order.executed.comm))
  #           self.sellprice = order.executed.price
  #           self.sellcomm = order.executed.comm

            
  def stop(self):
      # pdb.set_trace()
      pass
