import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pdb 


class PostModelingAnalysis:

  @staticmethod
  def plot_history(history_orig, skip):
    history = {}
    history['loss'] = history_orig.history['loss'][skip:]
    history['val_loss'] = history_orig.history['val_loss'][skip:]
    history['accuracy'] = history_orig.history['accuracy'][skip:]
    history['val_accuracy'] = history_orig.history['val_accuracy'][skip:]

    plt.plot(history['accuracy'])
    plt.plot(history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


  @staticmethod
  def get_predictions_with_threshold(probs, threshold):
    predictions_with_threshold = [] 
    for x in probs:
      argmx = np.argmax(x)

      if argmx == 0 or x[argmx] < threshold:
        predictions_with_threshold.append(0)

      else:
        predictions_with_threshold.append(argmx)

    return predictions_with_threshold

  @staticmethod
  def get_win_rate(probs, t_valid, threshold):
    predictions_with_threshold = PostModelingAnalysis.get_predictions_with_threshold(probs, threshold)

    conf_matrix = tf.math.confusion_matrix(
      t_valid, 
      predictions_with_threshold
    ).numpy()

    numerator = conf_matrix[1][1] + conf_matrix[2][2]
    denominator = conf_matrix[0][1] + conf_matrix[0][2] + conf_matrix[1][2] + conf_matrix[2][1] + numerator
    
    return (numerator / denominator), numerator

  @staticmethod
  def get_wins_array_and_best_info(probs, t_valid):
    best_wr_n = 0
    best_threshold = -1

    wins = []
    for x in np.arange(0, 1, 0.05):
      y, n = PostModelingAnalysis.get_win_rate(probs, t_valid, x)
      wins.append([x, y])

      loss_rate = 1 - y

      delta = y - loss_rate
      wr_n = (delta * n) / y

      if wr_n > best_wr_n:
        best_threshold = x
        best_wr_n = wr_n

      # print(f'x: {x}, y: {y}, n: {n}')
    return wins, best_wr_n, best_threshold
 
  @staticmethod
  def get_sharpe_array_and_best_sharpe(probs, t_valid, config):
    best_sharpe = np.NINF
    best_threshold = np.NINF

    sharpe_array = []
    for x in np.arange(0, 1, 0.05):
      pwt = PostModelingAnalysis.get_predictions_with_threshold(probs, x)
      sharpe = PostModelingAnalysis.analysis(pwt, t_valid, 'sharpe', config)
      sharpe_array.append(sharpe)
      if sharpe > best_sharpe:
        best_sharpe = sharpe

    return sharpe_array, best_sharpe, best_threshold


  @staticmethod
  def analysis(predictions_with_threshold, t_valid, analysis_type, config):
    daily_risk_free = 0.035 / 365
    Y = predictions_with_threshold
    Y_true = t_valid
    profit_arr = []
    # negative_profit = []

    for y, y_true in zip(Y, Y_true):
      if (y == 1 and y_true == 1) or (y == 2 and y_true == 2):
        profit_arr.append(config['profit'])
      elif y == 0:
        profit_arr.append(0)
      else:
        neg_p = -1*config['profit']
        profit_arr.append(neg_p)
        # negative_profit.append(neg_p)

    profit_abs = sum(profit_arr)
    profit_perc = profit_abs / config['capital']

    daily_returns = []
    daily_negative_returns = []

    days = int(np.ceil(len(profit_arr) / (config['five_minutes_in_day']*config['pairs'])))

    for i in range(days):
      s = i*config['five_minutes_in_day']*config['pairs']
      f = (i+1)*config['five_minutes_in_day']*config['pairs']

      daily_profit = sum(profit_arr[s:f])
      neg_daily_profit = 0 if daily_profit >= 0 else daily_profit

      daily_returns.append(daily_profit / config['capital'])
      daily_negative_returns.append(neg_daily_profit / config['capital'])
      # pdb.set_trace()

    daily_std = np.std(daily_negative_returns if analysis_type=='sortino' else daily_returns)
    daily_returns_mean = np.mean(daily_returns)

    sortino = (daily_returns_mean - daily_risk_free) / daily_std
    
    print(daily_returns)

    # pdb.set_trace()
    
    return sortino