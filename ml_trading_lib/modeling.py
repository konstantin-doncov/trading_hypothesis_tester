from pandas.core.arrays.numeric import T
from post_modeling_analysis import PostModelingAnalysis

import tensorflow.keras.layers
import tensorflow as tf
import keras
import numpy as np
import random, os
import matplotlib.pyplot as plt
import pdb

class ImprovedCCE(tf.keras.losses.Loss):
  def __init__(self):
    super().__init__()

  def call(self, Y_true, Y):
    
    output = Y
    target = Y_true
    axis = -1
    from_logits = False

    # scale preds so that the class probas of each sample sum to 1
    output = output / tf.reduce_sum(output, axis, True)
    # Compute cross entropy from probabilities.
    output = tf.clip_by_value(output, tf.keras.backend.epsilon(), 1. - tf.keras.backend.epsilon())

    # sum_not_zero = -tf.reduce_sum(tf.gather(target, [1, 2], axis=1) * tf.math.log(tf.gather(output, [1, 2], axis=1)), -1)
    # sum_zero = -tf.reduce_sum(tf.math.maximum(0, tf.gather(target, 0, axis=1) - (tf.gather(target, 0, axis=1) / 2)) * tf.math.log(tf.gather(output, 0, axis=1)), -1)


    sum_not_zero = -tf.reduce_sum(tf.gather(target, [1, 2], axis=1) * tf.math.log(tf.gather(output, [1, 2], axis=1)), -1)
    sum_zero = -tf.math.maximum(0.0, tf.gather(target, 0, axis=1) - (tf.gather(output, 0, axis=1) / 2)) * tf.math.log(tf.gather(output, 0, axis=1))
    res = sum_not_zero + sum_zero


    return res

class SaveCallback(tf.keras.callbacks.Callback):
  def __init__(self, path, is_tpu):
    super().__init__()
    self.path = path
    self.is_tpu = is_tpu

  def on_epoch_end(self, epoch, logs=None): 
    Modeling.save_model(self.model, self.path, self.is_tpu)
    

class MyCallback(tf.keras.callbacks.Callback):
  def __init__(self, epoch_pattern, f_valid, f2_valid, t_valid, config):
    super().__init__()
    self.epoch_pattern = epoch_pattern
    # self.f = f
    # self.f2 = f2
    # self.t = t

    self.f_valid = f_valid
    self.f2_valid = f2_valid
    self.t_valid = t_valid

    self.config = config

  def get_average_wins(self, wins):
    first_wr = wins[0][1]
    wr_list = [first_wr]
    tail_started = False
    for wr in wins:
      if not tail_started and not first_wr == wr[1]:
        tail_started = True
      
      if tail_started and not np.isnan(wr[1]):
        wr_list.append(wr[1])
    return np.nanmean(wr_list), max(wr_list)

  def calculate_wins_array(self, f, f2, t):
      probs = self.model.predict((f, f2))
      wins, best_wr_n, best_threshold = PostModelingAnalysis.get_wins_array_and_best_info(probs, t)
      avg_win, max_win = self.get_average_wins(wins)
      return avg_win, max_win

  def calculate_sharpe_array(self, f, f2, t):
    probs = self.model.predict((f, f2))
    sharpe_array, best_sharpe, _ = PostModelingAnalysis.get_sharpe_array_and_best_sharpe(probs, t, self.config)
    return np.nanmean(sharpe_array), best_sharpe

  def on_epoch_end(self, epoch, logs=None):
    epoch_minus_ten = epoch % 10
    if any([ep == epoch_minus_ten for ep in self.epoch_pattern]):
      probs = self.model.predict((self.f_valid, self.f2_valid))
      # wins, best_wr_n, best_threshold = PostModelingAnalysis.get_wins_array_and_best_info(probs, self.t_valid)
      # avg_win = self.process_one_set(self.f, self.f2, self.t)
      val_avg_win, val_max_win = self.calculate_wins_array(self.f_valid, self.f2_valid, self.t_valid)
      val_avg_sharpe, val_max_sharpe = self.calculate_sharpe_array(self.f_valid, self.f2_valid, self.t_valid)

      # print(f'avg_win:', avg_win)
      # print(f'val_avg_win:', val_avg_win)

      # logs['avg_win'] = avg_win
      logs['val_avg_win'] = val_avg_win
      logs['val_max_win'] = val_max_win

      # logs['val_avg_sharpe'] = val_avg_sharpe
      logs['val_max_sharpe'] = val_max_sharpe

      tf.summary.scalar('val_avg_win', data=val_avg_win, step=epoch)
      tf.summary.scalar('val_max_win', data=val_max_win, step=epoch)
      # tf.summary.scalar('val_avg_sharpe', data=val_avg_sharpe, step=epoch)
      tf.summary.scalar('val_max_sharpe', data=val_max_sharpe, step=epoch)
      # pdb.set_trace()
      # plt.plot(wins)
      # plt.show()


class Modeling:
  @staticmethod 
  def save_model(model, path, is_tpu):
    save_locally = tf.saved_model.SaveOptions(experimental_io_device='/job:localhost') if is_tpu else None
    model.save(path, options=save_locally)

  @staticmethod
  def _load_model(path, is_tpu):
    load_locally = tf.saved_model.LoadOptions(experimental_io_device='/job:localhost') if is_tpu else None

    model = tf.keras.models.load_model(path, compile=False, options=load_locally,  custom_objects={'ImprovedCCE': ImprovedCCE})
    if compile:
      model.compile(loss=ImprovedCCE(), optimizer='adam', run_eagerly=False if is_tpu else True, metrics=['accuracy'])
    return model

  @staticmethod 
  def load_model(path, strategy, compile=False):
    if strategy is not None:
      with strategy.scope():
        return Modeling._load_model(path, True)
    else:
        return Modeling._load_model(path, False)
    
  @staticmethod
  def tf_seed(seed=0):
    os.environ['PYTHONHASHSEED'] = str(seed)
    # For working on GPUs from "TensorFlow Determinism"
    os.environ["TF_DETERMINISTIC_OPS"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)

  @staticmethod
  def get_model(loss, len_bars, len_cols):
    Modeling.tf_seed(1)
    
    i1 = keras.layers.Input(shape=(len_bars, len_cols))

    ts1 = keras.layers.LSTM(64, return_sequences=True, recurrent_regularizer=tf.keras.regularizers.l2(0.4))(i1)
    ts1 = keras.layers.Dropout(0.4)(ts1)

    ts1 = keras.layers.LSTM(64, recurrent_regularizer=tf.keras.regularizers.l2(0.4))(ts1)
    ts1 = keras.layers.Dropout(0.4)(ts1)

    # d = Attention(30)(ts1_n)


    i2 = tf.keras.layers.Input(shape=(9))
    ohl = tf.keras.layers.Dense(units=8, activation='relu')(i2)

    merge = tf.keras.layers.concatenate([ts1, ohl])
    d = tf.keras.layers.Dense(units=32, activation='relu')(merge)
    d = keras.layers.Dropout(0.5039079864662757)(d)

    o = keras.layers.core.Dense(3, activation='softmax')(d)

    model = tf.keras.Model(inputs=[i1, i2], outputs=o)

    if loss == 'ImprovedCCE':
      loss = ImprovedCCE()
    
    model.compile(loss=loss, optimizer='adam', run_eagerly=True, metrics=['accuracy'])

    return model

  @staticmethod # Sharpe 0.45
  def get_experimental_model(len_bars, len_cols):
    Modeling.tf_seed(1)
     
    i1 = keras.layers.Input(shape=(len_bars, len_cols))

    ts1 = keras.layers.LSTM(64, return_sequences=True, recurrent_regularizer=tf.keras.regularizers.l2(0.4))(i1)
    ts1 = keras.layers.Dropout(0.4)(ts1)

    ts1 = keras.layers.LSTM(64, return_sequences=True, recurrent_regularizer=tf.keras.regularizers.l2(0.4))(i1)
    ts1 = keras.layers.Dropout(0.4)(ts1)

    ts1 = keras.layers.LSTM(64, recurrent_regularizer=tf.keras.regularizers.l2(0.4))(ts1)
    ts1 = keras.layers.Dropout(0.4)(ts1)

    # d = Attention(30)(ts1_n)


    i2 = tf.keras.layers.Input(shape=(58))
    ohl = tf.keras.layers.Dense(units=24, activation='relu')(i2)

    merge = tf.keras.layers.concatenate([ts1, ohl])
    d = tf.keras.layers.Dense(units=32, activation='relu')(merge)
    d = keras.layers.Dropout(0.5039079864662757)(d)

    o = keras.layers.core.Dense(3, activation='softmax')(d)

    model = tf.keras.Model(inputs=[i1, i2], outputs=o)
    
    model.compile(loss=ImprovedCCE(), optimizer='adam', run_eagerly=True, metrics=['accuracy'])

    return model

  @staticmethod
  def get_experimental_model2(len_bars, len_cols):
    Modeling.tf_seed(1)
     
    i1 = keras.layers.Input(shape=(len_bars, len_cols))

    ts1 = keras.layers.LSTM(256, return_sequences=True, recurrent_regularizer=tf.keras.regularizers.l2(0.3))(i1)
    ts1 = keras.layers.BatchNormalization()(ts1)
    ts1 = keras.layers.Dropout(0.3)(ts1)

    ts1 = keras.layers.LSTM(256, return_sequences=True, recurrent_regularizer=tf.keras.regularizers.l2(0.3))(ts1)
    ts1 = keras.layers.BatchNormalization()(ts1)
    ts1 = keras.layers.Dropout(0.3)(ts1)

    ts1 = keras.layers.LSTM(256, recurrent_regularizer=tf.keras.regularizers.l2(0.3))(ts1)
    ts1 = keras.layers.BatchNormalization()(ts1)
    ts1 = keras.layers.Dropout(0.3)(ts1)

    # d = Attention(30)(ts1_n)

    i2 = tf.keras.layers.Input(shape=(58))
    ohl = tf.keras.layers.Dense(units=32, activation='relu')(i2)

    merge = tf.keras.layers.concatenate([ts1, ohl])
    d = tf.keras.layers.Dense(units=64, activation='relu')(merge)
    # d = keras.layers.Dropout(0.5039079864662757)(d)

    o = keras.layers.core.Dense(3, activation='softmax')(d)

    model = tf.keras.Model(inputs=[i1, i2], outputs=o)
    
    model.compile(loss=ImprovedCCE(), optimizer='adam', run_eagerly=True, metrics=['accuracy'])

    return model

  @staticmethod
  def get_model_without_tk_and_sl(len_bars, len_cols):
    i1 = keras.layers.Input(shape=(len_bars, len_cols))

    ts1 = keras.layers.LSTM(64, return_sequences=True, recurrent_regularizer=tf.keras.regularizers.l2(0.4))(i1)
    ts1 = keras.layers.Dropout(0.4)(ts1)

    ts1 = keras.layers.LSTM(64, recurrent_regularizer=tf.keras.regularizers.l2(0.4))(ts1)
    ts1 = keras.layers.Dropout(0.4)(ts1)

    d = tf.keras.layers.Dense(units=32, activation='relu')(ts1)
    d = keras.layers.Dropout(0.5039079864662757)(d)

    o = keras.layers.core.Dense(3, activation='softmax')(d)

    model = tf.keras.Model(inputs=[i1], outputs=o)
    
    model.compile(loss=ImprovedCCE(), optimizer='adam', run_eagerly=True, metrics=['accuracy'])

    return model

  @staticmethod
  def get_convlstm(len_bars, len_cols, strategy):
    with strategy.scope():

      Modeling.tf_seed(1)
      
      i1 = keras.layers.Input(shape=(10, len_cols, 5, 1), name='feature')

      ts1 = keras.layers.ConvLSTM2D(128, kernel_size=(1,3), return_sequences=True, padding='same', recurrent_regularizer=tf.keras.regularizers.l2(0.4))(i1)
      ts1 = keras.layers.Dropout(0.4)(ts1)

      ts1 = keras.layers.ConvLSTM2D(128, kernel_size=(1,3), return_sequences=True, padding='same', recurrent_regularizer=tf.keras.regularizers.l2(0.4))(i1)
      ts1 = keras.layers.Dropout(0.4)(ts1)

      ts1 = keras.layers.ConvLSTM2D(128, kernel_size=(1,3), padding='same', recurrent_regularizer=tf.keras.regularizers.l2(0.4))(ts1)
      ts1 = keras.layers.Dropout(0.4)(ts1)

      ts1 = tf.keras.layers.Flatten()(ts1)

      i2 = tf.keras.layers.Input(shape=(58), name='feature2')
      ohl = tf.keras.layers.Dense(units=24, activation='relu')(i2)

      merge = tf.keras.layers.concatenate([ts1, ohl])
      d = tf.keras.layers.Dense(units=32, activation='relu')(merge)
      d = keras.layers.Dropout(0.5039079864662757)(d)

      o = keras.layers.core.Dense(3, activation='softmax', name='label')(d)

      model = tf.keras.Model(inputs=[i1, i2], outputs=o)
      
      model.compile(loss=ImprovedCCE(), optimizer='adam', metrics=['accuracy'])

    return model
