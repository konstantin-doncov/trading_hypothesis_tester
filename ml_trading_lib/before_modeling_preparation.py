class BeforeModelingPreparation:
  def features_dataframe_to_lstm_dataframe(f, len_bars, len_cols):
    return f.reshape(len(f), len_bars, len_cols)