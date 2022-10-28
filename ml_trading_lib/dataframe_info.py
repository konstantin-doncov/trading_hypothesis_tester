class DataframeInfo:
  def __init__(self, df, name, min_p, max_p, size):
    self.df = df
    self.name = name
    self.min_p = min_p
    self.max_p = max_p
    self.size = size