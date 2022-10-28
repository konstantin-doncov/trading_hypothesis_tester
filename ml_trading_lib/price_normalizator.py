import tqdm
import pandas as pd

class PriceNormalizator:
  @staticmethod
  def normalize_price(price, min_p, max_p):
    return (price - min_p) / (max_p - min_p)
    
  @staticmethod
  def unnormalize_price(normalized_price, min_p, max_p):
    return normalized_price * (max_p - min_p) + min_p
