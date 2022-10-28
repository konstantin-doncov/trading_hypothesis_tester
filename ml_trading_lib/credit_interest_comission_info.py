import backtrader as bt

class CreditInterestCommissionInfo(bt.CommissionInfo):
  iter = 0
  opening_calculated = False


  def get_credit_interest(self, data, pos, dt):
    '''Calculates the credit due for short selling or product specific'''
    size, price = pos.size, pos.price

    if size > 0 and not self.p.interest_long:
        return 0.0  # long positions not charged
    
    four_hours = 14400
    self.iter += (dt - pos.datetime).seconds

    if self.iter == four_hours:
      self.iter = 0      
      return pos.price * abs(pos.size) * rollover_fee

    if not self.opening_calculated:
      self.opening_calculated = True
      
      return pos.price * abs(pos.size) * opening_fee

    return 0.0
        
