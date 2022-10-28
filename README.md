# Trading hypothesis tester

This project is aimed at testing the trading hypothesis.

### Trading hypothesis
Using LSTM and a custom loss function, we can predict whether to buy, sell or do nothing based on OHLC data, indicator data, and take profit and stop loss data that equals the highest OHLC in the last x OHLC.

### Result
Many experiments were carried out with different configurations of data, indicators, neural networks, loss functions. 
The result was on average at the Sharp level of 0.1, which is not a significant result.

### Analysis
Most likely, the main problem is the simplicity and unbiasedness of the strategy, because next more complex strategies have a much higher Sharp coefficient. So this is the main room for improvement.
