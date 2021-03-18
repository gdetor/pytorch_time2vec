# pytorch_time2vec

This repository contains a Pytorch implementation of the Time2Vec Algorithm [1]. It
makes use of the Punta Salute 2009 dataset (historical levels of water in Venice)
as in [2]. The dataset shows the hourly water level (cm). Furthermore, we have
followed the analysis provided in [2] and thus a Bootstrapping script can be 
found in the current repository. 


## Code organization

  - **main.py** - Implements a simple time series prediction training and testing example
                  on the Livelo dataset.
  - **analysis.py** Implements a Bootstrapping and estimates the distributions of
                    the samples on the test dataset.
  - **model/network.py** Implements an LSTM and an LSTM equipped with a Time2Vec layer
  - **model/time2vec** Implents the Time2Vec layer
  - **data/livelo.npy** Raw dataset of Punta Salute 2009


## Example of usage
The time2vec layer can be used at will. In this repository, we provide a simple
example of how to use it along with an LSTM to predict the hourly water level
of Venice based on historical data (2009). 

To run the simple LSTM model type in:
```
$ python (or python3 depending on your system`s configuration) main.py lstm 
```

And to run the T2V-LSTM:
```
$ python main.py tv-lstm
```

When you run either of the two aforementioned examples the script will store 
the results of the test prediction to the directory **results** (you will need 
to create this directory before running the main.py script). 

If you'd like to run the Bootstrapping to get the distributions of the
test predictions you can run the **analysis.py** script (this script requires 
the results of the test predictions for both LSTM and T2V-LSTM models).
```
$ python analysis.py
```

## Requirements
 - Python 3
 - Numpy
 - Matplotlib
 - Pytorch
 

## Platform specifications
The current implementation has been tested and ran on the following software
configuration:
- GCC 8.3.0
- Ubuntu Linux 5.3.0-40-generic
- Python 3.6.9
- Numpy 1.18.2
- Matplotlib 3.1.1
- Pytorch 1.4.0 (No GPU)


## References

[1] "Time2Vec: Learning a vector representation of time", Kazemi et al., 2019

[2] [Time2Vec for Time Series features encoding](https://towardsdatascience.com/time2vec-for-time-series-features-encoding-a03a4f3f937e)
