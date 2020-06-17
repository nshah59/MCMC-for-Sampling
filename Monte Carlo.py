import math
import random
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.stats import norm


def getvolatility(ticker):
    returnlist = getreturns(ticker)
    return returnlist.std()


def getreturns(ticker):
    data_df = yf.download(ticker, start='2015-03-24', end='2020-03-24')['Adj Close']
    return data_df.pct_change().dropna()


def metropolis(current, data, volatility):
    x = np.random.normal(0,volatility)
    trial = current + x
    print(current)
    print(trial)
    ratio_likelihood = likelihood(trial, current, data, volatility)
    prior_trial = norm.pdf(trial, np.mean(data), volatility)
    prior_current = norm.pdf(current, np.mean(data), volatility)
    ratio = ratio_likelihood * prior_trial / prior_current
    ratio = ratio_likelihood
    if random.uniform(0, 1) < ratio:
        return trial
    else:
        return current


def prior(x):
    return norm.pdf(x)


def likelihood(trial, current, data, volatility):
    ratio = 1
    for rtn in data.values:
        prob_trial = norm.pdf(rtn, trial, volatility)
        prob_current = norm.pdf(rtn, current, volatility)
        ratio *= (prob_trial / prob_current)
        print(rtn)
        print(prob_trial)
        print(prob_current)
        print("Ratio")
        print(ratio)
    return ratio


def sampler(iterations, first, data, volatility):
    accepted = [first]
    for i in range(1, iterations):
        accept_val = metropolis(accepted[i - 1],  data, volatility)
        accepted += [accept_val]
    return accepted


def main():
    x = sampler(20, np.mean(getreturns("AAPL")), getreturns("AAPL"), getvolatility("AAPL"))
    print(x)
    plt.hist(x, bins=10)
    plt.show()
    print(getreturns("AAPL"))
    plt.xlim(-.1, .1)
    plt.hist(getreturns("AAPL"), bins=10)
    plt.show()


if __name__ == "__main__":
    main()
