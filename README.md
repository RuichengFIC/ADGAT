# Modeling the Momentum Spillover Effect for Stock Prediction via Attribute-Driven Graph Attention Networks
Code & Data for the stock prediction model in our paper: Modeling the Momentum Spillover Effect for Stock Prediction via
Attribute-Driven Graph Attention Networks.

## Environment
Python 3.7.6 & Pytorch 1.5.1 

## Run
```sh
$ python main.py --device=0
```
Make sure that the GPU is used to reproduce our experiments.

## Data
All the preprocessed data can be found at  ./data. 

##### Selected Stock
The selected 198 tickers can be found at ./raw_data/stocks.txt

##### Transcational Data
The raw market data can be download from https://drive.google.com/file/d/14B3EubjMzNYGAYDkTEIMeTJ3weXQycw6/view?usp=sharing.
Please refer to the notebook "rawdata/marketdata_preprocessing.ipynb" for preprocessing. 

##### Sentiment Indicators
The news data are not provided in our repository due to copyright issues.

We use financial news from Reuters and Bloomberg over the period from 2011 to 2013, released by Ding et al. ("Using structured events to predict stock price movement: An empirical investigation." EMNLP. 2014.)  

The Loughran-McDonald Master Dictionary (https://sraf.nd.edu/textual-analysis/resources/) is used to extract sentiment from financial articles.

##### Company Relations
Firm relations can be found at ./raw_data/relations/*.
The Company Relations are collected from S&P Capical IQ (https://www.capitaliq.com/). 
The first row is the target stock tickers, and following rows are firms that has specific relation with that frim.

## Contact
chengrui0108@hotmail.com
