import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import datetime
from kaggle.competitions import nflrush
import tqdm
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import keras

sns.set_style('darkgrid')
mpl.rcParams['figure.figsize'] = [15,10]

train = pd.read_csv('C://NFL_train.csv')

print(train.shape)
print(train.head())