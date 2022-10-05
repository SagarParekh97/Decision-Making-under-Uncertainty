from pgmpy.estimators import BDsScore, K2Score, BicScore
from pgmpy.models import BayesianModel
import pandas as pd
import numpy as np


data = pd.read_csv('heart.csv')


