import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error as MSE
from sklearn.linear_model import LinearRegression


def get_lib():
    return np, pd, StandardScaler, LinearRegression, MSE,
