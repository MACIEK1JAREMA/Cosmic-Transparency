'''
Attempt including a variety of H_0 in the H(z) model. Produce 2D chi^2
from this, plot it, get the 2D likelihood, plot it. Marginalise over H_0
with flat and gaussian priors. Plot 1D lieklihoods over Om
''' 

# import modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import Codes.Module as module
import matplotlib.cm as cm
import time

# %%

