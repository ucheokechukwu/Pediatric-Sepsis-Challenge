from helper_functions.imputer import CustomIterativeImputer
from helper_functions.visualizations import *
from helper_functions.feature_transformer import *
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from rich import pretty
import warnings
warnings.filterwarnings("ignore")

pretty.install()


data = pd.read_csv(
    "../../preliminar_phase_submision/training_data/SyntheticData_Training.csv")
dictionary = pd.read_csv("tmp_datasets/DataDictionary.csv")
target = "inhospital_mortality"

# multi_imputer = CustomIterativeImputer()
X = data.drop(columns=['inhospital_mortality', 'studyid_adm', 'lengthadm'],
              errors='ignore')
y = data["inhospital_mortality"]
print("""Loaded: 
      data: the entire dataset.
      dictionary: the data dictionary.
      X: the features columns with 'inhospital_mortality', 'studyid_adm', 'lengthadm' dropped.
      y: the target column.""")
