# S2.1: Open Sublime text editor, create a new Python file, copy the following code in it and save it as 'glass_type_app.py'.
# You have already created this ML model in ones of the previous classes.

# Importing the necessary Python modules.
import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score 

# ML classifier Python modules
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Loading the dataset.
@st.cache()
def load_data():
  df = pd.read_csv('glass-types.csv', header = None)
  df.drop(columns = 0, inplace = True)
  list1 = ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'GlassType']
  for i in range(1, 11):
    df.rename(columns = {i : list1[i - 1]}, inplace = True)
  return df

glass_df = load_data()
# Creating the features data-frame holding all the columns except the last column.
x = glass_df.drop(columns = 'GlassType')
# Creating the target series that holds last column.
y = glass_df['GlassType']
# Spliting the data into training and testing sets.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.33, random_state = 12)

@st.cache()
def prediction(model, RI, Na, Mg, Al, Si, K, Ca, Ba, Fe):
  glass_type = model.predict([[RI, Na, Mg, Al, Si, K, Ca, Ba, Fe]])
  glass_type = glass_type[0]
  if glass_type == 1:
    return 'building windows float processed'.upper()
  elif glass_type == 2:
    return 'building windows non float processed'.upper()
  elif glass_type == 3:
    return 'vehicle windows float processed'.upper()
  elif glass_type == 4:
    return 'vehicle windows non float processed'.upper()
  elif glass_type == 5:
    return 'containers'.upper()
  elif glass_type == 6:
    return 'tableware'.upper()
  else:
    return 'headlamp'.upper() 

st.title('Glass Type Predictor')
st.sidebar.title('Exploratory Data Anaylsis')
if st.sidebar.checkbox('Show raw data'):
  st.subheader('Full Dataset')
  #st.write(glass_df)
  st.dataframe(glass_df)
  #st.table(glass_df)
# Add a subheader in the sidebar with label "Scatter Plot".
st.sidebar.subheader('Scatter Plot')
# Choosing x-axis values for the scatter plot.
# Add a multiselect in the sidebar with the 'Select the x-axis values:' label
# and pass all the 9 features as a tuple i.e. ('RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe') as options.
# Store the current value of this widget in the 'features_list' variable.
feature = st.sidebar.multiselect('Select the x-axis values', ('RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe'))
# Remove deprecation warning.
st.set_option('deprecation.showPyplotGlobalUse', False)
for i in feature:
  st.subheader(f'Scatter plot beetween {i} and GlassType')
  plt.figure(figsize = (20,8))
  sns.scatterplot(x = glass_df[i], y = glass_df['GlassType'])
  st.pyplot()
# Sidebar for histograms.
st.sidebar.subheader('Histogram')
# Choosing features for histograms.
hist_feature = st.sidebar.multiselect('Select features to create histograms', ('RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe'))
# Create histograms.
for i in hist_feature:
  st.subheader(f'Histogram for {i}')
  plt.figure(figsize = (20,8))
  plt.hist(glass_df[i], bins = 'sturges', edgecolor = 'red')
  st.pyplot()
# Sidebar for box plots.
st.sidebar.subheader('Box Plot')
# Choosing columns for box plots.
box_feature = st.sidebar.multiselect('Select features to create boxplots', ('RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe'))
# Create box plots.
for i in box_feature:
  st.subheader(f'Boxplot for {i}')
  plt.figure(figsize = (20,8))
  sns.boxplot(glass_df[i])
  st.pyplot()