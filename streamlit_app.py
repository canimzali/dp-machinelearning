import streamlit as st
import pandas as pd 

st.title('🤖 Predict of Insurance Price')

st.write('This is a machine learning app')

df=pd.read_csv("https://raw.githubusercontent.com/canimzali/dp-machinelearning/master/insurance.csv")
df
