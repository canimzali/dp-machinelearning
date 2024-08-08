streamlit run streamlit_app.py


import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle

# 1. Veri setini yükleme
st.title('Sigorta Fiyatı Tahmin Uygulaması')
st.write("Sigorta veri setini yükleyin ve modeli eğitin.")

uploaded_file = st.file_uploader("Bir CSV dosyası yükleyin", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv('https://raw.githubusercontent.com/canimzali/dp-machinelearning/master/insurance.csv')
    st.write(df.head())

    # 2. Veri ön işleme
    st.write("Veri ön işleme yapılıyor...")
    
    le_sex = LabelEncoder()
    le_smoker = LabelEncoder()
    le_region = LabelEncoder()
    
    df['sex'] = le_sex.fit_transform(df['sex'])
    df['smoker'] = le_smoker.fit_transform(df['smoker'])
    df['region'] = le_region.fit_transform(df['region'])

    X = df.drop('charges', axis=1)
    y = df['charges']

    # 3. Veriyi eğitim ve test setlerine ayırma
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 4. Modeli eğitme
    st.write("Model eğitiliyor...")
    model = XGBRegressor()
    model.fit(X_train, y_train)

    st.write("Model başarıyla eğitildi!")

    # 5. Modeli kaydetme
    with open('trained_model.pkl', 'wb') as file:
        pickle.dump(model, file)
    st.write("Model kaydedildi.")

    

