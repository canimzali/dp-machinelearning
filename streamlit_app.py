

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
from sklearn.ensemble import RandomForestRegressor


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
    model=RandomForestRegressor()
    model.fit(X_train, y_train)

    st.write("Model başarıyla eğitildi!")

    # 5. Modeli kaydetme
    with open('trained_model.pkl', 'wb') as file:
        pickle.dump(model, file)
    st.write("Model kaydedildi.")

    # 6. Tahmin yapma
    st.write("Yeni müşteri bilgilerini girin ve tahmini sigorta fiyatını öğrenin.")
    
    age = st.number_input('Yaş', min_value=18, max_value=100, value=30)
    sex = st.selectbox('Cinsiyet', le_sex.classes_)
    bmi = st.number_input('BMI (Vücut Kitle İndeksi)', min_value=10.0, max_value=50.0, value=25.0)
    children = st.number_input('Çocuk Sayısı', min_value=0, max_value=10, value=0)
    smoker = st.selectbox('Sigara Kullanımı', le_smoker.classes_)
    region = st.selectbox('Bölge', le_region.classes_)

    input_data = pd.DataFrame({
        'age': [age],
        'sex': [le_sex.transform([sex])[0]],
        'bmi': [bmi],
        'children': [children],
        'smoker': [le_smoker.transform([smoker])[0]],
        'region': [le_region.transform([region])[0]]
    })

    if st.button('Tahmin Et'):
        prediction = model.predict(input_data)[0]
        st.write(f"Tahmini Sigorta Fiyatınız: ${prediction:.2f}")
    

