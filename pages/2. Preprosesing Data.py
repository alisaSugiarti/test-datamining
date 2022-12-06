import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import streamlit as st
import plotly.express as px
import numpy as np
from sklearn.utils.validation import joblib
from io import StringIO, BytesIO
import urllib.request
import joblib
import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import os,sys
from scipy import stats

# intial template
px.defaults.template = "plotly_dark"
px.defaults.color_continuous_scale = "reds"

st.markdown("# 2. Preprosesing Data")


# st.header("Input Data Sample")
# datain = st.text_input('Masukkan dataset', '')
st.title("Preprosesing Data")
st.write("Data preprocessing adalah proses mengubah data mentah ke dalam bentuk yang lebih mudah dipahami. Proses ini diperlukan untuk memperbaiki kesalahan pada data mentah yang seringkali tidak lengkap dan memiliki format yang tidak teratur.")
st.header("Import Data")
uploaded_files = st.file_uploader("Upload file CSV", accept_multiple_files=True)
for uploaded_file in uploaded_files:
    # uplod file
    data = pd.read_csv(uploaded_file)
    st.write(" **Nama File yang Di-Upload :** ", uploaded_file.name)
   
    # view dataset asli
    st.header("Dataset Asli")
    # X = data.drop(['No'])
    st.dataframe(data)
    row, col = data.shape 
    st.caption(f"({row} rows, {col} cols)")

    #diskritisasi data
    st.header("Diskritisasi Data")
    st.write("""
    Deskritisasi atau binning digunakan untuk mengubah atribut numerik menjadi atribut kategorikal. Perubahan tersebut dilakukan dengan mengkategorikan atribut numerik menjadi beberapa tingkatan atribut kategorikal.
    \n**Diskritisasi Data dengan Equal Frequency Intervals**
    \nEqual-frequency intervals adalah discretization yang membagi data numerik menjadi beberapa kelompok dengan jumlah anggota yang kurang lebih sama besar.
    \ndata akan dikelompokkan menjadi tiga band seimbang berdasarkan harga unit area menjadi :
    \n* 33% terbawah
    \n* 33% tengah
    \n* 33% teratas 

    \n**Dataset Hasil Diskritisasi Data**
    """)
    data['Price Band'] = pd.qcut(data['Y house price of unit area'], 3, labels=['bottom 33', 'middle 33', 'top 33'])
    st.dataframe(data)
    row, col = data.shape 
    st.caption(f"({row} rows, {col} cols)")

    #fitur yang akan digunakan untuk prediksi
    st.header("Pemisahan Fitur Yang Akan Digunakan Untuk Prediksi ")
    st.write("Dari beberapa variabel/fitur diatas fitur(variabel independen) yang akan digunakan untuk memprediksi harga rumah adalah latitude, longitude, dan house price of unit area")
    data_without_column_for_convert = pd.DataFrame(data, columns = ['X5 latitude',	'X6 longitude',	'X2 house age'])
    st.dataframe(data_without_column_for_convert)
    row, col = data.shape 
    st.caption(f"({row} rows, {col} cols)")

    #Normalisasi Data
    st.header("Normalisasi Data")
    st.write("""
    Normalisasi disini bukan normalisasi yang dilakukan pada database. Normalisasi disini merupakan normalisasi pada Data Mining proses penskalaan nilai atribut dari data sehingga bisa jatuh pada range tertentu.
    \n**Min-Max**
    \nMetode Min-Max merupakan metode normalisasi dengan melakukan transformasi linier terhadap data asli
    \n**rumus newdata = (data-min)*(newmax-newmin)/(max-min)+newmin**
    \nketerangan:
    \nnewdata= data hasil normalisasi 
    \nmin = nilai minimum dari data per kolom 
    \nmax = nilai maximum dari data per kolom 
    \nnewmin = batas minimum yang kita berikan 
    \nnewmax = batas maximum yang kita berikan

    \n**Data Hasil Normalisasi**
    """)
    from sklearn.preprocessing import MinMaxScaler
    data_for_minmax_scaler=pd.DataFrame(data, columns = ['X5 latitude',	'X6 longitude',	'X2 house age'])
    data_for_minmax_scaler.to_numpy()
    scaler = MinMaxScaler()
    data_hasil_minmax_scaler=scaler.fit_transform(data_for_minmax_scaler)

    import joblib
    filename = "normalisasi_realEstatePrice.sav"
    joblib.dump(scaler, filename) 

    data_hasil_minmax_scaler = pd.DataFrame(data_hasil_minmax_scaler,columns = ['X5 latitude',	'X6 longitude',	'X2 house age'])
    st.dataframe(data_hasil_minmax_scaler)
    row, col = data.shape 
    st.caption(f"({row} rows, {col} cols)")

    #view target class prediksi
    st.header("Target Class Prediksi")
    data_PriceBand = pd.DataFrame(data, columns = ['Price Band'])
    st.dataframe(data_PriceBand)
    row, col = data.shape 
    st.caption(f"({row} rows, {col} cols)")


    # view dataset hasil preprocessing
    st.header("Dataset Hasil Preprocessing")
    data_new = pd.concat([data_hasil_minmax_scaler,data_PriceBand], axis=1)
    st.dataframe(data_new)
    row, col = data.shape 
    st.caption(f"({row} rows, {col} cols)")


  