import pandas as pd
import streamlit as st
import plotly.express as px
import numpy as np
from sklearn.utils.validation import joblib

# intial template
px.defaults.template = "plotly_dark"
px.defaults.color_continuous_scale = "reds"

st.markdown("# 1. Informasi Data")
# create content
st.title("Real estate price prediction")
st.container()
st.write("Website ini bertujuan untuk memprediksi harga perumahan berdasarkan latitude, longitude, dan house age")

st.header("Informasi Data")

# read data
st.write("""
Dalam dataset yang digunakan terdapat beberapa variabel/fitur yaitu:
* transaction date    : tahun pembelian rumah
* house age           : umur rumah dalam hitungan tahun
* distance to the nearest MRT station : jarak perumahan ke stasiun MRT (satuan M)
* number of convenience stores        : jumlah minimarket terdekat dari perumahan
* latitude            : ukuran lintang/panjang tanah (satuan meter persegi)
* longitude           : ukuran bujur/lebar tanah (satuan meter persegi)
* house price of unit area            : harga rumah per unit

Dari beberapa variabel/fitur diatas fitur(variabel independen) yang akan digunakan untuk memprediksi harga rumah adalah latitude, longitude, dan house price of unit area

Karena klasifikasi KNN bergantung pada voting mayoritas, memiliki data yang tidak seimbang akan menyulitkan algoritma untuk memilih apa pun selain kelas mayoritas. 
oleh karena itu akan membuat tiga band seimbang sebagai gantinya yang mengelompokkan pengamatan berdasarkan harga unit area menjadi :
* 33% terbawah
* 33% tengah
* 33% teratas 
""")

st.header("Sumber Data")
st.write("Dataset yang digunakan dalam percobaan ini diambil dari Kaggle dengan jumlah data sebanyak 414 data dengan 7 fitur")
st.caption('link datasets : https://www.kaggle.com/datasets/quantbruce/real-estate-price-prediction?select=Real+estate.csv')
