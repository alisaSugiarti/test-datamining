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

st.markdown("#3. Modeling Data")

    # ambil data
data = pd.read_csv("https://raw.githubusercontent.com/alisaSugiarti/streamlit/main/Real%20estate.csv")

#diskritisasi data
data['Price Band'] = pd.qcut(data['Y house price of unit area'], 3, labels=['bottom 33', 'middle 33', 'top 33'])

#Normalisasi MinMax 
from sklearn.preprocessing import MinMaxScaler
data_for_minmax_scaler=pd.DataFrame(data, columns = ['X5 latitude',	'X6 longitude',	'X2 house age'])
data_for_minmax_scaler.to_numpy()
scaler = MinMaxScaler()
data_hasil_minmax_scaler=scaler.fit_transform(data_for_minmax_scaler)

import joblib
filename = "normalisasi_realEstatePrice.sav"
joblib.dump(scaler, filename) 

#hasil preprocessing
data_hasil_minmax_scaler = pd.DataFrame(data_hasil_minmax_scaler,columns = ['X5 latitude',	'X6 longitude',	'X2 house age'])
data_PriceBand = pd.DataFrame(data, columns = ['Price Band'])
data_new = pd.concat([data_hasil_minmax_scaler,data_PriceBand], axis=1)

X = data_hasil_minmax_scaler
y_baru = data_PriceBand
# split data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X, y_baru, test_size=0.2, random_state=1)

# inisialisasi knn
my_param_grid = {'n_neighbors':[2,3,5,7], 'weights': ['distance', 'uniform']}
GridSearchCV(estimator=KNeighborsClassifier(), param_grid=my_param_grid, refit=True, verbose=3, cv=3)
knn = GridSearchCV(KNeighborsClassifier(), param_grid=my_param_grid, refit=True, verbose=3, cv=3)
knn.fit(X_train, y_train)

pred_test = knn.predict(X_test)

vknn = f'Hasil akurasi dari pemodelan K-Nearest Neighbour : {accuracy_score(y_test, pred_test) * 100 :.2f} %'

filenameModelKnn = 'modelKnn.pkl'
joblib.dump(knn, filenameModelKnn)

# inisialisasi model gausian
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)

filenameModelGau = 'modelGauNB.pkl'
joblib.dump(gnb, filenameModelGau)

y_pred = gnb.predict(X_test)

vg = f'Hasil akurasi dari pemodelan Gausian : {accuracy_score(y_test, y_pred) * 100 :.2f} %'

# inisialisasi model decision tree
from sklearn.tree import DecisionTreeClassifier
d3 = DecisionTreeClassifier()
d3.fit(X_train, y_train)

filenameModelDT = 'modelDT.pkl'
joblib.dump(d3, filenameModelDT)

y_pred = d3.predict(X_test)

vd3 = f'Hasil akurasi dari pemodelan decision tree : {accuracy_score(y_test, y_pred) * 100 :.2f} %'

#bagging random forest
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=14, max_depth=2, random_state=0)
clf = clf.fit(X_train, y_train)

filenameModelrmf = 'modelrmf.pkl'
joblib.dump(d3, filenameModelrmf)

y_test_pred = clf.predict(X_test)
rmf = f'Hasil akurasi dari pemodelan Random Forest Classifier : {accuracy_score(y_test, y_test_pred) * 100 :.2f} %'

# tampilan tabs
K_Nearest_Naighbour, Gausian, Decision_Tree, Random_Forest = st.tabs(["K-Nearest aighbour", "Naive Bayes Gausian", "Decision Tree", "Random Forest"])


with K_Nearest_Naighbour:
    st.header("K-Nearest Neighbour")
    st.write("""
    K-Nearest Neighbor (KNN) adalah suatu metode yang menggunakan algoritma supervised dimana hasil dari query instance yang baru diklasifikan berdasarkan mayoritas dari kategori pada KNN. Tujuan dari algoritma KNN adalah untuk mengklasifikasi objek baru berdasarkan atribut dan training samples. Dimana hasil dari sampel uji yang baru diklasifikasikan berdasarkan mayoritas dari kategori pada KNN.
    \n**Langkah-Langkah Algoritma KNN Langkah-langkah pada algoritma KNN:**
    \n* Tentukan jumlah tetangga (K) yang akan digunakan untuk pertimbangan penentuan kelas.
    \n* Hitung jarak dari data baru ke masing-masing data point di dataset. Ambil sejumlah K data dengan jarak terdekat, kemudian tentukan kelas dari data baru tersebut.
    \n* Untuk mencari dekat atau jauhnya jarak antar titik pada kelas k biasanya dihitung menggunakan jarak Euclidean. Jarak Euclidean adalah formula untuk mencari jarak antara 2 titik dalam ruang dua dimensi.
    """)
    st.header("Pengkodean")
    st.text("""
    my_param_grid = {'n_neighbors':[2,3,5,7], 'weights': ['distance', 'uniform']}
    GridSearchCV(estimator=KNeighborsClassifier(), param_grid=my_param_grid, refit=True, verbose=3, cv=3)
    knn = GridSearchCV(KNeighborsClassifier(), param_grid=my_param_grid, refit=True, verbose=3, cv=3)
    knn.fit(X_train, y_train)

    filenameModelKnnNorm = 'modelKnn.pkl'
    joblib.dump(knn, filenameModelKnnNorm)

    pred_test = knn.predict(X_test)

    vknn = f'Hasil akurasi dari pemodelan K-Nearest Neighbour : {accuracy_score(y_test, pred_test) * 100 :.2f} %'
    """)
    st.header("Hasil Akurasi")
    st.write(vknn)
    
    

with Gausian:
    st.header("Naive Bayes Gausian")
    st.write("""
    Naive Bayes adalah algoritma machine learning untuk masalah klasifikasi. Ini didasarkan pada teorema probabilitas Bayes. Hal ini digunakan untuk klasifikasi teks yang melibatkan set data pelatihan dimensi tinggi. Beberapa contohnya adalah penyaringan spam, analisis sentimental, dan klasifikasi artikel berita.
    Algoritma Naive Bayes adalah teknik klasifikasi berdasarkan penerapan teorema Bayes dengan asumsi kuat bahwa semua prediktor independen satu sama lain. Secara sederhana dengan kata lain, asumsinya adalah bahwa kehadiran fitur di kelas tidak tergantung pada kehadiran fitur lain di kelas yang sama.
    """)
    st.header("Pengkodean")
    st.text("""
    from sklearn.naive_bayes import GaussianNB
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)

    filenameModelGau = 'modelGauNB.pkl'
    joblib.dump(gnb, filenameModelGau)

    y_pred = gnb.predict(X_test)

    vg = f'Hasil akurasi dari pemodelan Gausian : {accuracy_score(y_test, y_pred) * 100 :.2f} %'
        """)
    st.header("Hasil Akurasi")
    st.write(vg)   

with Decision_Tree:
    st.header("Decision Tree")
    st.write("Decision tree merupakan alat pendukung keputusan dengan struktur seperti pohon yang memodelkan kemungkinan hasil, biaya sumber daya, utilitas, dan kemungkinan konsekuensi. Konsepnya adalah dengan cara menyajikan algoritma dengan pernyataan bersyarat yang meliputi cabang untuk mewakili langkah-langkah pengambilan keputusan yang dapat mengarah pada hasil yang menguntungkan. Biasanya decision tree dimulai dari satu node atau satu simpul. Kemudian node tersebut bercabang untuk memberikan pilihan-pilihan Tindakan yang lain. Selanjutnya node tersebut akan memiliki cabang-cabang baru. Dalam pembuatan node atau cabang baru akan terus di ulang sampai kriteria berhenti dipenuhi. Decision tree biasanya dapat memperoses dataset yang berisi atribut nominal atau numerik. Label attribute harus berbentuk nominal untuk proses klasifikasi dan berbentuk numerik untuk regresi.")
    st.header("Pengkodean")
    st.text(""" 
    from sklearn.tree import DecisionTreeClassifier
    d3 = DecisionTreeClassifier()
    d3.fit(X_train, y_train)

    filenameModelDT = 'modelDT.pkl'
    joblib.dump(d3, filenameModelDT)

    y_pred = d3.predict(X_test)

    vd3 = f'Hasil akurasi dari pemodelan decision tree : {accuracy_score(y_test, y_pred) * 100 :.2f} %'
    """)
    st.header("Hasil Akurasi")
    st.write(vd3)   

with Random_Forest:
    st.header("Random Forest")
    st.write("Prinsip utama K-Means adalah menyusun k prototype atau pusat massa (centroid) dari sekumpulan data berdimensi. \nSebelum diterapkan proses algoritma K-means, dokumen akan di preprocessing terlebih dahulu. Kemudian dokumen direpresentasikan sebagai vektor yang memiliki term dengan nilai tertentu.")
    st.header("Pengkodean")
    st.text(""" 
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=14, max_depth=2, random_state=0)
    clf = clf.fit(X_train, y_train)

    filenameModelrmf = 'modelrmf.pkl'
    joblib.dump(d3, filenameModelrmf)

    y_test_pred = clf.predict(X_test)
    rmf = f'Hasil akurasi dari pemodelan Random Forest Classifier : {accuracy_score(y_test, y_test_pred) * 100 :.2f} %'
    """)
    st.header("Hasil Akurasi")
    st.write(rmf)