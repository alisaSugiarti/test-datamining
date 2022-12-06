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

st.markdown("#4. Implementasi Model")


# st.header("Input Data Sample")
# datain = st.text_input('Masukkan dataset', '')
st.title("Implementasi Model")
st.write("Sebagai bahan eksperimen silahkan inputkan beberapa data yang akan digunakan sebagai data testing untuk pengklasifikasian")

st.header("Input Data Testing")
# create input
t = st.number_input("Transaction Date")
ha = st.number_input("House Age")
d = st.number_input("Distance to the Nearest MRT Station")
n = st.number_input("Number of Convenience Stores")
la = st.number_input("Latitude")
lo = st.number_input("Longitude")
hp = st.number_input("House Price of Unit Area")

def submit():
    # input
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

    # olah Inputan
    a = np.array([[ha, la, lo]])

    test_data = np.array(a).reshape(1, -1)
    test_data = pd.DataFrame(test_data)

    scaler = joblib.load(filename)
    test_d = scaler.fit_transform(test_data)
    # pd.DataFrame(test_d)

    # load knn
    knn = joblib.load(filenameModelKnn)
    pred = knn.predict(test_d)

    # load gausian
    gnb = joblib.load(filenameModelGau)
    pred = gnb.predict(test_d)

    # load gdecision tree
    d3 = joblib.load(filenameModelDT)
    pred = d3.predict(test_d)

    # load kmean
    km = joblib.load(filenameModelrmf)
    pred = km.predict(test_d)





    # button
    st.header("Data Input")
    st.write("Berikut ini tabel hasil input data testing yang akan diklasifikasi:")
    st.dataframe(a)
    
    st.header("Hasil Prediksi")
    K_Nearest_Naighbour, Naive_Bayes, Decision_Tree, Random_Forest = st.tabs(["K-Nearest aighbour", "Naive Bayes Gausian", "Decision Tree", "Random Forest"])
    
    with K_Nearest_Naighbour:
        st.subheader("Model K-Nearest Neighbour")
        pred = knn.predict(test_d)
        if pred[0]== 'bottom 33' :
            st.write("Hasil Klasifikaisi : bottom 33")
        elif pred[0]== 'middle 33' :
            st.write("Hasil Klasifikaisi : middle 33")
        elif pred[0]== 'top 33' :
            st.write("Hasil Klasifikaisi : top 33")
        # else:
        #     st.write("Hasil Klasifikaisi : top 33")
        

    with Naive_Bayes:
        st.subheader("Model Naive Bayes Gausian")
        pred = gnb.predict(test_d)
        if pred[0]== 'bottom 33' :
            st.write("Hasil Klasifikaisi : bottom 33")
        elif pred[0]== 'middle 33' :
            st.write("Hasil Klasifikaisi : middle 33")
        elif pred[0]== 'top 33' :
            st.write("Hasil Klasifikaisi : top 33")
        # else:
        #     st.write("Hasil Klasifikaisi : top 33")

    with Decision_Tree:
        st.subheader("Model Decision Tree")
        pred = d3.predict(test_d)
        if pred[0]== 'bottom 33' :
            st.write("Hasil Klasifikaisi : bottom 33")
        elif pred[0]== 'middle 33' :
            st.write("Hasil Klasifikaisi : middle 33")
        elif pred[0]== 'top 33' :
            st.write("Hasil Klasifikaisi : top 33")
        # else:
        #     st.write("Hasil Klasifikaisi : top 33")

    with Random_Forest:
        st.subheader("Model Random Forest")
        pred = km.predict(test_d)
        if pred[0]== 'bottom 33' :
            st.write("Hasil Klasifikaisi : bottom 33")
        elif pred[0]== 'middle 33' :
            st.write("Hasil Klasifikaisi : middle 33")
        elif pred[0]== 'top 33' :
            st.write("Hasil Klasifikaisi : top 33")
        # else:
        #     st.write("Hasil Klasifikaisi : top 33")

submitted = st.button("Submit")
if submitted:
    submit()





