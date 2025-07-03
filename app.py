import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier, plot_importance
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb

# ----------------- CONFIG
st.set_page_config(page_title="Prediksi Status Stok", layout="wide")

# ----------------- HEADER
st.markdown("""
    <h1 style='text-align: center;'>📦 Prediksi Status Stok Barang (XGBoost)</h1>
    <p style='text-align: center; font-size: 18px;'>Upload data sparepart, analisis korelasi, dan latih model XGBoost untuk klasifikasi status stok</p>
""", unsafe_allow_html=True)

# ----------------- UPLOAD SECTION
st.subheader("📤 Upload File CSV")
uploaded_file = st.file_uploader("Pilih file dataset sparepart (.csv)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, sep=';')
    st.success("✅ File berhasil diupload!")

    # ----------------- PREVIEW DATA
    with st.expander("👁️ Preview Data (5 Baris Pertama)", expanded=True):
        st.dataframe(df.head(), use_container_width=True)

    # ----------------- DATA CLEANING & PREP
    df['Demand'] = df['Penggunaan Terakhir (2024)']
    df['Forecast'] = df['Forecast (2025)']
    df['Lead Time'] = df['Lead Time (Month)']
    df['Pola Pergerakan'] = df['MOVEMENT'].map({'Fast Moving': 0, 'Slow Moving': 1, 'Non Moving': 2})
    df['Pola Permintaan'] = df['Kategori'].map({'Smooth': 0, 'Erratic': 1, 'Lumpy': 2})
    df['Klasifikasi ABC'] = df['Kelas'].map({'A': 0, 'B': 1, 'C': 2})
    df['Status'] = df['Status'].map({'Normal': 0, 'Understock': 1, 'Overstock': 2})
    df['Inventory Level'] = df['SOH']
    df = df.drop(columns=['Lead Time (Month)', 'Kategori', 'Kelas'])

    df = df[['Demand', 'Forecast', 'Inventory Level', 'Safety Stock',
             'Lead Time', 'Pola Pergerakan', 'Klasifikasi ABC', 'Status']]

    # ----------------- BUTTONS SECTION
    col1, col2 = st.columns(2)
    show_corr = col1.button("📊 Tampilkan Korelasi")
    run_model = col2.button("🚀 Latih Model XGBoost")

    # ----------------- KORELASI
    if show_corr:
        st.subheader("📈 Korelasi Antar Fitur")
        fig_corr, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(df.corr(), annot=True, cmap="BrBG", ax=ax)
        st.pyplot(fig_corr)

    # ----------------- MODELING
    if run_model or 'model' in st.session_state:
        st.subheader("🧠 Pelatihan Model XGBoost")

        if 'model' not in st.session_state:
            X = df.drop(columns=['Status'])
            y = df['Status']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                                stratify=y, random_state=42)

            xgb_params = dict(
                objective='multi:softprob',
                eval_metric=['mlogloss', 'merror'],
                n_estimators=1000,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.5,
                colsample_bytree=1,
                reg_lambda=50.0,
                reg_alpha=10.0,
                random_state=42)

            model = XGBClassifier(**xgb_params)
            eval_set = [(X_train, y_train), (X_test, y_test)]
            model.fit(X_train, y_train, eval_set=eval_set, verbose=False)

            st.session_state.model = model
            st.session_state.X_train = X_train
            st.session_state.X_test = X_test
            st.session_state.y_train = y_train
            st.session_state.y_test = y_test
            st.session_state.results = model.evals_result()

        model = st.session_state.model
        X_train = st.session_state.X_train
        X_test = st.session_state.X_test
        y_train = st.session_state.y_train
        y_test = st.session_state.y_test
        results = st.session_state.results

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        acc_train = accuracy_score(y_train, y_train_pred)
        acc_test = accuracy_score(y_test, y_test_pred)

        col3, col4 = st.columns(2)
        col3.metric("🎯 Akurasi Training", f"{acc_train:.2%}")
        col4.metric("📊 Akurasi Testing", f"{acc_test:.2%}")

        with st.expander("📜 Classification Report"):
            st.text(classification_report(y_test, y_test_pred))

        st.subheader("📌 Confusion Matrix")
        fig_cm, ax = plt.subplots()
        cm = confusion_matrix(y_test, y_test_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=['Normal', 'Understock', 'Overstock'],
                    yticklabels=['Normal', 'Understock', 'Overstock'], ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig_cm)

        st.subheader("📉 Learning Curve")
        fig_lc, ax = plt.subplots(1, 2, figsize=(12, 4))
        ax[0].plot(results['validation_0']['mlogloss'], label='Train')
        ax[0].plot(results['validation_1']['mlogloss'], label='Test')
        ax[0].set_title("Log Loss")
        ax[0].legend()
        ax[1].plot(results['validation_0']['merror'], label='Train')
        ax[1].plot(results['validation_1']['merror'], label='Test')
        ax[1].set_title("Error Rate")
        ax[1].legend()
        st.pyplot(fig_lc)

        st.subheader("📌 Feature Importance")
        fig_fi, ax = plt.subplots(figsize=(10, 5))
        plot_importance(model, ax=ax)
        st.pyplot(fig_fi)

        # ----------------- PREDIKSI MANUAL
        st.subheader("🔍 Coba Prediksi Manual")
        with st.form("manual_input"):
            st.markdown("Masukkan nilai fitur untuk memprediksi status stok:")

            col1, col2, col3 = st.columns(3)
            with col1:
                demand = st.number_input("Demand (Penggunaan 2024)", step=0.1, format="%.2f")
                forecast = st.number_input("Forecast (2025)", step=0.1, format="%.2f")
                inventory = st.number_input("Inventory Level (SOH)", step=0.1, format="%.2f")
            with col2:
                safety = st.number_input("Safety Stock", step=0.1, format="%.2f")
                leadtime = st.number_input("Lead Time (Month)", step=0.1, format="%.2f")
            with col3:
                movement_label = st.selectbox("Pola Pergerakan", ['Fast Moving', 'Slow Moving', 'Non Moving'])
                abc_label = st.selectbox("Klasifikasi ABC", ['A', 'B', 'C'])

            submitted = st.form_submit_button("🔎 Prediksi Status Stok")

            if submitted:
                movement = {'Fast Moving': 0, 'Slow Moving': 1, 'Non Moving': 2}[movement_label]
                abc = {'A': 0, 'B': 1, 'C': 2}[abc_label]

                input_data = pd.DataFrame([{
                    'Demand': demand,
                    'Forecast': forecast,
                    'Inventory Level': inventory,
                    'Safety Stock': safety,
                    'Lead Time': leadtime,
                    'Pola Pergerakan': movement,
                    'Klasifikasi ABC': abc
                }])

                prediction = model.predict(input_data)[0]
                label_map = {0: 'Normal', 1: 'Understock', 2: 'Overstock'}
                pred_label = label_map.get(prediction, 'Tidak Diketahui')

                st.success(f"📦 Prediksi Status Stok: **{pred_label}**")
