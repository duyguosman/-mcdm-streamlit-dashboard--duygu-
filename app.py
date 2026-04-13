import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pymcdm import weights as w
from pymcdm.methods import TOPSIS, MABAC, ARAS, WSM
from pymcdm.helpers import rrankdata
from pymcdm import visuals
from pymcdm.normalization import minmax_normalization

# Alias SAW to WSM
SAW = WSM

st.set_page_config(page_title="MCDM Dashboard", layout="wide")
st.title("Multi-Criteria Decision Making (MCDM) Dashboard")

# --- 1. DATA INPUT ---
st.sidebar.header("1. Upload or Edit Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=['csv'])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    data = {
        'Alternative': ['A1', 'A2', 'A3'],
        'i1': [2.5, 3.0, 4.0],
        'i2': [50, 60, 80],
        'i10': [0.9, 0.0, 0.1] # Örnek 0 değeri
    }
    df = pd.DataFrame(data)

st.subheader("Decision Matrix")
edited_df = st.data_editor(df, num_rows="dynamic", use_container_width=True)

# --- DATA CLEANING ---
try:
    alts_names = edited_df.iloc[:, 0].astype(str).tolist()
    numeric_df = edited_df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')
    
    if numeric_df.isnull().values.any():
        st.error("Matrix sayısal olmayan değerler içeriyor!")
        st.stop()

    alts_data = numeric_df.to_numpy()
    criteria_names = numeric_df.columns
except Exception as e:
    st.error(f"Veri hatası: {e}")
    st.stop()

# --- 2. WEIGHTS & TYPES ---
st.sidebar.header("2. Criteria Configuration")
weights_list = []
types_list = []

for col in criteria_names:
    st.sidebar.markdown(f"**{col}**")
    c1, c2 = st.sidebar.columns(2)
    with c1:
        weight = st.slider(f"Weight", 0.0, 1.0, 1.0/len(criteria_names), key=f"w_{col}")
        weights_list.append(weight)
    with c2:
        ctype = st.radio("Type", ["Benefit", "Cost"], key=f"t_{col}")
        types_list.append(1 if ctype == "Benefit" else -1)

weights = np.array(weights_list)
if np.sum(weights) > 0: weights = weights / np.sum(weights)
types = np.array(types_list)

# --- 3. METHOD SELECTION ---
available_methods = {
    'TOPSIS': TOPSIS(),
    'SAW': SAW(normalization=minmax_normalization),
    'MABAC': MABAC(),
    'ARAS': ARAS(),
    'WSM': WSM(normalization=minmax_normalization)
}

selected_method_names = st.sidebar.multiselect(
    "Methods:", list(available_methods.keys()), default=['TOPSIS', 'SAW', 'MABAC']
)

# --- 4. EXECUTE ---
if st.button("Run MCDM Analysis"):
    if not selected_method_names:
        st.warning("Lütfen yöntem seçin.")
    else:
        # KRİTİK DÜZELTME: 0 değerlerini MABAC ve diğerleri için küçük bir sayıya çekiyoruz
        # Bu işlem orijinal 'edited_df'i bozmaz, sadece hesaplamada kullanılır.
        clean_data = np.where(alts_data == 0, 1e-9, alts_data)
        
        prefs, ranks, successful_methods = [], [], []
        
        for name in selected_method_names:
            try:
                method = available_methods[name]
                # Temizlenmiş veriyi kullanıyoruz
                pref = method(clean_data, weights, types)
                rank = rrankdata(pref)
                
                prefs.append(pref)
                ranks.append(rank)
                successful_methods.append(name)
            except Exception as e:
                st.error(f"{name} hatası: {e}")

        if successful_methods:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Preference Table")
                st.dataframe(pd.DataFrame(zip(*prefs), columns=successful_methods, index=alts_names).round(4))
            with col2:
                st.subheader("Ranking Table")
                st.dataframe(pd.DataFrame(zip(*ranks), columns=successful_methods, index=alts_names).astype(int))

            st.subheader("Polar Ranking Plot")
            fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(projection='polar'))
            visuals.polar_plot(ranks, labels=successful_methods, ax=ax)
            st.pyplot(fig)
