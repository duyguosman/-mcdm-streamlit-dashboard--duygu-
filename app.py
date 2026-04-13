import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pymcdm import weights as w
from pymcdm.methods import TOPSIS, MABAC, ARAS, WSM
from pymcdm.helpers import rrankdata
from pymcdm import visuals
from pymcdm.normalization import minmax_normalization  # Normalizasyon hatasını çözen import

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
    # Varsayılan veri seti (CSV yoksa)
    data = {
        'Alternative': ['A1', 'A2', 'A3'],
        'Criterion 1': [2.5, 3.0, 4.0],
        'Criterion 2': [50, 60, 80],
        'Criterion 3': [0.9, 0.6, 0.1]
    }
    df = pd.DataFrame(data)

st.subheader("Decision Matrix")
st.markdown("Edit the matrix directly below or upload a new CSV file from the sidebar.")
edited_df = st.data_editor(df, num_rows="dynamic", use_container_width=True)

# --- DATA CLEANING & VALIDATION ---
try:
    # İlk sütun isimlerini/alternatifleri al
    alts_names = edited_df.iloc[:, 0].astype(str).tolist()
    # Geri kalan sütunları sayısal veriye çevir
    numeric_df = edited_df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')
    
    if numeric_df.isnull().values.any():
        st.error("The matrix contains non-numeric values. Please check your data.")
        st.stop()

    alts_data = numeric_df.to_numpy()
    criteria_names = numeric_df.columns
except Exception as e:
    st.error(f"Error processing data: {e}")
    st.stop()

# --- 2. WEIGHTS & TYPES CONFIGURATION ---
st.sidebar.header("2. Criteria Configuration")

weights_list = []
types_list = []

for col in criteria_names:
    st.sidebar.markdown(f"**{col}**")
    c1, c2 = st.sidebar.columns(2)
    with c1:
        weight = st.slider(f"Weight", min_value=0.0, max_value=1.0, value=1.0/len(criteria_names), key=f"w_{col}")
        weights_list.append(weight)
    with c2:
        ctype = st.radio("Type", options=["Benefit", "Cost"], key=f"t_{col}")
        types_list.append(1 if ctype == "Benefit" else -1)

# Ağırlıkları normalize et (toplamı 1 olmalı)
weights = np.array(weights_list)
if np.sum(weights) > 0:
    weights = weights / np.sum(weights)
else:
    weights = np.ones(len(criteria_names)) / len(criteria_names)
    
types = np.array(types_list)

# --- 3. METHOD SELECTION ---
st.sidebar.header("3. Select MCDM Methods")

# ÇÖZÜM: SAW ve WSM için minmax_normalization kullanarak 0 değerleri için hatayı engelliyoruz.
available_methods = {
    'TOPSIS': TOPSIS(),
    'SAW': SAW(normalization=minmax_normalization),
    'MABAC': MABAC(),
    'ARAS': ARAS(),
    'WSM': WSM(normalization=minmax_normalization)
}

selected_method_names = st.sidebar.multiselect(
    "Choose evaluation methods:",
    list(available_methods.keys()),
    default=['TOPSIS', 'SAW']
)

# --- 4. EXECUTE & DISPLAY RESULTS ---
if st.button("Run MCDM Analysis"):
    if not selected_method_names:
        st.warning("Please select at least one method from the sidebar.")
    else:
        prefs = []
        ranks = []
        successful_methods = []
        
        for name in selected_method_names:
            try:
                method = available_methods[name]
                
                # Artık epsilon eklemeye gerek yok, minmax_normalization bunu halleder.
                pref = method(alts_data, weights, types)
                
                rank = rrankdata(pref)
                prefs.append(pref)
                ranks.append(rank)
                successful_methods.append(name)
            except Exception as e:
                st.error(f"Method {name} failed: {e}")

        if successful_methods:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Preference Table")
                pref_df = pd.DataFrame(zip(*prefs), columns=successful_methods, index=alts_names).round(4)
                st.dataframe(pref_df, use_container_width=True)
                
            with col2:
                st.subheader("Ranking Table")
                rank_df = pd.DataFrame(zip(*ranks), columns=successful_methods, index=alts_names).astype(int)
                st.dataframe(rank_df, use_container_width=True)

            # Polar chart görselleştirme
            st.subheader("Polar Ranking Plot")
            fig, ax = plt.subplots(figsize=(7, 7), dpi=150, tight_layout=True, subplot_kw=dict(projection='polar'))
            visuals.polar_plot(ranks, labels=successful_methods, legend_ncol=2, ax=ax)
            st.pyplot(fig)
