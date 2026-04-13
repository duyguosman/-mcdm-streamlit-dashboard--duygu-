import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pymcdm import weights as w
from pymcdm.methods import TOPSIS, MABAC, ARAS, WSM
from pymcdm.helpers import rrankdata
from pymcdm import visuals
from pymcdm.normalization import minmax_normalization  # Hatayı çözen import

# SAW, WSM ile aynıdır
SAW = WSM

st.set_page_config(page_title="MCDM Dashboard", layout="wide")
st.title("Multi-Criteria Decision Making (MCDM) Dashboard")

# --- 1. DATA INPUT ---
st.sidebar.header("1. Upload or Edit Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=['csv'])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    # Varsayılan veri seti
    data = {
        'alternative': ['A1', 'A2', 'A3', 'A4'],
        'criteria_1': [2.5, 3.0, 4.0, 3.5],
        'criteria_2': [50, 60, 80, 45],
        'criteria_3': [0.9, 0.6, 0.1, 0.4],
        'criteria_4': [0.1, 0.6, 0.3, 0.8],
        'criteria_5': [0.17, 0.83, 0.50, 0.20]
    }
    df = pd.DataFrame(data)

st.subheader("Decision Matrix")
st.markdown("Edit the matrix directly below or upload a new CSV file from the sidebar.")
# Kullanıcının tabloyu düzenlemesine izin ver
edited_df = st.data_editor(df, num_rows="dynamic", use_container_width=True)

# Alternatif ve kriterleri ayır
alts_names = edited_df.iloc[:, 0].astype(str).tolist()
criteria_names = edited_df.columns[1:]
alts_data = edited_df.iloc[:, 1:].to_numpy()

# --- 2. WEIGHTS & TYPES CONFIGURATION ---
st.sidebar.header("2. Criteria Configuration")
st.sidebar.markdown("Set weights and types for each criterion.")

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
types = np.array(types_list)

# --- 3. METHOD SELECTION ---
st.sidebar.header("3. Select MCDM Methods")

# NOT: minmax_normalization kullanımı 0 içeren verilerde hatayı engeller
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
    elif alts_data.size == 0:
        st.error("Data matrix is empty!")
    else:
        try:
            methods = [available_methods[name] for name in selected_method_names]
            prefs = []
            ranks = []
            
            for method in methods:
                # Hesaplama
                pref = method(alts_data, weights, types)
                rank = rrankdata(pref)
                
                prefs.append(pref)
                ranks.append(rank)
                
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Preference Table")
                pref_df = pd.DataFrame(zip(*prefs), columns=selected_method_names, index=alts_names).round(4)
                st.dataframe(pref_df, use_container_width=True)
                
            with col2:
                st.subheader("Ranking Table")
                rank_df = pd.DataFrame(zip(*ranks), columns=selected_method_names, index=alts_names).astype(int)
                st.dataframe(rank_df, use_container_width=True)

            # Görselleştirme
            st.subheader("Polar Ranking Plot")
            fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
            visuals.polar_plot(ranks, labels=selected_method_names, ax=ax)
            st.pyplot(fig)
            
        except Exception as e:
            st.error(f"An error occurred during analysis: {e}")
            st.info("Tip: Ensure your data doesn't contain negative values for 'Sum' or 'MinMax' normalization unless the method supports it.")
