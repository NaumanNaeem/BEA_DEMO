import streamlit as st
from streamlit_option_menu import option_menu
from single_pair import single_pair_page
from batch_upload import batch_upload_page

st.set_page_config(page_title="Mistake‑ID Demo", layout="wide")

st.markdown("<h2 style='text-align:center'>Mistake Identification Demo</h2>",
            unsafe_allow_html=True)

choice = option_menu(
    None,
    ["Single Pair", "Batch Upload"],
    icons=["bi bi-chat", "bi bi-table"],
    default_index=0,
    orientation="horizontal"
)

if choice == "Single Pair":
    single_pair_page()
else:
    batch_upload_page()
