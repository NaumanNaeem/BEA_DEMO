import streamlit as st
from pathlib import Path
import sys, os

sys.path.append(str(Path(__file__).resolve().parent.parent / "backend"))
from predictor import predict_single

def single_pair_page():
    st.write("Enter a **conversation history** and **tutor response**")
    history  = st.text_area("Conversation History", height=150)
    response = st.text_area("Tutor Response",       height=100)

    if st.button("Predict"):
        if not history or not response:
            st.warning("Both fields are required.")
            return

        result = predict_single(history, response)
        st.success(f"Prediction: **{result['prediction']}** "
                   f"(confidence {result['confidence']})")
        st.json(result["probs"])
