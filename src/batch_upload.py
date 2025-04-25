import streamlit as st
import pandas as pd
import json
from sklearn.metrics import classification_report
from pathlib import Path
import sys

# Add backend path
sys.path.append(str(Path(__file__).resolve().parent.parent / "backend"))
from predictor import predict_batch

# Label mapping
label_map = {"No": 0, "Yes": 1, "To some extent": 2}
inv_label_map = {v: k for k, v in label_map.items()}


def batch_upload_page():
    st.markdown("### Upload MRBench-style JSON")
    uploaded_file = st.file_uploader("Upload `.json` file", type=["json"])

    if uploaded_file:
        try:
            data = json.load(uploaded_file)

            # Step 1: Flatten tutor responses
            rows = []
            for entry in data:
                history = entry["conversation_history"]
                conv_id = entry["conversation_id"]

                for model_name, details in entry["tutor_responses"].items():
                    response = details.get("response", "")
                    label = details.get("annotation", {}).get("Mistake_Identification", None)
                    rows.append({
                        "conversation_id": conv_id,
                        "model": model_name,
                        "conversation_history": history,
                        "response": response,
                        "label_text": label,
                        "label": label_map.get(label) if label else None,
                    })

            df = pd.DataFrame(rows)
            st.write("Parsed", len(df), "model responses.")
            st.dataframe(df[["conversation_id", "model", "label_text"]].head())

            # Step 2: Predict
            predictions = predict_batch(df["conversation_history"].tolist(), df["response"].tolist())
            pred_df = pd.DataFrame(predictions)

            # Step 3: Merge and format
            df["predicted_label"] = pred_df["prediction"]
            df["confidence"] = pred_df["confidence"]
            df["probs"] = pred_df["probs"]

            st.success("✅ Inference complete!")
            st.dataframe(df[["conversation_id", "model", "predicted_label", "confidence"]])

            # Step 4: Show classification report if labels are available
            if df["label"].notna().all():
                y_true = df["label"]
                y_pred = df["predicted_label"].map(label_map)
                report = classification_report(y_true, y_pred, target_names=list(label_map.keys()), digits=3)
                st.markdown("### 📊 Classification Report")
                st.code(report)

            # Step 5: Downloadable results
            output_json = df.to_dict(orient="records")
            json_str = json.dumps(output_json, indent=2)
            st.download_button(
                "📥 Download Results (JSON)",
                data=json_str,
                file_name="mistake_predictions.json",
                mime="application/json"
            )

        except Exception as e:
            st.error(f"Error parsing JSON: {e}")
