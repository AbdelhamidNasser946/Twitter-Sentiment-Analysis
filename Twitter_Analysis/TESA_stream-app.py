import streamlit as st
import tempfile
import os
import pickle
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf

st.set_page_config(page_title="Sentiment Analyzer", layout="centered")

st.title("Sentiment Analysis — Streamlit App")
st.write("Load your trained Keras model and tokenizer (pickle), enter text or upload a CSV, and get predictions.")


@st.cache_resource
def load_keras_model(model_path: str):
    """Try loading a Keras model from several common file formats:
    - .h5
    - .keras (Keras native format)
    - SavedModel directory (or a ZIP containing a SavedModel)

    Returns the loaded model or raises the original exception.
    """
    # First try the straightforward load
    try:
        return tf.keras.models.load_model(model_path)
    except Exception as e:
        # If it's a zip containing a SavedModel, extract and try again
        try:
            if model_path.endswith('.zip'):
                import zipfile
                import tempfile
                tmpdir = tempfile.mkdtemp()
                with zipfile.ZipFile(model_path, 'r') as z:
                    z.extractall(tmpdir)
                # locate a directory with saved_model.pb
                for root, dirs, files in os.walk(tmpdir):
                    if 'saved_model.pb' in files:
                        return tf.keras.models.load_model(root)
        except Exception:
            pass
        # Fallback: try using keras API directly (sometimes helps for .keras files)
        try:
            from tensorflow import keras
            return keras.models.load_model(model_path)
        except Exception:
            # re-raise the original exception for clearer error messaging
            raise e


@st.cache_resource
def load_tokenizer(tokenizer_path: str):
    with open(tokenizer_path, 'rb') as f:
        tok = pickle.load(f)
    return tok


def save_uploaded_file(uploaded_file):
    # Save uploaded file to a temporary path and return it
    suffix = os.path.splitext(uploaded_file.name)[1]
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.getbuffer())
    tmp.flush()
    tmp.close()
    return tmp.name


def preprocess_texts(texts, tokenizer, maxlen):
    # basic cleaning
    texts = [str(t).strip() for t in texts]
    seqs = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(seqs, maxlen=maxlen, padding='post', truncating='post')
    return padded


st.sidebar.header("Model & Tokenizer")
model_file = st.sidebar.file_uploader(
    "Upload Keras model (.h5, .keras, or a zip of a SavedModel)",
    type=["h5", "zip", "keras"]
)

tokenizer_file = st.sidebar.file_uploader("Upload tokenizer (pickle .pkl)", type=["pkl", "pickle"])

maxlen = st.sidebar.number_input("Max sequence length (maxlen)", min_value=10, max_value=2000, value=100, step=10)

# Class labels: default to the four requested labels but allow customization
labels_input = st.sidebar.text_input("Class labels (comma-separated, order = model output order)",
                                    value="Positive,Negative,Neutral,Irrelative")
label_names = [l.strip() for l in labels_input.split(',') if l.strip()]

# Optional: allow user to enter path instead of upload (for local run)
st.sidebar.write("If you're running this app locally you can also provide local paths below.")
local_model_path = st.sidebar.text_input("Local model path (optional)")
local_tokenizer_path = st.sidebar.text_input("Local tokenizer path (optional)")

model = None
tokenizer = None

# Load model
try:
    if model_file is not None:
        model_path = save_uploaded_file(model_file)
        with st.spinner("Loading model..."):
            model = load_keras_model(model_path)
    elif local_model_path:
        model = load_keras_model(local_model_path)
except Exception as e:
    st.sidebar.error(f"Failed to load model: {e}")

# Load tokenizer
try:
    if tokenizer_file is not None:
        tok_path = save_uploaded_file(tokenizer_file)
        with st.spinner("Loading tokenizer..."):
            tokenizer = load_tokenizer(tok_path)
    elif local_tokenizer_path:
        tokenizer = load_tokenizer(local_tokenizer_path)
except Exception as e:
    st.sidebar.error(f"Failed to load tokenizer: {e}")


st.header("Single prediction")
input_text = st.text_area("Enter text to analyze", placeholder="Type a sentence here...", height=120)

col1, col2 = st.columns([1,1])
with col1:
    if st.button("Predict"):
        if model is None or tokenizer is None:
            st.error("Please upload both a Keras model and a tokenizer (or provide local paths).")
        elif not input_text.strip():
            st.error("Please enter some text to predict.")
        else:
            x = preprocess_texts([input_text], tokenizer, maxlen)
            with st.spinner("Running prediction..."):
                preds = model.predict(x)
            # Normalize preds to 2D array for easier handling
            preds = np.asarray(preds)
            if preds.ndim == 1:
                preds = preds.reshape(1, -1)

            n_outputs = preds.shape[-1]

            # Case: single-output sigmoid (n_outputs == 1)
            if n_outputs == 1:
                prob = float(preds[0][0])
                # Use first two labels if available, otherwise default to Positive/Negative
                pos_label = label_names[0] if len(label_names) >= 1 else "Positive"
                neg_label = label_names[1] if len(label_names) >= 2 else "Negative"
                label = pos_label if prob >= 0.5 else neg_label
                st.success(f"Prediction: {label} — probability: {prob:.4f}")

            # Multi-class outputs
            else:
                if len(label_names) != n_outputs:
                    st.warning(f"Number of provided labels ({len(label_names)}) does not match model outputs ({n_outputs}). Using default numeric class indices.")
                probs = preds[0]
                class_idx = int(np.argmax(probs))
                if len(label_names) == n_outputs:
                    label = label_names[class_idx]
                else:
                    label = f"class_{class_idx}"
                prob_str = ', '.join([f"{(label_names[i] if i < len(label_names) else 'c'+str(i))}: {probs[i]:.4f}" for i in range(len(probs))])
                st.success(f"Prediction: {label} — probs: [{prob_str}]")

st.markdown("---")
st.write("**Notes / Troubleshooting:**")
st.write("- This app expects a Keras model (`.h5`, `.keras`, or SavedModel) trained for text classification.")
st.write("- Tokenizer should be the `Tokenizer` instance saved with pickle (e.g. `pickle.dump(tokenizer, file)`).")
st.write(f"- Default `maxlen` is {maxlen}. Change it to the same value you used during training for best results.")
st.write("- If you upload a zip containing a SavedModel the app will extract it and look for `saved_model.pb`.")
st.write("- To get outputs among Positive / Negative / Neutral / Irrelative, either use a 4-class model (with outputs in that order) or edit the 'Class labels' field in the sidebar to match your model's output order.")

st.write("---")
st.write("If you want, paste here a small snippet of your model architecture (or the relevant cells from the notebook) and I can adapt the app to exactly match the input preprocessing and labels used during training.")
