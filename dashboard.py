import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

LOG_DIR = "logs"

st.set_page_config(page_title="📊 Log Dashboard", layout="wide")
st.title("📊 Chatbot Log Dashboard")

# Fájlok betöltése
log_files = {
    "CLI Chat Log": "chat.log",
    "Streamlit Chat Log": "streamlit_chat.log",
    "Document Processing Log": "process.log"
}

def load_log(log_path):
    data = []
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('|')
            if len(parts) < 3:
                continue
            timestamp = parts[0].strip()
            level = parts[1].strip()
            message = "|".join(parts[2:]).strip()
            data.append((timestamp_from_log(timestamp=parts[0]), level, parts[2]))
    return pd.DataFrame(data, columns=["timestamp", "level", "message"])

@st.cache_data
def timestamp_to_datetime(df):
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

def parse_chat_log(df):
    questions, responses, inference_times = [], [], []
    for i, row in df.iterrows():
        if "Kérdés:" in row.message:
            questions.append(row.message.split("Kérdés: ")[1])
        elif "Válasz:" in row['message']:
            responses.append(row['message'].split("Válasz: ")[1])
        elif "Inferencia idő:" in row["message"]:
            t = float(row["message"].split(": ")[1].replace('s',''))
            inference_times.append(t)
    return questions, responses, inference_times

selected_log = st.sidebar.selectbox("Válassz logfájlt:", list(logs for logs in log_path.keys()))

log_path = os.path.join(LOG_DIR, log_files[selected_log])

if os.path.exists(log_path):
    df = load_log(log_path)
    df = timestamp_to_datetime(df)

    st.subheader(f"{selected_log} tartalma")
    st.dataframe(df.head(100), height=250)

    if "Chat Log" in selected_log:
        questions, responses, inference_times = parse_chat_log(df)

        st.subheader("🚀 Kérdések száma és válaszidők")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Összes kérdés", len(questions))
            st.metric("Átlagos válaszidő (s)", f"{pd.Series(inference_times).mean():.2f}")

        with col2:
            st.metric("Legrövidebb válaszidő (s)", f"{pd.Series(inference_times).min():.2f}")
            st.metric("Leghosszabb válaszidő (s)", f"{pd.Series(inference_times).max():.2f}")

        st.subheader("📈 Válaszidők eloszlása")
        fig, ax = plt.subplots()
        sns.histplot(inference_times, kde=True, bins=30, ax=ax)
        ax.set_xlabel("Válaszidő (s)")
        ax.set_ylabel("Gyakoriság")
        st.pyplot(fig)

    if selected_log == "Document Processing Log":
        process_times = df[df.message.str.contains("Idő:")]
        process_times['processing_time'] = process_times['message'].str.extract(r'Idő: ([\d.]+) s').astype(float)

        st.subheader("📚 Dokumentum feldolgozási idők")
        st.line_chart(process_times.set_index("timestamp")["processing_time"])

else:
    st.error(f"A(z) {selected_log} logfájl nem található.")