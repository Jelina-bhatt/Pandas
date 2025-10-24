"""
moodbridge_mega.py

Improved MoodBridge — mega prototype with enhanced Streamlit analytics,
timezone-aware timestamps, and extra visualizations.

Author: Jelina Bhatt
Date: 2025-10-24 (updated)

How to run:
- CLI mode:
    python moodbridge_mega.py --mode cli
- Streamlit UI:
    streamlit run moodbridge_mega.py -- --mode streamlit
"""

import argparse
import random
import sqlite3
import time
import os
import json
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List, Tuple
import base64
import io

# --- ML imports with graceful fallback ---
try:
    from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
    HF_AVAILABLE = True
except Exception as e:
    print("Warning: transformers library not available or failed to import.", e)
    HF_AVAILABLE = False

# Optional audio libs
try:
    import numpy as np
    import librosa
    AUDIO_AVAILABLE = True
except Exception as e:
    AUDIO_AVAILABLE = False

# Optional streamlit
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except Exception as e:
    STREAMLIT_AVAILABLE = False

# plotting and data
import pandas as pd
import matplotlib.pyplot as plt

# Local config
DB_PATH = "moodbridge_logs.db"
DEFAULT_TEXT_MODEL = "j-hartmann/emotion-english-distilroberta-base"
DEFAULT_REPLY_MODEL = "microsoft/DialoGPT-small"  # optional
EMOTION_LABELS = ["joy", "sadness", "anger", "fear", "surprise", "disgust", "neutral"]

# ---------------------------
# Data classes
# ---------------------------
@dataclass
class Interaction:
    timestamp: str
    input_text: str
    detected_emotion: str
    model_confidence: float
    response_text: str
    suggestion: str
    metadata: Optional[str] = None

# ---------------------------
# Database logger
# ---------------------------
class MoodLogger:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        initialize = not os.path.exists(self.db_path)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        if initialize:
            # create table if not exists
            self._ensure_table()
        else:
            self._ensure_table()

    def _ensure_table(self):
        c = self.conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                input_text TEXT,
                detected_emotion TEXT,
                model_confidence REAL,
                response_text TEXT,
                suggestion TEXT,
                metadata TEXT,
                message_length INTEGER
            )
        """)
        self.conn.commit()

    def log(self, interaction: Interaction):
        c = self.conn.cursor()
        c.execute("""
            INSERT INTO interactions 
            (timestamp, input_text, detected_emotion, model_confidence, response_text, suggestion, metadata, message_length)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (interaction.timestamp, interaction.input_text, interaction.detected_emotion,
              interaction.model_confidence, interaction.response_text, interaction.suggestion, interaction.metadata,
              len(interaction.input_text)))
        self.conn.commit()

    def fetch_recent(self, limit: int = 200) -> List[Interaction]:
        c = self.conn.cursor()
        c.execute("SELECT timestamp, input_text, detected_emotion, model_confidence, response_text, suggestion, metadata FROM interactions ORDER BY id DESC LIMIT ?", (limit,))
        rows = c.fetchall()
        return [Interaction(*row) for row in rows]

    def fetch_all_df(self) -> pd.DataFrame:
        df = pd.read_sql_query("SELECT timestamp, input_text, detected_emotion, model_confidence, response_text, suggestion, metadata, message_length FROM interactions ORDER BY id ASC", self.conn)
        if df.empty:
            return df
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['date'] = df['timestamp'].dt.date
        return df

    def fetch_aggregate(self) -> pd.DataFrame:
        df = self.fetch_all_df()
        if df.empty:
            return pd.DataFrame()
        summary = df.groupby(['date', 'detected_emotion']).size().unstack(fill_value=0)
        return summary

    def clear(self):
        c = self.conn.cursor()
        c.execute("DELETE FROM interactions")
        self.conn.commit()

# ---------------------------
# Emotion detection
# ---------------------------
class EmotionDetector:
    def __init__(self, model_name: str = DEFAULT_TEXT_MODEL):
        self.model_name = model_name
        self.pipeline = None
        self._load_model()

    def _load_model(self):
        if not HF_AVAILABLE:
            print("transformers not available — EmotionDetector will use simple keyword fallback.")
            self.pipeline = None
            return
        try:
            print(f"Loading emotion detection model: {self.model_name} ...")
            self.pipeline = pipeline("text-classification", model=self.model_name, return_all_scores=False)
            print("Model loaded.")
        except Exception as e:
            print("Failed to load HF pipeline, falling back to keyword-based detector.", e)
            self.pipeline = None

    def detect(self, text: str) -> Tuple[str, float]:
        text = text.strip()
        if not text:
            return "neutral", 1.0
        if self.pipeline:
            try:
                res = self.pipeline(text)[0]
                label = res['label'].lower()
                score = float(res.get('score', 0.0))
                label = label if label in EMOTION_LABELS else "neutral"
                return label, score
            except Exception as e:
                print("Error running HF pipeline:", e)
        # keyword fallback
        lowered = text.lower()
        keyword_map = {
            "joy": ["happy", "glad", "excited", "great", "joy", "yay", "wonderful", "amazing", "celebrate", "proud"],
            "sadness": ["sad", "alone", "unhappy", "depressed", "sorrow", "lonely", "cry", "tired"],
            "anger": ["angry", "mad", "furious", "hate", "annoyed", "irritated", "frustrat"],
            "fear": ["scared", "afraid", "fear", "anxious", "nervous", "worried", "panic"],
            "surprise": ["surprise", "shocked", "unexpected", "wow", "surprised"],
            "disgust": ["disgust", "gross", "nasty", "sick", "yuck"],
        }
        for label, kws in keyword_map.items():
            for kw in kws:
                if kw in lowered:
                    return label, 0.65
        return "neutral", 0.60

# ---------------------------
# Voice / Audio processing (optional)
# ---------------------------
class VoiceEmotionExtractor:
    def __init__(self):
        if not AUDIO_AVAILABLE:
            print("Audio libs (librosa) not available. Voice emotion disabled.")
        self.available = AUDIO_AVAILABLE

    def extract_from_file(self, file_path: str) -> Dict[str, float]:
        if not self.available:
            return {"error": "audio lib not available"}
        try:
            y, sr = librosa.load(file_path, sr=16000)
            zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))
            rmse = float(np.mean(librosa.feature.rms(y)))
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            features = {"zcr": zcr, "energy": rmse, "tempo": float(tempo)}
            # very rough mapping
            if features["energy"] > 0.04 and features["zcr"] > 0.1:
                guess = "anger"
            elif features["energy"] < 0.01:
                guess = "sadness"
            else:
                guess = "neutral"
            features["guess_emotion"] = guess
            return features
        except Exception as e:
            return {"error": str(e)}

# ---------------------------
# Response generator
# ---------------------------
class ResponseGenerator:
    def __init__(self, use_generator: bool = False, gen_model_name: str = DEFAULT_REPLY_MODEL):
        self.use_generator = use_generator and HF_AVAILABLE
        self.gen_model = None
        self.gen_tokenizer = None
        if self.use_generator:
            try:
                print(f"Loading generative reply model: {gen_model_name} ...")
                self.gen_tokenizer = AutoTokenizer.from_pretrained(gen_model_name)
                self.gen_model = AutoModelForSeq2SeqLM.from_pretrained(gen_model_name)
                print("Generative model loaded.")
            except Exception as e:
                print("Failed to load generative model. Falling back to templates.", e)
                self.use_generator = False

        self.templates = {
            "joy": [
                "That's wonderful to hear! Keep shining — would you like to celebrate by sharing or saving this moment?",
                "You sound joyful! Want to jot down what's making you happy so you can revisit it later?"
            ],
            "sadness": [
                "I'm really sorry you're going through this. If you'd like, we can try a short grounding exercise together.",
                "It makes sense to feel sad — want a small suggestion that might help lift your mood?"
            ],
            "anger": [
                "I hear that you're angry. Do you want to vent, or would you prefer a short calming exercise first?",
                "Anger can be intense. Let's try a breathing trick or a short walk — which would you prefer?"
            ],
            "fear": [
                "That sounds worrying. Taking a few steady breaths can help. Want me to guide you?",
                "I understand this scares you. Would you like a reassuring message or a distraction?"
            ],
            "surprise": [
                "Wow — that's unexpected! How are you feeling about it?",
                "That's a surprise. Want to unpack it together?"
            ],
            "disgust": [
                "That seems unpleasant. Want to talk it out or shift focus to something nice?",
                "I get why you'd feel disgusted. Small self-care might help right now — would you like a suggestion?"
            ],
            "neutral": [
                "I'm here if you want to share more. Want to reflect on your day?",
                "Sounds calm. Would you like a small task or a quote to keep the day smooth?"
            ]
        }

    def generate(self, user_text: str, detected_emotion: str) -> str:
        if self.use_generator and self.gen_model and self.gen_tokenizer:
            try:
                prompt = f"Emotion: {detected_emotion}\nUser: {user_text}\nAssistant:"
                inputs = self.gen_tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=512)
                outputs = self.gen_model.generate(inputs, max_length=80, num_return_sequences=1, do_sample=True, top_k=50, top_p=0.95)
                reply = self.gen_tokenizer.decode(outputs[0], skip_special_tokens=True)
                return reply.strip()
            except Exception as e:
                print("Error generating reply:", e)
        pool = self.templates.get(detected_emotion, self.templates["neutral"])
        reply = random.choice(pool)
        if random.random() < 0.3:
            reply += " If you'd like, tell me more."
        return reply

# ---------------------------
# Recommendation engine
# ---------------------------
class Recommender:
    def __init__(self):
        self.mapping = {
            "joy": {
                "music": ["Upbeat playlist", "Dance your favorite track"],
                "action": ["Share the moment with someone", "Save a gratitude note"]
            },
            "sadness": {
                "music": ["Calm acoustic playlist", "Lo-fi relaxing mix"],
                "action": ["Try journaling for 10 minutes", "Take a short walk outside"]
            },
            "anger": {
                "music": ["Slow down with calm piano", "Breathing music"],
                "action": ["5-minute breathing exercise", "Take a break from the trigger"]
            },
            "fear": {
                "music": ["Ambient calming music", "Nature sounds"],
                "action": ["Grounding exercise: name 5 things you see", "Call a trusted person"]
            },
            "surprise": {
                "music": ["Something new and exploratory", "A discovery playlist"],
                "action": ["Write what surprised you", "Reflect on expectations"]
            },
            "disgust": {
                "music": ["Uplifting neutral sounds", "Soft instrumentals"],
                "action": ["Change environment for a while", "Do a small cleansing ritual"]
            },
            "neutral": {
                "music": ["Instrumental background", "Chill focus playlists"],
                "action": ["Try a micro-goal for the next 20 minutes", "Practice quick gratitude"]
            }
        }

    def suggest(self, emotion: str) -> Tuple[str, str]:
        emo = emotion if emotion in self.mapping else "neutral"
        music = random.choice(self.mapping[emo]["music"])
        action = random.choice(self.mapping[emo]["action"])
        suggestion_text = f"Music: {music} — Action: {action}"
        return music, action

# ---------------------------
# Mood index utilities
# ---------------------------
EMOTION_SCORE_MAP = {
    "joy": 1.0,
    "neutral": 0.5,
    "surprise": 0.6,
    "sadness": 0.0,
    "anger": 0.1,
    "fear": 0.15,
    "disgust": 0.05
}

def compute_mood_index(df: pd.DataFrame, window: int = 5) -> pd.Series:
    """
    Compute a simple positivity/mood index based on emotion mapping and model confidence.
    mood_value = EMOTION_SCORE_MAP[emotion] * confidence
    Then take a rolling mean (window).
    """
    if df.empty:
        return pd.Series(dtype=float)
    df2 = df.copy()
    df2['emotion_score'] = df2['detected_emotion'].map(EMOTION_SCORE_MAP).fillna(0.5)
    df2['mood_raw'] = df2['emotion_score'] * df2['model_confidence']
    mood_series = df2['mood_raw'].rolling(window=window, min_periods=1).mean()
    return mood_series

# ---------------------------
# Application logic
# ---------------------------
class MoodBridgeApp:
    def __init__(self, enable_generator=False):
        self.detector = EmotionDetector()
        self.voice_extractor = VoiceEmotionExtractor()
        self.generator = ResponseGenerator(use_generator=enable_generator)
        self.recommender = Recommender()
        self.logger = MoodLogger()

    def _now_iso(self) -> str:
        # timezone-aware UTC ISO format
        return datetime.now(timezone.utc).isoformat()

    def handle_message(self, text: str, metadata: Optional[dict] = None) -> Interaction:
        emotion, conf = self.detector.detect(text)
        response = self.generator.generate(text, emotion)
        music, action = self.recommender.suggest(emotion)
        suggestion = f"Music: {music} — Action: {action}"
        inter = Interaction(
            timestamp=self._now_iso(),
            input_text=text,
            detected_emotion=emotion,
            model_confidence=float(conf),
            response_text=response,
            suggestion=suggestion,
            metadata=json.dumps(metadata) if metadata else None
        )
        self.logger.log(inter)
        return inter

    # CLI chat
    def run_cli(self):
        print("\n=== MoodBridge (CLI) ===")
        print("Type 'exit' to quit. Type 'history' to show recent interactions.")
        while True:
            try:
                text = input("You: ")
            except (KeyboardInterrupt, EOFError):
                print("\nGoodbye!")
                break
            if text.strip().lower() in ["exit", "quit"]:
                print("MoodBridge: Take care! 💫")
                break
            if text.strip().lower() == "history":
                rows = self.logger.fetch_recent(10)
                for r in rows:
                    print(f"[{r.timestamp}] You: {r.input_text} -> Emotion: {r.detected_emotion} | Resp: {r.response_text}")
                continue
            inter = self.handle_message(text)
            print(f"\n🧠 Detected emotion: {inter.detected_emotion} (confidence: {inter.model_confidence:.2f})")
            print(f"MoodBridge: {inter.response_text}")
            print(f"💡 Suggestion: {inter.suggestion}\n")

    # Helper to create downloadable CSV bytes for Streamlit
    def _to_csv_bytes(self, df: pd.DataFrame) -> bytes:
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        return buf.getvalue().encode('utf-8')

    # Streamlit UI
    def run_streamlit(self):
        if not STREAMLIT_AVAILABLE:
            raise RuntimeError("Streamlit not installed. Install via `pip install streamlit`")

        st.set_page_config(page_title="MoodBridge", layout="wide")
        st.title("MoodBridge — AI Social Companion")
        st.write("An emotion-aware assistant that replies empathetically and suggests helpful actions.")
        # Sidebar controls
        st.sidebar.header("Settings")
        enable_gen = st.sidebar.checkbox("Enable generative replies (may be slower)", value=False)
        clear_db = st.sidebar.button("Clear logs (danger!)")
        if clear_db:
            self.logger.clear()
            st.sidebar.success("Logs cleared. Refreshing page.")
            st.experimental_rerun()

        st.sidebar.markdown("---")
        st.sidebar.markdown("Export / Download:")
        all_df = self.logger.fetch_all_df()
        if not all_df.empty:
            csv_bytes = self._to_csv_bytes(all_df)
            st.sidebar.download_button("Download logs CSV", csv_bytes, file_name=f"moodbridge_logs_{int(time.time())}.csv")
        else:
            st.sidebar.info("No logs to export yet.")

        # Chat and quick inputs
        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader("Chat")
            if "chat_history" not in st.session_state:
                st.session_state.chat_history = []

            user_input = st.text_input("Say something to MoodBridge:", key="user_text")
            if st.button("Send", key="send_btn"):
                if user_input.strip():
                    self.generator.use_generator = enable_gen and HF_AVAILABLE
                    inter = self.handle_message(user_input)
                    st.session_state.chat_history.append(inter)
                    st.experimental_rerun()

            # Display latest chats
            for inter in reversed(st.session_state.chat_history[-50:]):
                st.markdown(f"**You** ({inter.timestamp.split('T')[0]}): {inter.input_text}")
                st.markdown(f"**MoodBridge**: *{inter.response_text}*")
                st.markdown(f"**Detected**: `{inter.detected_emotion}` (conf {inter.model_confidence:.2f}) — **Suggestion:** {inter.suggestion}")
                st.markdown("---")

        with col2:
            st.subheader("Quick actions")
            if st.button("I feel stressed"):
                inter = self.handle_message("I am really stressed and overloaded")
                st.session_state.chat_history.append(inter)
                st.experimental_rerun()
            if st.button("I am happy today"):
                inter = self.handle_message("I am feeling very happy and excited")
                st.session_state.chat_history.append(inter)
                st.experimental_rerun()
            st.markdown("**Voice input** (optional)")
            st.markdown("If you have recorded audio, you can drop an audio file (wav/mp3) into the server folder and mention its filename in the chat to analyze voice features (audio support is optional & requires librosa).")

        # Analytics pane
        st.header("Analytics & Mood Insights")
        df = self.logger.fetch_all_df()
        if df.empty:
            st.info("No interactions logged yet — talk to MoodBridge to start generating data.")
        else:
            # Emotion distribution pie chart
            st.subheader("Emotion distribution")
            emo_counts = df['detected_emotion'].value_counts().reindex(EMOTION_LABELS, fill_value=0)
            fig1, ax1 = plt.subplots(figsize=(5, 4))
            ax1.pie(emo_counts.values, labels=emo_counts.index, autopct='%1.1f%%', startangle=140)
            ax1.axis('equal')
            st.pyplot(fig1)

            # Stacked daily bar chart
            st.subheader("Daily emotion timeline (stacked)")
            agg = df.groupby(['date', 'detected_emotion']).size().unstack(fill_value=0)
            fig2, ax2 = plt.subplots(figsize=(10, 4))
            agg.plot(kind="bar", stacked=True, ax=ax2)
            ax2.set_xlabel("Date")
            ax2.set_ylabel("Count")
            st.pyplot(fig2)

            # Mood index and moving average
            st.subheader("Positivity / Mood index (rolling)")
            mood_series = compute_mood_index(df, window=5)
            fig3, ax3 = plt.subplots(figsize=(10, 3))
            ax3.plot(df['timestamp'], mood_series, marker='o')
            ax3.set_title("Rolling mood index (higher = more positive)")
            ax3.set_ylim(0, 1.05)
            ax3.set_ylabel("Mood index")
            ax3.set_xlabel("Timestamp")
            st.pyplot(fig3)

            # Correlation: message length vs model confidence
            st.subheader("Message length vs. model confidence")
            fig4, ax4 = plt.subplots(figsize=(6, 4))
            ax4.scatter(df['message_length'], df['model_confidence'])
            ax4.set_xlabel("Message length (chars)")
            ax4.set_ylabel("Model confidence")
            st.pyplot(fig4)

            # Show table and allow quick filtering
            st.subheader("Interactions table")
            st.dataframe(df[['timestamp', 'input_text', 'detected_emotion', 'model_confidence', 'suggestion']].sort_values(by='timestamp', ascending=False).reset_index(drop=True))

        # Advanced: allow user to export filtered CSV
        st.markdown("---")
        st.subheader("Export filtered data")
        if not df.empty:
            filtered = df.copy()
            # small filters
            emotion_filter = st.multiselect("Choose emotions", options=EMOTION_LABELS, default=EMOTION_LABELS)
            if emotion_filter:
                filtered = filtered[filtered['detected_emotion'].isin(emotion_filter)]
            date_min = st.date_input("From date", value=filtered['date'].min() if not filtered.empty else None)
            date_max = st.date_input("To date", value=filtered['date'].max() if not filtered.empty else None)
            if (not filtered.empty) and (date_min is not None) and (date_max is not None):
                mask = (filtered['date'] >= date_min) & (filtered['date'] <= date_max)
                filtered = filtered[mask]
            if not filtered.empty:
                csv_bytes = self._to_csv_bytes(filtered)
                st.download_button("Download filtered CSV", csv_bytes, file_name=f"moodbridge_filtered_{int(time.time())}.csv")
            else:
                st.write("No data for selected filters.")

        st.markdown("**Notes:** This analytics pane uses a simple 'mood index' heuristic combining detected emotion and model confidence. It is for demonstration and exploration rather than clinical use.")

# ---------------------------
# CLI argument parser + main
# ---------------------------
def build_arg_parser():
    p = argparse.ArgumentParser(description="MoodBridge — mega prototype (improved)")
    p.add_argument("--mode", choices=["cli", "streamlit"], default="cli", help="Run mode")
    p.add_argument("--enable-gen", action="store_true", help="Enable generative reply model (requires HF models)")
    return p

def main():
    parser = build_arg_parser()
    args, unknown = parser.parse_known_args()
    app = MoodBridgeApp(enable_generator=args.enable_gen)
    if args.mode == "cli":
        app.run_cli()
    elif args.mode == "streamlit":
        app.run_streamlit()
    else:
        print("Unknown mode.")

if __name__ == "__main__":
    main()
