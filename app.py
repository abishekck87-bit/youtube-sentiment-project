import streamlit as st
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns

from googleapiclient.discovery import build
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ---- CONFIG ----
st.set_page_config(page_title="YouTube Sentiment Analysis", layout="centered")

# ---- API KEY (NO WIDGET) ----
YOUTUBE_API_KEY = st.secrets.get("YOUTUBE_API_KEY", "")

# ---- FUNCTIONS ----
def extract_video_id(url):
    if "youtu.be/" in url:
        return url.split("/")[-1].split("?")[0]
    if "watch?v=" in url:
        return url.split("v=")[1].split("&")[0]
    return url

def fetch_comments(video_id, max_comment=100):
    youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
    comments = []

    request = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults=max_comment,
        textFormat="plainText"
    )

    response = request.execute()

    for item in response["items"]:
        comments.append(item["snippet"]["topLevelComment"]["snippet"]["textDisplay"])

    return pd.DataFrame(comments, columns=["Comment"])

def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^A-Za-z ]", "", text)
    return text.lower()

def analyze_sentiment(df):
    analyzer = SentimentIntensityAnalyzer()
    df["Cleaned_Comment"] = df["Comment"].apply(clean_text)

    def classify(text):
        score = analyzer.polarity_scores(text)["compound"]
        if score >= 0.1:
            return "Positive"
        elif score <= -0.1:
            return "Negative"
        else:
            return "Neutral"

    df["Sentiment"] = df["Cleaned_Comment"].apply(classify)
    return df

# ---- UI ----
st.title("📊 YouTube Comment Sentiment Analysis")

url = st.text_input("Enter YouTube Video URL")

if st.button("Analyze"):
    if not url:
        st.error("Please enter a YouTube URL")
    elif not YOUTUBE_API_KEY:
        st.error("API key not found")
    else:
        video_id = extract_video_id(url)
        df = fetch_comments(video_id)

        df = analyze_sentiment(df)

        st.success(f"{len(df)} comments analyzed")

        st.dataframe(df.head())

        # Bar chart
        fig1, ax1 = plt.subplots()
        sns.countplot(x="Sentiment", data=df, ax=ax1)
        st.pyplot(fig1)

        # Pie chart
        fig2, ax2 = plt.subplots()
        df["Sentiment"].value_counts().plot.pie(
            autopct="%1.1f%%", ax=ax2
        )
        st.pyplot(fig2)





