import streamlit as st
import joblib
import numpy as np
import pandas as pd
import re
import string
from textblob import TextBlob
from datetime import datetime
from googleapiclient.discovery import build
from nltk.stem import PorterStemmer

ps = PorterStemmer()

# -------------------------------
# LOAD API KEY FROM STREAMLIT SECRETS
# -------------------------------
API_KEY = st.secrets["API_KEY"]
youtube = build("youtube", "v3", developerKey=API_KEY)

# -------------------------------
# LOAD MODELS
# -------------------------------
clf = joblib.load('clf.pkl')
xgb = joblib.load('xgb_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')
stopwords = joblib.load('stopwords.pkl')

# -------------------------------
# FUNCTIONS
# -------------------------------

def extract_video_id(url):
    regex_list = [
        r"(?:https?://)?(?:www\.)?youtu\.be/([a-zA-Z0-9_-]{11})",
        r"(?:https?://)?(?:www\.)?youtube\.com/watch\?v=([a-zA-Z0-9_-]{11})",
        r"v=([a-zA-Z0-9_-]{11})",
        r"embed/([a-zA-Z0-9_-]{11})",
        r"shorts/([a-zA-Z0-9_-]{11})"
    ]
    for regex in regex_list:
        match = re.search(regex, url)
        if match:
            return match.group(1)
    raise ValueError("Invalid YouTube link format.")


def get_video_details(video_url):
    video_id = extract_video_id(video_url)

    response = youtube.videos().list(
        part="snippet,statistics",
        id=video_id
    ).execute()

    if not response["items"]:
        raise ValueError("Video not found")

    item = response["items"][0]
    snippet = item["snippet"]
    stats = item["statistics"]

    details = {
        "video_id": video_id,
        "title": snippet.get("title", ""),
        "publish_time": snippet.get("publishedAt", ""),
        "channelTitle": snippet.get("channelTitle", ""),
        "thumbnail": snippet["thumbnails"]["high"]["url"],
        "tags": snippet.get("tags", []),
        "views": int(stats.get("viewCount", 0)),
        "likes": int(stats.get("likeCount", 0)),
        "comments": int(stats.get("commentCount", 0)),
        "description": snippet.get("description", "")
    }

    return details


def get_top_comments(video_id, max_comments=5):
    comments = []

    response = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults=max_comments,
        textFormat="plainText"
    ).execute()

    for item in response["items"]:
        text = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
        comments.append(text)

    return comments


def get_comment_sentiment(video_id, max_comments=30):
    sentiments = []

    response = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults=max_comments,
        textFormat="plainText"
    ).execute()

    for item in response["items"]:
        text = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
        polarity = TextBlob(text).sentiment.polarity
        sentiments.append(polarity)

    avg_sentiment = np.mean(sentiments) if sentiments else 0
    return avg_sentiment


def preprocess_for_engagement(details):
    views = details["views"]
    likes = details["likes"]
    comments = details["comments"]
    tags = details["tags"]
    title = details["title"]
    publish_time = details["publish_time"]

    rs = np.random.RandomState(42)
    frac = rs.beta(2.0, 6.0, size=1)
    frac = np.clip(frac, 0.05, 0.6)[0]

    first24_views = int(views * frac)
    first24_likes = int(likes * frac)
    first24_comments = int(comments * frac)
    like_view_ratio_24h = first24_likes / (first24_views + 1)
    comment_view_ratio_24h = first24_comments / (first24_views + 1)

    tag_count = len(tags)
    title_length = len(str(title))

    try:
        publish_hour = datetime.fromisoformat(
            publish_time.replace("Z", "")
        ).hour
    except:
        publish_hour = 0

    title_sentiment = TextBlob(str(title)).sentiment.polarity
    engagement_24h = first24_likes + first24_comments

    row = {
        "first24_views": first24_views,
        "first24_likes": first24_likes,
        "first24_comments": first24_comments,
        "like_view_ratio_24h": like_view_ratio_24h,
        "comment_view_ratio_24h": comment_view_ratio_24h,
        "tag_count": tag_count,
        "title_length": title_length,
        "publish_hour": publish_hour,
        "title_sentiment": title_sentiment,
        "engagement_24h": engagement_24h
    }

    return pd.DataFrame([row])


def preprocess_for_title_with_clf(details):
    title = details["title"]

    def clean_text(text):
        text = "".join([w.lower() for w in text if w not in string.punctuation])
        tokens = re.split('\W+', text)
        text = [ps.stem(word) for word in tokens if word not in stopwords]
        return " ".join(text)

    cleaned_title = clean_text(title)
    title_vec = vectorizer.transform([cleaned_title])
    return title_vec


def predict_video_virality(video_url):
    details = get_video_details(video_url)
    video_id = details["video_id"]

    sentiment_score = get_comment_sentiment(video_id)
    top_comments = get_top_comments(video_id)

    X_engage = preprocess_for_engagement(details)
    pred_engage = xgb.predict(X_engage)[0]

    title_vec = preprocess_for_title_with_clf(details)
    pred_title = clf.predict(title_vec)[0]

    if sentiment_score > 0.15 or pred_engage == 1 or pred_title == 1:
        final = "🔥 RECOMMENDED for Trending Section"
    else:
        final = "❄️ NOT Recommended"

    if pred_engage == 1 or pred_title == 1:
        final = "🔥 RECOMMENDED for Trending Section"
    else:
        if sentiment_score > 0.40:
           final = "⚠️ Potentially Good, but Models Predict NOT Trending"
        else:
           final = "❄️ NOT Recommended"

    return details, sentiment_score, pred_engage, pred_title, final, top_comments


# -------------------------------
# STREAMLIT UI
# -------------------------------
st.title("🎥 YouTube Trending Prediction App")
st.write("Paste a YouTube link to check if it can reach the trending section.")

url = st.text_input("Enter YouTube Video Link")

if st.button("Predict"):
    try:
        details, sentiment, pred_engage, pred_title, final, top_comments = predict_video_virality(url)

        st.subheader("📺 Video Preview")
        st.video(f"https://www.youtube.com/watch?v={details['video_id']}")

        st.subheader("📌 Video Title")
        st.write(details["title"])

        st.subheader("📊 Stats")
        st.write(f"Views: {details['views']}")
        st.write(f"Likes: {details['likes']}")
        st.write(f"Comments: {details['comments']}")

        st.subheader("🧠 Model Outputs")
        st.write(f"Sentiment Score: {sentiment:.3f}")
        st.write(f"Engagement Model Prediction: {pred_engage}")
        st.write(f"Title Model Prediction: {pred_title}")

        st.subheader("🔮 Final Recommendation")
        st.success(final)

        st.subheader("💬 Top 5 Comments")
        if top_comments:
            for i, c in enumerate(top_comments, 1):
                st.write(f"**{i}.** {c}")
        else:
            st.write("No comments found.")

    except Exception as e:
        st.error(f"Error: {str(e)}")

