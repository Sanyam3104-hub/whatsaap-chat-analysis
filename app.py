# app.py — WhatsApp Chat Analyzer (Rich Dashboard)
import streamlit as st
import pandas as pd
import plotly.express as px
import preprosser
import help as hp
import datetime
import io, os

st.set_page_config(page_title="WhatsApp Chat Analyzer", layout="wide")

# ------------------ Custom CSS ------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
.stApp {background-color: #17a589;}
* {font-family: 'Poppins', sans-serif;}
#MainMenu, footer {visibility: hidden;}
.hero {text-align: left; padding: 40px; color: white;}
.hero h1 {font-size: 36px; font-weight: 700; margin: 0;}
.hero p {font-size: 16px; margin-top: 8px;}
.upload-box {width: 60%; margin: 0 auto; padding: 20px;
             border: 2px dashed rgba(255,255,255,0.3); border-radius: 10px; text-align:center; color:white;}
.stat-card {background: white; color: #033; border-radius: 10px;
            padding: 18px; text-align:center; box-shadow: 0 4px 12px rgba(0,0,0,0.1);}
.stat-title {font-weight: 600; color:#0e6b61;}
</style>
""", unsafe_allow_html=True)

# ------------------ Hero ------------------
st.markdown("""
<div class="hero">
  <h1>Analyze your WhatsApp Chat in Seconds</h1>
  <p>Instant insights: discover who’s most active, peak times, hidden patterns, emoji trends and more — all offline, safe, and fast.</p>
</div>
""", unsafe_allow_html=True)

# ------------------ Upload ------------------
st.markdown('<div class="upload-box">📂 Drag & Drop your WhatsApp chat (.txt) here</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("", type=["txt"])

if uploaded_file is None:
    st.stop()

# ------------------ Preprocess ------------------
raw = uploaded_file.read()
text = raw.decode("utf-8", errors="ignore")
df = preprosser.preprocess(text)

# Sidebar filters
users = sorted([u for u in df['user'].unique() if u != 'group_notification'])
user_list = ["Overall"] + users
selected_user = st.sidebar.selectbox("Select User", user_list)

date_filter = st.sidebar.checkbox("Filter by Date")
if date_filter:
    # Ensure only_date is datetime type
    if not pd.api.types.is_datetime64_any_dtype(df['only_date']):
        df['only_date'] = pd.to_datetime(df['only_date'])

    # Convert column to date for comparison
    df['only_date'] = df['only_date'].dt.date

    # Sidebar date pickers
    start = st.sidebar.date_input("Start", value=min(df['only_date']))
    end = st.sidebar.date_input("End", value=max(df['only_date']))

    # Apply filter
    df = df[(df['only_date'] >= start) & (df['only_date'] <= end)]


# ------------------ Stats cards ------------------
num_messages, words, num_media, num_links = hp.fetch_stats(selected_user, df)
c1, c2, c3, c4 = st.columns(4)
c1.markdown(f"<div class='stat-card'><div class='stat-title'>💬 Messages</div><h2>{num_messages}</h2></div>", unsafe_allow_html=True)
c2.markdown(f"<div class='stat-card'><div class='stat-title'>📝 Words</div><h2>{words}</h2></div>", unsafe_allow_html=True)
c3.markdown(f"<div class='stat-card'><div class='stat-title'>📷 Media</div><h2>{num_media}</h2></div>", unsafe_allow_html=True)
c4.markdown(f"<div class='stat-card'><div class='stat-title'>🔗 Links</div><h2>{num_links}</h2></div>", unsafe_allow_html=True)

# ------------------ Tabs ------------------
tabs = st.tabs(["📆 Timelines", "📊 Activity", "🔤 Words & N-grams",
                "😊 Emotions", "😂 Emojis", "🧑‍🤝‍🧑 Users", "⚠ Toxicity",
                "🏆 Leaderboards", "📑 Comparison"])

# Timelines
with tabs[0]:
    st.subheader("Monthly Timeline")
    timeline = hp.monthly_timeline(selected_user, df)
    fig = px.line(timeline, x="time", y="message", markers=True)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Daily Timeline")
    daily = hp.daily_timeline(selected_user, df)
    fig = px.line(daily, x="only_date", y="message")
    st.plotly_chart(fig, use_container_width=True)

# Activity
with tabs[1]:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Most Busy Day")
        busy_day = hp.week_activity_map(selected_user, df)
        st.bar_chart(busy_day)
    with col2:
        st.subheader("Most Busy Month")
        busy_month = hp.month_activity_map(selected_user, df)
        st.bar_chart(busy_month)

    st.subheader("Heatmap")
    heatmap = hp.activity_heatmap(selected_user, df)
    st.dataframe(heatmap)

# Words
with tabs[2]:
    st.subheader("WordCloud")
    wc = hp.create_wordcloud(selected_user, df)
    st.image(wc.to_array())

    st.subheader("Most Common Words")
    common = hp.most_common_words(selected_user, df)
    st.bar_chart(common.set_index('word')['count'])

    st.subheader("Top Bigrams & Trigrams")
    bigrams = hp.top_ngrams(df, n=2, k=15)
    trigrams = hp.top_ngrams(df, n=3, k=15)
    c1, c2 = st.columns(2)
    c1.dataframe(bigrams)
    c2.dataframe(trigrams)

# Emotions
with tabs[3]:
    df2, emotions = hp.emotion_detection(df)
    st.subheader("Emotion Counts")
    st.bar_chart(emotions.set_index('emotion')['count'])

# Emojis
with tabs[4]:
    st.subheader("Top Emojis")
    e_df = hp.emoji_helper(df)
    st.dataframe(e_df.head(20))
    st.plotly_chart(px.pie(e_df.head(10), names="emoji", values="count"))

    st.subheader("Emoji Trends")
    et = hp.emoji_trends_over_time(df)
    st.line_chart(et.pivot(index="period", columns="emoji", values="count").fillna(0))

# Users
with tabs[5]:
    if selected_user == "Overall":
        x, df_percent = hp.most_busy_users(df)
        st.subheader("Most Busy Users")
        st.bar_chart(x)
        st.dataframe(df_percent)

    st.subheader("Interaction Network")
    path = hp.build_interaction_network(df)
    with open(path, "r", encoding="utf-8") as f:
        st.components.v1.html(f.read(), height=600)

# Toxicity
with tabs[6]:
    st.subheader("Toxic Messages")
    tox_msgs, tox_summary = hp.detect_toxic_messages(df)
    st.dataframe(tox_msgs[['date','user','message','toxic']].head(50))
    st.dataframe(tox_summary)

# Leaderboards
with tabs[7]:
    lb = hp.leaderboard(df)
    for k, v in lb.items():
        st.subheader(k.title())
        st.dataframe(v)

# Comparison
with tabs[8]:
    if len(users) >= 2:
        u1 = st.selectbox("User 1", users)
        u2 = st.selectbox("User 2", users)
        st.dataframe(hp.compare_users(df, u1, u2))

    st.subheader("Before vs After")
    a1 = st.date_input("Start A")
    b1 = st.date_input("End A")
    a2 = st.date_input("Start B")
    b2 = st.date_input("End B")
    if st.button("Compare"):
        result = hp.compare_time_ranges(df, a1, b1, a2, b2)
        st.write(result)