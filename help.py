# help.py — WhatsApp Chat Analyzer (ALL-IN-ONE)
# Contains all analytics: stats, timelines, text, emotions, toxicity, interactions, comparisons, fun features

import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from urlextract import URLExtract
from wordcloud import WordCloud
import emoji
import re
import datetime
from sklearn.feature_extraction.text import CountVectorizer
from langdetect import detect_langs, DetectorFactory
import networkx as nx
from pyvis.network import Network

DetectorFactory.seed = 0
extract = URLExtract()

# -----------------------------
# 📊 Basic Stats
# -----------------------------
def fetch_stats(selected_user, df):
    if selected_user != "Overall":
        df = df[df['user'] == selected_user]

    num_messages = df.shape[0]
    words = []
    for msg in df['message']:
        words.extend(str(msg).split())

    num_media = df[df['message'] == '<Media omitted>\n'].shape[0]
    links = []
    for msg in df['message']:
        links.extend(extract.find_urls(str(msg)))

    return num_messages, len(words), num_media, len(links)

def most_busy_users(df):
    x = df['user'].value_counts().head()
    df_percent = round((df['user'].value_counts()/df.shape[0])*100, 2).reset_index()
    df_percent.columns = ['name','percent']
    return x, df_percent

# -----------------------------
# 📆 Timelines & Activity
# -----------------------------
def monthly_timeline(selected_user, df):
    if selected_user != "Overall":
        df = df[df['user'] == selected_user]
    timeline = df.groupby(['year','month_num','month']).count()['message'].reset_index()
    timeline['time'] = timeline['month'] + "-" + timeline['year'].astype(str)
    return timeline

def daily_timeline(selected_user, df):
    if selected_user != "Overall":
        df = df[df['user'] == selected_user]
    return df.groupby('only_date').count()['message'].reset_index()

def week_activity_map(selected_user, df):
    if selected_user != "Overall":
        df = df[df['user'] == selected_user]
    return df['day_name'].value_counts()

def month_activity_map(selected_user, df):
    if selected_user != "Overall":
        df = df[df['user'] == selected_user]
    return df['month'].value_counts()

def activity_heatmap(selected_user, df):
    if selected_user != "Overall":
        df = df[df['user'] == selected_user]
    heatmap = df.pivot_table(index='day_name', columns='period', values='message', aggfunc='count').fillna(0)
    ordered = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    return heatmap.reindex(ordered)

# -----------------------------
# 📊 Advanced Analytics
# -----------------------------
def build_interaction_network(df, min_edge=1, output_html="chat_network.html"):
    df = df.sort_values('date').reset_index(drop=True)
    df['next_user'] = df['user'].shift(-1)
    transitions = df[(df['user'] != df['next_user']) & df['next_user'].notna()]
    pairs = list(zip(transitions['user'], transitions['next_user']))
    counts = Counter(pairs)
    G = nx.DiGraph()
    for (u,v), w in counts.items():
        if w >= min_edge:
            G.add_edge(u, v, weight=w, title=str(w))
    net = Network(directed=True, height="600px", width="100%")
    net.from_nx(G)
    net.repulsion(node_distance=200, central_gravity=0.1)
    net.save_graph(output_html)
    return output_html

def detect_bursts(df, window_minutes=10, threshold=50):
    times = df['date'].sort_values().reset_index(drop=True)
    bursts = []
    for i in range(len(times)):
        window_end = times[i] + pd.Timedelta(minutes=window_minutes)
        j = times.searchsorted(window_end, side='right') - 1
        count = j - i + 1
        if count >= threshold:
            bursts.append({'start': times[i], 'end': times[j], 'count': count})
    return pd.DataFrame(bursts)

def calculate_response_times(df):
    df = df.sort_values('date').reset_index(drop=True)
    df['next_user'] = df['user'].shift(-1)
    df['next_time'] = df['date'].shift(-1)
    df['resp_sec'] = (df['next_time'] - df['date']).dt.total_seconds()
    replies = df[(df['user'] != df['next_user']) & (df['resp_sec']>0)].copy()
    avg = replies.groupby('next_user')['resp_sec'].agg(['mean','median','count']).reset_index()
    avg.rename(columns={'next_user':'user'}, inplace=True)
    return replies, avg

def media_counts_per_day(df):
    temp = df.copy()
    temp['is_media'] = temp['message'].str.contains(r'<Media omitted>|image|video|sticker|attached', case=False, na=False)
    grouped = temp.groupby('only_date').agg(text=('is_media', lambda x:(~x).sum()), media=('is_media','sum'))
    grouped['total'] = grouped['text']+grouped['media']
    return grouped.reset_index()

# -----------------------------
# 🔤 Text & Language
# -----------------------------
def create_wordcloud(selected_user, df, stopwords_path='stop_hinglish.txt'):
    stop_words = []
    try:
        with open(stopwords_path, 'r', encoding='utf-8') as f:
            stop_words = set([w.strip() for w in f.readlines()])
    except: pass
    if selected_user != "Overall":
        df = df[df['user'] == selected_user]
    temp = df[(df['user']!='group_notification') & (df['message']!='<Media omitted>\n')]
    def clean(msg): return " ".join([w for w in str(msg).lower().split() if w not in stop_words])
    text = " ".join(temp['message'].apply(clean))
    wc = WordCloud(width=500,height=500,min_font_size=10,background_color='white')
    return wc.generate(text)

def most_common_words(selected_user, df, n=20):
    if selected_user != "Overall":
        df = df[df['user'] == selected_user]
    words = []
    for msg in df['message']:
        words.extend(str(msg).lower().split())
    common = Counter(words).most_common(n)
    return pd.DataFrame(common, columns=['word','count'])

def detect_languages(df, sample=2000):
    msgs = df['message'].astype(str).tolist()[:sample]
    langs = []
    for m in msgs:
        try:
            langs.append(detect_langs(m)[0].lang)
        except: pass
    counts = Counter(langs)
    total = sum(counts.values())
    return pd.DataFrame([(k,v,round(v/total*100,2)) for k,v in counts.items()], columns=['lang','count','percent'])

def message_length_distribution(df):
    df['char_len'] = df['message'].astype(str).apply(len)
    df['word_len'] = df['message'].astype(str).apply(lambda x: len(str(x).split()))
    return df[['char_len','word_len']]

def top_ngrams(df, n=2, k=20):
    texts = df['message'].astype(str).tolist()
    vect = CountVectorizer(ngram_range=(n,n), stop_words='english')
    X = vect.fit_transform(texts)
    sums = np.asarray(X.sum(axis=0)).ravel()
    terms = vect.get_feature_names_out()
    pairs = sorted(zip(terms, sums), key=lambda x:x[1], reverse=True)[:k]
    return pd.DataFrame(pairs, columns=['ngram','count'])

# -----------------------------
# 😊 Sentiment & Emotions
# -----------------------------
EMOTION_LEXICON = {
    'joy': {'happy','yay','awesome','great','love','lol','hurray'},
    'sadness': {'sad','miss','cry','sorry','down','tears'},
    'anger': {'angry','mad','hate','stupid','idiot','fuck','damn'},
    'fear': {'scared','afraid','panic'},
    'surprise': {'wow','omg','whoa'},
    'love': {'love','darling','babe','❤','💕'}
}

def emotion_detection(df):
    def detect(msg):
        found = []
        txt = str(msg).lower()
        for emo, words in EMOTION_LEXICON.items():
            if any(w in txt for w in words): found.append(emo)
        return found
    df['emotions'] = df['message'].apply(detect)
    flat = Counter([e for lst in df['emotions'] for e in lst])
    return df, pd.DataFrame(flat.items(), columns=['emotion','count'])

# -----------------------------
# ⚠️ Toxicity Detection
# -----------------------------
BAD_WORDS = {
    'insult': ['idiot','stupid','moron','dumb'],
    'swear': ['fuck','shit','bitch','bastard'],
    'hate': ['hate','racist','terrorist']
}

def detect_toxic_messages(df):
    def check(msg):
        res = {}
        for cat,words in BAD_WORDS.items():
            found = [w for w in words if w in str(msg).lower()]
            if found: res[cat]=found
        return res
    df['toxic'] = df['message'].apply(check)
    df['is_toxic'] = df['toxic'].apply(lambda d: bool(d))
    toxic_msgs = df[df['is_toxic']]
    summary = []
    for _,row in toxic_msgs.iterrows():
        for cat,words in row['toxic'].items():
            summary.append((row['user'],cat,len(words)))
    return toxic_msgs, pd.DataFrame(summary, columns=['user','category','count'])

# -----------------------------
# 😂 Emoji Analysis
# -----------------------------
def emoji_helper(df):
    emojis = []
    for msg in df['message']:
        emojis.extend([c for c in str(msg) if emoji.is_emoji(c)])
    return pd.DataFrame(Counter(emojis).most_common(), columns=['emoji','count'])

def emoji_trends_over_time(df, freq='M'):
    df['period'] = df['date'].dt.to_period(freq)
    rows = []
    for _,row in df.iterrows():
        for c in str(row['message']):
            if emoji.is_emoji(c):
                rows.append((str(row['period']),c))
    return pd.DataFrame(rows, columns=['period','emoji']).value_counts().reset_index(name='count')

# -----------------------------
# 🏆 Fun Features
# -----------------------------
def first_and_last_message(df):
    first = df.sort_values('date').iloc[0]
    last = df.sort_values('date').iloc[-1]
    return first, last

def most_active_hour(df):
    df['hour'] = df['date'].dt.hour
    return df['hour'].value_counts()

def leaderboard(df):
    wordiest = df.groupby('user')['message'].apply(lambda s: sum(len(str(m).split()) for m in s)).reset_index(name='words')
    emojis = df.groupby('user')['message'].apply(
        lambda s: sum([1 for m in s for c in str(m) if emoji.is_emoji(c)])
    ).reset_index(name='emojis')

    longest = df.groupby('user')['message'].apply(lambda s: np.mean([len(str(m).split()) for m in s])).reset_index(name='avg_len')
    return {'wordiest':wordiest.sort_values('words',ascending=False),
            'most_emojis':emojis.sort_values('emojis',ascending=False),
            'longest_avg':longest.sort_values('avg_len',ascending=False)}

# -----------------------------
# 📑 Comparison Features
# -----------------------------
def compare_users(df, user1, user2):
    d1 = df[df['user']==user1]
    d2 = df[df['user']==user2]
    stats = {
        'msg_count':[len(d1), len(d2)],
        'avg_len':[d1['message'].str.split().map(len).mean(), d2['message'].str.split().map(len).mean()]
    }
    return pd.DataFrame(stats, index=[user1,user2])

def compare_time_ranges(df, start1,end1,start2,end2):
    d1 = df[(df['only_date']>=start1)&(df['only_date']<=end1)]
    d2 = df[(df['only_date']>=start2)&(df['only_date']<=end2)]
    return {'range1':fetch_stats("Overall",d1), 'range2':fetch_stats("Overall",d2)}
