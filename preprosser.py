# preprosser.py — Preprocess WhatsApp exported chat into structured DataFrame

import re
import pandas as pd

# Regex for 12-hour and 24-hour formats
PATTERNS = [
    r'(\d{1,2}/\d{1,2}/\d{2,4}), (\d{1,2}:\d{2} (?:am|pm)) - ',  # e.g. 12/03/2022, 10:30 pm -
    r'(\d{1,2}/\d{1,2}/\d{2,4}), (\d{1,2}:\d{2}) - '             # e.g. 12/03/2022, 22:30 -
]

def detect_pattern(data):
    """Detect which timestamp pattern is present in chat."""
    for pat in PATTERNS:
        if re.search(pat, data):
            return pat
    raise ValueError("Unsupported date/time format in chat export.")

def preprocess(data):
    """Convert raw WhatsApp chat export text into a clean pandas DataFrame."""

    # Choose correct regex
    pattern = detect_pattern(data)

    # Split messages using regex
    messages = re.split(pattern, data)[1:]
    # Example output: [date, time, rest_of_msg, date, time, rest_of_msg, ...]

    dates, times, messages_clean = [], [], []

    for i in range(0, len(messages), 3):
        date = messages[i]
        time = messages[i+1]
        message = messages[i+2].strip()
        dates.append(date + " " + time)
        messages_clean.append(message)

    # Separate user and message text
    users, texts = [], []
    for msg in messages_clean:
        if ": " in msg:
            user, text = msg.split(": ", 1)
            users.append(user)
            texts.append(text)
        else:
            # System notification (no user)
            users.append("group_notification")
            texts.append(msg)

    # Create DataFrame
    df = pd.DataFrame({
        "date": pd.to_datetime(dates, dayfirst=True, errors="coerce"),
        "user": users,
        "message": texts
    })

    df.dropna(subset=["date"], inplace=True)

    # Add useful breakdown columns
    df["only_date"] = df["date"].dt.date
    df["year"] = df["date"].dt.year
    df["month_num"] = df["date"].dt.month
    df["month"] = df["date"].dt.month_name()
    df["day_name"] = df["date"].dt.day_name()
    df["hour"] = df["date"].dt.hour
    df["minute"] = df["date"].dt.minute
    df["period"] = df["hour"].astype(str) + "-" + (df["hour"]+1).astype(str)

    return df
