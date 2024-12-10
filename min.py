import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import string
import nltk
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import spacy
import streamlit as st

nltk.download("vader_lexicon")

# ============================
# 1. إعداد واجهة المستخدم باستخدام Streamlit
# ============================
st.title("Text Analysis and Visualization Tool")

# إدخال النص من المستخدم
text_data = st.text_area("Enter your text:", value="""
Python is an amazing programming language. Python is popular for data analysis and machine learning.
Machine learning is the future, and Python plays a key role in that future.
""")

# ============================
# 2. تنظيف وتحليل النصوص
# ============================
def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text

cleaned_text = clean_text(text_data)

# تقسيم النص إلى كلمات
words = cleaned_text.split()

# حساب الكلمات الأكثر تكرارًا
word_counts = Counter(words)
word_df = pd.DataFrame(word_counts.items(), columns=["Word", "Count"]).sort_values(by="Count", ascending=False)

# عرض الكلمات الأكثر تكرارًا
st.subheader("Most Frequent Words")
st.write(word_df)

# ============================
# 3. تحليل المشاعر باستخدام TextBlob و VADER
# ============================
# TextBlob لتحليل المشاعر
blob = TextBlob(text_data)
textblob_sentiment = blob.sentiment.polarity  # مدى إيجابية النص

# VADER لتحليل المشاعر
vader = SentimentIntensityAnalyzer()
vader_sentiment = vader.polarity_scores(text_data)

st.subheader("Sentiment Analysis")
st.write("**TextBlob Sentiment Polarity:**", textblob_sentiment)
st.write("**VADER Sentiment Scores:**", vader_sentiment)

# ============================
# 4. استخراج الكيانات المسماة باستخدام spaCy
# ============================
# تحميل نموذج spaCy
nlp = spacy.load("en_core_web_sm")
doc = nlp(text_data)

entities = [(ent.text, ent.label_) for ent in doc.ents]
entity_df = pd.DataFrame(entities, columns=["Entity", "Type"])

st.subheader("Named Entity Recognition (NER)")
st.write(entity_df)

# ============================
# 5. الرسوم البيانية
# ============================
st.subheader("Visualizations")

# الكلمات الأكثر تكرارًا
st.write("**Top 10 Frequent Words**")
fig1, ax1 = plt.subplots(figsize=(10, 6))
sns.barplot(data=word_df.head(10), x="Count", y="Word", palette="viridis", ax=ax1, hue="Word", legend=False)
ax1.set_title("Top 10 Words in the Text")
st.pyplot(fig1)

# توزيع المشاعر باستخدام VADER
st.write("**VADER Sentiment Distribution**")
fig2, ax2 = plt.subplots(figsize=(8, 6))
sentiment_labels = list(vader_sentiment.keys())
sentiment_values = list(vader_sentiment.values())
sns.barplot(x=sentiment_labels, y=sentiment_values, palette="coolwarm", ax=ax2)
ax2.set_title("Sentiment Scores")
st.pyplot(fig2)

# ============================
# 6. تحليل البيانات العددية (اختياري)
# ============================
data = {
    "Category": ["A", "B", "C", "D", "E"],
    "Values": [23, 45, 56, 78, 12]
}
df = pd.DataFrame(data)

st.subheader("Numerical Data Analysis")
fig3, ax3 = plt.subplots(figsize=(8, 8))
ax3.pie(df["Values"], labels=df["Category"], autopct="%1.1f%%", startangle=140, colors=sns.color_palette("pastel"))
ax3.set_title("Category Distribution")
st.pyplot(fig3)
