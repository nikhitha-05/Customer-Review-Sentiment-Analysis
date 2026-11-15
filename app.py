
import pandas as pd

# Load dataset
df = pd.read_csv("dataset/Dataset-SA.csv")

print("Original Dataset Shape:", df.shape)

# Remove duplicate rows
df = df.drop_duplicates()

# Remove rows with missing review or sentiment
df = df.dropna(subset=["Review", "Sentiment"])

# Clean text columns
df["Review"] = df["Review"].astype(str).str.strip()
df["Sentiment"] = df["Sentiment"].astype(str).str.strip().str.lower()

# Remove empty reviews
df = df[df["Review"] != ""]

# Save cleaned dataset
df.to_csv("output/cleaned_reviews.csv", index=False)

print("\nDATA CLEANING COMPLETED!")
print("Cleaned Dataset Shape:", df.shape)
print("\nMissing Values:")
print(df.isnull().sum())
print("\nSentiment Counts:")
print(df["Sentiment"].value_counts())

import matplotlib.pyplot as plt
import seaborn as sns

# Sentiment count
sentiment_counts = df["Sentiment"].value_counts()

# Bar chart
plt.figure(figsize=(8, 5))
sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values)

plt.title("Customer Review Sentiment Analysis")
plt.xlabel("Sentiment")
plt.ylabel("Number of Reviews")

plt.savefig("output/sentiment_analysis.png")
plt.show()

print("Sentiment visualization completed!")

from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Combine all customer reviews
text = " ".join(df["Review"].dropna().astype(str))

# Generate Word Cloud
wordcloud = WordCloud(
    width=1200,
    height=600,
    background_color="white",
    max_words=100
).generate(text)

# Display and save
plt.figure(figsize=(12, 6))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Customer Review Word Cloud")

plt.savefig("output/wordcloud.png")
plt.show()

print("Word Cloud generated successfully!")

# Sentiment Percentage Analysis

sentiment_percentage = df["Sentiment"].value_counts(normalize=True) * 100

print("\nSENTIMENT PERCENTAGE ANALYSIS")
print(sentiment_percentage.round(2))

# Create Pie Chart
plt.figure(figsize=(8, 8))

plt.pie(
    sentiment_percentage.values,
    labels=sentiment_percentage.index,
    autopct="%1.1f%%",
    startangle=90
)

plt.title("Customer Sentiment Percentage")
plt.savefig("output/sentiment_percentage.png")
plt.show()

print("Sentiment percentage analysis completed!")

# FINAL PROJECT INSIGHTS

total_reviews = len(df)

sentiment_counts = df["Sentiment"].value_counts()
sentiment_percentages = df["Sentiment"].value_counts(normalize=True) * 100

print("\n========== FINAL PROJECT INSIGHTS ==========")

print("Total Customer Reviews:", total_reviews)

print("\nPositive Reviews:",
      sentiment_counts.get("positive", 0))

print("Negative Reviews:",
      sentiment_counts.get("negative", 0))

print("Neutral Reviews:",
      sentiment_counts.get("neutral", 0))

print("\nSentiment Percentages:")
print(sentiment_percentages.round(2))

print("\nKEY FINDINGS:")
print("1. Positive reviews are the highest.")
print("2. Negative reviews are lower than positive reviews.")
print("3. Neutral reviews are the lowest.")

print("\nPROJECT ANALYSIS COMPLETED SUCCESSFULLY!")

