import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------
# Load Dataset
# ----------------------------
df = pd.read_csv("Instagram data.csv", encoding="latin1")

print("Dataset Shape:", df.shape)
print("\nColumns:")
print(df.columns)

print("\nFirst 5 rows:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())

# ----------------------------
# Feature Engineering
# ----------------------------
df["Engagement"] = (
    df["Likes"] +
    df["Comments"] +
    df["Shares"] +
    df["Saves"]
)

df["Engagement_Rate"] = df["Engagement"] / df["Impressions"]

# ----------------------------
# Basic Statistics
# ----------------------------
print("\nBasic Stats:")
print(df.describe())

# ----------------------------
# Reach Source Analysis
# ----------------------------
reach_cols = [
    "From Home",
    "From Hashtags",
    "From Explore",
    "From Other"
]

reach_sum = df[reach_cols].sum()
print("\nTotal Impressions by Source:")
print(reach_sum)

reach_sum.plot(kind="bar", title="Impressions by Source")
plt.ylabel("Total Impressions")
plt.tight_layout()
plt.show()

# ----------------------------
# Engagement vs Impressions
# ----------------------------
plt.scatter(df["Impressions"], df["Engagement"])
plt.xlabel("Impressions")
plt.ylabel("Engagement")
plt.title("Impressions vs Engagement")
plt.show()

# ----------------------------
# Likes vs Follows
# ----------------------------
plt.scatter(df["Likes"], df["Follows"])
plt.xlabel("Likes")
plt.ylabel("Follows")
plt.title("Likes vs Follows")
plt.show()

# ----------------------------
# Correlation Heatmap
# ----------------------------
numeric_df = df.select_dtypes(include=np.number)

plt.figure(figsize=(10, 6))
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# ----------------------------
# Top Performing Posts
# ----------------------------
top_posts = df.sort_values(by="Engagement_Rate", ascending=False).head(5)
print("\nTop 5 Posts by Engagement Rate:")
print(top_posts[["Impressions", "Engagement", "Engagement_Rate"]])

print("\nEDA Completed Successfully")
