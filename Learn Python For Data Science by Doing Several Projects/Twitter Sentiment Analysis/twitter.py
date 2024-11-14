import tweepy
from textblob import TextBlob
import pandas as pd
import os

# Step 1 - Authenticate (use Bearer Token for API v2)
bearer_token = os.getenv('BEARER_TOKEN')

client = tweepy.Client(bearer_token=bearer_token)

# Step 3 - Retrieve Tweets using the v2 search_recent_tweets method
query = 'MSI'  # Enter your query
tweets_response = client.search_recent_tweets(query=query, max_results=10)  # Adjust max_results as needed

tweets, sentiment = [], []

for tweet in tweets_response.data:
    tweets.append(tweet.text)
    analysis = TextBlob(tweet.text).sentiment.polarity
    sentiment.append('Positive' if analysis >= 0.5 else 'Negative')

df = pd.DataFrame({'Tweets': tweets, 'Sentiment': sentiment})
df.to_csv('./Learn Python For Data Science by Doing Several Projects/tweets.csv', index=False)
