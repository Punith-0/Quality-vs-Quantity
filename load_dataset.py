
import pandas as pd

imdb_url = "https://raw.githubusercontent.com/laxmimerit/All-CSV-ML-Data-Files-Download/master/IMDB-Dataset.csv"
df_data = pd.read_csv(imdb_url)
df_data = df_data.sample(10000, random_state=42)
texts = df_data['review'].tolist()
labels = df_data['sentiment'].map({'positive': 1, 'negative': 0}).tolist()
df_data.to_csv('imdb_dataset.csv', index=False)

twitter_url = "https://raw.githubusercontent.com/dD2405/Twitter_Sentiment_Analysis/master/train.csv"
twitter_df = pd.read_csv(twitter_url)
twitter_df = twitter_df[['tweet', 'label']]
twitter_df = twitter_df.sample(10000, random_state=42)
twitter_df.rename(columns={'tweet': 'text', 'label': 'sentiment'}, inplace=True)
twitter_df.to_csv('twitter_dataset.csv', index=False)