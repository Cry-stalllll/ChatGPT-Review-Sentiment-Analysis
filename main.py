# Import necessary libraries
import pandas as pd
from textblob import TextBlob
import re
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from collections import Counter

file_path = 'C:/Users/user/OneDrive/Desktop/MMU file/2024 August/social media/chatgpt1.csv'  # Update with actual path
data = pd.read_csv(file_path)

print("Dataset Head:")
print(data.head())
print("\nDataset Info:")
print(data.info())

# --- Preprocessing function to clean text ---
def preprocess_text(text):
    text = text.lower()  # Convert text to lowercase
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text


text_column = 'Text' if 'Text' in data.columns else data.columns[0]
data['cleaned_text'] = data[text_column].apply(preprocess_text)

# --- Function to classify sentiment using TextBlob ---
def get_sentiment(text):
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0:
        return 'positive'
    elif analysis.sentiment.polarity < 0:
        return 'negative'
    else:
        return 'neutral'
    
data['Sentiment'] = data['cleaned_text'].apply(get_sentiment)

print("\nModified Dataset with Sentiment Labels:")
print(data.head())

print("\nSentiment Distribution:")
print(data['Sentiment'].value_counts())

# Save the modified dataset to a new CSV file
output_file_path = 'C:/Users/user/OneDrive/Desktop/MMU file/2024 August/social media/modified_dataset_with_sentiment.csv' # Update with actual path
data = pd.read_csv(file_path)
data.to_csv(output_file_path, index=False)

# --- 1. Sentiment Distribution Bar Chart ---
plt.figure(figsize=(8, 5))
data['Sentiment'].value_counts().plot(kind='bar', color=['green', 'red', 'blue'], edgecolor='black')
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.show()

# --- 2. Word Cloud for Each Sentiment ---
custom_stopwords = set(STOPWORDS)
custom_stopwords.update(["chatgpt", "ChatGPT", "chat", "gpt", "de", "que"]) 

def generate_wordcloud(text, sentiment):
    wordcloud = WordCloud(
        width=800, 
        height=400, 
        background_color='white', 
        collocations=False,
        stopwords=custom_stopwords 
    ).generate(text)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Word Cloud - {sentiment.capitalize()} Sentiment')
    plt.show()

positive_text = ' '.join(data[data['Sentiment'] == 'positive']['cleaned_text'].dropna())
negative_text = ' '.join(data[data['Sentiment'] == 'negative']['cleaned_text'].dropna())
neutral_text = ' '.join(data[data['Sentiment'] == 'neutral']['cleaned_text'].dropna())

generate_wordcloud(positive_text, 'positive')
generate_wordcloud(negative_text, 'negative')
generate_wordcloud(neutral_text, 'neutral')

# --- 3. Sentiment by Language ---
top_languages = data['Language'].value_counts().nlargest(5).index

top_language_data = data[data['Language'].isin(top_languages)]
top_language_sentiment_counts = top_language_data.groupby(['Language', 'Sentiment']).size().unstack().fillna(0)

plt.figure(figsize=(14, 6))
top_language_sentiment_counts.plot(kind='bar', stacked=True, colormap='viridis', ax=plt.gca())
plt.title('Sentiment Distribution by Top 5 Languages')
plt.xlabel('Language')
plt.ylabel('Number of Tweets')
plt.legend(title='Sentiment')
plt.show()

# --- 4. Top Hashtags by Sentiment ---
data['hashtag_list'] = data['hashtag'].apply(lambda x: eval(x) if pd.notna(x) else [])
positive_hashtags = data[data['Sentiment'] == 'positive']['hashtag_list'].explode().dropna()
negative_hashtags = data[data['Sentiment'] == 'negative']['hashtag_list'].explode().dropna()
neutral_hashtags = data[data['Sentiment'] == 'neutral']['hashtag_list'].explode().dropna()

positive_hashtag_counts = Counter(positive_hashtags).most_common(10)
negative_hashtag_counts = Counter(negative_hashtags).most_common(10)
neutral_hashtag_counts = Counter(neutral_hashtags).most_common(10)

positive_hashtags_df = pd.DataFrame(positive_hashtag_counts, columns=['Hashtag', 'Frequency'])
negative_hashtags_df = pd.DataFrame(negative_hashtag_counts, columns=['Hashtag', 'Frequency'])
neutral_hashtags_df = pd.DataFrame(neutral_hashtag_counts, columns=['Hashtag', 'Frequency'])

plt.figure(figsize=(14, 6))

plt.subplot(1, 3, 1)
plt.barh(positive_hashtags_df['Hashtag'], positive_hashtags_df['Frequency'], color='green')
plt.gca().invert_yaxis()
plt.title('Positive Sentiment - Top Hashtags')
plt.xlabel('Frequency')

plt.subplot(1, 3, 2)
plt.barh(negative_hashtags_df['Hashtag'], negative_hashtags_df['Frequency'], color='red')
plt.gca().invert_yaxis()
plt.title('Negative Sentiment - Top Hashtags')
plt.xlabel('Frequency')

plt.subplot(1, 3, 3)
plt.barh(neutral_hashtags_df['Hashtag'], neutral_hashtags_df['Frequency'], color='blue')
plt.gca().invert_yaxis()
plt.title('Neutral Sentiment - Top Hashtags')
plt.xlabel('Frequency')

plt.tight_layout()
plt.show()
