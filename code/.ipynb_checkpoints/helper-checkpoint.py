import pandas as pd
import numpy as np
import json
import time
import requests
import datetime
import praw
import re
import string
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer


# Using PRAW to scrape

#Loading Reddit credentials
def load_credentials():
    with open('credentials.json') as f:
        data = json.load(f)
    return data["reddit"]

credentials = load_credentials()
client_id = credentials['client_id']
client_secret = credentials['client_secret']
username = credentials['username']
password = credentials['password']
user_agent = credentials['user_agent']


# Authenticating with Reddit API
reddit = praw.Reddit(
    client_id=client_id,
    client_secret=client_secret,
    username=username,
    password=password,
    user_agent=user_agent
)



#Defining getting top comment from post
def get_top_comments(post):
    '''

    
    '''
    top_comments=[]
    post.comments.replace_more(limit=None)  # Retrieve all comments (including nested ones)
    comments = post.comments.list()
    if len(comments) > 0:
        comments.sort(key=lambda c: c.score, reverse=True)  # Sort comments by score in descending order
        top_comments= [comment.body for comment in comments[:5]]
    return top_comments



#Defining scraping function
def get_subreddit_posts(subreddit_name, post_limit=1000):
    """
     Scrapes the subreddit of choice and returns the title, post datetime, post contents, 
     number of upvotes, downvotes, if author is a premium reddit user, no of comments, subreddit name,
     url suffix and awards received on post.
    
     :param subreddit_name: The name of the subreddit without prefix 'r/'
     :type subreddit_name: string
    
     :param post_limit: Number of posts to scrape. Function scrapes just slightly over the limit.
     :type post_limit: int
    
     :return: A list of dictionaries with the post data
     :rtype: list of dicts
    """
    posts_list = []
    
    subreddit = reddit.subreddit(subreddit_name)
    print(f"Post limit: {post_limit}")
    posts = subreddit.new(limit=post_limit)
    for post in posts:
        post_item = {
                "time": datetime.datetime.fromtimestamp(post.created),
                "title": post.title,
                "content": post.selftext,
                "top_10_comments": get_top_comments(post),
                "upvotes": post.score,
                "no_of_comments": post.num_comments,
                "url": post.url
            }

        posts_list.append(post_item)
        print(f"Post id added:{post.id}")
        
        if len(posts_list) % 10 == 0:
            print(f"Number of posts successfully scrapped from r/{subreddit_name}: {len(posts_list)}")


    return posts_list


def extract_tags(row):
    match = re.search(r'\[.*?\]',row)
    if match:
        return match.group(0)
    else:
        return ''

    

def tokenize_text(text):
    '''
    
    '''
    text = text.lower()
    text = "".join([char for char in text if char not in string.punctuation])
    tokens = word_tokenize(text)
    custom_stopwords = ['use', 'would','see','2023','info','given','using','open', 'know','trying','non','would']
    stop_words = stopwords.words('english') + custom_stopwords
    tokens = [token for token in tokens if token not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return tokens


def get_top_keywords(tfidf_matrix, clusters, feature_names, n_terms):
    df = pd.DataFrame(tfidf_matrix.todense()).groupby(clusters).mean()
    for i, r in df.iterrows():
        print('\nCluster {}'.format(i))
        print(','.join([feature_names[t] for t in np.argsort(r)[-n_terms:]]))


def extract_top_words_tfidf(df, text_column, feature_variable, top_n=20):
    """
    Extracts the top n words from the TF-IDF representation of a column with tokenized words and appends them to the feature variable.

    Args:
        df (pandas DataFrame): The DataFrame containing the text data.
        text_column (str): The name of the column containing the tokenized words.
        feature_variable (str): The name of the feature variable to append the extracted top words.
        top_n (int): The number of top words to extract (default: 20).

    Returns:
        pandas DataFrame: The updated DataFrame with the appended feature variable.
    """
    # Initialize the TF-IDF vectorizer
    tfidf_vectorizer = TfidfVectorizer()

    # Fit the vectorizer on the tokenized words
    tfidf_vectorizer.fit([row.split(' ') for row in df['tokenized_title']])

    # Extract the top words from the TF-IDF vectors and append them to the feature variable
    for i in range(top_n):
        col_name = f"top_word_{i+1}"
        top_words = [tfidf_vectorizer.get_feature_names()[idx] for idx in tfidf_vectorizer.transform(df[text_column]).toarray().argsort(axis=1)[:, -i-1]]
        df[col_name] = top_words

    return df


def OHE_catcolumn(df, col):
    '''
    
    '''
    X_OHE = pd.get_dummies(df[col],drop_first=True)
    return X_OHE