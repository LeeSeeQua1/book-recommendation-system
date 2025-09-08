import re

import pandas as pd
import numpy as np
import string
from sklearn.metrics.pairwise import cosine_similarity


def is_good_title(title):
    return all(ch.isalpha()
               or ch == ' '
               or ch.isdigit()
               or ch in string.punctuation for ch in title)


def normalize(text: str) -> str:
    text = re.sub(r'[^A-Za-z0-9\s]', '', text)
    return text.lower()


def get_matching_titles(book_title: str) -> list[str]:
    title_words = list(map(normalize, book_title.split()))
    return (top_books[top_books.to_series()
            .apply(lambda x: all(s in normalize(str(x)).split() for s in title_words))]
            .to_list())


def recommend(book_title: str, k: int = 5) -> list[str]:
    indices = np.where(book_user_table.index == book_title)
    if len(indices) == 0 or indices[0].size == 0:
        print("Unknown book")
        return []
    idx = indices[0][0]

    enum_sim = list(enumerate(pairwise_sim[idx]))
    sorted_sim = list(sorted(enum_sim, key=lambda x: x[1], reverse=True))
    neighbors = list(map(lambda x: x[0], sorted_sim[1:k + 1]))
    similarities = list(map(lambda x: x[1], sorted_sim[1:k + 1]))
    ans = book_user_table.index[neighbors]
    return list(zip(ans.to_list(), similarities))


MIN_BOOK_REVIEWS = 25
MIN_USER_RATINGS = 100

data_path = './data/'
books = pd.read_csv(data_path + 'Books.csv')
ratings = pd.read_csv(data_path + 'Ratings.csv')
users = pd.read_csv(data_path + 'Users.csv')

books.dropna(inplace=True)
books.drop(columns=['Image-URL-S', 'Image-URL-L'], inplace=True)
books = books[books['Book-Title'].apply(is_good_title)]

new_ratings = ratings.merge(books, on='ISBN')
new_ratings.drop(columns=[
    'Year-Of-Publication',
    'Publisher',
    'Image-URL-M',
    'Book-Author'],
    inplace=True)
user_ratings = ratings.groupby('User-ID').size()
book_ratings = new_ratings.groupby('Book-Title')['Book-Rating'].agg(
    avg_rating='mean',
    num_ratings='count'
)

top_books = book_ratings[book_ratings['num_ratings'] > MIN_BOOK_REVIEWS].index
top_users = user_ratings[user_ratings > MIN_USER_RATINGS].index
filtered_ratings = new_ratings[new_ratings['Book-Title'].isin(top_books) & new_ratings['User-ID'].isin(top_users)]
book_user_table = filtered_ratings.pivot_table(index='Book-Title', columns='User-ID', values='Book-Rating')
book_user_table.fillna(0, inplace=True)

pairwise_sim = cosine_similarity(book_user_table)
