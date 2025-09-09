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


class CFModel:

    # TODO shouldn't init return object created
    def __init__(self, books_dataset: pd.DataFrame, users_dataset: pd.DataFrame):
        self.top_books = None
        self.book_user_table = None
        self.pairwise_sim = None
        self.books = books_dataset
        self.users = users_dataset

        self.books.dropna(inplace=True) #TODO will this change the original dataset? Probably will
        self.books.drop(columns=['Image-URL-S', 'Image-URL-L'], inplace=True)
        self.books = self.books[self.books['Book-Title'].apply(is_good_title)]

    def fit(self, ratings: pd.DataFrame,
            min_book_reviews:int = 50,
            min_user_ratings: int = 200):

        new_ratings = ratings.merge(self.books, on='ISBN')
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

        self.top_books = book_ratings[book_ratings['num_ratings'] > min_book_reviews].index
        top_users = user_ratings[user_ratings > min_user_ratings].index
        filtered_ratings = new_ratings[
            new_ratings['Book-Title'].isin(self.top_books) & new_ratings['User-ID'].isin(top_users)]
        self.book_user_table = filtered_ratings.pivot_table(index='Book-Title', columns='User-ID', values='Book-Rating')
        self.book_user_table.fillna(0, inplace=True)
        self.pairwise_sim = cosine_similarity(self.book_user_table)

    def recommend(self, book_title: str, num_books: int = 5) -> list[str]:
        indices = np.where(self.book_user_table.index == book_title)
        if len(indices) == 0 or indices[0].size == 0:
            print("Unknown book")
            return []
        idx = indices[0][0]

        enum_sim = list(enumerate(self.pairwise_sim[idx]))
        sorted_sim = list(sorted(enum_sim, key=lambda x: x[1], reverse=True))
        neighbors = list(map(lambda x: x[0], sorted_sim[1 : num_books + 1]))
        similarities = list(map(lambda x: x[1], sorted_sim[1 : num_books + 1]))
        ans = self.book_user_table.index[neighbors]
        return list(zip(ans.to_list(), similarities))

    def get_matching_titles(self, book_title: str) -> list[str]:
        title_words = list(map(normalize, book_title.split()))
        return (self.top_books[self.top_books.to_series()
                .apply(lambda x: all(s in normalize(str(x)).split() for s in title_words))]
                .to_list())
