import re

from typing import Final, Pattern
import pandas as pd
import numpy as np
import string
from sklearn.metrics.pairwise import cosine_similarity

num_word: Final[dict[str, str]] = {
    '1': 'one',
    '2': 'two',
    '3': 'three',
    '4': 'four',
    '5': 'five',
    '6': 'six',
    '7': 'seven',
    '8': 'eight',
    '9': 'nine',
    '10': 'ten',
    '11': 'eleven',
    '12': 'twelve',
    '13': 'thirteen',
    '14': 'fourteen',
    '15': 'fifteen',
    '16': 'sixteen',
    '17': 'seventeen',
    '18': 'eighteen',
    '19': 'nineteen',
    '20': 'twenty',
    '30': 'thirty',
    '40': 'forty',
    '50': 'fifty',
    '60': 'sixty',
    '70': 'seventy',
    '80': 'eighty',
    '90': 'ninety'
}

roman_arabic: Final[dict[str, str]] = {
    'i': 'one',
    'ii': 'two',
    'iii': 'three',
    'iv': 'four',
    'v': 'five',
    'vi': 'six',
    'vii': 'seven',
    'viii': 'eight',
    'ix': 'nine',
    'x': 'ten',
    'xi': 'eleven',
    'xii': 'twelve',
    'xiii': 'thirteen',
    'xiv': 'fourteen',
    'xv': 'fifteen',
    'xvi': 'sixteen',
    'xvii': 'seventeen',
    'xviii': 'eighteen',
    'xix': 'nineteen',
    'xx': 'twenty',
    'xxi': 'twenty one'
}

ordinal_to_number: Final[dict[str, str]] = {
    'first': 'one',
    'second': 'two',
    'third': 'three',
    'fourth': 'four',
    'fifth': 'five',
    'sixth': 'six',
    'seventh': 'seven',
    'eighth': 'eight',
    'ninth': 'nine',
    'tenth': 'ten',
    'eleventh': 'eleven',
    'twelfth': 'twelve',
    'thirteenth': 'thirteen',
    'fourteenth': 'fourteen',
    'fifteenth': 'fifteen',
    'sixteenth': 'sixteen',
    'seventeenth': 'seventeen',
    'eighteenth': 'eighteen',
    'nineteenth': 'nineteen',
    'twentieth': 'twenty',
    'twenty-first': 'twenty one',
    'twenty-second': 'twenty two',
    'twenty-third': 'twenty three',
    'twenty-fourth': 'twenty four',
    'twenty-fifth': 'twenty five',
    'twenty-sixth': 'twenty six',
    'twenty-seventh': 'twenty seven',
    'twenty-eighth': 'twenty eight',
    'twenty-ninth': 'twenty nine',
    'thirtieth': 'thirty'
}

ordinal_pattern: Final[Pattern[str]] = re.compile(r'(\d+)(st|nd|rd|th)')
non_alphanum_pattern: Final[Pattern[str]] = re.compile(r'[^a-z0-9\s]')


def is_good_title(title: str) -> bool:
    return all(ch.isalpha()
               or ch == ' '
               or ch.isdigit()
               or ch in string.punctuation for ch in title)


def normalize(text: str) -> list[str]:
    text = text.lower()
    text = text.replace("'", "")
    text = re.sub(non_alphanum_pattern, ' ', text)
    text = re.sub(ordinal_pattern, r'\1', text)
    norm_words = []
    for w in text.split():
        norm_words += num_to_word(w)
    norm_words = list(map(lambda s: roman_arabic[s] if s in roman_arabic else s, norm_words))
    norm_words = list(map(lambda s: ordinal_to_number[s] if s in ordinal_to_number else s, norm_words))
    return norm_words


def num_to_word(s: str) -> list[str]:
    if s in num_word:
        return [num_word[s]]
    if s[:-2] in num_word and s[-2:] == '00':
        return [num_word[s[:-2]], 'hundred']
    if s[:-3] in num_word and s[-3:] == '000':
        return [num_word[s[:-3]], 'thousand']
    return [s]


class CFModel:

    def __init__(self, books_dataset: pd.DataFrame, users_dataset: pd.DataFrame) -> None:
        self.pictures = None
        self.top_users = None
        self.title_dict = None
        self.top_books = None
        self.book_user_table = None
        self.pairwise_sim = None
        self.books = books_dataset
        self.users = users_dataset

        self.books.dropna(inplace=True)
        self.books.drop(columns=['Image-URL-S', 'Image-URL-M'], inplace=True)
        self.books = self.books[self.books['Book-Title'].apply(is_good_title)]

    def fit(self, ratings: pd.DataFrame,
            min_book_reviews: int = 50,
            min_user_ratings: int = 200) -> None:
        new_ratings = ratings.merge(self.books, on='ISBN')
        new_ratings.drop(columns=[
            'Year-Of-Publication',
            'Publisher',
            'Book-Author'],
            inplace=True)
        user_ratings = ratings.groupby('User-ID').size()
        book_ratings = new_ratings.groupby('Book-Title')['Book-Rating'].agg(
            avg_rating='mean',
            num_ratings='count'
        )
        self.top_books = book_ratings[book_ratings['num_ratings'] > min_book_reviews].index.to_series()
        books_by_title = self.books.groupby('Book-Title')['Image-URL-L'].first()
        self.pictures = books_by_title[books_by_title.index.isin(self.top_books)]
        self.top_users = user_ratings[user_ratings > min_user_ratings].index
        filtered_ratings = new_ratings[
            new_ratings['Book-Title'].isin(self.top_books) & new_ratings['User-ID'].isin(self.top_users)]
        self.book_user_table = filtered_ratings.pivot_table(index='Book-Title', columns='User-ID', values='Book-Rating')

        self.title_dict = dict(zip(self.top_books.apply(lambda t: " ".join(normalize(t))), self.top_books))

        self.book_user_table.fillna(0, inplace=True)
        self.book_user_table = self.book_user_table.loc[(self.book_user_table != 0).any(axis=1)]
        self.pairwise_sim = cosine_similarity(self.book_user_table)

    def recommend(self, book_title: str, num_books: int = 5) -> list[str]:
        indices = np.where(self.book_user_table.index == book_title)
        if len(indices) == 0 or indices[0].size == 0:
            return []
        idx = indices[0][0]

        enum_sim = list(enumerate(self.pairwise_sim[idx]))
        sorted_sim = list(sorted(enum_sim, key=lambda x: x[1], reverse=True))
        neighbors = list(map(lambda x: x[0], sorted_sim[1: num_books + 1]))
        ans = self.book_user_table.index[neighbors]
        return ans.to_list()

    def get_matching_titles(self, book_title: str) -> list[str]:
        title_words = normalize(book_title)
        processed_words = []
        for w in title_words:
            processed_words += num_to_word(w)
        return [self.title_dict[k] for k in self.title_dict if all(s in k.split() for s in processed_words)]

    def get_top_users(self) -> list[str]:
        return self.top_users.index.to_list()

    def get_pictures(self, titles: list[str]) -> list[str]:
        pic_series = self.pictures[self.pictures.index.isin(titles)]
        title_pic_list = list(zip(pic_series.index, pic_series))
        return list(map(lambda t: t[1], sorted(title_pic_list, key=lambda t: titles.index(t[0]))))
