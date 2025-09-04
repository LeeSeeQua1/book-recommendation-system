import pandas as pd
import numpy as np

data_path = './data/'
books = pd.read_csv(data_path + 'Books.csv')
ratings = pd.read_csv(data_path + 'Ratings.csv')
users = pd.read_csv(data_path + 'Users.csv')

books.dropna(inplace=True)
books.drop(columns=['Image-URL-S', 'Image-URL-L'], inplace=True)

user_ratings = ratings.groupby('User-ID').size()
book_ratings = ratings.groupby('ISBN').size()

top_books = book_ratings[book_ratings > 50].index
top_users = user_ratings[user_ratings > 200].index
filtered_ratings = ratings[ratings['ISBN'].isin(top_books) & ratings['User-ID'].isin(top_users)]
book_user_table = filtered_ratings.pivot_table(index='ISBN', columns='User-ID', values='Book-Rating')
book_user_table.fillna(0, inplace=True)

from sklearn.metrics.pairwise import cosine_similarity
pairwise_sim = cosine_similarity(book_user_table)


def get_matching_titles(book_title: str) -> list[str]:
    title_words = book_title.split()
    title_words = list(map(lambda s: s.lower(), title_words))
    return books[books['Book-Title'].apply(lambda x: all(s in str(x).lower() for s in title_words))]['Book-Title'].to_list()
    # return books[books['Book-Title'].str.contains(
    #     book_title,
    #     case=False,
    #     na=False
    # )]['Book-Title'].to_list()


def recommend(book_title: str, k: int = 5) -> list[str] | None:
    match = books[books['Book-Title'] == book_title]
    if match.empty:
        print("Unknown book")
        return None

    id = match['ISBN'].iloc[0]
    if id not in book_user_table.index:
        print("This book has too little reviews")
        return None

    idx = np.where(book_user_table.index == id)[0][0]
    enum_sim = list(enumerate(pairwise_sim[idx]))
    sorted_sim = list(sorted(enum_sim, key=lambda x: x[1], reverse=True))
    neighbors = list(map(lambda x: x[0], sorted_sim[1:k+1]))
    neighbors_ids = book_user_table.index[neighbors]
    neighbours_df = books[books['ISBN'].isin(neighbors_ids)]
    return list(zip(neighbours_df['Book-Title'], neighbours_df['Image-URL-M']))