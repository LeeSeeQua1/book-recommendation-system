import pytest
import pandas as pd
from model import CFModel

data_path = './data/'
books = pd.read_csv(data_path + 'Books.csv')
ratings = pd.read_csv(data_path + 'Ratings.csv')
users = pd.read_csv(data_path + 'Users.csv')

model = CFModel(books, users)
model.fit(ratings)

with open("dump.txt", "w") as f:
    for k, v in model.title_dict.items():
        f.write(f"{k}: {v}\n")

titles = {
    'rich dad, poor dad',
    'rich dad poor dad',
    '52 deck series',
    '9 11',
    '9-11',
    'chicken soup for the soul 2',
    '2 helping of chicken soup'
}


def check(test_dict: dict[str, set[str]]):
    for k, v in test_dict.items():
        assert set(model.get_matching_titles(k)) == v


def test_base():
    test_dict = {
        "life of pi": {'Life of Pi'},
        "life pi": {'Life of Pi'},
        'tell no one': {'Tell No One'},
        'sushi': {'Sushi for Beginners'},
        'crimson petal': {'The Crimson Petal and the White'},
        'crush': {'The Crush'}
    }
    check(test_dict)


def test_multiple():
    test_dict = {
        'memory': {'A Traitor to Memory',
                   'Breath, Eyes, Memory',
                   'False Memory'},
        'harry potter': {'Harry Potter and the Chamber of Secrets (Book 2)',
                         'Harry Potter and the Goblet of Fire (Book 4)',
                         'Harry Potter and the Order of the Phoenix (Book 5)',
                         'Harry Potter and the Prisoner of Azkaban (Book 3)',
                         'Harry Potter and the Sorcerer\'s Stone (Book 1)',
                         'Harry Potter and the Sorcerer\'s Stone (Harry Potter (Paperback))', },
        'walk in the woods': {'A Walk in the Woods: Rediscovering America on the Appalachian Trail',
                              'A Walk in the Woods: Rediscovering America on the Appalachian Trail '
                              '(Official Guides to the Appalachian Trail)'}
    }
    check(test_dict)


def test_numbers():
    test_dict = {
        'harry potter book 5': {'Harry Potter and the Order of the Phoenix (Book 5)'},
        '100 years of solitude': {'One Hundred Years of Solitude',
                                  'One Hundred Years of Solitude (Oprah\'s Book Club)'},
        'battlefield 3000': {'Battlefield Earth: A Saga of the Year 3000'},
        '10 lb penalty': {'10 Lb. Penalty'},
        '1984': {'1984'},
        '16 lighthouse road': {'16 Lighthouse Road'},
        '84 charing cross': {'84 Charing Cross Road'}
    }
    check(test_dict)


def test_ordinal_indicators():
    test_dict = {
        "21 century comedy": {'Vernon God Little: A 21st Century Comedy in the Presence of Death'},
        "2 chance": {'2nd Chance', 'No Second  Chance'},
        '506 regiment': {'Band of Brothers : E Company, 506th Regiment, '
                         '101st Airborne from Normandy to Hitler\'s Eagle\'s Nest'},
        '101 airborne': {'Band of Brothers : E Company, 506th Regiment, '
                         '101st Airborne from Normandy to Hitler\'s Eagle\'s Nest'},
        '1st to die': {'1st to Die: A Novel'},
        '3 degree': {'3rd Degree'}
    }
    check(test_dict)


def test_ordinal_words():
    test_dict = {
        'seventh heaven': {'Seventh Heaven'},
        '7 heaven': {'Seventh Heaven'},
        '7th heaven': {'Seventh Heaven'},
        '10th insight': {'The Tenth Insight : Holding the Vision'},
        'hour 11': {'Eleventh Hour: An FBI Thriller (FBI Thriller (Jove Paperback))'}
    }
    check(test_dict)


def test_roman():
    test_dict = {
        'vampire chronicles 2': {'The Vampire Lestat (Vampire Chronicles, Book II)'},
        'world war 2': {'Farewell to Manzanar: A True Story of Japanese American '
                        'Experience During and  After the World War II Internment'},
        'chicken soup 2': {'A 2nd Helping of Chicken Soup for the Soul (Chicken Soup for the Soul Series (Paper))',
                           'A Second Chicken Soup for the Woman\'s Soul (Chicken Soup for the Soul Series)',
                           'Chicken Soup for the Teenage Soul II (Chicken Soup for the Soul Series)'
                           },
    }
    check(test_dict)


def test_punctuation():
    test_dict = {
        "rich dad poor dad": {'Rich Dad, Poor Dad: What the Rich Teach Their '
                              'Kids About Money--That the Poor and Middle Class Do Not!'},
        "rich dad, poor dad": {'Rich Dad, Poor Dad: What the Rich Teach Their '
                               'Kids About Money--That the Poor and Middle Class Do Not!'},
        "don't sweat the small stuff": {'Don\'t Sweat the Small Stuff and It\'s All Small Stuff '
                                        ': Simple Ways to Keep the Little Things from Taking Over Your Life '
                                        '(Don\'t Sweat the Small Stuff Series)'},
        "angela's ashes": {'Angela\'s Ashes: A Memoir',
                           'Angelas Ashes', 'Angela\'s Ashes (MMP) : A Memoir'},
    }
    check(test_dict)
