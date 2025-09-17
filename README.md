## Book Recommendation Telegram Bot

### Overview

This is a Telegram bot that recommends books based on a 
collaborative filtering algorithm. When users provide a book title, the bot 
suggests similar books based on user rating patterns.

### Core components

+ **Exploratory data analysis** (EDA.ipynb);
+ **Telegram Bot Handler** (main.py). Handles user interactions using python-telegram-bot module;
+ **Recommendation Engine** (model.py). Implements a collaborative filtering
model using cosine similarity;
+ **Title matching system** (model.py, title_matching_test.py). Adds flexible matching for
partial titles.

### Setup

1. Clone the repository
    
    ```git clone https://github.com/LeeSeeQua1/book-recommendation-system.git```;

2. Install dependencies 

   ```pip install python-telegram-bot pandas scikit-learn numpy```;
3. Download data, place it to the data folder;
4. Obtain a bot token from @BotFather in Telegram and add it to the token variable;
in main
5. Run the bot by calling
    ```python main.py```

**Data Source**:
https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset