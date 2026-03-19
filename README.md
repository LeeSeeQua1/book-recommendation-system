# Book Recommendation Telegram Bot

A Telegram bot that recommends books using collaborative filtering.  
Built with Python, pandas, scikit-learn, and dockerized for easy deployment.

## Features
- Search books by partial title
- Collaborative filtering recommendations (cosine similarity)
- Docker support for local dev & production

## Quick Start

### Required Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `BOT_TOKEN` | Telegram bot token from @BotFather | `123456:AAHdqTcv...` |


### Option A: Run with Docker
```bash
# 1. Clone repo & prepare config
git clone https://github.com/LeeSeeQua1/book-recommendation-system.git
cd book-recommendation-system

# 2. Edit .env with your BOT_TOKEN
echo "BOT_TOKEN={BOT_TOKEN}" > .env

# 3. Build and start
docker compose build
docker compose up -d

# 4. View logs
docker compose logs -f
```

### Option B: Run Locally

```bash
# 1. Clone repo
git clone https://github.com/LeeSeeQua1/book-recommendation-system.git
cd book-recommendation-system

# 2. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Edit .env with your BOT_TOKEN
echo "BOT_TOKEN={BOT_TOKEN}" > .env

# Download data from https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset
# Place CSV files in ./data/

# 5. Run the bot
python main.py
```

## Data source
[Book Recommendation Dataset](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset?spm=a2ty_o01.29997173.0.0.507d5171LDVM0s)
