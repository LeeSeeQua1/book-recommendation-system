import logging
from telegram import InputMediaPhoto, Update
from telegram.ext import Application, MessageHandler, filters, CommandHandler, ContextTypes, ConversationHandler
from model import CFModel
import pandas as pd
logger = logging.getLogger(__name__)

CHOOSE_TITLE, WAIT_FOR_TITLE = 0, 1
MAX_TITLES = 10

data_path = './data/'
books = pd.read_csv(data_path + 'Books.csv')
ratings = pd.read_csv(data_path + 'Ratings.csv')
users = pd.read_csv(data_path + 'Users.csv')

model = CFModel(books, users)
model.fit(ratings)


async def rec_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    title = " ".join(context.args)
    return await recommend(title, update, context)


async def recommend(title: str, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    if title == "":
        await update.message.reply_text("Title can't be empty, please try again")
        return ConversationHandler.END
    try:
        possible_titles = model.get_matching_titles(title)
    except Exception as e:
        logger.warning(f"Error while getting matching titles for {title}:\n{str(e)}")
        await update.message.reply_text("Error while getting matching titles")
        return ConversationHandler.END
    if not possible_titles:
        await update.message.reply_text("Couldn't find your book")
        return ConversationHandler.END
    elif len(possible_titles) == 1:
        await get_recommendations(update, possible_titles[0])
        return ConversationHandler.END
    elif len(possible_titles) > MAX_TITLES:
        await update.message.reply_text("Too many matching titles. Please enter more specific title")
        return WAIT_FOR_TITLE
    else:
        return await get_title(possible_titles, update, context)


async def get_title(possible_titles: list[str],
                    update: Update,
                    context: ContextTypes.DEFAULT_TYPE) -> int:
    context.user_data['possible_titles'] = possible_titles
    message = "I found several possible titles. Please choose one of the following by sending its number:\n"
    for i, t in enumerate(possible_titles, 1):
        message += f"{i}. {t}\n"
    await update.message.reply_text(message)
    return CHOOSE_TITLE


async def get_recommendations(update: Update, book_title: str) -> None:
    try:
        recommended = model.recommend(book_title)
    except Exception as e:
        logger.warning(f"Error while generating recommendations for {book_title}:\n{str(e)}")
        await update.message.reply_text("Error while generating recommendations")
        return
    if recommended is None:
        await update.message.reply_text("Couldn't generate recommendations for your book")
    else:
        message = f"My recommendations for {book_title}:\n\n"
        for book in recommended:
            message += book + "\n"
        await update.message.reply_text(message)
        try:
            pictures = model.get_pictures(recommended)
            media_group = []
            for book, url in zip(recommended, pictures):
                media_group.append(InputMediaPhoto(
                    media=url,
                    caption=book
                ))
            await update.message.reply_media_group(media=media_group)
        except Exception as e:
            logger.warning(f"Error while loading covers for {recommended}:\n{str(e)}")
            await update.message.reply_text("Couldn't load book covers")


async def choose_title(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    possible_titles = context.user_data.get('possible_titles')
    try:
        num = int(update.message.text)
    except ValueError:
        logger.warning("User has entered incorrect number while choosing title")
        await update.message.reply_text(f'Please enter number from 1 to {len(possible_titles)}')
        return CHOOSE_TITLE
    if 1 <= num <= len(possible_titles):
        context.user_data.pop('possible_titles')
        await get_recommendations(update, possible_titles[num - 1])
        return ConversationHandler.END
    else:
        await update.message.reply_text(f'Please enter number from 1 to {len(possible_titles)}')
        return CHOOSE_TITLE


async def choose_again(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    possible_titles = context.user_data.get('possible_titles')
    await update.message.reply_text(f'Please enter number from 1 to {len(possible_titles)}')
    return CHOOSE_TITLE


async def wait_for_title(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    title = update.message.text
    return await recommend(title, update, context)


async def wait_again(update: Update, _context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text("Please enter title")
    return WAIT_FOR_TITLE


async def cancel(update: Update, _context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text("You've cancelled current conversation. "
                                    "Please start a new one by writing"
                                    "/recommend <book title>")
    return ConversationHandler.END


async def start(update: Update, _context: ContextTypes.DEFAULT_TYPE) -> None:
    message = '''
    Syntax: 
        /recommend <book_title> - get book recommendations
        /cancel - cancel current request
    '''
    await update.message.reply_text(message)


async def error(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.warning("Update %s caused error %s", update, context.error)


def main():
    token = ""

    logging.basicConfig(filename='bot.log', level=logging.WARNING)

    application = Application.builder().token(token).concurrent_updates(False).build()
    application.add_handler(CommandHandler("start", start))
    cancel_handler = CommandHandler("cancel", cancel)
    application.add_handler(ConversationHandler(
        entry_points=[CommandHandler("recommend", rec_handler)],
        states={
            CHOOSE_TITLE: [MessageHandler(~filters.UpdateType.EDITED_MESSAGE
                                          & filters.TEXT
                                          & ~filters.COMMAND, choose_title),
                           MessageHandler(~filters.UpdateType.EDITED_MESSAGE, choose_again)],
            WAIT_FOR_TITLE: [MessageHandler(~filters.UpdateType.EDITED_MESSAGE
                                            & filters.TEXT, wait_for_title),
                             MessageHandler(~filters.UpdateType.EDITED_MESSAGE, wait_again)]
        },
        fallbacks=[cancel_handler]
    ))
    application.add_handler(CommandHandler("cancel", cancel))
    application.add_handler(MessageHandler(~filters.COMMAND, start))
    print("Bot started!", flush=True)
    application.run_polling()


if __name__ == "__main__":
    main()
