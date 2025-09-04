from telegram import InputMediaPhoto, Update
from telegram.ext import Application, MessageHandler, filters, CommandHandler, ContextTypes, ConversationHandler
from model import recommend, get_matching_titles


CHOOSE_TITLE = 0


async def rec_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    partial_title = " ".join(context.args)
    possible_titles = get_matching_titles(partial_title)
    print(possible_titles)
    if not possible_titles:
        await update.message.reply_text("Couldn't find your book")
        return ConversationHandler.END
    elif len(possible_titles) == 1:
        await get_recommendations(update, possible_titles[0])
        return ConversationHandler.END
    else:
        return await get_title(possible_titles, update, context)


async def get_title(possible_titles: list[str],
                    update: Update,
                    context: ContextTypes.DEFAULT_TYPE) -> int:
    context.user_data['possible_titles'] = possible_titles
    message = "I found several possible titles. Please, choose one of the following by sending its number:\n"
    for i, t in enumerate(possible_titles, 1):
        message += f"{i}. {t}\n"
    await update.message.reply_text(message)
    return CHOOSE_TITLE


async def get_recommendations(update: Update, book_title: str) -> None:
    ans = recommend(book_title)
    if ans is None:
        message = "Couldn't generate recommendations for your book"
        await update.message.reply_text(message)
    else:
        message = "I suggest you to read the following:\n"
        for book, _ in ans:
            message += book + "\n"
        try:
            media_group = []
            for book, url in ans:
                media_group.append(InputMediaPhoto(
                    media=url,
                    caption=book
                ))
            await update.message.reply_text(message)
            await update.message.reply_media_group(media=media_group)
        except Exception as e:
            print("Error while sending images:", e)
            await update.message.reply_text("Couldn't load book covers")


async def choose_title(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # TODO error handling
    num = int(update.message.text)
    possible_titles = context.user_data.get('possible_titles')
    if 1 <= num <= len(possible_titles):
        context.user_data.pop('possible_titles')
        await get_recommendations(update, possible_titles[num - 1])
        return ConversationHandler.END
    else:
        await update.message.reply_text(f'Please enter number from 1 to {len(possible_titles)}')
        return CHOOSE_TITLE


async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    pass


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = '''
    Syntax: /recommend <book_title> - Get book recommendations
    '''
    await update.message.reply_text(message)


def main():
    token = ""
    application = Application.builder().token(token).concurrent_updates(False).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(ConversationHandler(
        entry_points=[CommandHandler("recommend", rec_handler)],
        states={
            CHOOSE_TITLE: [MessageHandler(filters.TEXT & ~filters.COMMAND, choose_title)]
        },
        fallbacks=[CommandHandler("cancel", cancel)]
    ))
    print("Bot started!", flush=True)
    application.run_polling()


if __name__ == "__main__":
    main()
