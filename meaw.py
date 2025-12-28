import os
import sys
import re
import json
import logging
import requests
import gdown
import numpy as np
from flask import Flask, request
from PIL import Image, ImageOps
import telebot
from tensorflow.keras.models import load_model
import tensorflow as tf
from telebot import util

logging.basicConfig(level=logging.INFO)
API_TOKEN = os.getenv('API_TOKEN')
if not API_TOKEN:
    sys.exit('ошибка API_TOKEN не задан в переменных окружения')

bot = telebot.TeleBot(API_TOKEN)
app = Flask(__name__)


MAX_LEN = 4096


def convert_markdown_to_html(text: str) -> str:
    text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'\*(.*?)\*', r'<i>\1</i>', text)
    text = re.sub(r'__(.*?)__', r'<u>\1</u>', text)
    text = re.sub(r'~~(.*?)~~', r'<s>\1</s>', text)
    text = re.sub(r'`([^`]*)`', r'<code>\1</code>', text)
    text = re.sub(r'\[(.*?)\]\((.*?)\)', r'<a href="\2">\1</a>', text)
    return text


def send_long_message(chat_id, text, parse_mode='HTML'):
    try:
        safe_text = convert_markdown_to_html(text or "")
        for part in util.smart_split(safe_text, MAX_LEN):
            bot.send_message(chat_id, part, parse_mode=parse_mode)
    except Exception as e:
        logging.error(f"Ошибка: {e}")


history_file = "history.json"
history = {}


if os.path.exists(history_file):
    try:
        with open(history_file, 'r', encoding='utf-8') as f:
            history = json.load(f)
    except Exception:
        history = {}


def save_history():
    try:
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False,indent=2)
    except Exception as e:
        logging.error('Ошибка сохранения истории: %s', e)



AI_KEY = os.getenv('AI_KEY')
if not AI_KEY:
    logging.warning('API_KEY не задан: чат-модель будет недоступна')



@app.route("/")
def index():
    return 'бот запущен'

@app.route(f'/{API_TOKEN}',methods= ['POST'])
def webhook():
    try:
        json_str = request.get.data(as_text= True)
        update = telebot.types.Update.be_json(json_str)
        if update:
            bot.process_new_updates([update])
    except Exception as e:
        app.logger.exception('unknown error', str(e))
    return '', 200

def load_photo(message, name):
    photo = message.photo[-1]
    file_info = bot.get_file(photo.file_id)
    downloaded_file = bot.download_file(file_info.file_path)
    save_path = name
    with open(save_path, 'wb') as new_file:
        new_file.write(downloaded_file)

def chat(user_id, text):
    try:
        if str(user_id) not in history:
            history[str(user_id)] = [
                {"role": "system", "content": "Ты — дружелюбный помощник."}
            ]

        history[str(user_id)].append({"role": "user", "content": text})

        if len(history[str(user_id)]) > 16:
            history[str(user_id)] = [history[str(user_id)][0]] + history[str(user_id)][-15:]

        url = "https://api.intelligence.io.solutions/api/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {AI_KEY}" if AI_KEY else ""
        }
        data = {
            "model": "deepseek-ai/DeepSeek-R1-0528",
            "messages": history[str(user_id)]
        }

        response = requests.post(url, headers=headers, json=data, timeout=300)
        data = response.json()

        if isinstance(data, dict) and data.get('choices'):
            content = data['choices'][0]['message']['content']
            history[str(user_id)].append({"role": "assistant", "content": content})

            if len(history[str(user_id)]) > 16:
                history[str(user_id)] = [history[str(user_id)][0]] + history[str(user_id)][-15:]

            save_history()

            if '</think>' in content:
                return content.split('</think>', 1)[1]
            return content
        else:
            logging.error(f"Ошибка API: {json.dumps(data, ensure_ascii=False)}")
    except Exception as e:
        logging.error(f"Ошибка при запросе: {e}")
        send_long_message(user_id, f"Ошибка при запросе: {e}, повторите попытку позже")


def load_db():
    try:
        with open('db.jason', 'r', encoding= 'utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def save_db(data):
    with open('db.jason', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii= False, indent=2)

db = load_db()


TFLITE_PATH = "cat_dog_model.tflite"
TFLITE_URL = os.getenv("CAT_DOGS_TFLITE_URL")
_interpreter = None
_input_details = None
_output_details = None

def ensure_catdog_tflite():
    global _interpreter, _input_details, _output_details
    if _interpreter is None:
        if not os.path.exists(TFLITE_PATH):
            if not TFLITE_URL:
                raise RuntimeError("CAT_DOGS_TFLITE_URL не задан, а локальной модели нет")
            gdown.download(TFLITE_URL, TFLITE_PATH, quiet=False)

        _interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
        _interpreter.allocate_tensors()
        _input_details = _interpreter.get_input_details()
        _output_details = _interpreter.get_output_details()
    return _interpreter, _input_details, _output_details


def cat_dog(photo):
    try:
        interpreter, input_details, output_details = ensure_catdog_tflite()

        image = Image.open(photo).convert("RGB")
        image = ImageOps.fit(image, (150, 150), method=Image.Resampling.LANCZOS)
        x = (np.asarray(image).astype(np.float32) / 255.0)[None, ...]

        interpreter.set_tensor(input_details[0]['index'], x)
        interpreter.invoke()
        pred = interpreter.get_tensor(output_details[0]['index'])

        if pred.ndim == 2 and pred.shape[1] == 1:
            confidence = float(pred[0][0])
        elif pred.ndim == 1:
            confidence = float(pred[0])
        else:
            confidence = float(np.ravel(pred)[0])

        return (f"На изображении собака (точность: {confidence:.2f})"
                if confidence >= 0.5 else
                f"На изображении кот (точность: {1 - confidence:.2f})")
    except Exception as e:
        return f"Ошибка при распознавании: {e}"


def ident_number(message):
    load_photo(message, "Number.jpg")
    answer_number = number_identification("Number.jpg")
    bot.send_message(message.chat.id, f"Цифра на фото: {answer_number}")


def ident_cat_dog(message):
    load_photo(message, "cat_dog.jpg")
    answer = cat_dog("cat_dog.jpg")
    bot.send_message(message.chat.id, answer)


MNIST_PATH = "mnist_model.h5"
_mnist_model = None

def ensure_mnist():
    global _mnist_model
    if _mnist_model is None:
        if not os.path.exists(MNIST_PATH):
            raise RuntimeError("MNIST модель не найдена: mnist_model.h5")
        _mnist_model = load_model(MNIST_PATH, compile=False)
    return _mnist_model

def number_identification(photo):
    try:
        model = ensure_mnist()
        image = Image.open(photo).convert("L")
        image = ImageOps.invert(image)
        image = ImageOps.fit(image, (28, 28), method=Image.Resampling.LANCZOS)
        x = (np.asarray(image).astype(np.float32) / 255.0).reshape(1, 28, 28, 1)
        pred = model.predict(x, verbose=0)
        return str(int(np.argmax(pred)))
    except Exception as e:
        return f"Ошибка распознавания цифры: {e}"

@bot.message_handler(commands=['start'])
def start(message):
    user_id = str(message.from_user.id)

    if user_id not in db:
        db[user_id] = {'name': {}, "age": {}, 'money': '100000'}
        save_db(db)
        bot.send_message(message.chat.id, 'hello, whats ur name')
        return


    keyboardReply = telebot.types.KeyboardButton('помощь')

    helpButton = telebot.types.KeyboardButton('помощь')
    infoButton = telebot.types.KeyboardButton('информация')
    aboutButton = telebot.types.KeyboardButton('о боте')
    slotMachine = telebot.types.KeyboardButton('игровой автомат')
    diceButton = telebot.types.KeyboardButton('игра в кубик')



    keyboardReply.add(helpButton, infoButton, aboutButton, slotMachine, diceButton)
    bot.send_message(message.chat.id, 'hello bot-world', reply_markup=keyboardReply)

bot.message_handler(commands=['help'])

def help(message):
    bot.send.message(message.chat.id, 'информация о боте')

def info(message):
    bot.send_message(message.chat.id, 'info')

@bot.message_handler(content_types=['text'])
def text_event(message):
    try:
        text = message.text
        user_id = str(message.from_user.id)

        if 'awaiting_name' in db.get(user_id, {'state'}):
            name = message.text.strip()
            db[user_id]['name'] = name
            db[user_id]['name'] = None
            save_db(db)
            bot.send_message(message.chat.id, 'nice to meet u', [name])
            start(message)
            return



        if message.text == 'помощь':
            pass
        elif message.text == 'info':
            print('этот бот простой. тут ты можешь просто быть собой')
        elif message.text == 'привет':
            print('привет! чем я могу помочь?')
        elif message.text == 'игровой автомат':
            if db[user_id]['money'] == 1000:
                value = bot.send_dice(message.chat.id, emoji='🎲').dice.value

                if value in (1, 22, 43):
                    db[user_id]['money'] += 2000
                    bot.send_message(message.chat.id, 'u win, ur balance is:', db[user_id]['money'])
                elif value in (16, 32, 48):
                    db[user_id]['money'] += 4000
                    bot.send_message(message.chat.id, 'u win, ur balance is:', db[user_id]['money'])
                elif value == 64:
                    db[user_id]['money'] += 1000
                    bot.send_message(message.chat.id, 'u win, ur balance is:', db[user_id]['money'])
                else:
                    bot.send_message(message.chat.id, 'lose')
        elif message.text == 'игра в кубики':
            inlineKeyboard = telebot.types.InlineKeyboardButton


            btn1 = telebot.typesInlineKeyboardButton('1', callback_data='1')
            btn2 = telebot.typesInlineKeyboardButton('2', callback_data='2')
            btn3 = telebot.typesInlineKeyboardButton('3', callback_data='3')
            btn4 = telebot.typesInlineKeyboardButton('4', callback_data='4')
            btn5 = telebot.typesInlineKeyboardButton('5', callback_data='5')
            btn6 = telebot.typesInlineKeyboardButton('6', callback_data='6')
            inlineKeyboard.add(btn1, btn2, btn3, btn4, btn5, btn6)

            bot.send_message(message.chat.id, 'угадай число', reply_markup=inlineKeyboard)
        elif text == 'Распознавание цифр':
            send1 = bot.send.message(message.chat.id, 'Загрузите изображение цифры')
            bot.register_next_step_handler(send1, ident_number)
        elif text == 'Распознавание цифр':
            send2 = bot.send.message(message.chat.id, 'Загрузите изображение кошки или собаки')
            bot.register_next_step_handler(send2, ident_number)
        else:
            msg = bot.send_message(message.chat.id, 'думаю над ответом')
            try:
                answer = chat(message.chat.id, message.text)
                send_long_message(message.chat.id, answer)
            finally:
                try:
                    bot.delete_message(message.chat.id, msg.message_id)
                except Exception:
                    pass
    except Exception as e:
        bot.send_message(message.chat.id, f"Ошибка: {str(e)}")

@bot.callback_query_handler(func=lambda call: call.data in('1', '2', '3', '4', '5','6'))
def dice_callback(call):
    value = bot.send.dice(call.message.chat.id, emoji='🎲').dice.value
    if str(value) == call.data:
        bot.send_message(call.message.chat.id,'win')
    else:
        bot.send_message(call.message.chat.id, 'lose')


if __name__ == '__main__':
    server_url = os.getenv('RENDER_EXTERNAL_URL')
    if server_url and API_TOKEN:
        webhook_url = f"{server_url.rstrip('/')}/{API_TOKEN}"


        try:
            r = requests.get(f"https://api.telegram.org/bot{API_TOKEN}/setWebhook",
                             params={"url": webhook_url}, timeout=10)
            logging.info(f"векбук установлен,: {r.text}")
        except Exception:
            logging.exception('ошибка при установлении webhook')
        port = int(os.getenv("PORT", 10000))
        logging.info(f'запуск на порте {port}')
        app.run(host='0.0.0.0', port=port)
    else:
        logging.info('запуск бота в режиме polling')
        bot.remove_webhook()
        bot.infinity_polling(timeout=60)
