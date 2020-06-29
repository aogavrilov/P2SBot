import asyncio
import logging
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
import json
from aiogram import Bot, Dispatcher, executor, types
from PIL import Image
import os
import sklearn
import sklearn.preprocessing
from aiogram.types import ChatActions
from asgiref.sync import sync_to_async
from styletr import StyleTransfer
import numpy as np

tgTok = None
deepmxTok = None

import deepmux
from torchvision import transforms

#logging.basicConfig(filename='app.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')
with open("config.json", "r") as readcfg:
    data = json.load(readcfg)
    tgTok = data['tgTok']
    deepmxTok = data['deepmxTok']
tgTok = '1159135693:AAFf_A3pLRsBfdkVVzRYRMbVeVEvzE-RFjQ'
bot = Bot(token=tgTok)
disp = Dispatcher(bot)

styles = set()
cycle = set()
styles_ = set()


class SomeModel:
    def __init__(self, model_name):
        # Создаем модель в DeepMux
        self.model = deepmux.get_model(model_name=model_name, token=deepmxTok)

        self._imagenet_mean = [0.5, 0.5, 0.5]
        self._imagenet_std = [0.5, 0.5, 0.5]

    def __call__(self, image: Image) -> np.ndarray:
        input_batch = self._preprocess_image(image)
        model_outputs = self.model.run(input_batch)
        category_probs = model_outputs[0]
        init_shape = category_probs.shape
        category_probs = sklearn.preprocessing.MinMaxScaler() \
            .fit_transform(np.ravel(category_probs)[:, None]) \
            .reshape(init_shape)
        return category_probs

    def _preprocess_image(self, image) -> np.ndarray:
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=self._imagenet_mean,
                                 std=self._imagenet_std),
        ])
        input_tensor = preprocess(image)
        input_batch = input_tensor.unsqueeze(0)
        return input_batch.numpy()


model = SomeModel("P2S")
model2 = SomeModel("P2A")


@disp.message_handler(commands="start")
async def start_dialog(message: types.Message):
    poll_keyboard = types.ReplyKeyboardMarkup(resize_keyboard=True)
    poll_keyboard.add(types.KeyboardButton("О приложении"))
    poll_keyboard.add(types.KeyboardButton("Style Transfer"))
    poll_keyboard.add(types.KeyboardButton("Преобразовать фото в скетч(криво увы)"))
    # poll_keyboard.add(types.KeyboardButton("Поддержать"))
    poll_keyboard.add(types.KeyboardButton("Отмена"))
    await message.answer("Выберите действие, которое хотите совершить", reply_markup=poll_keyboard)


@disp.message_handler(commands="log")
async def return_log(message: types.Message):
    user_id = message.from_user.id
    await bot.send_chat_action(user_id, ChatActions.UPLOAD_DOCUMENT)
    await asyncio.sleep(1)  # скачиваем файл и отправляем его пользователю
    TEXT_FILE = open("app.log", "rb")
    await bot.send_document(user_id, TEXT_FILE)


@disp.message_handler()
async def echo(message: types.Message):
    if (message.text == "О приложении"):
        await message.answer("Данное приложение создано как выпускной проект из DLS. "
                             "\nВыберете на клавиатуре, что хотите сделать.\n"
                             "Построение изображений может занять некоторое время, после чего вам придет сообщение "
                             "с результатом\n"
                             )
    elif (message.text == "Преобразовать фото в скетч(криво увы)"):
        if (message.from_user.id in styles_):
            styles_.remove(message.from_user.id)
        if (message.from_user.id in styles):
            styles.remove(message.from_user.id)
        await message.answer("Отправьте ОДНО фото, которое хотите преобразовать в скетч")
        cycle.add(message.from_user.id)

    elif (message.text == "Style Transfer"):
        if (message.from_user.id in cycle):
            cycle.remove(message.from_user.id)
        await message.answer("Отправьте ОДНО фото, на которое хотите наложить стиль")
        styles.add(message.from_user.id)

    elif (message.text == "Отмена"):
        tmp = types.ReplyKeyboardRemove()
        await message.answer(
            "Спасибо, что воспользовались нашим приложением. Вы всегда можете ввести /start и продолжить развлекаться.",
            reply_markup=tmp)
        if (message.from_user.id in styles_):
            styles_.remove(message.from_user.id)
        if (message.from_user.id in styles):
            styles.remove(message.from_user.id)
        if (message.from_user.id in cycle):
            cycle.remove(message.from_user.id)

    else:
        await message.answer("Данная команда неизвестна. Введите /start для отображения меню.")
        print(message.text)

def sendimg(id):
    pass
@disp.message_handler(content_types=["photo"])
async def photo_ed(message: types.Message):
    if (message.from_user.id in cycle):

        await message.photo[-1].download("images/" + str(message.from_user.id) + '.jpg')
        img = Image.open('images/' + str(message.from_user.id) + '.jpg')
        # open("images/" + str(message.from_user.id) + '.jpg', 'rb')
        h, w = img.size
        # print(img.shape)
        t1 = transforms.Compose([
            transforms.Resize((512, 512)),
            #  transforms.ToTensor()
        ])
        t2 = transforms.Compose([
            transforms.Resize((256, 256)),
            #  transforms.ToTensor()
        ])
        img = t1(img)
        img = model(img)

        img = np.transpose(img, (1, 2, 0))
        img = Image.fromarray((img * 255).astype(np.uint8))
        t = transforms.Compose([
            transforms.Resize((w, h)),
            transforms.ToTensor()
        ])
        img = t(img).numpy()
        img = np.transpose(img, (1, 2, 0))
        img = Image.fromarray(np.uint8(img * 255))
        img.save('images/' + str(message.from_user.id) + '.jpg')
        img2 = t2(img)
        img2 = model2(img2)
        img2 = np.transpose(img2, (1, 2, 0))
        img2 = Image.fromarray((img2 * 255).astype(np.uint8))
        t = transforms.Compose([
            transforms.Resize((w, h)),
            transforms.ToTensor()
        ])
        img2 = t(img2).numpy()
        img2 = np.transpose(img2, (1, 2, 0))
        img2 = Image.fromarray(np.uint8(img2 * 255))
        img2.save('images/' + str(message.from_user.id) + '_2.jpg')
        img_ = open('images/' + str(message.from_user.id) + '.jpg', 'rb')
        img_2 = open('images/' + str(message.from_user.id) + '_2.jpg', 'rb')
        media = [types.InputMediaPhoto(img_, "Преобразованное фото"), types.InputMediaPhoto(img_2)]
        await bot.send_media_group(message.from_user.id, media)
        cycle.remove(message.from_user.id)
        os.remove("images/" + str(message.from_user.id) + '.jpg')

    elif (message.from_user.id in styles):
        await message.photo[-1].download("images/" + str(message.from_user.id) + '.jpg')
        styles.remove(message.from_user.id)
        styles_.add(message.from_user.id)
        await message.answer("Принято! Отправь еще фотку стиля, который нужно перенести.")

    elif (message.from_user.id in styles_):
        await message.photo[-1].download("images/" + str(message.from_user.id) + 'style.jpg')
        # img = Image.open('images/' + str(message.from_user.id) + '.jpg')
        await message.answer("Отлично! Осталось подождать 5-10 минут и бот пришлет результат.")
        #async def asynctransf():


        nm = StyleTransfer("images/" + str(message.from_user.id) + '.jpg',
                           "images/" + str(message.from_user.id) + 'style.jpg')
        x = nm.getRes()
        x = Image.fromarray((x.detach().numpy().squeeze(0).transpose((1, 2, 0)) * 255).astype(np.uint8))
        x.save("images/" + str(message.from_user.id) + '.jpg')
        with open('images/' + str(message.from_user.id) + '.jpg', 'rb') as img_:
            await bot.send_photo(message.from_user.id, img_, caption="Преобразованное фото")
        styles_.remove(message.from_user.id)
        os.remove("images/" + str(message.from_user.id) + '.jpg')
        os.remove("images/" + str(message.from_user.id) + 'style.jpg')

        #await asynctransf()
        #pool = ThreadPoolExecutor(max_workers=multiprocessing.cpu_count())
        #loop = asyncio.get_event_loop()
        #loop.run_in_executor(pool, asynctransf)
        #loop.close()

    else:
        await message.answer(
            "Ой, ты не выбрал, что хочешь сделать с картинкой. Введи /start и выбери на клавиатуре действие!")


#


if __name__ == "__main__":
    files = os.listdir('images/')
    for file in files:
        os.remove('images/' + str(file))
    executor.start_polling(disp, skip_updates=True)
