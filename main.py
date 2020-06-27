import logging
import json
from aiogram import Bot, Dispatcher, executor, types
from PIL import Image
import os

tgTok = None
deepmxTok = None
import numpy as np
import sklearn
import sklearn.preprocessing
import deepmux
#import torch
#from torchvision import transforms
logging.basicConfig(filename='app.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')
with open("config.json", "r") as readcfg:
    data = json.load(readcfg)
    tgTok = data['tgTok']
    deepmxTok = data['deepmxTok']
tgTok = '1139411645:AAFLTOA3AoWkoLjdUvzXtRZpDGx3mOJPTe8'
bot = Bot(token=tgTok)
disp = Dispatcher(bot)


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


@disp.message_handler(commands="start")
async def start_dialog(message: types.Message):
    poll_keyboard = types.ReplyKeyboardMarkup(resize_keyboard=True)
    poll_keyboard.add(types.KeyboardButton("О приложении"))
    poll_keyboard.add(types.KeyboardButton("Преобразовать фото в скетч"))
    # poll_keyboard.add(types.KeyboardButton("Поддержать"))
    poll_keyboard.add(types.KeyboardButton("Отмена"))
    await message.answer("Выберите действие, которое хотите совершить", reply_markup=poll_keyboard)


@disp.message_handler()
async def echo(message: types.Message):
    if (message.text == "О приложении"):
        await message.answer("Данное приложение делает из фото скетч, применяя технологию cycleGAN. \nКогда вы нажмете "
                             "кнопку ''Преобразовать фото в скетч'' вы получите инструкции для этого преобразования\n"
                             "Построение изображения может занять пару секунд. Когда приложение завершит свою работу, "
                             "оно пришлет вам результат. \nДанное приложение создано в рамках выпускного проекта DLS.\n"
                             "Пример работы программы вы можете видеть ниже:")
    elif (message.text == "Преобразовать фото в скетч"):
        await message.answer("Отправьте ОДНО фото, которое хотите преобразовать в скетч")

    elif (message.text == "Отмена"):
        tmp = types.ReplyKeyboardRemove()
        await message.answer(
            "Спасибо, что воспользовались нашим приложением. Вы всегда можете ввести /start и продолжить развлекаться.",
            reply_markup=tmp)
    else:
        await message.answer("Данная команда неизвестна. Введите /start для отображения меню.")
        print(message.text)


@disp.message_handler(content_types=["photo"])
async def photo_ed(message: types.Message):
    model = SomeModel("P2S")
    await message.photo[-1].download("images/" + str(message.from_user.id) + '.jpg')
    img = open("images/" + str(message.from_user.id) + '.jpg', 'rb')
    h, w = img.size
    # print(img.shape)
    img = model(img)
    img = np.transpose(img, (1, 2, 0))
    img = Image.fromarray((img * 255).astype(np.uint8))
    t = transforms.Compose([
        transforms.Resize((h, w)),
        transforms.ToTensor()
    ])
    img = t(img)
    img = np.transpose(img, (1, 2, 0))
    await bot.send_photo(message.from_user.id, img, caption="Преобразованное фото")


if __name__ == "__main__":
    executor.start_polling(disp, skip_updates=True)
