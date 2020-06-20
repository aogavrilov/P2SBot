import logging
import json
from aiogram import Bot, Dispatcher, executor, types
from PIL import Image
import os
data = None
with open("config.json", "r") as readcfg:
    data = json.load(readcfg)


bot = Bot(token=data)
disp = Dispatcher(bot)


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
    if(message.text == "О приложении"):
        await message.answer("Данное приложение делает из фото скетч, применяя технологию cycleGAN. \nКогда вы нажмете "
                             "кнопку ''Преобразовать фото в скетч'' вы получите инструкции для этого преобразования\n"
                             "Построение изображения может занять пару секунд. Когда приложение завершит свою работу, "
                             "оно пришлет вам результат. \nДанное приложение создано в рамках выпускного проекта DLS.\n"
                             "Пример работы программы вы можете видеть ниже:")
    elif(message.text == "Преобразовать фото в скетч"):
        await message.answer("Отправьте ОДНО фото, которое хотите преобразовать в скетч")

    elif(message.text == "Отмена"):
        tmp = types.ReplyKeyboardRemove()
        await message.answer("Спасибо, что воспользовались нашим приложением. Вы всегда можете ввести \start и продолжить развлекаться.", reply_markup=tmp)
    else:
        await message.answer("Данная команда неизвестна. Введите /start для отображения меню.")
        print(message.text)

@disp.message_handler(content_types=["photo"])
async def photo_ed(message: types.Message):
    await message.photo[-1].download("images/" + str(message.from_user.id) + '.jpg')
    img = open("images/" + str(message.from_user.id) + '.jpg', 'rb')
    await bot.send_photo(message.from_user.id, img, caption="Преобразованное фото")

if __name__ == "__main__":

    executor.start_polling(disp, skip_updates=True)