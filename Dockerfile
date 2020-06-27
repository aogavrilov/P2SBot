FROM python:3
WORKDIR /telegrambotp2s/
ADD *.py /telegrambotp2s/
ADD app.log /telegrambotp2s/
ADD /images/ /telegrambotp2s/
ADD requirements.txt /telegrambotp2s/
ADD config.json /telegrambotp2s/
RUN python3 -m pip install -r /telegrambotp2s/requirements.txt
CMD ["python3", "main.py"]