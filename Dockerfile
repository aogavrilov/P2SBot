FROM python:3
WORKDIR /P2SBot/
ADD *.py /P2SBot/
ADD /images /P2SBot/
ADD requirements.txt /P2SBot/
ADD config.json /P2SBot/
RUN python3 -m pip install -r /P2SBot/requirements.txt
CMD ["python3", "main.py"]