FROM python:3
WORKDIR /P2SBot/
COPY . ./
RUN python3 -m pip install -r /P2SBot/requirements.txt
CMD ["python3", "main.py"]