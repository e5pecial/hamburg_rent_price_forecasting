FROM python:3-alpine3.9

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV FLASK_APP service.py

CMD [ "flask", "run", "--port", "80", "--host", "0.0.0.0"]