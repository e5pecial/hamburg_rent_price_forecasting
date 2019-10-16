FROM python:3.7-alpine3.9

WORKDIR /usr/src/app

RUN apk add --no-cache python3-dev libstdc++ && \
    apk add --no-cache g++ && \
    ln -s /usr/include/locale.h /usr/include/xlocale.h

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV FLASK_APP service.py

CMD [ "flask", "run", "--port", "2282", "--host", "0.0.0.0"]