FROM tiangolo/python-machine-learning:python3.7

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV FLASK_APP service.py

CMD [ "flask", "run", "--port", "2282", "--host", "0.0.0.0"]