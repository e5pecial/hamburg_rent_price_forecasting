FROM tiangolo/python-machine-learning:python3.7

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . forecaster/.

ENV FLASK_APP forecaster/service.py

CMD [ "flask", "run", "--port", "2282", "--host", "0.0.0.0"]
#CMD ["python", "service.py"]