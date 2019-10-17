import json
import requests
import pandas as pd


def send_test_example():
    data = pd.read_csv('ml_eng_ay_data.csv')

    example = data.loc[0:200]

    print(example.to_dict())
    resp = requests.post("http://localhost:2282/predict",
                         json=example.to_dict())

    print(resp.status_code)
    print(resp.text)


if __name__ == '__main__':
    send_test_example()
