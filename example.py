import requests
import pandas as pd


def send_test_example():
    data = pd.read_csv('ml_eng_ay_data.csv')

    # You can send as pd.DataFrame converted to_dict():
    example = data.loc[[439, 3658]]
    example.drop(['rent_total'], axis=1, inplace=True)

    message_to_send = example.to_dict()
    # Or send just as dict

    resp = requests.post("http://localhost:2282/predict",
                         json=message_to_send)

    print(resp.status_code)
    print(resp.text)


if __name__ == '__main__':
    send_test_example()
