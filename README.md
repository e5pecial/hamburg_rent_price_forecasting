## FORECASTER FOR PRICES IN HAMBURG

In this project you can find some things for forecasting prices.

* [Jupyter-notebook](aboutyou_eda.ipynb) with some EDA, data processing, training some models;
* Service with API for predict prices by features; [forecaster](forecaster)
* Example script with instruction for using that.  [example.py](example.py)

### Some assumptions:
In the dataset we have feature `rent_base` -- I think it's a leak feature :)
But I am really not sure will be it in the test dataset -- in the notebook I
try different models with this feature and without that.

I don't know quality of Seasonal Naive model -- and also 
I'm not sure that my solution better or not. 

Also, I could not formulate this problem in terms of time series,
 so I solved it as a regression problem. 
 
Data was sometimes noisy (flat_type with mistake, huge differences between some `base` and `total_rent` prices. And I think we have only one year --
 also it's not enough for good validation.
 
And (imho) this this quality is quite low -- 
but I would like to know how to solve this problem with the best results.
 
## Simple usage

Start service:
`docker-compose up --build`

It will be started in `http://0.0.0.0:2282`

After that you can trigger API method `/preidct` and send to it
test pandas Dataframe in `to_dict()` view. (see [example.py](example.py) 
for more details)

```python
import requests


message_to_send = {'cnt_rooms': 1,
                   'flat_area': 37.0,
                   'rent_base': 462.0,
                   'flat_type': 'apartment',
                   'flat_interior_quality': 'average',
                   'flat_condition': 'good',
                   'flat_age': '60+',
                   'has_elevator': 'f',
                   'has_balcony': 'f',
                   'has_garden': 'f',
                   'has_kitchen': 't',
                   'has_guesttoilet': 'f',
                   'geo_city_part': 'wandsbek',
                   'date': '2018-02-2',
                   'flat_thermal_characteristic': 'None',
                   'geo_city': 'hamburg',
                   'weekday': 2,
                   'weekofyear': 39,
                   'month': 9,
                   'dayofyear': 269}

resp = requests.post("http://localhost:2282/predict",
                     json=message_to_send)

print(resp.status_code)
print(resp.text)
```
#### Answer:
```python
200
{"pred":{"0":566.0894462133544}}
```

## Ideas and TODO:
I have some ideas how to make it better and what should by tried:
* Target Encoding for categorial features
* Quantile Transformation for prices
* Try RNN models
* Try to drop more outliers & use ordinal features too
* Also should add tests to service, refactor, change some structure, maybe add webUI
