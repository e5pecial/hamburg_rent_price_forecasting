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
 
## Simple usage

Start service:
`docker-compose up --build`

It will be started in `http://0.0.0.0:2282`

After that you can trigger API method `/preidct` and send to it
test pandas Dataframe in `to_dict()` view. (see [example.py](example.py) 
for more details)