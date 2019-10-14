#!/usr/bin/env python
from flask import Flask
from flask import request
from flask import jsonify
from werkzeug.exceptions import BadRequest

from forecaster.exceptions import BasicValidationError

app = Flask(__name__)


@app.route('/')
def root():
    message = {
        "hi": "this is not the endpoint you're looking for, "
              "maybe try /predict"
    }
    return jsonify(message), 200


@app.route('/predict', methods=['POST'])
def predict():
    content = request.get_json(force=True)  # eh, why require content-type?
    required_keys = ['date', 'cnt_rooms', 'flat_area', 'rent_base',
                     'rent_total',
                     'flat_type', 'flat_interior_quality', 'flat_condition',
                     'flat_age',
                     'flat_thermal_characteristic', 'has_elevator',
                     'has_balcony',
                     'has_garden', 'has_kitchen', 'has_guesttoilet',
                     'geo_city',
                     'geo_city_part']

    if not all([key in content.keys() for key in required_keys]):
        raise BasicValidationError("Data for predict missing a required key")

    response_body = ''

    return jsonify(response_body)


@app.errorhandler(BadRequest)
def error_me_out(error):
    yer_error = {
        'error': 'Failed some bad request or something. ' + str(error),
        'status': '400'
    }
    return jsonify(yer_error), 400


@app.errorhandler(404)
def not_found(error):
    yer_error = {
        'error': 'not found',
        'status': 404
    }
    return jsonify(yer_error), 404


@app.errorhandler(Exception)
def error_all_the_things(error):
    yer_error = {
        'exception': str(error),
        'status': 500
    }
    return jsonify(yer_error), 500