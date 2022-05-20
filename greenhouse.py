import time
import datetime
import board
import busio
import adafruit_am2320
import Adafruit_DHT
import adafruit_bh1750
import adafruit_ads1x15.ads1115 as ADS
from adafruit_ads1x15.analog_in import AnalogIn
import RPi.GPIO as GPIO
import csv

import tensorflow as tf
import pandas as pd
import numpy as np

#############################################################################################
DHTSensor = Adafruit_DHT.DHT11
BCM_DHT11 = 16

i2c = board.I2C()

sensor_am2320 = adafruit_am2320.AM2320(i2c)

sensor_bh1750 = adafruit_bh1750.BH1750(i2c)

ads = ADS.ADS1115(i2c)
chan0 = AnalogIn(ads, ADS.P0)
chan1 = AnalogIn(ads, ADS.P1)
chan2 = AnalogIn(ads, ADS.P2)
chan3 = AnalogIn(ads, ADS.P3)
DRY_VOLTAGE = 17000
WATER_VOLTAGE = 10000

BCM_HOTTER = 17
BCM_COOLER = 27
BCM_VALVE = 22
GPIO.setmode(GPIO.BCM)  # GPIO Numbers instead of board numbers
GPIO.setup(BCM_HOTTER, GPIO.OUT)
GPIO.setup(BCM_COOLER, GPIO.OUT)
GPIO.setup(BCM_VALVE, GPIO.OUT)
GPIO.output(BCM_VALVE, 1)  # to prevent short moment of opening caused by setup

AIR_T_LOW = 20.0
AIR_T_OPT = 25.0
AIR_T_HIGH = 30.0

AIR_H_LOW = 35.0
AIR_H_OPT = 55.0
AIR_H_HIGH = 75.0

SOIL_H_LOW = 30.0
SOIL_H_OPT = 60.0
SOIL_H_HIGH = 80.0

ON = 1
OFF = 0

STEP_TIME = 300

HOTTER_TIME = 93
COOLER_TIME = 200
VALVE_TIME = 3

HISTORY_SIZE = 224
FUTURE_TARGET = 1

FEATURES = ['temp_out', 'hum_out', 'temp_in', 'hum_in', 'lum', 'hotter', 'cooler']

MEAN_FROM_TRAINING = {'temp_out': 17.24900,
                      'hum_out': 25.04000,
                      'temp_in': 26.918800000000005,
                      'hum_in': 33.72485,
                      'lum': 10464.31000,
                      'hotter': 0.30900,
                      'cooler': 0.51200}

STD_FROM_TRAINING = {'temp_out': 9.512439,
                     'hum_out': 22.277114,
                     'temp_in': 7.731576129388101,
                     'hum_in': 12.648902114499986,
                     'lum': 11139.511054,
                     'hotter': 0.462312,
                     'cooler': 0.500106}

model_air_t = tf.keras.models.load_model('model_temp_in')
model_air_h = tf.keras.models.load_model('model_hum_in')


def manipulate_actuators_usual(_inside_air_temperature, _inside_air_humidity,
                               _outside_air_temperature, _outside_air_humidity,
                               _soil_humidity,
                               _illuminance):
    _hotter = OFF
    _cooler = OFF
    _valve = OFF

    if _soil_humidity < SOIL_H_LOW:
        _valve = ON
    if _soil_humidity > SOIL_H_HIGH:
        #         hotter = ON
        if _outside_air_humidity < _inside_air_humidity:
            _cooler = ON

    if _inside_air_temperature > AIR_T_HIGH:
        if _outside_air_temperature < _inside_air_temperature:
            _cooler = ON
        _hotter = OFF
    if _inside_air_temperature < AIR_T_LOW:
        _hotter = ON
        _cooler = OFF
        if _outside_air_temperature > _inside_air_temperature:
            _cooler = ON

    if _inside_air_humidity < AIR_H_LOW:
        if _outside_air_humidity > _inside_air_humidity:
            _cooler = ON
    if _inside_air_humidity > AIR_H_HIGH:
        #         if (outside_air_humidity < inside_air_humidity):
        #             cooler = ON
        _hotter = ON

    return _hotter, _cooler, _valve


def create_slices(dataset, target):
    data = []
    labels = []

    for i in range(0, HISTORY_SIZE * 2 + 10):
        indices = range(i-HISTORY_SIZE, i)
        data.append(dataset[indices])

        labels.append(target[i+FUTURE_TARGET])

    return np.array(data), np.array(labels)


def normalize(data):
    for feature in FEATURES:
        data[feature] = (data[feature] - MEAN_FROM_TRAINING[feature]) / STD_FROM_TRAINING[feature]

    return data


def predict_parameter(parameter_idx, model, history):
    x, y = create_slices(history, history[:, parameter_idx])
    history_set = tf.data.Dataset.from_tensor_slices((x, y))
    history_set = history_set.batch(32)
    predicted = None
    for x, y in history_set.take(1):
        predict = model.predict(x)
        predicted = predict[0][0]

    return predicted


def hampel(vals_orig):
    vals = vals_orig.copy()
    difference = np.abs(vals.median()-vals)
    median_abs_deviation = difference.median()
    threshold = 3 * median_abs_deviation
    outlier_idx = difference > threshold
    vals[outlier_idx] = np.nan
    return(vals)


def make_prediction(_hotter, _cooler,
                    _inside_air_temperature, _inside_air_humidity,
                    _outside_air_temperature, _outside_air_humidity,
                    _soil_humidity,
                    _illuminance
                    ):

    all_past = pd.read_csv('data.csv')
    considered_past_all_features = all_past.tail(HISTORY_SIZE * 2 + 10)
    features_considered = ["temp_out", "hum_out", "temp_in", "hum_in", "lum", "hotter", "cooler"]
    considered_past = considered_past_all_features[features_considered]
    considered_past.fillna(0, inplace=True)

    columns = {"temp_out": [_outside_air_temperature], "hum_out": [_outside_air_humidity], "temp_in": [_inside_air_temperature], "hum_in": [_inside_air_humidity], "lum": [_illuminance], "hotter": [_hotter], "cooler": [_cooler]}
    now = pd.DataFrame(data=columns)
    now_with_past = considered_past.append(now, ignore_index=True)

    now_with_past["hum_out"] = hampel(now_with_past["hum_out"])
    outlier_idx = now_with_past["hum_out"].isna()
    now_with_past["hum_out"].interpolate(inplace=True)

    now_with_past["temp_out"] = hampel(now_with_past["temp_out"])
    now_with_past["temp_out"][outlier_idx] = np.nan
    now_with_past["temp_out"].interpolate(inplace=True)

    now_with_past["temp_in"] = hampel(now_with_past["temp_in"])
    now_with_past["temp_in"].interpolate(inplace=True)

    now_with_past["hum_in"] = hampel(now_with_past["hum_in"])
    now_with_past["hum_in"].interpolate(inplace=True)

    now_with_past_norm = normalize(now_with_past)

    now_with_past_ = now_with_past_norm.to_numpy()

    predicted_temp = predict_parameter(2, model_air_t, now_with_past_)
    predicted_hum = predict_parameter(3, model_air_h, now_with_past_)

    predicted_temp_denorm = predicted_temp * STD_FROM_TRAINING['temp_in'] + MEAN_FROM_TRAINING['temp_in']
    predicted_hum_denorm = predicted_hum * STD_FROM_TRAINING['hum_in'] + MEAN_FROM_TRAINING['hum_in']

    return predicted_temp_denorm, predicted_hum_denorm


def calculate_deviation(temperature, humidity):
    deviation = 0
    normalized_temperature = abs((temperature - AIR_T_OPT) / (AIR_T_HIGH - AIR_T_LOW))
    normalized_humidity = abs((humidity - AIR_H_OPT) / (AIR_H_HIGH - AIR_H_LOW))
    deviation = (normalized_temperature + normalized_humidity) / 2

    return deviation


def manipulate_actuators_neural(inside_air_temperature, inside_air_humidity,
                                outside_air_temperature, outside_air_humidity,
                                soil_humidity,
                                illuminance):
    _hotter = OFF
    _cooler = OFF
    _valve = OFF

    choices = {'hotter OFF, cooler OFF': {'cooler': OFF,
                                          'hotter': OFF,
                                          'deviation': 0},
               'hotter OFF, cooler ON': {'cooler': OFF,
                                         'hotter': ON,
                                         'deviation': 0},
               'hotter ON, cooler OFF': {'cooler': ON,
                                         'hotter': OFF,
                                         'deviation': 0},
               'hotter ON, cooler ON': {'cooler': ON,
                                        'hotter': ON,
                                        'deviation': 0}}
    minimal = 1
    best_choice = None
    for choice in choices:
        temperature, humidity = make_prediction(choices[choice]['hotter'], choices[choice]['cooler'],
                                                inside_air_temperature, inside_air_humidity,
                                                outside_air_temperature, outside_air_humidity,
                                                soil_humidity,
                                                illuminance)
        print("Predicted temperature with " + choice + ": " + format(temperature, '.2f'))
        print("Predicted humidity with " + choice + ": " + format(humidity, '.2f'))
        choices[choice]['deviation'] = calculate_deviation(temperature, humidity)
        print("Predicted deviation with " + choice + ": " + format(choices[choice]['deviation'], '.2f'))

        if minimal > choices[choice]['deviation']:
            minimal = choices[choice]['deviation']
            best_choice = choice
    print()
    print("Best choice is " + best_choice)

    return choices[best_choice]['hotter'], choices[best_choice]['cooler'], OFF




while True:
    print(datetime.datetime.now())
    try:
        outside_air_humidity, outside_air_temperature = Adafruit_DHT.read_retry(DHTSensor, BCM_DHT11)
        print("Outside air temperature: " + str(outside_air_temperature))
        print("Outside air humidity: " + str(outside_air_humidity))
    except:
        continue

    try:
        inside_air_temperature = sensor_am2320.temperature
        inside_air_humidity = sensor_am2320.relative_humidity
        print("Inside air temperature: " + str(inside_air_temperature))
        print("Inside air humidity: " + str(inside_air_humidity))
    except:
        continue

    try:
        illuminance = int(sensor_bh1750.lux)
        print("Illuminance: " + str(illuminance))
    except:
        continue

    try:
        soil_value = chan0.value
        if soil_value > DRY_VOLTAGE:
            soil_value = DRY_VOLTAGE

        if soil_value < WATER_VOLTAGE:
            soil_value = WATER_VOLTAGE

        soil_value = soil_value - WATER_VOLTAGE
        soil_value = soil_value / (DRY_VOLTAGE - WATER_VOLTAGE)

        soil_humidity = 50  # int((1 - soil_value) * 100)
        print("Soil humidity: " + str(soil_humidity))
    except:
        continue

    ##############################################################################

    # hotter = OFF
    # cooler = OFF
    # valve = OFF

    # hotter, cooler, valve = manipulate_actuators_usual(inside_air_temperature, inside_air_humidity,
    #                                                    outside_air_temperature, outside_air_humidity,
    #                                                    soil_humidity,
    #                                                    illuminance)

    hotter, cooler, valve = manipulate_actuators_neural(inside_air_temperature, inside_air_humidity,
                                                        outside_air_temperature, outside_air_humidity,
                                                        soil_humidity,
                                                        illuminance)

    ##############################################################################

    GPIO.output(BCM_HOTTER, hotter)
    print("Hotter condition is " + str(hotter))
    GPIO.output(BCM_COOLER, (cooler - 1) * -1)
    print("Cooler condition is " + str(cooler))
    GPIO.output(BCM_VALVE, (valve - 1) * -1)
    print("Valve condition is " + str(valve))

    with open('data.csv', mode='a', newline='') as data:
        data_writer = csv.writer(data, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        data_writer.writerow([str(outside_air_temperature), str(outside_air_humidity), str(inside_air_temperature),
                              str(inside_air_humidity), str(soil_humidity), str(illuminance), str(hotter), str(cooler),
                              str(valve)])
        data.close()

    with open('time.txt', 'w') as timer:
        timer.write(str(datetime.datetime.now()))

    PASSED_TIME = 0

    time.sleep(VALVE_TIME - PASSED_TIME)
    PASSED_TIME = PASSED_TIME + VALVE_TIME
    GPIO.output(BCM_VALVE, (OFF - 1) * 1)

    time.sleep(HOTTER_TIME - PASSED_TIME)
    PASSED_TIME = PASSED_TIME + HOTTER_TIME
    GPIO.output(BCM_HOTTER, OFF)

    time.sleep(COOLER_TIME - PASSED_TIME)
    PASSED_TIME = PASSED_TIME + COOLER_TIME
    GPIO.output(BCM_COOLER, (OFF - 1) * 1)

    time.sleep(STEP_TIME - PASSED_TIME)

    ######################################################################

    print()
