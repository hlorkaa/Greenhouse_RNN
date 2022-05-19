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

DHTSensor = Adafruit_DHT.DHT11

GPIO_Pin = 16
i2c = board.I2C()
#Create an object from the AM2320 class called "sensor"
sensor_am2320 = adafruit_am2320.AM2320(i2c)

sensor_bh1750 = adafruit_bh1750.BH1750(i2c)

# Create the I2C bus
#i2c = busio.I2C(board.SCL, board.SDA)
# Create the ADC object using the I2C bus
ads = ADS.ADS1115(i2c)
# Create single-ended input on channels
chan0 = AnalogIn(ads, ADS.P0)
chan1 = AnalogIn(ads, ADS.P1)
chan2 = AnalogIn(ads, ADS.P2)
chan3 = AnalogIn(ads, ADS.P3)

hotter_relay_bcm_number = 17
cooler_relay_bcm_number = 27
valve_relay_bcm_number = 22
dry = 17000
water = 10000
GPIO.setmode(GPIO.BCM) # GPIO Numbers instead of board numbers
GPIO.setup(hotter_relay_bcm_number, GPIO.OUT)
GPIO.setup(cooler_relay_bcm_number, GPIO.OUT)
GPIO.setup(valve_relay_bcm_number, GPIO.OUT)
GPIO.output(valve_relay_bcm_number, 1)

low_air_temperature = 20.0
optimal_air_temperature = 25.0
high_air_temperature = 30.0

low_air_humidity = 35.0
optimal_air_humidity = 55.0
high_air_humidity = 75.0

low_soil_humidity = 30.0
optimal_soil_humidity = 60.0
high_soil_humidity = 80.0

ON = 1
OFF = 0

# GPIO.output(hotter_relay_bcm_number, ON)
# time.sleep(5)
# GPIO.output(hotter_relay_bcm_number, OFF)
# GPIO.output(cooler_relay_bcm_number, ON)
# time.sleep(5)
# GPIO.output(cooler_relay_bcm_number, OFF)
# GPIO.output(valve_relay_bcm_number, ON)
# time.sleep(1)
# GPIO.output(valve_relay_bcm_number, OFF)

STEP_TIME = 300

HOTTER_TIME = 93
COOLER_TIME = 200
VALVE_TIME = 3

while True:
    print(datetime.datetime.now())

    outside_air_humidity, outside_air_temperature = Adafruit_DHT.read_retry(DHTSensor, GPIO_Pin)
    print("Outside air temperature: " + str(outside_air_temperature))
    print("Outside air humidity: " + str(outside_air_humidity))
    try:
        inside_air_temperature = sensor_am2320.temperature
        inside_air_humidity = sensor_am2320.relative_humidity
        print("Inside air temperature: " + str(inside_air_temperature))
        print("Inside air humidity: " + str(inside_air_humidity))
    except:
        continue

    illuminance = int(sensor_bh1750.lux)
    print("Illuminance: " + str(illuminance))

    soil_value = chan0.value
    if (soil_value > dry):
        soil_value = dry
    
    if (soil_value < water):
        soil_value = water
    
    soil_value = soil_value - water
    soil_value = soil_value / (dry - water)

    soil_humidity = 50 #int((1 - soil_value) * 100)
    print("Soil humidity: " + str(soil_humidity))

    ##############################################################################

    hotter = OFF
    cooler = OFF
    valve = OFF

    if (low_soil_humidity > soil_humidity):
        valve = ON
    if (high_soil_humidity < soil_humidity):
#         hotter = ON
        if (outside_air_humidity < inside_air_humidity):
            cooler = ON

    if (high_air_temperature < inside_air_temperature):
        if (outside_air_temperature < inside_air_temperature):
            cooler = ON
        hotter = OFF
    if (low_air_temperature > inside_air_temperature):
        hotter = ON
        cooler = OFF
        if (outside_air_temperature > inside_air_temperature):
            cooler = ON

    if (low_air_humidity > inside_air_humidity):
        if (outside_air_humidity > inside_air_humidity):
            cooler = ON
    if (high_air_humidity < inside_air_humidity):
#         if (outside_air_humidity < inside_air_humidity):
#             cooler = ON
        hotter = ON

    ##############################################################################


    GPIO.output(hotter_relay_bcm_number, hotter)
    print("Hotter condition is " + str(hotter))
    GPIO.output(cooler_relay_bcm_number, (cooler - 1) * -1)
    print("Cooler condition is " + str(cooler))
    GPIO.output(valve_relay_bcm_number, (valve - 1) * -1)
    print("Valve condition is " + str(valve))
    
    with open('data.csv', mode='a', newline='') as data:
        data_writer = csv.writer(data, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        
        data_writer.writerow([str(outside_air_temperature), str(outside_air_humidity), str(inside_air_temperature), str(inside_air_humidity), str(soil_humidity), str(illuminance), str(hotter), str(cooler), str(valve)])
        data.close()
        
    with open('time.txt', 'w') as timer:
        timer.write(str(datetime.datetime.now()))
        
    PASSED_TIME = 0

    time.sleep(VALVE_TIME - PASSED_TIME)
    PASSED_TIME = PASSED_TIME + VALVE_TIME
    GPIO.output(valve_relay_bcm_number, (OFF - 1) * 1)

    time.sleep(HOTTER_TIME - PASSED_TIME)
    PASSED_TIME = PASSED_TIME + HOTTER_TIME
    GPIO.output(hotter_relay_bcm_number, OFF)

    time.sleep(COOLER_TIME - PASSED_TIME)
    PASSED_TIME = PASSED_TIME + COOLER_TIME
    GPIO.output(cooler_relay_bcm_number, (OFF - 1) * 1)

    time.sleep(STEP_TIME - PASSED_TIME)
    
    ######################################################################

    print()
    