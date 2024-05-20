import RPi.GPIO as GPIO
import time

while True:
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(10, GPIO.OUT)
    time.sleep(1)
    GPIO.cleanup()
    time.sleep(1)
