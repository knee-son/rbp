import RPi.GPIO as GPIO
import time
from datetime import datetime
import pytz

GPIO.setmode(GPIO.BCM)
GPIO.setup(3, GPIO.OUT)

tz = pytz.timezone('Asia/Manila')
start = datetime.now(tz)

while True:
    now = datetime.now(tz)
    # Check if it's in mod 4 hours
    if now.hour%4==start.hour%4 and now.minute == start.minute:
        GPIO.output(3, GPIO.HIGH)  # Perform your task
        time.sleep(180)  # Wait for 3 minutes
        GPIO.output(3, GPIO.LOW)  # Finish the task

    # Delay to avoid unnecessary CPU load
    time.sleep(60)  # Check every minute
