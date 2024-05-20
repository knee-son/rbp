print('Importing camera and firebase...')
from picamera2 import Picamera2
import time
import random
import threading
import datetime
import firebase_admin
from firebase_admin import credentials, firestore, storage
import pytz
import board
import adafruit_dht
import RPi.GPIO as GPIO
from PIL import Image

from firebase_admin.exceptions import ResourceExhaustedError

GPIO.setmode(GPIO.BCM)
GPIO.setup(2, GPIO.OUT)
GPIO.output(2, GPIO.HIGH)
GPIO.setup(4, GPIO.IN)

location_raw = '/home/phenom/Pictures/raw.png'
location_processed = '/home/phenom/Pictures/final.png'
location_cred  = '/home/phenom/Documents/phenom-c0a0c-firebase-adminsdk-3yjmf-17bb5b392b.json'

cred = credentials.Certificate(location_cred)
firebase_admin.initialize_app(cred, {'storageBucket': 'phenom-c0a0c.appspot.com'})
db = firestore.client()
ref_sensor = db.collection('data').document('sensor')
ref_camera = db.collection('data').document('camera')

'''=========================================================
OBJECT DETECTION
========================================================='''
print('Importing Mask_RCNN...')
import os
if os.getcwd().split('/')[-1] != 'Mask_RCNN':
    os.chdir('Mask_RCNN')
ROOT_DIR = os.path.abspath('')

import sys
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import colorsys
from skimage.measure import find_contours
from matplotlib.patches import Polygon
import random

sys.path.append(ROOT_DIR)
from mrcnn import utils
import mrcnn.model as modellib

from mrcnn import visualize
sys.path.append(os.path.join(ROOT_DIR, 'samples/coco'))
import coco

MODEL_PATH = os.path.join(ROOT_DIR, 'mask_rcnn_coco.h5')
if not os.path.exists(MODEL_PATH):
    utils.download_trained_weights(MODEL_PATH)

class InferenceConfig(coco.CocoConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
config = InferenceConfig()

print("Loading inference model...")
model = modellib.MaskRCNN(mode='inference', model_dir=MODEL_PATH, config=config)
model.load_weights(MODEL_PATH, by_name=True)
print("Finished loading model!")

import colorsys
from skimage.measure import find_contours
from matplotlib.patches import Polygon

class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'BG', 'banana', 'egg',
               'sandwich', 'egg', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush'
              ]

def display_instances(image, boxes, masks, class_ids, class_names,
                      colors=None, captions=None):
    N = np.shape(boxes)[0]

    height, width, _ = image.shape
    factor = 2
    figsize = (width/100*factor, height/100*factor)
    fig, ax = plt.subplots(1, figsize=figsize)
    auto_show = True

    hsv = [(i/N, 1, 1.0) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)

    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    ax.axis('off')

    print('captions:', captions)
    print('n boxes:', N)

    for i in range (N):
        color = colors[i]
        if not np.any(boxes[i]):
            continue
        y1, x1, y2, x2 = boxes[i]

        p = matplotlib.patches.Rectangle(
            (x1, y1),
            x2-x1, y2-y1,
            linewidth=2,
            alpha=0.7, linestyle='dashed',
            edgecolor=color, facecolor='none')
        ax.add_patch(p)

        caption = captions[i]

        ax.text(x1, y1+8, caption, weight='bold',
                color='w', size=9, backgroundcolor='k')

        mask = masks[:, :, i]
        padded_mask = np.zeros(
            (mask.shape[0]+2, mask.shape[1]+2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        contours = find_contours(padded_mask, 0.5)
        for verts in contours:
            verts = np.fliplr(verts)-1
            p = Polygon(verts, facecolor='none', edgecolor=color)
            ax.add_patch(p)
    ax.imshow(image.astype(np.uint8))

    plt.margins(0,0)
    plt.savefig(location_processed, bbox_inches='tight', pad_inches=0)
    # plt.show()

def remove_classes_instances(class_ids, rois, masks, scores, class_names, whitelist_class):
    whitelist_class_id = class_names.index(whitelist_class)

    filtered_class_ids = []
    filtered_rois = []
    filtered_masks = []
    filtered_scores = []

    for i in range(len(class_ids)):
        if class_ids[i] == whitelist_class_id:
            filtered_class_ids.append(class_ids[i])
            filtered_rois.append(rois[i])
            filtered_masks.append(masks[:, :, i])
            filtered_scores.append(scores[i])

    filtered_rois = np.array(filtered_rois)
    if filtered_masks:
        filtered_masks = np.stack(filtered_masks, axis=2)
    else:
        filtered_masks = np.empty((masks.shape[0], masks.shape[1], 0),
                                   dtype=masks.dtype)
    filtered_scores = np.array(filtered_scores)

    return filtered_rois, filtered_masks, filtered_scores, filtered_class_ids

def perform_segmentation():
    ref_sensor.set({'message': 'Performing image segmentation...'}, merge=True)

    image = skimage.io.imread(location_raw)[:, :, :3]
    r = model.detect([image], verbose=0)[0]

    ref_sensor.set({'message': 'Preparing image overlays...'}, merge=True)

    filtered_rois, filtered_masks, filtered_scores, filtered_class_ids = \
        remove_classes_instances(r['class_ids'], r['rois'], r['masks'],
                                 r['scores'], class_names, 'egg')

    captions = []
    num_masks = filtered_masks.shape[-1]

    for k in range(num_masks):
        three_channel_mask = np.stack(
            (filtered_masks[:, :, k],)*3, axis=-1).astype(np.uint8)
        masked_im = image*three_channel_mask

        count = np.count_nonzero(filtered_masks[:, :, k])

        red = np.sum(masked_im[:, :, 0]*0.2126)
        green = np.sum(masked_im[:, :, 1]*0.7152)
        blue = np.sum(masked_im[:, :, 2]*0.0722)
        wtd_lmn = (red+green+blue)/count

        c = f'egg {k+1} ({filtered_scores[k]*100:.2f}%): '
        c += 'INFERTILE ' if wtd_lmn > 60 else 'FERTILE '
        c += f'({wtd_lmn:.2f})'
        captions.append(c)

    display_instances(image, filtered_rois, filtered_masks,
                      [0]*num_masks, [''], captions=captions)
#    display_instances(image, r['rois'], r['masks'], r['class_ids'],
#                      class_names, captions=['',]*len(r['masks']))

'''======================================================'''
tz = pytz.timezone('Asia/Manila')

picam2 = Picamera2()
#camera_config = picam2.create_preview_configuration()
camera_config = picam2.create_still_configuration()
picam2.configure(camera_config)
picam2.start()

dhtDevice = adafruit_dht.DHT22(board.D14)

# Camera
def capture_image():
    ref_sensor.set({'message': 'Capturing image...'}, merge=True)

    tz = pytz.timezone('Asia/Manila')
    current_time = datetime.datetime.now(tz)

    filename = current_time.strftime('%Y-%m-%d_%H-%M-%S.png')

    GPIO.setup(4,GPIO.OUT)
    time.sleep(0.5) # give lens time to adjust to light
    picam2.capture_file(location_raw)
    GPIO.setup(4,GPIO.IN)

    downsample(location_raw)

    mean_brightness = 0.0

    perform_segmentation()

    ref_sensor.set({'message': 'Uploading image...'}, merge=True)
    bucket = storage.bucket()
    blob = bucket.blob(filename)
#    blob.upload_from_filename(location_raw)
    blob.upload_from_filename(location_processed)
    image_url = blob.generate_signed_url(datetime.timedelta(seconds=300), method='GET')

    ref_record = ref_camera.collection('records')
    update_time, timeref = ref_record.add({
        'brightness': mean_brightness,
    })
    timeref.set({'time': update_time}, merge=True)

    ref_camera.set({
        'capturing': False,
        'filename': filename,
        'brightness': mean_brightness,
        'last_id': timeref.id,
    })
    ref_sensor.set({'message': 'An image has been uploaded!'}, merge=True)

def downsample(location):
    with Image.open(location) as img:
        divide = 5
        new_width = img.width // divide
        new_height = img.height // divide
        resized_img = img.resize((new_width, new_height), Image.LANCZOS)

        resized_img.save(location)

    print(f"Image saved successfully at {location} with new dimensions: {new_width}x{new_height}")


# camera interrupt
callback_done = threading.Event()
def on_snapshot(doc_snapshot, changes, read_time):
    for change in changes:
         i_want_to_capture = change.document.to_dict()['capturing']
    if i_want_to_capture: capture_image()
    callback_done.set()
camera_watch = ref_camera.on_snapshot(on_snapshot)

ref_sensor.set({'message': 'Raspberry Pi booted successfully!'}, merge=True)

while True:
    current_time = datetime.datetime.now(tz)

    # Temperature and Humidity Sensor
    try:
        temperature = dhtDevice.temperature + random.gauss(0, 0.05)
        humidity = dhtDevice.humidity + random.gauss(0, 0.05)
        msg = ''
    except Exception as error:
        no_sensor = False
        print(error)
        msg = str(error).replace(' Try again.', '')
        time.sleep(1.0)
        continue

    print("Temp: {:.2f} C    Humidity: {:.2f}% Time:{:}"
        .format(temperature, humidity, current_time))
    if temperature <= 37.9:
        GPIO.output(2,GPIO.HIGH)
    else:
        GPIO.output(2,GPIO.LOW)

    ref_record = ref_sensor.collection('records')

    try:
        timestamp, timeref = ref_record.add({
            'temperature': temperature,
            'humidity': humidity,
            'time': current_time,
        })

        ref_sensor.set({
            'time': current_time,
            'temperature': temperature,
            'humidity': humidity,
            'last_id': timeref.id
        }, merge=True)

    except Exception as e:
        print('firebase limit exceeded:', e)

    # waiting 5 seconds. interrupt when camera is needed
    time_sleeping = 5.0
    segs = 100
    time_interval = time_sleeping/segs

    for i in range(segs):
        time.sleep(time_interval)
