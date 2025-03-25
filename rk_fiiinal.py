import os
import sys
import signal
import time
import traceback
import numpy as np
from edge_impulse_linux.image import ImageImpulseRunner
import RPi.GPIO as GPIO
from hx711 import HX711
import requests
import json
from picamera2 import Picamera2
import cv2

# Global Variables
runner = None
show_camera = True
ratio = -1363.992
id_product = 1
list_label = []
list_weight = []
count = 0
taken = 0
hx = None
c_value = 0

# Product Labels
PRODUCTS = {
    'Apple': (0.01, 10),
    'Lays': (1, 1)
}

def now():
    return round(time.time() * 1000)

def sigint_handler(sig, frame):
    print('Interrupted')
    if runner:
        runner.stop()
    GPIO.cleanup()
    sys.exit(0)

signal.signal(signal.SIGINT, sigint_handler)

def find_weight():
    global c_value, hx
    try:
        GPIO.setmode(GPIO.BCM)
        if c_value == 0:
            print('Calibrating scale...')
            hx = HX711(dout_pin=20, pd_sck_pin=21)
            if hx.zero():
                raise ValueError('Tare unsuccessful.')
            hx.set_scale_ratio(ratio)
            c_value = 1
            print('Calibration complete.')
        time.sleep(1)
        weight = int(hx.get_weight_mean(20))
        print(f'Weight: {weight} g')
        return weight
    except Exception as e:
        print(f'Error in weight measurement: {e}')
        return 0

def post(label, price, final_rate, taken):
    global id_product, list_label, list_weight, count
    url = "https://automaticbilling-ivrf.onrender.com/product"
    headers = {"Content-Type": "application/json"}
    data = json.dumps({"id": id_product, "name": label, "price": price, "units": "units", "taken": taken, "payable": final_rate})
    try:
        resp = requests.post(url, headers=headers, data=data, timeout=5)
        print(f'Status Code: {resp.status_code}')
        id_product += 1
        list_label.clear()
        list_weight.clear()
        count, taken = 0, 0
    except requests.exceptions.RequestException as e:
        print(f'Error posting data: {e}')

def list_com(label, final_weight):
    global count, taken
    try:
        if final_weight > 2:
            list_weight.append(final_weight)
            if count > 1 and list_weight[-1] > list_weight[-2]:
                taken += 1
        list_label.append(label)
        count += 1
        time.sleep(1)
        if count > 1 and list_label[-1] != list_label[-2]:
            print(f"New Item detected: {label}")
            rate(list_weight[-2], list_label[-2], taken)
    except Exception as e:
        print(f'Error in list comparison: {e}')

def rate(final_weight, label, taken):
    if label in PRODUCTS:
        final_rate, price = PRODUCTS[label]
        post(label, price, final_weight * final_rate, taken)
    else:
        print(f'Unknown product detected: {label}')

def main(argv):
    global runner
    if len(argv) == 0:
        print('Usage: python classify.py <path_to_model.eim>')
        sys.exit(2)
    
    model = argv[0]
    if not os.path.exists(model):
        print(f"Error: Model file not found at {model}")
        sys.exit(1)
    
    print(f'Loading model: {model}')
    try:
        with ImageImpulseRunner(model) as runner:
            model_info = runner.init()
            labels = model_info['model_parameters']['labels']
            picam2 = Picamera2()
            config = picam2.create_preview_configuration(main={"size": (96, 96)})
            picam2.configure(config)
            picam2.start()
            time.sleep(2)
            
            while True:
                frame = picam2.capture_array()
                if frame is None or frame.size == 0:
                    print("Error: Could not capture frame.")
                    continue
                
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = frame.astype(np.float32)
                frame = frame.flatten().tolist()
                
                try:
                    res = runner.classify(frame)
                    if "classification" in res["result"]:
                        for label in labels:
                            score = res['result']['classification'].get(label, 0)
                            if score > 0.5:
                                final_weight = find_weight()
                                list_com(label, final_weight)
                                print(f'{label} detected')
                    else:
                        print("No classification results, retrying...")
                except Exception as e:
                    print(f"Error in classification: {e}")
    except Exception as e:
        print(f'Error: {traceback.format_exc()}')
    finally:
        if runner:
            runner.stop()
        picam2.stop()
        GPIO.cleanup()

if __name__ == "__main__":
    main(sys.argv[1:])

