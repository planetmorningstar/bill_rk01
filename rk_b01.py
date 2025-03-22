import os
import sys
import getopt
import signal
import time
from edge_impulse_linux.image import ImageImpulseRunner
import RPi.GPIO as GPIO
from hx711 import HX711
import requests
import json
from requests.structures import CaseInsensitiveDict
from picamera2 import Picamera2
import numpy as np

runner = None
show_camera = True

# Global variables
c_value = 0
flag = 0
ratio = -1363.992
id_product = 1
list_label = []
list_weight = []
count = 0
final_weight = 0
taken = 0
hx = None  # Global HX711 object

# Product labels
a = 'Apple'
b = 'Banana'
l = 'Lays'
c = 'Coke'

def now():
    return round(time.time() * 1000)

def sigint_handler(sig, frame):
    print('Interrupted')
    if runner:
        runner.stop()
    sys.exit(0)

signal.signal(signal.SIGINT, sigint_handler)

def help():
    print('python classify.py <path_to_model.eim>')

def find_weight():
    global c_value, hx
    if c_value == 0:
        print('Calibration starts')
        try:
            # Disable GPIO warnings to avoid the channel in use message
            GPIO.setwarnings(False)
            GPIO.setmode(GPIO.BCM)
            hx = HX711(dout_pin=20, pd_sck_pin=21)
            err = hx.zero()
            if err:
                raise ValueError('Tare is unsuccessful.')
            hx.set_scale_ratio(ratio)
            c_value = 1
        except (KeyboardInterrupt, SystemExit):
            print('Bye :)')
        except Exception as e:
            print(f"Error during calibration: {e}")
        print('Calibrate ends')
    else:
        try:
            # No need to set GPIO mode again if already set
            time.sleep(1)
            weight = int(hx.get_weight_mean(20))
            print(weight, 'g')
            return weight
        except (KeyboardInterrupt, SystemExit):
            print('Bye :)')
        except Exception as e:
            print(f"Error getting weight: {e}")
            return 0  # Return 0 if there's an error

def post(label, price, final_rate, taken_items):
    global id_product, list_label, list_weight, count, final_weight, taken
    url = "https://automaticbilling-ivrf.onrender.com/product"
    headers = CaseInsensitiveDict()
    headers["Content-Type"] = "application/json"
    
    # Convert numpy values to Python native types to ensure JSON serialization works
    if isinstance(final_rate, np.generic):
        final_rate = final_rate.item()
    if isinstance(price, np.generic):
        price = price.item()
    if isinstance(taken_items, np.generic):
        taken_items = taken_items.item()
    
    data_dict = {
        "id": id_product, 
        "name": label, 
        "price": price, 
        "units": "units", 
        "taken": taken_items, 
        "payable": final_rate
    }
    
    try:
        data = json.dumps(data_dict)
        resp = requests.post(url, headers=headers, data=data)
        print(f"API Response: {resp.status_code}")
        
        # Reset global variables
        id_product += 1
        list_label.clear()
        list_weight.clear()
        count = 0
        final_weight = 0
        taken = 0
    except Exception as e:
        print(f"Error in post request: {e}")
    
    time.sleep(1)

def list_com(label, final_weight):
    global count, taken
    
    if final_weight is None:
        final_weight = 0
        
    # Only add to list if weight is significant
    if final_weight > 2:
        list_weight.append(final_weight)
        if count > 1 and len(list_weight) > 1 and list_weight[-1] > list_weight[-2]:
            taken += 1
    else:
        # Add current weight even if it's low or zero to maintain index alignment with labels
        list_weight.append(final_weight)
    
    list_label.append(label)
    count += 1
    print('count is', count)
    time.sleep(1)
    
    if count > 1 and len(list_label) > 1 and len(list_weight) > 1:
        if list_label[-1] != list_label[-2]:
            print("New Item detected")
            print("Final weight is", list_weight[-1])
            rate(list_weight[-2], list_label[-2], taken)

def rate(final_weight, label, taken_items):
    print("Calculating rate")
    
    # Ensure final_weight is a valid number
    if final_weight is None or final_weight <= 0:
        final_weight = 1  # Default to 1 to avoid zero-price items
    
    if label == a:
        print("Calculating rate of", label)
        final_rate_a = float(final_weight) * 0.01
        price = 10
        post(label, price, final_rate_a, taken_items)
    elif label == b:
        print("Calculating rate of", label)
        final_rate_b = float(final_weight) * 0.02
        price = 20
        post(label, price, final_rate_b, taken_items)
    elif label == l:
        print("Calculating rate of", label)
        final_rate_l = 1.0
        price = 1
        post(label, price, final_rate_l, taken_items)
    else:
        print("Calculating rate of", label)
        final_rate_c = 2.0
        price = 2
        post(label, price, final_rate_c, taken_items)

def main(argv):
    global flag, final_weight
    if flag == 0:
        find_weight()
        flag = 1

    try:
        opts, args = getopt.getopt(argv, "h", ["--help"])
    except getopt.GetoptError:
        help()
        sys.exit(2)

    for opt, arg in opts:
        if opt in ('-h', '--help'):
            help()
            sys.exit()

    if len(args) == 0:
        help()
        sys.exit(2)

    model = args[0]

    dir_path = os.path.dirname(os.path.realpath(__file__))
    modelfile = os.path.join(dir_path, model)

    print('MODEL: ' + modelfile)

    with ImageImpulseRunner(modelfile) as runner:
        try:
            model_info = runner.init()
            print('Loaded runner for "' + model_info['project']['owner'] + ' / ' + model_info['project']['name'] + '"')
            labels = model_info['model_parameters']['labels']

            # Initialize Picamera2
            picam2 = Picamera2()
            config = picam2.create_preview_configuration(main={"size": (640, 480)})
            picam2.configure(config)
            picam2.start()
            time.sleep(2)  # Allow camera to warm up

            next_frame = 0  # limit to ~10 fps here

            while True:
                if next_frame > now():
                    time.sleep((next_frame - now()) / 1000)

                # Capture frame using Picamera2
                try:
                    frame = picam2.capture_array()
                    if frame is None or frame.size == 0:
                        print("Error: Could not capture frame.")
                        continue

                    # Convert frame to RGB format
                    frame_rgb = np.ascontiguousarray(frame[:, :, :3])  # Ensure the frame is in RGB format

                    # Perform inference with proper error handling
                    try:
                        # Make sure the frame has the correct dimensions for the model
                        res = runner.classify(frame_rgb)
                        
                        if "result" in res and "classification" in res["result"]:
                            print('Result (%d ms.) ' % (res['timing']['dsp'] + res['timing']['classification']), end='')
                            
                            for label in labels:
                                if label in res['result']['classification']:
                                    score = res['result']['classification'][label]
                                    if score > 0.9:
                                        # A high confidence detection occurred
                                        current_weight = find_weight()
                                        list_com(label, current_weight)
                                        if label == a:
                                            print('Apple detected')
                                        elif label == b:
                                            print('Banana detected')
                                        elif label == l:
                                            print('Lays detected')
                                        else:
                                            print('Coke detected')
                            print('', flush=True)
                    except Exception as e:
                        print(f"Classification error: {e}")
                        continue
                        
                except Exception as e:
                    print(f"Camera error: {e}")
                    continue
                    
                next_frame = now() + 100

        except Exception as e:
            print(f"Error: {e}")
        finally:
            if runner:
                runner.stop()
            try:
                picam2.stop()
            except:
                pass

if __name__ == "__main__":
    main(sys.argv[1:])import os
import sys
import getopt
import signal
import time
from edge_impulse_linux.image import ImageImpulseRunner
import RPi.GPIO as GPIO
from hx711 import HX711
import requests
import json
from requests.structures import CaseInsensitiveDict
from picamera2 import Picamera2
import numpy as np

runner = None
show_camera = True

# Global variables
c_value = 0
flag = 0
ratio = -1363.992
id_product = 1
list_label = []
list_weight = []
count = 0
final_weight = 0
taken = 0
hx = None  # Global HX711 object

# Product labels
a = 'Apple'
b = 'Banana'
l = 'Lays'
c = 'Coke'

def now():
    return round(time.time() * 1000)

def sigint_handler(sig, frame):
    print('Interrupted')
    if runner:
        runner.stop()
    sys.exit(0)

signal.signal(signal.SIGINT, sigint_handler)

def help():
    print('python classify.py <path_to_model.eim>')

def find_weight():
    global c_value, hx
    if c_value == 0:
        print('Calibration starts')
        try:
            # Disable GPIO warnings to avoid the channel in use message
            GPIO.setwarnings(False)
            GPIO.setmode(GPIO.BCM)
            hx = HX711(dout_pin=20, pd_sck_pin=21)
            err = hx.zero()
            if err:
                raise ValueError('Tare is unsuccessful.')
            hx.set_scale_ratio(ratio)
            c_value = 1
        except (KeyboardInterrupt, SystemExit):
            print('Bye :)')
        except Exception as e:
            print(f"Error during calibration: {e}")
        print('Calibrate ends')
    else:
        try:
            # No need to set GPIO mode again if already set
            time.sleep(1)
            weight = int(hx.get_weight_mean(20))
            print(weight, 'g')
            return weight
        except (KeyboardInterrupt, SystemExit):
            print('Bye :)')
        except Exception as e:
            print(f"Error getting weight: {e}")
            return 0  # Return 0 if there's an error

def post(label, price, final_rate, taken_items):
    global id_product, list_label, list_weight, count, final_weight, taken
    url = "https://automaticbilling-ivrf.onrender.com/product"
    headers = CaseInsensitiveDict()
    headers["Content-Type"] = "application/json"
    
    # Convert numpy values to Python native types to ensure JSON serialization works
    if isinstance(final_rate, np.generic):
        final_rate = final_rate.item()
    if isinstance(price, np.generic):
        price = price.item()
    if isinstance(taken_items, np.generic):
        taken_items = taken_items.item()
    
    data_dict = {
        "id": id_product, 
        "name": label, 
        "price": price, 
        "units": "units", 
        "taken": taken_items, 
        "payable": final_rate
    }
    
    try:
        data = json.dumps(data_dict)
        resp = requests.post(url, headers=headers, data=data)
        print(f"API Response: {resp.status_code}")
        
        # Reset global variables
        id_product += 1
        list_label.clear()
        list_weight.clear()
        count = 0
        final_weight = 0
        taken = 0
    except Exception as e:
        print(f"Error in post request: {e}")
    
    time.sleep(1)

def list_com(label, final_weight):
    global count, taken
    
    if final_weight is None:
        final_weight = 0
        
    # Only add to list if weight is significant
    if final_weight > 2:
        list_weight.append(final_weight)
        if count > 1 and len(list_weight) > 1 and list_weight[-1] > list_weight[-2]:
            taken += 1
    else:
        # Add current weight even if it's low or zero to maintain index alignment with labels
        list_weight.append(final_weight)
    
    list_label.append(label)
    count += 1
    print('count is', count)
    time.sleep(1)
    
    if count > 1 and len(list_label) > 1 and len(list_weight) > 1:
        if list_label[-1] != list_label[-2]:
            print("New Item detected")
            print("Final weight is", list_weight[-1])
            rate(list_weight[-2], list_label[-2], taken)

def rate(final_weight, label, taken_items):
    print("Calculating rate")
    
    # Ensure final_weight is a valid number
    if final_weight is None or final_weight <= 0:
        final_weight = 1  # Default to 1 to avoid zero-price items
    
    if label == a:
        print("Calculating rate of", label)
        final_rate_a = float(final_weight) * 0.01
        price = 10
        post(label, price, final_rate_a, taken_items)
    elif label == b:
        print("Calculating rate of", label)
        final_rate_b = float(final_weight) * 0.02
        price = 20
        post(label, price, final_rate_b, taken_items)
    elif label == l:
        print("Calculating rate of", label)
        final_rate_l = 1.0
        price = 1
        post(label, price, final_rate_l, taken_items)
    else:
        print("Calculating rate of", label)
        final_rate_c = 2.0
        price = 2
        post(label, price, final_rate_c, taken_items)

def main(argv):
    global flag, final_weight
    if flag == 0:
        find_weight()
        flag = 1

    try:
        opts, args = getopt.getopt(argv, "h", ["--help"])
    except getopt.GetoptError:
        help()
        sys.exit(2)

    for opt, arg in opts:
        if opt in ('-h', '--help'):
            help()
            sys.exit()

    if len(args) == 0:
        help()
        sys.exit(2)

    model = args[0]

    dir_path = os.path.dirname(os.path.realpath(__file__))
    modelfile = os.path.join(dir_path, model)

    print('MODEL: ' + modelfile)

    with ImageImpulseRunner(modelfile) as runner:
        try:
            model_info = runner.init()
            print('Loaded runner for "' + model_info['project']['owner'] + ' / ' + model_info['project']['name'] + '"')
            labels = model_info['model_parameters']['labels']

            # Initialize Picamera2
            picam2 = Picamera2()
            config = picam2.create_preview_configuration(main={"size": (640, 480)})
            picam2.configure(config)
            picam2.start()
            time.sleep(2)  # Allow camera to warm up

            next_frame = 0  # limit to ~10 fps here

            while True:
                if next_frame > now():
                    time.sleep((next_frame - now()) / 1000)

                # Capture frame using Picamera2
                try:
                    frame = picam2.capture_array()
                    if frame is None or frame.size == 0:
                        print("Error: Could not capture frame.")
                        continue

                    # Convert frame to RGB format
                    frame_rgb = np.ascontiguousarray(frame[:, :, :3])  # Ensure the frame is in RGB format

                    # Perform inference with proper error handling
                    try:
                        # Make sure the frame has the correct dimensions for the model
                        res = runner.classify(frame_rgb)
                        
                        if "result" in res and "classification" in res["result"]:
                            print('Result (%d ms.) ' % (res['timing']['dsp'] + res['timing']['classification']), end='')
                            
                            for label in labels:
                                if label in res['result']['classification']:
                                    score = res['result']['classification'][label]
                                    if score > 0.9:
                                        # A high confidence detection occurred
                                        current_weight = find_weight()
                                        list_com(label, current_weight)
                                        if label == a:
                                            print('Apple detected')
                                        elif label == b:
                                            print('Banana detected')
                                        elif label == l:
                                            print('Lays detected')
                                        else:
                                            print('Coke detected')
                            print('', flush=True)
                    except Exception as e:
                        print(f"Classification error: {e}")
                        continue
                        
                except Exception as e:
                    print(f"Camera error: {e}")
                    continue
                    
                next_frame = now() + 100

        except Exception as e:
            print(f"Error: {e}")
        finally:
            if runner:
                runner.stop()
            try:
                picam2.stop()
            except:
                pass

if __name__ == "__main__":
    main(sys.argv[1:])
