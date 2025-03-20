import requests as r
import json
import time

# URL of the phyphox experiment
url = "http://10.250.173.209"
what_to_get = ['magX', 'magY', 'magZ', 'mag']

def phyphox_data():
    # Get the data from phyphox
    response = r.get(url + '&'.join(what_to_get)).text
    data = json.loads(response)
    for item in what_to_get:
        mag_data = data['buffer'][item]['buffer'][0]
        print(f'{mag_data:10.8}', end ='\t')
    print()
    
    while True:
        phyphox_data()
        time.sleep(0.1)