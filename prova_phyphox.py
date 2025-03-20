import requests as r
import json
import time

# URL of the phyphox experiment
url = "http://10.250.173.209/get?channels="  # Adjust this URL based on phyphox's API
what_to_get = ['magX', 'magY', 'magZ', 'mag']

def get_phyphox_data():
    # Construct the URL with the channels
    full_url = url + ','.join(what_to_get)
    
    try:
        # Get the data from phyphox
        response = r.get(full_url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        data = response.json()
        
        # Process the data
        for item in what_to_get:
            # Assuming 'buffer' is a key in the response and it contains a list of values
            # Adjust this based on the actual structure of the response
            mag_data = data.get('buffer', {}).get(item, [])[0] if data.get('buffer', {}).get(item) else None
            print(f'{mag_data:10.8}', end='\t')
        print()
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
    except json.JSONDecodeError:
        print("Failed to decode JSON.")

def main():
    while True:
        get_phyphox_data()
        time.sleep(0.1)

if __name__ == "__main__":
    main()
