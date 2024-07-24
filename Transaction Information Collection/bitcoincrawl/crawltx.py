#encoding=utf-8
"""
@author=gang wang
crawl tx useing wallet address.
"""
import requests
import json
def get_transactions(address):
    api_url=f'https://blockchain.info/rawaddr/{address}'
    try:
        response=requests.get(api_url)
        response.raise_for_status()
        data=response.json()
        print(data)
        print(len(data))
    except requests.exceptions.HTTPError as err:
        print(f"HTTP Error: {err}")
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
if __name__ == '__main__':
    bitcoin_address='16uHQwNUvkDBzNwgYb4fPkjMMxWddAULCE'
    get_transactions(bitcoin_address)