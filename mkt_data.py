import requests
import json

def load_api_key() -> dict:
    with open('conf\\eod_data_api_key.json') as file: 
        cdp_api_key = json.load(file)
    
    return cdp_api_key

# change these to load data
symbol = 'SPX.LSE'
frm = '2024-01-02'
to = '2024-09-03'

api_key = load_api_key()['key']
url = f'https://eodhd.com/api/eod/{symbol}?from={frm}&to={to}&period=d&api_token={api_key}&fmt=json'
# url = f'https://eodhd.com/api/exchanges-list/?api_token={api_key}&fmt=json'

# use this to search by isin / ticker
# url = f'https://eodhistoricaldata.com/api/search/SPXD?api_token={api_key}'
    #   {
    #     "Code": "VIXL",
    #     "Exchange": "F",
    #     "Name": "S&P 500 VIX Short-term Futures Index",
    #     "Type": "ETF",
    #     "Country": "Germany",
    #     "Currency": "EUR",
    #     "ISIN": None,
    #     "previousClose": 0.0068,
    #     "previousCloseDate": "2024-06-28"
    # },


data = requests.get(url).json()

print(data)