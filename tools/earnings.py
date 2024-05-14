"""Earnings Checker"""

from datetime import datetime, timedelta

import pandas as pd
import requests


def get_earnings_by_date(date: datetime = None):
    url = "https://api.nasdaq.com/api/calendar/earnings"

    headers = {
        "authority": "api.nasdaq.com",
        "accept": "application/json, text/plain, */*",
        "user-agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.116 Safari/537.36",
        "origin": "https://www.nasdaq.com",
        "referer": "https://www.nasdaq.com/",
        "accept-language": "en-US,en;q=0.9",
    }

    if date:
        datestr = date.strftime("%Y-%m-%d")
    else:
        datestr = datetime.now().strftime("%Y-%m-%d")

    params = {"date": datestr}

    response = requests.get(url, headers=headers, params=params)
    data = response.json()["data"]
    return pd.DataFrame(data["rows"], columns=data["headers"])


def get_symbols_with_earnings():
    earnings = []
    for add_days in range(0, 8):
        earnings.append(get_earnings_by_date(datetime.now() + timedelta(days=add_days)))

    return pd.concat(earnings, ignore_index=True).symbol.str.lower().to_list()
