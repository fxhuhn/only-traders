"""One-Pager of a Candle Screener"""

import datetime
import os
import pickle
from typing import Dict, List

import pandas as pd
import yfinance as yf

from tools import adx, atr, doji, resample_week, sma


def get_symbols() -> List[str]:
    """
    If not locally available, load stocks from alphavantage.
    Filter the List for stock symbols, which a located at NASDAQ or NYSE

    Returns:
        List[str]: List of stock symbols
    """

    URL: str = "https://www.alphavantage.co/query?function=LISTING_STATUS&apikey=demo"
    filename: str = "stocks.pkl"

    try:
        symbols = pd.read_pickle(filename)
    except FileNotFoundError:
        stocks = pd.read_csv(URL)
        symbols = stocks.query(
            'exchange in ["NASDAQ", "NYSE"] and assetType == "Stock"'
        ).symbol
        symbols.to_pickle(filename)

    return symbols


def get_stocks(symbols: List[str]) -> Dict[str, pd.DataFrame]:
    """_summary_

    Args:
        symbols (List[str]): _description_

    Returns:
        Dict[str, pd.DataFrame]: _description_
    """

    filename: str = "yahoo.pkl"
    dfs = {}
    try:
        # delete stock data of older than 12h
        file_time = os.path.getmtime(filename)
        file_age = datetime.datetime.now() - datetime.datetime.fromtimestamp(file_time)
        if file_age.seconds / 3600 > 12:
            os.remove(filename)

        # use current stock data
        with open(filename, "rb") as file:
            dfs = pickle.load(file)

    except FileNotFoundError:
        # in case no stock data exists, load them from yahoo
        for _, group in symbols.groupby(symbols.symbol.str[0]):
            stock_data = yf.download(
                group.symbol.values.tolist(),
                rounding=2,
                progress=False,
                group_by="ticker",
            )

            # perform some pre preparation
            for symbol in stock_data.columns.get_level_values(0).unique():
                # drop unclear items
                df = stock_data[symbol]
                df = df[~(df.High == df.Low) & ~(df.Open == df.Close)]
                df = df.dropna()
                df.index = pd.to_datetime(df.index)

                dfs[symbol.lower()] = df

        # save the current stock data for later activities
        with open(filename, "wb") as file:
            pickle.dump(dfs, file)
    return dfs


def main():
    export_list = []

    # update the stock data for the screening process
    dfs = get_stocks(symbols=get_symbols())

    # update stocklist with valid symbols
    pd.DataFrame(dfs.keys(), columns=["symbol"]).to_pickle("stocks.pkl")

    for symbol in dfs.keys():
        df = dfs[symbol.lower()]["2020-01-01":].copy()
        df_week = resample_week(df)

        # Minimum quantity of stockdata is 200 trading days
        if len(df) < 200:
            continue

        # Only high volume stocks with more than 1 Mio shares per day
        if sma(df.Volume, 10).iloc[-1] < 1_000_000:
            continue

        # Ignore stock with a price lower than 10 US$
        if df.Close.iloc[-1] < 10:
            continue

        # Apply the necessary indicators for stock data
        df["sma_5"] = sma(df.Close, 5)
        df["sma_200"] = sma(df.Close, 200)
        df["atr_10"] = atr(df, 10, "sma")
        df["adx_14"] = adx(df)

        df["close_pct_5"] = df.Close.pct_change(5)
        df["close_pct_60"] = df.Close.pct_change(60)

        df["high_max_3"] = df.High.rolling(window=3).max()
        df["low_min_3"] = df.Low.rolling(window=3).min()

        df["high_max_8"] = df.High.rolling(window=8).max()
        df["low_min_8"] = df.Low.rolling(window=8).min()

        df["doji"] = doji(df)
        df["prev_doji"] = df.doji.shift(1)

        df["prev_Close"] = df.Close.shift(1)
        df["prev_Open"] = df.Open.shift(1)
        df["prev_High"] = df.High.shift(1)
        df["prev_Low"] = df.Low.shift(1)

        df["close_above_sma_200"] = df.Close > df.sma_200
        df["atr_distance_high_3"] = (df.high_max_3 - df.Close) / df.atr_10
        df["atr_distance_low_3"] = (df.Close - df.low_min_3) / df.atr_10
        df["atr_distance_high_8"] = (df.high_max_8 - df.High) / df.atr_10
        df["atr_distance_low_8"] = (df.Low - df.low_min_8) / df.atr_10

        df["down_volume_5"] = (
            df[df.Close < df.sma_5]
            .Volume.dropna()
            .rolling(5)
            .mean()
            .reindex(df.index, method="pad")
        )
        df["up_volume_5"] = (
            df[df.Close > df.sma_5]
            .Volume.dropna()
            .rolling(5)
            .mean()
            .reindex(df.index, method="pad")
        )

        df_week["adx_14"] = adx(df_week)

        day = df.iloc[-1].to_dict()
        week = df_week.iloc[-1].to_dict()

        # Pattern for long:
        long_condition = [
            day["close_above_sma_200"] is True,
            day["doji"] is False,
            day["close_pct_5"] < 0,
            day["Close"] > day["Open"],
            not (
                (day["prev_Open"] < day["prev_Close"])
                and (day["prev_High"] < day["High"])
                and (day["prev_doji"] is False)
            ),
            day["close_pct_60"] > 0,
            day["atr_distance_high_8"] > 1.8,
            day["atr_distance_low_3"] < 1.5,
            day["up_volume_5"] > day["down_volume_5"],
            week["adx_14"] > 25,
        ]

        # If the long pattern matches, add the symbol to the daily screener
        if all(long_condition):
            export_list.append(
                {
                    "direction": "LONG",
                    "symbol": symbol,
                    "kk": day["High"],
                    "sl": round(day["High"] - 0.9 * day["atr_10"], 2),
                    "tp": round(day["High"] + 1.8 * day["atr_10"], 2),
                    "distance_tp_atr": round(day["atr_distance_high_8"], 1),
                    "adx_day": round(day["adx_14"]),
                    "adx_week": round(week["adx_14"]),
                    "up_volume": int(day["up_volume_5"]),
                    "down_volume": int(day["down_volume_5"]),
                }
            )

        # Pattern for short
        short_condition = [
            day["close_above_sma_200"] is False,
            day["doji"] is False,
            day["close_pct_5"] > 0,
            day["Close"] < day["Open"],
            not (
                (day["prev_Open"] > day["prev_Close"])
                and (day["prev_Low"] > day["Low"])
                and (day["prev_doji"] is False)
            ),
            day["close_pct_60"] < 0,
            day["atr_distance_low_8"] > 1.8,
            day["atr_distance_high_3"] < 1.5,
            day["up_volume_5"] < day["down_volume_5"],
            week["adx_14"] > 25,
        ]

        # If the short pattern matches, add the symbol to the daily screener
        if all(short_condition):
            export_list.append(
                {
                    "direction": "SHORT",
                    "symbol": symbol,
                    "kk": day["Low"],
                    "sl": round(day["Low"] + 0.9 * day["atr_10"], 2),
                    "tp": round(day["Low"] - 1.8 * day["atr_10"], 2),
                    "distance_tp_atr": round(day["atr_distance_low_8"], 1),
                    "adx_day": round(day["adx_14"]),
                    "adx_week": round(week["adx_14"]),
                    "up_volume": int(day["up_volume_5"]),
                    "down_volume": int(day["down_volume_5"]),
                }
            )

    print(pd.DataFrame(export_list).sort_values(by="symbol"))
    pd.DataFrame(export_list).sort_values(by="symbol").to_csv(
        f"./data/screener/{datetime.datetime.now():%Y-%m-%d}.csv", index=False
    )


if __name__ == "__main__":
    main()
