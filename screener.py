"""One-Pager of a Candle Screener"""

import datetime
import os
import pickle
from typing import Dict, List

import pandas as pd
import yfinance as yf
from pandas_ta import adx

from tools import atr, doji, resample_week, roc, sma


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
        if file_age.days > 0 or file_age.seconds / 3600 > 12:
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
                df = df[~(df.High == df.Low)]
                df = df.dropna()
                df.index = pd.to_datetime(df.index)

                if len(df) > 0:
                    dfs[symbol.lower()] = df

        # save the current stock data for later activities
        with open(filename, "wb") as file:
            pickle.dump(dfs, file)
    return dfs


def triple_witching_day() -> str:
    today = datetime.date.today()
    first_day = datetime.date(today.year, int(today.month / 4) + 3, 1)
    offset = (first_day.weekday() - 4) % 7
    last_friday = (first_day + datetime.timedelta(days=3 * 7 + offset)).strftime(
        "%Y-%m-%d"
    )
    return last_friday


def get_symbol_metadata(symbol: str) -> Dict[str, str]:
    """
    returns sector and  country of symbol

    Args:
        symbol (str): _description_

    Returns:
        sector, country: _description_
    """
    stock_metadata = yf.Ticker(symbol).info
    return {
        "sector": stock_metadata.get("sector"),
        "country": stock_metadata.get("country"),
        "industry": stock_metadata.get("industry"),
    }


def main():
    export_list = []

    # update the stock data for the screening process
    dfs = get_stocks(symbols=get_symbols())

    # update stocklist with valid symbols
    pd.DataFrame(dfs.keys(), columns=["symbol"]).to_pickle("stocks.pkl")

    for symbol in dfs.keys():
        df = dfs[symbol.lower()]["2020-01-01":].copy()
        df_week = resample_week(df.copy())

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
        df["sma_3"] = sma(df.Close, 3)
        df["sma_200"] = sma(df.Close, 200)

        df["roc_60"] = roc(df.Close, 60)
        df["roc_5"] = roc(df.Close, 5)

        df["atr_10"] = atr(df, 10, "sma")
        df["atr_10_pct"] = atr(df, 10, "sma") / df.Close
        df["atr_20_pct"] = atr(df, 20, "sma") / df.Close

        df["adx_7"] = adx(df.High, df.Low, df.Close, 7)["ADX_7"]
        df["adx_10"] = adx(df.High, df.Low, df.Close, 10)["ADX_10"]

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

        df["sma_200"] = sma(df.Close, 200) / df.Close

        df["down_volume"] = (
            df[(df.Close < df.sma_3) & (df.index != triple_witching_day())]
            .Volume.dropna()
            .rolling(5)
            .mean()
            .reindex(df.index, method="pad")
        )
        df["up_volume"] = (
            df[(df.Close > df.sma_3) & (df.index != triple_witching_day())]
            .Volume.dropna()
            .rolling(5)
            .mean()
            .reindex(df.index, method="pad")
        )

        df_week["adx_10"] = adx(df_week.High, df_week.Low, df_week.Close, 10)["ADX_10"]

        day = df.iloc[-1].to_dict()
        week = df_week.iloc[-1].to_dict()

        # Pattern for long:
        long_condition = [
            day["close_above_sma_200"] is True,
            day["doji"] is False,
            day["Close"] < day["Open"],
            not ((day["prev_High"] < day["High"]) and (day["prev_doji"] is False)),
            day["atr_distance_high_8"] > 1.8,
            day["atr_distance_low_3"] < 1.5,
            day["up_volume"] > day["down_volume"],
            day["roc_60"] > 0,
            day["roc_60"] < 150,
            day["atr_20_pct"] > 0.04,
            day["atr_20_pct"] < 0.1,
            day["roc_5"] > -15,
            day["roc_5"] < -4,
            day["adx_7"] > 20,
        ]

        # If the long pattern matches, add the symbol to the daily screener
        if all(long_condition):
            symbol_metadata = get_symbol_metadata(symbol)

            if (
                symbol_metadata["sector"] != "Real Estate"
                and symbol_metadata["country"] == "United States"
            ):
                export_list.append(
                    {
                        "direction": "LONG",
                        "symbol": symbol,
                        "signal-date": df.iloc[-1].name.strftime("%Y-%m-%d"),
                        "kk": round(day["High"] + max(0.001 * day["Low"], 0.02), 2),
                        "sl": round(day["High"] - 0.9 * day["atr_10"], 2),
                        "tp": round(day["High"] + 1.8 * day["atr_10"], 2),
                        "distance_tp_atr": round(day["atr_distance_high_8"], 1),
                        "sma_200": round(day["sma_200"], 2),
                        "adx_day": round(day["adx_10"]),
                        "adx_week": round(week["adx_10"]),
                        "up_volume": int(day["up_volume"]),
                        "down_volume": int(day["down_volume"]),
                        "industry": symbol_metadata["industry"],
                    }
                )

        # Pattern for short
        short_condition = [
            day["close_above_sma_200"] is False,
            day["doji"] is False,
            day["Close"] > day["Open"],
            not ((day["prev_Low"] > day["Low"]) and (day["prev_doji"] is False)),
            day["atr_distance_low_8"] > 1.8,
            day["atr_distance_high_3"] < 1.5,
            day["up_volume"] < day["down_volume"],
            day["roc_60"] < -1,
            day["roc_60"] > -15,
            day["atr_20_pct"] > 0.045,
            day["atr_20_pct"] < 0.085,
            day["adx_10"] < 40,
            week["adx_10"] < 45,
        ]

        # If the short pattern matches, add the symbol to the daily screener
        if all(short_condition):
            symbol_metadata = get_symbol_metadata(symbol)

            if (
                symbol_metadata["sector"] != "Real Estate"
                and symbol_metadata["country"] == "United States"
            ):
                export_list.append(
                    {
                        "direction": "SHORT",
                        "symbol": symbol,
                        "signal-date": df.iloc[-1].name.strftime("%Y-%m-%d"),
                        "kk": round(day["Low"] - max(0.001 * day["Low"], 0.02), 2),
                        "sl": round(day["Low"] + 0.9 * day["atr_10"], 2),
                        "tp": round(day["Low"] - 1.8 * day["atr_10"], 2),
                        "distance_tp_atr": round(day["atr_distance_low_8"], 1),
                        "sma_200": round(day["sma_200"], 2),
                        "adx_day": round(day["adx_10"]),
                        "adx_week": round(week["adx_10"]),
                        "up_volume": int(day["up_volume"]),
                        "down_volume": int(day["down_volume"]),
                        "industry": symbol_metadata["industry"],
                    }
                )

    if len(export_list):
        df_screener = pd.DataFrame(export_list).sort_values(by="symbol")
        df_long = df_screener[df_screener.direction == "LONG"].sort_values(
            by=["sma_200"], ascending=[False]
        )
        df_short = df_screener[df_screener.direction == "SHORT"].sort_values(
            by=["sma_200"], ascending=[False]
        )
        df_screener = pd.concat([df_long, df_short])

        print(df_screener)
        df_screener["symbol"].to_csv(
            f"./data/screener/{datetime.datetime.now():%Y-%m-%d}.txt",
            header=None,
            index=None,
            sep=" ",
            mode="a",
        )
        df_screener.to_csv(
            f"./data/screener/{datetime.datetime.now():%Y-%m-%d}.csv", index=False
        )
    else:
        print("No Trades for today!")


if __name__ == "__main__":
    main()
