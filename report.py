"""One-Pager of a Screener Report"""

import datetime
import glob
from typing import Dict, List

import pandas as pd
import yfinance as yf


def load_screener() -> pd.DataFrame:
    screener = []
    for filename in glob.glob("./data/screener/*.csv"):
        df = pd.read_csv(filename)
        df["date"] = datetime.datetime.strptime(
            filename.split("/")[-1].split(".")[0], "%Y-%m-%d"
        )
        screener.append(df)
    return pd.concat(screener)


def get_stocks(symbols: List[str]) -> Dict[str, pd.DataFrame]:
    """_summary_

    Args:
        symbols (List[str]): _description_

    Returns:
        Dict[str, pd.DataFrame]: _description_
    """

    dfs = {}
    stock_data = yf.download(
        symbols,
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

        dfs[symbol.lower()] = df

    return dfs


def main():
    screener = load_screener()
    dfs = get_stocks(symbols=screener.symbol.unique().tolist())

    export_list = []

    for _, row in screener.iterrows():
        df = dfs[row["symbol"]][row["date"] :]
        df = df[:5]
        trade = {}
        trade["date"] = row["date"]
        trade["symbol"] = row["symbol"]
        trade["direction"] = row["direction"]
        trade["kk"] = row["kk"]
        trade["sl"] = row["sl"]
        trade["tp"] = row["tp"]
        trade["risk"] = abs(row["kk"] - row["sl"])

        if row["direction"] == "LONG":
            if df.iloc[0].High > row["kk"]:
                trade["entry"] = max(df.iloc[0].Open, row["kk"])

                if df.High.max() > row["tp"]:
                    trade["exit"] = row["tp"]
                    trade["status"] = "TP"
                elif df.Low.min() < row["sl"]:
                    trade["exit"] = row["sl"]
                    trade["status"] = "SL"
                else:
                    trade["exit"] = df.iloc[-1].Close
                    trade["status"] = "TE"
                trade["r"] = (trade["exit"] - trade["entry"]) / trade["risk"]

        if row["direction"] == "SHORT":
            if df.iloc[0].Low < row["kk"]:
                trade["entry"] = min(df.iloc[0].Open, row["kk"])

                if df.Low.min() < row["tp"]:
                    trade["exit"] = row["tp"]
                    trade["status"] = "TP"
                elif df.High.max() > row["sl"]:
                    trade["exit"] = row["sl"]
                    trade["status"] = "SL"
                else:
                    trade["exit"] = df.iloc[-1].Close
                    trade["status"] = "TE"

                trade["r"] = (trade["entry"] - trade["exit"]) / trade["risk"]

        export_list.append(trade)

    df_report = pd.DataFrame(export_list).sort_values(by="date")
    df_report["r_sum"] = df_report["r"].cumsum()
    df_report["r_sum"] = df_report["r_sum"].round(1)
    df_report["r"] = df_report["r"].round(1)
    df_report["risk"] = df_report["risk"].round(1)

    print(df_report)
    df_report.to_csv(
        f"./data/report/{datetime.datetime.now():%Y-%m-%d}.csv", index=False
    )


if __name__ == "__main__":
    main()
