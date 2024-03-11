import pandas as pd


def atr(df: pd.DataFrame, intervall: int = 14, smoothing: str = "sma") -> pd.Series:
    # Ref: https://stackoverflow.com/a/74282809/

    def rma(s: pd.Series, intervall: int) -> pd.Series:
        return s.ewm(
            alpha=1 / intervall, min_periods=intervall, adjust=False, ignore_na=False
        ).mean()

    def sma(s: pd.Series, intervall: int) -> pd.Series:
        return s.rolling(intervall).mean()

    def ema(s: pd.Series, intervall: int) -> pd.Series:
        return s.ewm(
            span=intervall, min_periods=10, adjust=False, ignore_na=False
        ).mean()

    high, low, prev_close = df["High"], df["Low"], df["Close"].shift()
    tr_all = [high - low, high - prev_close, low - prev_close]
    tr_all = [tr.abs() for tr in tr_all]
    tr = pd.concat(tr_all, axis=1).max(axis=1)

    if smoothing == "rma":
        return rma(tr, intervall)
    elif smoothing == "ema":
        return ema(tr, intervall)
    elif smoothing == "sma":
        return sma(tr, intervall)
    else:
        raise ValueError(f"unknown smothing type {smoothing}")


def resample_week(df: pd.DataFrame) -> pd.DataFrame:
    # expected columns Date, Open, High, Low, Close

    # TODO work with indices
    df.reset_index(inplace=True)

    try:
        df.Date = df["Date"].astype("datetime64[ns]")
    except Exception as e:
        print(df, e)

    # note calendar week and year
    df["week"] = df["Date"].dt.strftime("%y-%W")

    df = df.groupby("week").agg(
        Date=("Date", "last"),
        Low=("Low", "min"),
        High=("High", "max"),
        Open=("Open", "first"),
        Close=("Close", "last"),
        Volume=("Volume", "sum"),
    )
    df.reset_index(inplace=True)
    df.drop(columns=["week"], inplace=True)
    df.Date = pd.to_datetime(df.Date)

    return df.set_index("Date")


def sma(close: pd.Series, period: int = 200) -> pd.Series:
    return round(close.rolling(period).mean(), 2)


def roc(close: pd.Series, period: int = 10) -> pd.Series:
    return round((close - close.shift(period)) / close.shift(period) * 100)


def ema(close: pd.Series, period: int = 200) -> pd.Series:
    return round(
        close.ewm(
            span=period, min_periods=period, adjust=False, ignore_na=False
        ).mean(),
        2,
    )
