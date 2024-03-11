import numpy as np
import pandas as pd


def doji(df: pd.DataFrame) -> pd.Series:
    """
    Erkennung von Dojis aufgrund folgender Bedingung:
    - Docht und Lunte jeweils doppelt so lang wie der Body
    - Farbe & Größe des Bodys spielt keine Rollte
    """

    body = abs(df.Open - df.Close)
    shadow_up = df.High - df[["Open", "Close"]].max(axis=1)
    shadow_low = df[["Open", "Close"]].min(axis=1) - df.Low

    conditions = (
        # (body / (high - low) < 0.1) &
        (shadow_up >= 2 * body) & (shadow_low >= 2 * body)
    )

    return np.where(conditions, True, False)
