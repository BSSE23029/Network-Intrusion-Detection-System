import pandas as pd


def preprocess(df: pd.DataFrame, features: list):

    try:

        # keep only required columns
        missing_cols = [col for col in features if col not in df.columns]

        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")

        df = df[features]

        # clean data safely
        df = df.replace([float("inf"), float("-inf")], 0)
        df = df.fillna(0)

        return df

    except Exception as e:
        raise Exception(f"Preprocessing error: {e}")