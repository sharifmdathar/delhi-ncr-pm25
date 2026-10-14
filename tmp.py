import pandas as pd
df = pd.read_csv("data/raw/opencity/alipur-dpcc_2024-25.csv",
                 usecols=["Timestamp", "PM2.5 (µg/m³)"])
ts = pd.to_datetime(df["Timestamp"], utc=True)
print(df.groupby(ts.dt.hour)["PM2.5 (µg/m³)"].mean().round(0))
