import pandas as pd


csv_data = pd.read_csv("sanfran.csv")
for x in csv_data["accommodates"]:
    print(x)
