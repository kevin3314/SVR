import pandas as pd
import numpy as np


csv_data = pd.read_csv("sanfran.csv")
x_list = []
y_list = []
#属性としてとる列を指定するリスト
taget_list = ["accommodates", "bathrooms", "bedrooms", "beds", "availability_365", "number_of_reviews"]
list_of_paralist = []
for att in taget_list:
    list_of_paralist.append([x for x in csv_data[att]])

dim = len(list_of_paralist)
vec_n = len(list_of_paralist[0])

for i in range(vec_n): 
    a = np.zeros(dim)
    for j,l in enumerate(list_of_paralist):
        a[j] = float(l[i])
    x_list.append(a)

pricelist = [x[1:] for x in csv_data["price"]]

for i in range(vec_n):
    a = np.zeros(1)
    a[0] = float(pricelist[i].replace(',', ''))
    y_list.append(a)
