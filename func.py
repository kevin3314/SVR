import numpy as np
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import error
import pandas as pd

#データをスケーリングする関数。単純に[0,1]にスケーリングする
def normal_list(l):
    maxi = np.max(l)
    l = [x / maxi for x in l]
    return l

#リストを等分割する関数
def div_list(l, div_n):
    if div_n == 0:
        print("ERROR:分割する数は0より大きい必要があります")
        sys.exit()
    result_l = []
    for i in range(div_n):
        result_l.append([]) 
    for i in range(len(l)):
        index = i % div_n
        result_l[index].append(l[i]) 
    return result_l

def div_grid(larger, mini, number):
    #最大値最小値の間を等分割する。each_sub:幅
    each_sub = (larger - mini) / number
    #l:等分割した各点の値
    l = []
    for i in range(number):
        l.append(mini + each_sub * i)
    return l

#xに応じて色々なカーネルを返す関数。
def determine_kernel(x, p_dict):
    if x == 0: return dot_kernel()
    elif x == 1: return poly_kernel(p_dict)
    elif x == 2: return gauss_kernel(p_dict)
    elif x == 3: return sigmoid_kernel(p_dict)
    else:        raise error.DetermineError

#カーネルの親クラス
class kernel:
    def __init__(self):
        pass

    def set_pera(p_dict):
        #パラメタを設定し直す関数
        pass

    def return_fun(self):
        #関数を返すクラス
        pass

#内積に対応するカーネルのクラス
class dot_kernel(kernel):
    def __init__(self):
        pass

    def set_pera(p_dict):
        pass
    
    def return_fun(self):
        return np.dot

#多項式カーネルに対応するクラス
class poly_kernel(kernel):
    def __init__(self, p_dict):
        self.p1 = p_dict["p1"]

    def set_pera(p_dict):
        self.p1 = p_dict["p1"]

    def return_fun(self):
        def poly(x1, x2):
            naiseki = np.dot(x1, x2)
            return pow( 1+naiseki, self.p1)
        return poly

#ガウスカーネル
class gauss_kernel(kernel):
    def __init__(self, p_dict):
        self.p1 = p_dict["p1"]

    def set_pera(p_dict):
        self.p1 = p_dict["p1"]

    def return_fun(self):
        V = self.p1
        def gauss(x1, x2):
            t = np.dot( x1-x2, x1-x2)
            return np.exp( -1 * t * V) 
        return gauss

#シグモイドカーネル
class sigmoid_kernel(kernel):
    def __init__(self, p_dict):
        self.p1 = p_dict["p1"]
        self.p2 = p_dict["p2"]

    def set_pera(p_dict):
        self.p1 = p_dict["p1"]
        self.p2 = p_dict["p2"]

    def return_fun(self):
        def sigmoid(x1, x2):
            naiseki = np.dot(x1, x2)
            return np.tanh( self.p1 * naiseki + self.p2)
        return sigmoid

#ファイル名を受け取って、データのリストと各データの次元を返す関数
def get_datalist(file_name):
    """
    filename : データの入っているファイルの名前
    x_list : データのリスト
    y_list : 分類データのリスト
    """
    x_list = []
    y_list = []
    try:
        with open(file_name) as f:
            """
            n:データの次元数
            a:データを一時的に保存するベクトル
            """
            n = 0
            for s_line in f:
                #s_line -> [1.3, 3.3, ... , 1.4]
                s_line = s_line.split(",")
                n = len(s_line) -1
                a = np.zeros(n)
                for i, x in enumerate(s_line):
                    if i == n:
                        y_list.append(float(x))
                    else: 
                        a[i] = float(x) 
                x_list.append(a) 

    except FileNotFoundError:
        print("エラー:ファイル名が間違っています。正しいファイル名を入力してください")
        sys.exit()
    return (x_list, y_list, n)

def perse_csv(file_name):
    csv_data = pd.read_csv("sanfran.csv")
    x_list = []
    y_list = []
    #属性として指定するリスト
    taget_list = ["accommodates","availability_365", "number_of_reviews"]
    list_of_paralist = []
    for att in taget_list:
        tmp_list = [x for x in csv_data[att]]
        new_list = [x / max(tmp_list) for x in tmp_list]
        list_of_paralist.append(new_list)

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
    return (x_list[::60], y_list[::60], dim)
