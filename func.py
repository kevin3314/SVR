import numpy as np
import sys
import matplotlib
import matplotlib.pyplot as plt
import error
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

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

def perse_csv(file_name, div=500):
    csv_data = pd.read_csv("sanfran.csv")
    x_list = []
    y_list = []
    #属性として指定するリスト
    #人数
    taget_list = ["accommodates"]
    #緯度・経度
    taget_list3 = ["longitude","latitude"]
    #賃貸の形態（アパート、家など)
    taget_list4 = ["property_type"]
    taget_list2 = ["room_type"]

    list_of_paralist = []
    for att in taget_list:
        tmp_list = [x for x in csv_data[att]]
        list_of_paralist.append(tmp_list)

    for att in taget_list4:
        tmp_list = [x for x in csv_data[att]]
        ap_list = []
        hou_list = []
        cond_list = []
        gest_list = []
        other_list = []
        for element in tmp_list:
            if element == "Apartment":
                ap_list.append(1)
                hou_list.append(0)
                cond_list.append(0)
                gest_list.append(0)
                other_list.append(0)
            elif element == "House":
                ap_list.append(0)
                hou_list.append(1)
                cond_list.append(0)
                gest_list.append(0)
                other_list.append(0)
            elif element == "Condominium":
                ap_list.append(0)
                hou_list.append(0)
                cond_list.append(1)
                gest_list.append(0)
                other_list.append(0)
            elif element == "Guest suite":
                ap_list.append(0)
                hou_list.append(0)
                cond_list.append(0)
                gest_list.append(1)
                other_list.append(0)
            else:
                ap_list.append(0)
                hou_list.append(0)
                cond_list.append(0)
                gest_list.append(0)
                other_list.append(1)

        list_of_paralist.append(ap_list)
        list_of_paralist.append(hou_list)
        list_of_paralist.append(cond_list)
        list_of_paralist.append(gest_list)
        list_of_paralist.append(other_list)

    for att in taget_list2:
        tmp_list = [x for x in csv_data[att]]
        entire_list = []
        pri_list = []
        other_list = []
        for element in tmp_list:
            if element == "Entire home/apt":
                entire_list.append(1)
                pri_list.append(0)
                other_list.append(0)
            elif element == "Private room":
                entire_list.append(0)
                pri_list.append(1)
                other_list.append(0)
            else:
                entire_list.append(0)
                pri_list.append(0)
                other_list.append(1)

        list_of_paralist.append(entire_list)
        list_of_paralist.append(pri_list)
        list_of_paralist.append(other_list)

    for att in taget_list3:
        tmp_list = [x for x in csv_data[att]]
        tmp_list = [abs(x) for x in tmp_list]
        mini = min(tmp_list)
        tmp_list = [x-mini for x in tmp_list]
        list_of_paralist.append(tmp_list)

    #データの次元
    dim = len(list_of_paralist)
    #データの数
    vec_n = len(list_of_paralist[0])

    #各リストについて正規化する
    for i in range(len(list_of_paralist)):
        list_of_paralist = [normal_list(l) for l in list_of_paralist]

    #リストをベクトルに変換する
    for i in range(vec_n): 
        a = np.zeros(dim)
        for j,l in enumerate(list_of_paralist):
            a[j] = float(l[i])
        x_list.append(a)

    pricelist = [x[1:] for x in csv_data["price"]]

    #料金のベクトルを得る
    for i in range(vec_n):
        a = np.zeros(1)
        a[0] = float(pricelist[i].replace(',', ''))
        y_list.append(a)
    return (x_list[0:div], y_list[0:div], dim)

def eval_sim(y_list, real_y_list):
    ans = 0
    for x,y in zip(y_list, real_y_list):
        if(x < y):
            ans += x
    return ans

def eval_sim_lists(nor_list, les_list, real_y_list):
    nor_ans = 0
    les_ans = 0
    lis_ans = 0
    for x,y,z in zip(nor_list, les_list, real_y_list):
        if x<z and x<y:
            nor_ans += x
        elif y<x and y<z:
            les_ans += y
        else:
            lis_ans += z

    return (nor_ans, les_ans, lis_ans)
