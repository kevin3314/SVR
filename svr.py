import numpy as np
import cvxopt
import func 
from cvxopt import matrix, solvers 
import sys
import error

class Svr:
    def __init__(self, x_list, y_list, data_dim, kernel, write_name, epsilon, cost, p_dict={}): 
        """
        x_list : データのリスト
        y_list : 分類の値のリスト
        kernel_number : カーネルとして何を使うかを指定.
                 0  -> 内積
                 1  -> 多項式
                 2  -> ガウス
                 3  -> シグモイド
        kernel_class: カーネルのインスタンスを持つ。パラメタを変えられる
        kernel: カーネル関数
        N : データの数*2
        data_dim : それぞれのデータの次元
        write_name : 結果を書き出すファイル名
        epsilon : 許す誤差の幅epsilon
        cost: モデルの単純さ
        """
        self.x_list = x_list
        self.y_list = y_list
        try:
            self.kernel_class = func.determine_kernel(kernel, p_dict)
        except error.DetermineError:
            print("""エラー：第二引数の値が誤っています.
            0:カーネルなし
            1:多項式カーネル
            2:ガウスカーネル
            3:シグモイドカーネル""")
            sys.exit()            

        self.kernel = self.kernel_class.return_fun()
        self.kernel_number = kernel
        self.N = 2 * len(x_list)
        d_n = len(x_list)
        self.data_dim = data_dim
        self.write_name = write_name

        #各係数を設定する.
        self.q = np.zeros(self.N)
        for i in range(self.N):
            if i < d_n:
                self.q[i] = -y_list[i] + epsilon
            else:
                self.q[i] = y_list[i - (d_n)] + epsilon
        self.q = matrix(self.q)
        self.G = np.zeros((self.N, self.N))
        for i in range(self.N):
            for j in range(self.N):
                if i <= (d_n):
                    if i == j:
                        self.G[i,j] = -1
                else:
                    if (i-d_n) == j:
                        self.G[i,j] == 1 

        self.G = matrix(self.G)

        tmp_l1 = np.zeros((d_n))
        tmp_l2 = cost * np.ones((d_n))
        self.h = np.hstack((tmp_l1, tmp_l2))
        self.h = matrix(self.h)

        self.b = matrix(0.0)

        tmp_l1 = np.ones((1, d_n))
        tmp_l2 = -1 * np.ones((1, d_n))
        self.A = np.hstack((tmp_l1, tmp_l2))
        print(self.A)
        self.A = matrix(self.A)

        self.P = np.zeros((self.N, self.N))
        #バグあり
        for i in range(self.N):
            for j in range(self.N):
                if ( i < d_n and j < d_n or  i > d_n and j > d_n ):
                    if i < d_n:
                        self.P[i, j] = 0.5 * self.kernel( self.x_list[i], self.x_list[j])
                    else:
                        self.P[i, j] = 0.5 * self.kernel( self.x_list[i-d_n], self.x_list[j-d_n])
                else:
                    if i >= d_n and j < d_n:
                        self.P[i, j] = 0.5 * self.kernel( self.x_list[i-d_n], self.x_list[j])
                    elif i < d_n and j >= d_n:
                        self.P[i, j] = 0.5 * self.kernel( self.x_list[i], self.x_list[j-d_n])
                    else:
                        self.P[i, j] = 0.5 * self.kernel( self.x_list[i-d_n], self.x_list[j-d_n])

        self.P = matrix(self.P) 

    def set_pera(self, p_dict):
        #パラメタのハッシュを受け取ってパラメタを設定し直す。
        self.kernel_class = self.kernel_class.set_pera(p_dict)
        self.kernel = self.kernel_class.return_fun()
        self.P = np.zeros((self.N, self.N))
        for i in range(self.N):
            for j in range(self.N):
                self.P[i, j] = self.kernel( self.x_list[i], self.x_list[j]) * self.y_list[i] * self.y_list[j]
        self.P = matrix(self.P) 
        

    def solve(self):
        sol = solvers.qp(P=self.P, q=self.q, G=self.G, h=self.h, A=self.A, b=self.b) 
        #alphaのリストを作る
        alpha_list = []
        #サポートベクタの番号を覚えておく
        sup_number = 0
        for i in range(self.N):
            alpha_list.append(sol['x'][i,0]) 
            if sol['x'][i,0] > 0.1:
                sup_number = i
        self.alpha_list = alpha_list
        print(alpha_list)
        #重みを計算する
        w = np.zeros(self.data_dim)
        for i in range(d_n):
            w += self.x_list[i] * (alpha_list[i]-alpha_list[d_n+i])
        self.w = w
        #閾値を計算する
        if sup_number < d_n:
            self.shita = - self.y_list[sup_number] + epsilon  
            for i in range(d_n):
                self.shita += (alpha_list[i]-alpha_list[d_n+i]) * np.dot(x_list[sup_number], x_list[i])
        else:
            self.shita = - self.y_list[sup_number] - epsilon  
            for i in range(d_n):
                self.shita += (alpha_list[i]-alpha_list[d_n+i]) * np.dot(x_list[sup_number], x_list[i])
