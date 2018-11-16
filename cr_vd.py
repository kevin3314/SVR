import svr
import numpy as np
import func
import sys

class val_class():
    def __init__(self, x_list, y_list, data_dim, kernel_number,cross_n):
        #初期化
        self.x_list = x_list
        self.y_list = y_list
        self.data_dim = data_dim
        self.kernel_number = kernel_number
        self.cross_n = cross_n

    def validate(self, p_dict):
        """
        交差検定を行う関数。p_dictでパラメタを指定する。
        cross_n: データを分割する個数
        div_x_list: データのリストを等分割した結果得たリストを保持するリスト
        div_y_list: 同上
        learn_x_lsit: 学習を行うために用いるリスト
        learn_y_list: 同上
        cor_per: 正解率
        cor_per_list: 正解率を保持するリスト
        average: 識別率の平均を計算するための一時変数
        """
        #x_list,y_listをそれぞれ等分割
        try:
            cross_n = int(self.cross_n)
        except ValueError:
            print("分割する値は自然数である必要があります")
            sys.exit()
        div_x_list = func.div_list(self.x_list,cross_n)
        div_y_list = func.div_list(self.y_list,cross_n)
        #学習に用いるリスト、正解率のリストを作る
        learn_x_list= []
        learn_y_list= []
        cor_per_list = []
        for i in range(cross_n):
            #i番目のリストをテスト用とし他のデータで学習する。
            for m in range(cross_n):
            #i番目以外のリストを一つのリストにまとめる
                if m != i:  
                    learn_x_list.extend(div_x_list[m])
                    learn_y_list.extend(div_y_list[m])
                else: pass
            #学習データを用いて学習を行う
            inst = svr.Svr(learn_x_list,learn_y_list, self.data_dim, self.kernel_number, p_dict)
            inst.solve()
            #精度を計算しリストに加える
            cor_per = inst.eval(div_x_list[i], div_y_list[i])
            cor_per_list.append(cor_per)
            #学習リストを初期化する
            learn_x_list = []
            learn_y_list = []
        #精度の平均を計算する
        average = 0.0
        for x in cor_per_list:
            average += x
        return (average / len(cor_per_list))


    def sol_pera(self):
        """
        パラメタを探索する関数。最後に一番よかったパラメタを用いて学習しその結果を出力する。
        """
        if self.kernel_number == 0:
            #内積の時
            p_dict = {} 
            score_dict = {}
            for i in [1, 10, 100, 1000]:
                for j in [ 0.1*x for x in range(1,10,2)]:
                    p_dict["cost"] = i
                    p_dict["epsilon"] = j
                    x1 = str(i)
                    x2 = str(j)
                    score_dict["cost-"+x1+"/epsilon-"+x2] = self.validate(p_dict)
                    
            print("最良パラメタ:->" + min(score_dict, key=score_dict.get) + "最良スコア->" + str(min(score_dict.values()) ))
            tmp_list = min(score_dict, key=score_dict.get).split("/")
            p_dict2 = {}
            for l in tmp_list:
                t_list = l.split("-")
                p_dict2[t_list[0]] = float(t_list[1])
            print(p_dict2)
            inst = svr.Svr(self.x_list,self.y_list, self.data_dim, self.kernel_number, p_dict2)
            inst.solve()
            return (inst, min(score_dict.values()))


        elif(self.kernel_number == 2 or self.kernel_number == 1):
            #ガウス,多項式カーネルの時
            p_dict = {} 
            score_dict = {}
            for i in [100, 1000]:
                for j in [0.3]:
                    for k in range(50,200,5):
                        p_dict["cost"] = i
                        p_dict["epsilon"] = j
                        p_dict["p1"] = k
                        x1 = str(i)
                        x2 = str(j)
                        x3 = str(k)
                        score_dict["cost-"+x1+"/epsilon-"+x2+"/p1-"+x3] = self.validate(p_dict)
                    
            print("最良パラメタ:->" + min(score_dict, key=score_dict.get) + "最良スコア->" + str(min(score_dict.values()) ))

            tmp_list = min(score_dict, key=score_dict.get).split("/")
            p_dict2 = {}
            for l in tmp_list:
                t_list = l.split("-")
                p_dict2[t_list[0]] = float(t_list[1])
            inst = svr.Svr(self.x_list,self.y_list, self.data_dim, self.kernel_number, p_dict2)
            inst.solve()
            
            return (inst, min(score_dict.values()))

        else:
            #シグモイドカーネルの時 
            score_dict = {}
            #それぞれのパラメタについて順に検定を行う
            for c in [1,10,100,1000]:
                for e in [0.1*x for x in range(1,10,2)]:
                    for i in range(10,15,2):
                        for j in range(10,15,2):
                            p_dict = {"p1": i, "p2": j, "cost":c, "epsilon":e}
                            score = self.validate(p_dict)
                            x1 = str(c)
                            x2 = str(e)
                            x3 = str(i)
                            x4 = str(j)
                            
                            score_dict["Cost-"+x1+"epsilon-"+x2+"p1-"+x3+"p2-"+x4] = score
            #最良スコアの値とパラメタを表示する
            print("最良パラメタ:->" + min(score_dict, key=score_dict.get) + "最良スコア->" + str(min(score_dict.values()) ))

            tmp_list = min(score_dict, key=score_dict.get).split("/")
            p_dict2 = {}
            for l in tmp_list:
                t_list = l.split("-")
                p_dict2[t_list[0]] = float(t_list[1])
            inst = svr.Svr(self.x_list,self.y_list, self.data_dim, self.kernel_number, p_dict2)
            inst.solve()
            
            return (inst, min(score_dict.values()))
