import svr
import sys
import func
import datetime
import cr_vd
import re
import math

#コマンドラインから引数を受け取る
args = sys.argv

try:
    file_name = args[1]
except IndexError:
    print("エラー:第一引数にはデータのファイル名を入力してください")
    sys.exit()

#エージェントを1体とするか、2体とするか
try:
    situation = args[2]
except IndexError:
    print("エラー：第二引数にはシミュレーションするエージェントの数を入力してください:1or2")
    sys.exit()


try:
    div = int(args[3])
except IndexError:
    div = 500


#カーネル、分割数、及びデータ数を指定する。
kernel_number = 0
cross_n = 10

#データをパースする
(xlist, ylist, data_dim) = func.perse_csv(file_name, div)

i=0

#学習用のデータセットとシミュレーション用のデータセットに分ける
l_xlist = []
l_ylist = []
s_xlist = []
s_ylist = []

for x, y in zip(xlist, ylist):
    if i%2 == 0:
        l_xlist.append(x) 
        l_ylist.append(y)
        i+=1
    else:
        s_xlist.append(x) 
        s_ylist.append(y)
        i=0

#エージェントの数によって場合分け
if situation == "1":
    #交差検定を行い,一番よかったものと平均誤差を覚えておく
    cr_vd = cr_vd.val_class(l_xlist, l_ylist, data_dim, kernel_number, cross_n)
    (inst, score) = cr_vd.sol_pera()

    score = math.sqrt(score)

    #提示価格を決める関数。
    def pred_err(x):
        if x-score > 0: return x-score
        else: return x

    def plus_err(x):
        return x+(score/10)

    def minus_err(x):
        return x-(score/2)

    def iden(x):
        return x

    #シミュレーション用のデータについて予測値のリストを計算
    naive_list = inst.simulate(s_xlist, iden)
    ref_list = [sum(l_ylist) / len(l_ylist)]*len(s_xlist)

    #実際の利益を計算する
    naive_value = func.eval_sim(naive_list, s_ylist)
    ref_value = func.eval_sim(ref_list, s_ylist)

    print("SVR's score(naive) ->" + str(naive_value)+ "/" + str(sum(s_ylist)))
    print("Naive score ->" + str(ref_value)+ "/" + str(sum(s_ylist)))

else:
    #普通に学習するエージェント 
    cr_vd1 = cr_vd.val_class(l_xlist, l_ylist, data_dim, kernel_number, cross_n)
    (inst1, score1) = cr_vd1.sol_pera()

    div_4 = int(div/4)
    #少ないデータを用いて学習を行なうエージェント
    cr_vd2 = cr_vd.val_class(l_xlist[0:div_4], l_ylist[0:div_4], data_dim, kernel_number, cross_n)
    (inst2, score2) = cr_vd2.sol_pera()

    score1 = math.sqrt(score1)
    score2 = math.sqrt(score2)

    #提示価格を決める関数。
    def pred_err(x):
        if x-score > 0: return x-score
        else: return x

    def plus_err(x):
        return x+(score/10)

    def minus_err(x):
        return x-(score/2)

    def iden(x):
        return x

    #シミュレーション用のデータについて予測値のリストを計算
    nor_list = inst1.simulate(s_xlist, iden)
    les_list = inst2.simulate(s_xlist, iden)

    (nor_value, les_value, lis_value) = func.eval_sim_lists(nor_list, les_list, s_ylist)

    print("SVR's score(normal) ->" + str(nor_value)+ "/" + str(sum(s_ylist)))
    print("SVR's score(less-lerning) ->" + str(les_value)+ "/" + str(sum(s_ylist)))
    print("Listing score ->" + str(lis_value)+ "/" + str(sum(s_ylist)))
