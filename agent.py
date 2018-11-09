import svr
import sys
import func
import datetime
import cr_vd
import re

#コマンドラインから引数を受け取る
args = sys.argv

try:
    file_name = args[1]
except IndexError:
    print("エラー:第一引数にはデータのファイル名を入力してください")
    sys.exit()

kernel_number = 2
cross_n = 10
div = 1000

(xlist, ylist, data_dim) = func.perse_csv(file_name, div)

i=0

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

cr_vd = cr_vd.val_class(l_xlist, l_ylist, data_dim, kernel_number, cross_n)
(inst, score) = cr_vd.sol_pera()

def pred_err(x):
    if x-score > 0: return x-score
    else: return x

result_list = inst.simulate(s_xlist, pred_err)
test_list = [sum(s_ylist) / len(s_ylist)]*len(s_xlist)

main_value = func.eval_sim(result_list, s_ylist)
ref_value = func.eval_sim(test_list, s_ylist)

print("SVR's profit ->" + str(main_value))
print("Naive profit ->" + str(ref_value))

main_score = sum(s_ylist) - main_value
ref_score = sum(s_ylist) - ref_value

print("SVR's score ->" + str(main_score)+ "/" + str(sum(s_ylist)))
print("Naive score ->" + str(main_score)+ "/" + str(sum(s_ylist)))
