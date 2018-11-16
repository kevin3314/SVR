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

try:
    kernel_number = int(args[2])
except IndexError:
    print("""エラー:第二引数にはカーネルの種類を入力してください
0:カーネルなし
1:多項式カーネル
2:ガウスカーネル
3:シグモイドカーネル""")
    sys.exit()

try:
    cross_n = args[3]
except IndexError:
    print("エラー:第三引数には分割数を入力してください")
    sys.exit()

try:
    div_n = args[4]
except IndexError:
    pass

#csvかどうかを判定
pattern = ".*\.csv"
result = re.match(pattern, file_name)

if result:
    (xlist, ylist, data_dim) = func.perse_csv(file_name)
else:
    pritn("csvファイル名を入力してください")
    sys.exit()

cr_vd = cr_vd.val_class(xlist, ylist, data_dim, kernel_number, cross_n)
cr_vd.sol_pera()
