import svr
import sys
import func
import datetime

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

"""
try:
    cross_n = args[3]
except IndexError:
    print("エラー:第三引数には分割数を入力してください")
    sys.exit()
try:
    write_name = args[4]
except IndexError:
    today = datetime.datetime.today()
    day = str(today.year)+str(today.month)+str(today.day)+str(today.hour)+str(today.minute)+str(today.second)
    write_name = day
"""

(xlist, ylist, data_dim) = func.get_datalist(file_name)

svr = svr.Svr(xlist, ylist, data_dim, kernel_number, "tetete", 0.1, 1000, {})
svr.solve()
