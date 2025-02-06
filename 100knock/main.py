import pandas as pd
from dateutil.relativedelta import relativedelta
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn import linear_model
import sklearn.model_selection
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import math
import networkx as nx
import numpy as np
from itertools import product
from pulp import LpVariable, LpProblem, LpMinimize, lpSum, value
from ortoolpy import model_min, model_max, addvars, addvals, logistics_network
import cv2

print("")
print("===== Start =====")

# uriage_data = pd.read_csv("uriage_mini.csv")
# # print(uriage_data)

# # print(uriage_data["item_price"].isnull())
# flg_is_null = uriage_data["item_price"].isnull()
# # print(flg_is_null)
# '''pandas Dataframe.loc の使い方
# 抽出元のデータフレームと同じ長さ（行数）の真偽値リストを指定することで、
# Trueの行のみを抽出することができる

# ※ l = list(uriage_data.loc[]~~~~~) は
# flg_is_null 配列を指定することで、flg_is_null = True の行のみ指定している
# '''
# l = list(uriage_data.loc[flg_is_null, "item_name"].unique())
# # print(l)
# trg = uriage_data.loc[(uriage_data["item_name"] == "商品A"), "item_price"]
# # print(trg)

# # for trg in list(uriage_data.loc[flg_is_null, "item_name"].unique()):
# #     # print(trg)
# #     not_null = uriage_data.loc[flg_is_null]
# #     print(not_null)


# # data = {'Name': ['Alice', 'Bob', 'Charlie'],
# #         'Age': [25, 30, 35],
# #         'City': ['New York', 'Los Angeles', 'Chicago']}
# # df = pd.DataFrame(data, index=['a', 'b', 'c'])
# # # 行ラベル 'a' のデータを取得
# # # print(df)
# # print(df.loc['a', "Age"])


# data = {
#     "登録日": ["20230101", "2023-01-01", "abc123", "456789"],
#     "価格": [1500, "2980", "1351", 23000],
#     # "登録日": [42783, 42783, 42783, 42783]
# }
# kokyaku_data = pd.DataFrame(data)
# Dataframe.astype(type): 型を変換
# .str.isdigit(): 文字列に対して数字のみで構成されているかチェック
# price = kokyaku_data["価格"].astype(str).str.isdigit()
# print(price)
# # True のデータ数
# print(price.sum())

# unit=D: 42783 を 42783日で数える
# fromSerial = pd.to_timedelta(kokyaku_data.loc[:, "登録日"].astype("float") -2, unit="D") + pd.to_datetime("1900/1/1")
# print(fromSerial)


###### groupby #####
# g = uriage_data.groupby("item_name").count()["item_price"]
# print(g)

# ### 価格が 2000 以上のレコードのみ表示
# nums = pd.to_numeric(kokyaku_data["価格"])
# # print(nums)
# expensive = nums < 2000
# # print(expensive)
# e = kokyaku_data.loc[expensive]
# print(e)


##### 2つのデータを join #####
# sales_data = pd.read_csv("uriage.csv")
# customer_data = pd.read_excel("kokyaku_daicho.xlsx")
# # kokyaku_daicho の顧客名の揺れを解消
# customer_data["顧客名"] = customer_data["顧客名"].str.replace("　", "")
# customer_data["顧客名"] = customer_data["顧客名"].str.replace(" ", "")

# join_data = pd.merge(sales_data, customer_data, left_on="customer_name", right_on="顧客名", how="left")
# join_data = join_data.drop("customer_name", axis=1)
# print(join_data.head())
# `かな` 列が NaN でない行を抽出
# filtered_df = join_data[join_data["かな"].notna()]
# print(filtered_df)
# 浅田賢二 の行を表示
# asada = join_data.loc[join_data["customer_name"] == "浅田賢二"]
# asada_num = asada.count()["customer_name"]
# print(asada)
# print(asada_num)
# print(join_data)

# dump_data = join_data[["purchase_date", "登録日", "customer_name"]]
# print(dump_data)

##### pivot_table #####
# import_data = pd.read_csv("dump_data.csv")
# # 商品が月毎にどれくらい売れたか確かめる
# byItem = import_data.pivot_table(columns="item_name", index="purchase_month", aggfunc="size")
# print(byItem)

# byPrice = import_data.pivot_table(columns="item_name", index="purchase_month", values="item_price", aggfunc="sum")
# print(byPrice)

''' pd.Dataframe.unique()
指定したカラムの値のうち、重複を取り除いた値を配列で返す
'''
# item_name = import_data["item_name"].unique()
# # print(item_name)

# ## 誰が何を合計いくら買ったか
# ca = import_data.groupby(["顧客名", "item_name"]).sum()["item_price"]
# print(ca)

#### purchase_date を YYYYmm など任意の形式に変換する
# purchase_date = pd.to_datetime(import_data["purchase_date"]).dt.strftime("%Y/%m/%d")
# print(purchase_date)


## item_price の平均など
# p = import_data["item_price"].agg(["mean", "median", "max", "min"])
# print(p)


# weekday = pd.to_datetime(import_data["purchase_date"]).dt.weekday
# print(weekday)

'''.max()
全てのカラムに対して最大値を計算する
※ カラムが文字列の場合、辞書順での最大値になる
'''
# ma = import_data.groupby("顧客名").max()
# ma = import_data.groupby("item_name").max()
# print(ma.head())


# gr = import_data.groupby("item_name").count()
# print(gr.head())


# cheap カラムを作成し、初期は 0 に設定。item_price が 1000 以下の場合、cheap カラムを 1 に設定する
# import_data["cheap"] = 0
# print(import_data.head())
# import_data["cheap"] = import_data["cheap"].where(import_data["item_price"] >= 1000, 1)
# print(import_data.head())


# d = pd.to_datetime(import_data["purchase_date"])
# print(d)
# one_year_after = d + relativedelta(year=1)
# print(one_year_after)


''' pd.df.loc と pd.df.iloc の違い
loc: df.loc["行条件", "列名"]

iloc: df["列名"].iloc["行番号"]
'''
# import_data["one_month_after"] = None
# import_data["calc_date"] = pd.to_datetime("20190430")
# suzuki = import_data["顧客名"][1]
# import_data["登録日"] = pd.to_datetime(import_data["登録日"]).dt.strftime("%Y%m")
# for i in range(len(import_data)):
#     # 非推奨
#     # import_data["one_month_after"].loc[i] = pd.to_datetime(import_data["登録日"].loc[i]) + relativedelta(months=1)
#     # 推奨
#     import_data.loc[i, "one_month_after"] = pd.to_datetime(import_data.loc[i, "登録日"]) + relativedelta(months=1)
#     import_data["登録日"] = pd.to_datetime(import_data["登録日"])
#     delta = relativedelta(import_data.loc[i, "登録日"], import_data.loc[i, "one_month_after"])
#     print(delta)
#     # print("======================")
#     # print(import_data.loc[i, "one_month_after"])
#     # print(import_data["one_month_after"].iloc[i])
#     # print(one_month_after)
#     # import_data["one_month_after"].iloc[i] = pd.to_datetime(import_data["one_month_after"]).dt.strftime("%Y%m")
#     # print(s)
#     # import_data["one_month_after"].iloc[i] = one_month_after.dt.strftime("%Y%m")
#     # print(import_data.iloc[i]["かな"])

# import_data["one_month_after"] = pd.to_datetime(import_data["one_month_after"]).dt.strftime("%Y-%m-%d")
# print(import_data)


##### 4章 顧客の行動を予測する10本ノック #####
# customer = pd.read_csv("customer_join.csv")
# uselog = pd.read_csv("use_log.csv")

# ''' pd.df[] の使い方
# - リストや単一の列名を渡す => データフレームの列を選択
# - スライスやブール値の条件を指定 => 行を選択
# '''
# # 条件指定で行を選択
# # customer_clustering = customer[customer["customer_id"] == "OA974876"]
# # スライスで行を選択
# # customer_clustering = customer[0:10]
# # print(customer_clustering)
# customer_clustering = customer[["mean", "median", "max", "min", "membership_period"]]


# ##### K-means法 #####
# ### クラスタリング処理 ###
# '''
# mean とかは 1 ~ 8 程度だが、 membership_period は最大値が47だからこのままだと
# membership_period に引っ張られてしまうので標準化が必要
# => StandardScalar() を使う

# [標準化とは]
# - 各特徴量の値を平均0、分散1に変換する処理 (分散 = データのばらつき具合)。
# - 特徴量間のスケールの違いをなくし、クラスタリングの結果が特定の特徴量に偏らないようにする。
# '''
# sc = StandardScaler()
# # print(sc)
# customer_clustering_sc = sc.fit_transform(customer_clustering)
# # print(customer_clustering_sc)

# # sc.scale_ , sc.mean_ などは sc.fit_transform 後でないと求められないので注意
# # print("標準偏差: ", sc.scale_)
# # print("平均: ", sc.mean_)

# ''' KMeans
# - random_state: 乱数のシード値 (設定しない場合、実行ごとに異なる結果になる可能性がある)
# '''
# kmeans = KMeans(n_clusters=4, random_state=0)
# clusters = kmeans.fit(customer_clustering_sc)
# customer_clustering = customer_clustering.assign(cluster = clusters.labels_)
# # print(customer_clustering["cluster"].unique())
# # print(customer_clustering.head())

# customer_clustering.columns = ["月内平均値", "月内中央値", "月内最大値", "月内最小値", "会員期間", "cluster"]
# # print(customer_clustering.groupby("cluster").count())
# # print(customer_clustering.groupby("cluster").mean())


# ### 可視化 ###
# '''
# 今回の変数は5つ。5つの変数を2次元空間にプロットする場合、次元削除が必要。

# 次元削除: 情報をなるべく失わないように変数を削減して新しい軸を作り出す操作
# => 代表的な手法: 主成分分析 (Principal Component Analysis)
# '''
# X = customer_clustering_sc
# ''' PCA()
# n_components: 保持する主成分の数

# pca.fit(): 標準化されたデータに基づいて PCA モデルを学習。学習過程で以下が計算される
# - 主成分軸（分散が最大となる方向）を見つける。
# - 各主成分の分散や重要度（固有値）を算出する。

# pca.transform(): 計算された値を返す (ここで主成分が2つになっている)
# '''
# pca = PCA(n_components=2)
# pca.fit(X)
# x_pca = pca.transform(X)
# # print(x_pca)
# pca_df = pd.DataFrame(x_pca)
# pca_df["cluster"] = customer_clustering["cluster"]
# # print(pca_df.head())

# ## プロット
# # for i in customer_clustering["cluster"].unique():
# #     ''' pd.df.loc[] と pd.df[] の違い
# #     - pd.df.loc["行条件", "列条件"]
# #     - pd.df["列条件"]
# #     '''
# #     tmp = pca_df.loc[pca_df["cluster"] == i]
# #     # print(tmp)
# #     # scatter(): 散布図をプロット, plot(): 棒グラフや折れ線グラフをプロット
# #     plt.scatter(tmp[0], tmp[1])

# # plt.show()


# ### ノック35 ###
# ''' pd.concat()
# axis: 0: 行, 1: 列 を結合
# '''
# # customer_clustering と customer は index で紐付けされているので、concat で列の結合が可能
# customer_clustering = pd.concat([customer_clustering, customer], axis=1)
# # print(customer_clustering.head())
# # print(customer["is_deleted"].head())
# # print(customer.columns)
# # print(customer_clustering.groupby(["cluster", "is_deleted"], as_index=False).count()[["cluster", "is_deleted", "customer_id"]])

# only_member = customer_clustering[customer_clustering["is_deleted"] != 1]
# # print(only_member[["customer_id", "is_deleted"]].head())
# # print(customer_clustering.groupby(["cluster", "routine_flg"], as_index=False).count()[["cluster", "routine_flg", "customer_id"]])
# # print(only_member.groupby(["cluster", "routine_flg"], as_index=False).count()[["cluster", "routine_flg", "customer_id"]])


# ### ノック36 ###
# ## 回帰 (教師あり学習)
# # print(uselog["usedate"].head())
# uselog["usedate"] = pd.to_datetime(uselog["usedate"])
# # print(uselog["usedate"].head())
# uselog["年月"] = uselog["usedate"].dt.strftime("%Y%m")
# uselog_months = uselog.groupby(["年月", "customer_id"], as_index=False).count()
# # print(uselog_months)
# '''
# inplace=True の場合、元の df を変更する
# False の場合、コピー先の df を変更する
# '''
# uselog_months.rename(columns={"log_id":"count"}, inplace=True)
# del uselog_months["usedate"]
# # print(uselog_months.head())
# # year_months = uselog_months["年月"].unique()
# year_months = list(uselog_months["年月"].unique())
# # print(year_months)
# # print(year_months.dtype)
# predict_data = pd.DataFrame()
# # 6が start, year_month の length で stop
# # 6ヶ月目からスタート
# for i in range(6, len(year_months)):
#     tmp = uselog_months.loc[uselog_months["年月"] == year_months[i]].copy()
#     # count_pred: 今月の count (利用回数) だと思う
#     tmp.rename(columns={"count":"count_pred"}, inplace=True)
#     # 1 ~ 6 まで繰り返し (7 は入らない)
#     for j in range(1, 7):
#         # i - j = 5 になる => 前月のデータが取れる
#         # j が増えていくので i - j = 5, 4, 3, 2, 1 ... となる
#         # count_0: 当月, count_1: 1ヶ月前, count_2: 2ヶ月前 ...
#         tmp_before = uselog_months.loc[uselog_months["年月"] == year_months[i - j]].copy()
#         # print("================")
#         # print("i: ", i)
#         # print("j: ", j)
#         # print("i-j: ", i-j)
#         # print("tmp_before: ", tmp_before)
#         del tmp_before["年月"]
#         tmp_before.rename(columns={"count":"count_{}".format(j - 1)}, inplace=True)
#         tmp = pd.merge(tmp, tmp_before, on="customer_id", how="left")
#     predict_data = pd.concat([predict_data, tmp], ignore_index=True)
# # print(predict_data.head())
# # 欠損値の削除
# predict_data = predict_data.dropna()
# # print(predict_data.head())


# ### ノック37 ###
# predict_data = pd.merge(predict_data, customer[["customer_id", "start_date"]], on="customer_id", how="left")
# predict_data["now_date"] = pd.to_datetime(predict_data["年月"], format="%Y%m")
# predict_data["start_date"] = pd.to_datetime(predict_data["start_date"])
# # predict_data["period"] = predict_data["now_date"] - predict_data["start_date"]
# predict_data["period"] = None
# for i in range(len(predict_data)):
#     delta = relativedelta(predict_data.loc[i, "now_date"], predict_data.loc[i, "start_date"])
#     predict_data.loc[i, "period"] = delta.years*12 + delta.months
# # print(predict_data.head())


# ### ノック38 線形回帰モデルによる予測分析 ###
# # start_date が 20180401 より後のデータのみ使用 (古すぎると最近の使用頻度が落ち着いてしまっているから)
# predict_data = predict_data.loc[predict_data["start_date"] >= pd.to_datetime("20180401")]
# # モデルの準備
# model = linear_model.LinearRegression()
# # 説明変数を定義
# X = predict_data[["count_0", "count_1", "count_2", "count_3", "count_4", "count_5", "period"]]
# # 目的変数を定義
# y = predict_data["count_pred"]
# # 教師データ:テストデータの分割割合 => 無指定の場合 75%:25%
# # random_state: データを分割する際に用いるシード。シードが同じであれば何回実行しても同じように分割される
# X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, random_state=0)
# # モデルを使用した学習
# model.fit(X_train, y_train)
# # スコアの表示
# # print(model.score(X_train, y_train))
# # print(model.score(X_test, y_test))

# # print(predict_data[["start_date", "count_pred"]])


# ### ノック39 モデルに寄与している変数を確認しよう ###
# # DataFrame を作成 (coefficient: 係数)
# coef = pd.DataFrame({"feature_names":X.columns, "coefficient":model.coef_})
# # print(coef)


# ### ノック40 来月の利用回数を予測しよう ###
# # ["count_0", "count_1", "count_2", "count_3", "count_4", "count_5", "period"] に対応
# # # count_0: 当月, count_1: 1ヶ月前, count_2: 2ヶ月前 ...
# x1 = [3, 4, 4, 6, 8, 7, 8]
# x2 = [2, 2, 3, 3, 4, 6, 8]
# x_pred = pd.DataFrame(data=[x1, x2], columns=["count_0", "count_1", "count_2", "count_3", "count_4", "count_5", "period"])
# # 予測
# print(model.predict(x_pred))



##### 第5章 顧客の退会を予測する10本ノック #####
# ''' 使用技術
# - 教師あり学習
# - 決定木
# '''
# ### ノック41: データを読み込んで利用データを整形しよう ###
# customer = pd.read_csv("customer_join.csv")
# uselog_months = pd.read_csv("use_log_months.csv")

# year_months = list(uselog_months["年月"].unique())
# # print(year_months)
# # print(uselog_months[0])
# uselog = pd.DataFrame()
# # 当月のデータは使わないので 1 から始める
# for i in range(1, len(year_months)):
#     ''' .copy() の有無による違い
#     あり: 元のデータを意図せず変更せずに済む
#     なし: 元のデータを意図せず変更してしまう可能性がある
#     '''
#     # 年月ごとにまとめたデータ群 (当月)
#     '''
#     - year_months: [201804, 201805, 201806, 201807, 201808, 201809, 201810, 201811, 201812, 201901, 201902, 201903]
#     - uselog_months には 同一 customer_id が複数ある
#     - 1回目のループ: 年月 = 201805 のレコードを処理
#     '''
#     tmp = uselog_months.loc[uselog_months["年月"] == year_months[i]].copy()
#     # tmp = uselog_months.loc[uselog_months["年月"] == year_months[i]]
#     # tmp["count"] = tmp["count"] * 100
#     tmp.rename(columns={"count":"count_0"}, inplace=True)
#     # 1ヶ月前のデータ群
#     tmp_before = uselog_months.loc[uselog_months["年月"] == year_months[i - 1]].copy()
#     del tmp_before["年月"]
#     tmp_before.rename(columns={"count":"count_1"}, inplace=True)
#     # 当月データと前月データを customer_id で join
#     '''
#     #1: 201804, 201805 の count が入る
#     #2: 201805, 201806 の count が入る
#     => uselog には同一 customer_id が複数あり得る
#     '''
#     tmp = pd.merge(tmp, tmp_before, on="customer_id", how="left")
#     # 大元の df に結合していく (push のイメージ)
#     uselog = pd.concat([uselog, tmp], ignore_index=True)

# # print("元データ: ", uselog_months.head())
# # print(uselog.head())
# # print(uselog.iloc[[708, 719, 730, 770, 785]])



# ### ノック42 ###
# exit_customer = customer.loc[customer["is_deleted"] == 1].copy()
# exit_customer["exit_date"] = None
# exit_customer["end_date"] = pd.to_datetime(exit_customer["end_date"])
# for i in exit_customer.index:
#     # print("============")
#     # print("i: ", i)

#     ## exit_date カラムを追加し、end_date の1ヶ月前を設定
#     '''
#     exit_customer 自身に end_date の 1ヶ月前を保持する exit_date を追加
#     '''
#     exit_customer.loc[i, "exit_date"] = exit_customer.loc[i, "end_date"] - relativedelta(months=1)

# exit_customer["exit_date"] = pd.to_datetime(exit_customer["exit_date"])
# # print("-------------\ncustomer (customer_join.csv, 同一 customer_id につき1レコード)\n", customer.loc[customer["customer_id"] == "AS055680", ["customer_id", "start_date", "end_date"]])
# # print("-------------\nexit_customer (id : レコード = 1 : 1)\n", exit_customer.loc[exit_customer["customer_id"] == "AS055680", ["customer_id", "start_date", "end_date", "exit_date"]].head())

# ## exit_customer["年月"] の値を exit_date の値 (200601 形式)に更新
# ## 1:1
# exit_customer["年月"] = exit_customer["exit_date"].dt.strftime("%Y%m")
# ## 1:N
# uselog["年月"] = uselog["年月"].astype(str)
# ## 1:N
# exit_uselog = pd.merge(uselog, exit_customer, on=["customer_id", "年月"], how="left")
# # print(len(uselog))
# # print(exit_customer[["customer_id", "start_date", "end_date", "年月", "exit_date"]].head())
# # print("-------------\nexit_uselog (新規作成, 同一 customer_id につき複数レコード？)\n", exit_uselog.loc[exit_uselog["customer_id"] == "AS055680"])
# # print("-------------\nuselog (新規作成, 同一 customer_id につき複数レコード)\n", uselog.loc[uselog["customer_id"] == "AS055680"])
# # print("-------------\nexit_customer (customer_join.csv から is_deleted = 1 を抽出)\n", exit_customer.loc[exit_customer["customer_id"] == "AS055680", ["customer_id", "start_date", "end_date", "年月", "exit_date"]].head())
# # print(exit_uselog[exit_uselog["exit_date"].notna()][["customer_id", "start_date", "end_date", "年月", "exit_date"]].head())
# # a = exit_uselog.loc[exit_uselog["customer_id"] == "TS511179"] # レコードなし
# # a = exit_uselog.loc[exit_uselog["customer_id"] == "AS002855"] # レコードあり

# # name 列が NAN である場合のみ削除
# '''
# customer は end_date 以外に欠損値はない
# => 結合後、name 列が欠損値だった場合、退会前月データと結合できない不要なデータ
# => name を指定して dropna() すればいい
# '''
# exit_uselog = exit_uselog.dropna(subset=["name"])
# # print(exit_uselog[["customer_id", "start_date", "end_date", "年月", "exit_date", "count_0", "count_1"]].head())
# # print(len(exit_uselog))
# # print(len(exit_uselog["customer_id"].unique()))


# ### ノック43 ###
# conti_customer = customer.loc[customer["is_deleted"] == 0]
# '''
# uselog に is_deleted はない => conti_uselog["is_deleted"] = 0 | NaN
# ※ 値がないカラムを結合すると、NaN が入る
# ※ NaN (Not a Number), NaT (Not a Time)
# '''
# conti_uselog = pd.merge(uselog, conti_customer, on=["customer_id"], how="left")
# # print(conti_uselog[["customer_id", "is_deleted"]].head())
# # print(conti_uselog.loc[conti_uselog["is_deleted"] == 1, ["customer_id", "is_deleted"]].head())
# # print(conti_uselog.groupby("is_deleted").count()["customer_id"])
# # print(len(conti_uselog.loc[conti_uselog["is_deleted"] != 0]))
# # print(len(conti_uselog))
# conti_uselog = conti_uselog.dropna(subset=["name"])
# # print(len(conti_uselog))


# ## 退会顧客は顧客あたりデータが1件なので、継続顧客も同様に調整する (アンダーサンプリング)
# '''アンダーサンプリングとは
# 多数派クラス（データが多く含まれているクラス）のデータ数を減らして、少数派クラス（データが少ないクラス）とのバランスを取ることを目的としています。

# df.sample(): df からデータを抽出 (サンプリング) する
# - frac: サンプルする割合。1の場合、全ての行からサンプリングする

# df.reset_index(): df のインデックスをリセットする
# - drop=True: 元データのインデックスを削除する

# df.drop_duplicates()
# - subset: 重複を削除したいカラム名
#   同一カラム値が複数行にわたって存在する場合、最初の行を残し、それ以降の重複した行が削除されます。
# '''

# conti_uselog = conti_uselog.sample(frac=1, random_state=0).reset_index(drop=True)
# conti_uselog = conti_uselog.drop_duplicates(subset="customer_id")
# # print(len(conti_uselog))
# # print(conti_uselog.head())

# ''' pd.concat()
# axis : {0/'index', 1/'columns'}, default 0
# '''
# predict_data = pd.concat([conti_uselog, exit_uselog], ignore_index=True)
# # print(len(predict_data))
# # print(predict_data.head())


# ### ノック44 ###
# predict_data["period"] = 0
# predict_data["now_date"] = pd.to_datetime(predict_data["年月"], format="%Y%m")
# predict_data["start_date"] = pd.to_datetime(predict_data["start_date"])
# # print(predict_data.head())
# for i in range(len(predict_data)):
#     delta = relativedelta(predict_data.loc[i, "now_date"], predict_data.loc[i, "start_date"])
#     # print("=================")
#     # print("delta: ", delta)
#     predict_data.loc[i, "period"] = int(delta.years*12 + delta.months)
# # print(predict_data[["customer_id", "start_date", "now_date", "period"]].head())


# ### ノック45 ###
# # print("\nisna\n", predict_data.isna().sum())
# # print("\nisnull\n", predict_data.isnull().sum())

# # count_1 が欠損しているデータを削除
# predict_data = predict_data.dropna(subset=["count_1"])
# # print("\nisna\n", predict_data.isna().sum())


# ### ノック46 ###
# target_col = ["campaign_name", "class_name", "gender", "count_1", "routine_flg", "period", "is_deleted"]
# predict_data = predict_data[target_col]
# # print(predict_data.head())
# ''' 機械学習にて文字列データをどう扱うか
# - 性別などのデータを「カテゴリカル変数」と呼ぶ
#   => 機械学習に用いるときはフラグ化する。この処理を「ダミー変数化」という
# '''
# # ダミー変数を作成
# predict_data = pd.get_dummies(predict_data)
# # print(predict_data.head())
# # print(predict_data["campaign_name_入会費無料"].head())
# # print(predict_data.columns)

# # 全てのパターンでダミー変数化されているが、片方のパターンだけあればいいので不要なものを削除する
# del predict_data["campaign_name_通常"]
# del predict_data["class_name_ナイト"]
# del predict_data["gender_M"]
# # print(predict_data.head())


# ### ノック47 ###
# exit = predict_data.loc[predict_data["is_deleted"] == 1]
# # 継続顧客のデータサイズを退会顧客と合わせる
# conti = predict_data.loc[predict_data["is_deleted"] == 0].sample(len(exit), random_state=0)

# # 退会顧客と継続顧客のデータを結合 (説明変数)
# X = pd.concat([exit, conti], ignore_index=True)
# # print(X)
# # 目的変数
# y = X["is_deleted"]
# # 説明変数に is_deleted が入ってしまっているので削除
# del X["is_deleted"]
# X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, random_state=0)

# # 決定木インスタンスの作成
# model = DecisionTreeClassifier(random_state=0)
# model.fit(X_train, y_train)
# y_test_pred = model.predict(X_test)
# # print(y_test_pred)

# results_test = pd.DataFrame({"y_test":y_test, "y_pred":y_test_pred})
# # print(results_test.head())


# ### ノック48 ###
# # モデル性能の計測
# correct = len(results_test.loc[results_test["y_test"] == results_test["y_pred"]])
# data_count = len(results_test)
# score_test = correct / data_count
# # print(score_test)

# # テストデータ、教師データでの性能計測
# # print(model.score(X_test, y_test)) # 0.8916349809885932
# # print(model.score(X_train, y_train)) # 0.9759188846641318
# # 過学習気味なのでパラメータをいじる
# X = pd.concat([exit, conti], ignore_index=True)
# y = X["is_deleted"]
# del X["is_deleted"]
# X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, random_state=0)
# model = DecisionTreeClassifier(random_state=0, max_depth=5)
# model.fit(X_train, y_train)
# # print(model.score(X_test, y_test)) # 0.9201520912547528
# # print(model.score(X_train, y_train)) # 0.9252217997465145
# # 過学習気味だった傾向が改善された


# ### ノック49 ###
# importance = pd.DataFrame({"feature_names":X.columns, "coefficient":model.feature_importances_})
# # print(importance)

# '''
# plt.figure()
# - figsize=(width, height) (inches)

# tree.plot_tree(): 学習済みの決定木モデルを可視化
# - model: 学習済みの決定木モデル
# - feature_names:
#   決定木の各ノードに表示する特徴量名を指定。
#   ここでは X.columns（データフレーム X の列名）を使用して、特徴量の名前をラベルとして表示します。
# '''
# # plt.figure(figsize=(20, 8))
# # tree.plot_tree(model, feature_names=X.columns, fontsize=8)
# # plt.show()


# ### ノック50 ###
# count_1 = 10
# routine_flg = 1
# period = 1
# campaign_name = "入会費無料"
# class_name = "オールタイム"
# gender = "F"

# if campaign_name == "入会費半額":
#     campain_name_list = [1, 0]
# elif campaign_name == "入会費無料":
#     campain_name_list = [0, 1]
# elif campaign_name == "通常":
#     campain_name_list = [0, 0]

# if class_name == "オールタイム":
#     class_name_list = [1, 0]
# elif class_name == "デイタイム":
#     class_name_list = [1, 0]
# elif class_name == "ナイト":
#     class_name_list = [0, 0]

# if gender == "F":
#     gender_list = [1]
# elif gender == "M":
#     gender_list = [0]

# input_data = [count_1, routine_flg, period]
# # .extend(): リストに値を追加する
# input_data.extend(campain_name_list)
# # print(input_data)
# input_data.extend(class_name_list)
# input_data.extend(gender_list)
# input_data = pd.DataFrame(data=[input_data], columns=X.columns)
# # print(input_data)

# # sample = pd.DataFrame({"foo":["bar", "boo"], "hello": ["world", "goodbye"]})
# # print(sample)

# print(model.predict(input_data))
# # 確率を予測
# # [[0. 1.]] の場合、クラス0: 0%, クラス1: 100%
# # [[0.72222222 0.27777778]] とかにもなる
# print(model.predict_proba(input_data))




##### 第6章 物流の最適ルートをコンサルティングする10本ノック #####
### ノック51 ###
# data = pd.read_csv("./ch6_logistics/demand.csv")
# print(data)

# ''' pd.read_csv
# index_col=0: インデックスに df の 0 番目の列を指定する
# ※ 指定しない場合、0 から自動採番される
# '''
# factories = pd.read_csv("./ch6_logistics/tbl_factory.csv", index_col=0)
# # print(factories)

# warehouses = pd.read_csv("./ch6_logistics/tbl_warehouse.csv", index_col=0)
# # print(warehouses)

# cost = pd.read_csv("./ch6_logistics/rel_cost.csv", index_col=0)
# # print(cost.head())

# trans = pd.read_csv("./ch6_logistics/tbl_transaction.csv", index_col=0)
# # print(trans.head())

# ''' pd.merge()
# left_on: 左側の結合カラム名
# right_on: 右側の結合カラム名
# => 左、右のカラム名が異なる場合に使う
# '''
# join_data = pd.merge(trans, cost, left_on=["ToFC", "FromWH"], right_on=["FCID", "WHID"], how="left")
# # print(join_data.head())

# join_data = pd.merge(join_data, factories, left_on="ToFC", right_on="FCID", how="left")
# # print(join_data.head())

# join_data = pd.merge(join_data, warehouses, left_on="FromWH", right_on="WHID", how="left")
# join_data = join_data[["TransactionDate", "Quantity", "Cost", "ToFC", "FCName", "FCDemand", "FromWH", "WHName", "WHSupply", "WHRegion"]]
# # print(join_data.head())

# # print(join_data["TransactionDate"].dtype)
# # join_data["TransactionDate"] = pd.to_datetime(join_data["TransactionDate"])

# kanto = join_data.loc[join_data["WHRegion"] == "関東"]
# # print(kanto.head())
# tohoku = join_data.loc[join_data["WHRegion"] == "東北"]
# # print(tohoku.head())

# # 関東と東北のコストを比較
# kanto_cost_sum = kanto["Cost"].sum()
# tohoku_cost_sum = tohoku["Cost"].sum()
# # print("関東コスト: ", str(kanto_cost_sum))
# # print("東北コスト: ", tohoku_cost_sum)
# # 荷物1つ当たりの輸送コスト
# kanto_cost_per_baggage = kanto_cost_sum / kanto["Quantity"].sum()
# tohoku_cost_per_baggage = tohoku_cost_sum / tohoku["Quantity"].sum()
# # print(f"関東1つ当たり輸送コスト: {math.floor(kanto_cost_per_baggage * 10000)}円")
# # print(f"東北1つ当たり輸送コスト: {math.floor(tohoku_cost_per_baggage * 10000)}円")


# # 各支社の輸送コストの平均
# # print(join_data.head())
# # 各支社 (関東, 東北) のレコード数、コスト合計 が分かればいい
# kanto_num = kanto.count()["TransactionDate"]
# tohoku_num = tohoku.count()["TransactionDate"]
# # print(f"関東支社の取引数: ", kanto_num)
# # print(f"東北支社の取引数: ", tohoku_num)
# kanto_cost_mean = kanto["Cost"].mean()
# tohoku_cost_mean = tohoku["Cost"].mean()
# # print(f"関東支社の輸送コスト平均: {kanto_cost_mean}")
# # print(f"東北支社の輸送コスト平均: {tohoku_cost_mean}")

# cost_chk = pd.merge(cost, factories, on="FCID", how="left")
# # print(len(cost_chk))
# # print(cost_chk.head())
# kanto_cost_mean = cost_chk[cost_chk["FCRegion"] == "関東"]["Cost"].mean()
# tohoku_cost_mean = cost_chk[cost_chk["FCRegion"] == "東北"]["Cost"].mean()
# # print(kanto_cost_mean)
# # print(tohoku_cost_mean)


# ### ノック53: ネットワークを可視化してみよう ###
# # グラフオブジェクトの作詞
# G = nx.Graph()
# # 頂点の設定
# G.add_node("nodeA")
# G.add_node("nodeB")
# G.add_node("nodeC")
# # 辺の設定
# G.add_edge("nodeA", "nodeB")
# G.add_edge("nodeA", "nodeC")
# G.add_edge("nodeB", "nodeC")
# # 座標の設定
# pos={}
# pos["nodeA"]=(0,0)
# pos["nodeB"]=(1,1)
# pos["nodeC"]=(0,1)
# # 描画
# # nx.draw(G, pos)
# # 表示
# # plt.show()


# ### ノック54: ネットワークにノード(頂点)を追加してみよう ###
# # G.add_node("nodeD")
# # G.add_edge("nodeA", "nodeD")
# # pos["nodeD"]=(1,0)
# # nx.draw(G, pos, with_labels=True)
# # # plt.show()


# ### ノック55: ルートの重みづけを実施しよう ###
# df_w = pd.read_csv("./ch6_logistics/network_weight.csv")
# df_p = pd.read_csv("./ch6_logistics/network_pos.csv")
# # G = nx.Graph()
# # # 頂点の設定
# # ''' df_w
# # - columns: A, B, C, D, E
# # '''
# # for i in range(len(df_w.columns)):
# #     # print(df_w.loc[i])
# #     G.add_node(df_w.columns[i])

# # # 辺の設定&エッジの重みのリスト化
# # size = 10
# # edge_weights = []
# # for i in range(len(df_w.columns)):
# #     for j in range(len(df_w.columns)):
# #         '''
# #         i と j が違う場合に辺を追加する
# #         => A と B,C,D,E は辺で繋ぐ
# #         '''
# #         if not (i == j):
# #             # 辺の追加
# #             G.add_edge(df_w.columns[i], df_w.columns[j])
# #             # エッジの重みの追加
# #             # df_w.iloc[i][j]: float が入っている
# #             edge_weights.append(df_w.iloc[i][j] * size)

# # # print(edge_weights) # 同じ値が2つずつ入っているはず
# # '''
# # [
# #     1.4335300000000002, 9.44669, 5.21848, 0.0, 
# #     1.4335300000000002, 2.64556, 4.5615, 5.68434, 
# #     9.44669, 2.64556, 0.0, 6.17635, 
# #     5.21848, 4.5615, 0.0, 6.12096, 
# #     0.0, 5.68434, 6.17635, 6.12096
# # ]
# # '''

# # # 座標の設定
# # pos = {}
# # for i in range(len(df_w.columns)):
# #     # node: A ~ E
# #     node = df_w.columns[i]
# #     # pos[A] = (df_p[A][0], df_p[A][1]) => (0, 0)
# #     pos[node] = (df_p[node][0], df_p[node][1])
# #     # if node == "B":
# #     #     print(df_p[node][0])
# #     #     print(df_p[node][1])
# # # print(pos)
# # '''
# # {'A': (0, 0), 'B': (0, 2), 'C': (2, 0), 'D': (2, 2), 'E': (1, 1)}
# # '''

# # # 描画
# # # nx.draw(G, pos, with_labels=True, font_size=16, node_size=1000, node_color="k", font_color="w", width=edge_weights)
# # # plt.show()

# # # print(G.nodes) # ['A', 'B', 'C', 'D', 'E']
# # ''' G.edges
# # [
# #     ('A', 'B'), ('A', 'C'), ('A', 'D'), ('A', 'E'), 
# #     ('B', 'C'), ('B', 'D'), ('B', 'E'), 
# #     ('C', 'D'), ('C', 'E'), 
# #     ('D', 'E')]
# # '''
# # # print(G.edges)


# ### ノック56 ###
# # NOTE
# ''' networkx (nx) の使い方
# - 例えばエッジ数が3で、重みリストの要素数が12の場合、重みリストのうち先頭3つのみグラフに反映される
# - エッジリストと重みリストは順番が対応している (index 0-0, 1-1, 2-2 で対応する)
# - pos は順番とか気にしなくてよく、ノードごとに指定すればOK
# '''
# df_tr = pd.read_csv("./ch6_logistics/trans_route.csv", index_col="工場")
# df_tr_sample = pd.read_csv("./ch6_logistics/trans_route.csv")
# df_pos = pd.read_csv("./ch6_logistics/trans_route_pos.csv")
# # print(df_tr.head())


# ### ノック57 ###
# '''
# 1. W1 ~ W3 を回して F1 ~ F4 までエッジを追加する
# 2. 重み付け用のリストを作る
# 3. 1,2 で作ったエッジと、重み付け用リストを用いてグラフを作成する

# エッジ数: 12
# '''
# G = nx.Graph()
# size = 10
# edge_weights = []
# pos = {}
# # エッジの追加
# # print(df_tr.index)

# ## 改良前
# # for index, warehouse in enumerate(list(df_tr.index)):
# #     # 座標の設定
# #     pos[warehouse] = (0, 3 - index)
# #     # print(index)
# #     # print(i)
# #     # print(f"(i, index) = ({i}, {index})")
# #     for i, factory in enumerate(list(df_tr.columns)):
# #         # print(j)
# #         # 辺の追加
# #         G.add_edge(warehouse, factory)
# #         # warehouse の2週めで F, W が入れ替わっている
# #         print("=================")
# #         print(f"エッジ詳細: {G.edges}\n(エッジ数: {len(G.edges)})")

# #         # 座標の設定
# #         if index == 0:
# #             pos[factory] = (3, 4 - int(i))

# #         # エッジの重みの追加
# #         # print(f"(W, F) = ({warehouse}, {factory})")
# #         edge_weight = df_tr.loc[warehouse, factory]
# #         # print(edge_weight)
# #         # print(f"{i, j} の重み: {edge}")
# #         # df_w.iloc[i][j]: float が入っている
# #         # edge_weights.append(df_w.iloc[i][j] * size)
# #         edge_weights.append(edge_weight)


# # ## 改良後
# # for i in range(len(df_tr.index)):
# #     warehouse = df_tr.index[i] # W1, W2, W3
# #     # print(warehouse)

# #     # 座標の設定
# #     ''' df_pos
# #     W1,W2,W3,F1,F2,F3,F4
# #     0,0,0,4,4,4,4
# #     1,2,3,0.5,1.5,2.5,3.5
# #     '''
# #     pos[warehouse] = (df_pos[warehouse][0], df_pos[warehouse][1])
# #     # print(index)
# #     # print(i)
# #     # print(f"(i, index) = ({i}, {index})")
# #     for j in range(len(df_tr.columns)): # 0 1 2 3 0 1 2 3 0 1 2 3
# #         # print(j)
# #         factory = df_tr.columns[j]
# #         # print(factory)

# #         # 辺の追加
# #         G.add_edge(warehouse, factory)
# #         # # warehouse の2週めで F, W が入れ替わっている => 規則性は不明だがそういう仕様
# #         # print("=================")
# #         # print(f"エッジ詳細: {G.edges}\n(エッジ数: {len(G.edges)})")
# #         # print(f"(w, f) = ({warehouse}, {factory})")

# #         # # 座標の設定
# #         # if index == 0:
# #         #     pos[factory] = (3, 4 - int(i))

# #         # # エッジの重みの追加
# #         # # print(f"(W, F) = ({warehouse}, {factory})")
# #         # edge_weight = df_tr.loc[warehouse, factory]
# #         # # print(edge_weight)
# #         # # print(f"{i, j} の重み: {edge}")
# #         # # df_w.iloc[i][j]: float が入っている
# #         # # edge_weights.append(df_w.iloc[i][j] * size)
# #         # edge_weights.append(edge_weight)


# # # print(edge_weights)
# # '''
# # [15, 15, 0, 5, 5, 0, 30, 5, 10, 15, 2, 15]
# # '''
# # # print(pos) # W1: {} がないとダメそう
# # '''
# # {'W1': (0, 3), 'F1': (3, 4), 'F2': (3, 3), 'F3': (3, 2), 'F4': (3, 1), 'W2': (0, 2), 'W3': (0, 1)}
# # '''
# # print(f"エッジ詳細: {G.edges}\n(エッジ数: {len(G.edges)})")


# # 描画
# # nx.draw(G, pos, with_labels=True, font_size=16, node_size=1000, node_color="k", font_color="w", width=edge_weights)
# # plt.show()


# # ## 練習
# # sample_G = nx.Graph()
# # # pos = {'A': (0, 0), 'B': (0, 2), 'C': (2, 0), 'D': (2, 2), 'E': (1, 1)}
# # sample_pos = {'A': (0, 0), 'B': (0, 2), 'C': (2, 0)}
# # sample_edge_weights = [10, 5, 1, 20, 40, 60]
# # sample_G.add_edge("A", "B")
# # sample_G.add_edge("A", "C")
# # sample_G.add_edge("B", "C")
# # nx.draw(sample_G, sample_pos, with_labels=True, font_size=16, node_size=1000, node_color="k", font_color="w", width=sample_edge_weights)
# # plt.show()


# ## こたえ
# # グラフオブジェクトの作成
# G = nx.Graph()

# # 頂点の設定
# for i in range(len(df_pos.columns)):
#     G.add_node(df_pos.columns[i])

# # 辺の設定&エッジの重みのリスト化
# num_pre = 0
# edge_weights = []
# size = 0.1
# for i in range(len(df_pos.columns)):
#     for j in range(len(df_pos.columns)):
#         if not (i==j):
#             # 辺の追加
#             G.add_edge(df_pos.columns[i],df_pos.columns[j])
#             # エッジの重みの追加
#             if num_pre<len(G.edges):
#                 num_pre = len(G.edges)
#                 weight = 0
#                 if (df_pos.columns[i] in df_tr.columns)and(df_pos.columns[j] in df_tr.index):
#                     if df_tr[df_pos.columns[i]][df_pos.columns[j]]:
#                         weight = df_tr[df_pos.columns[i]][df_pos.columns[j]]*size
#                 elif(df_pos.columns[j] in df_tr.columns)and(df_pos.columns[i] in df_tr.index):
#                     if df_tr[df_pos.columns[j]][df_pos.columns[i]]:
#                         weight = df_tr[df_pos.columns[j]][df_pos.columns[i]]*size
#                 edge_weights.append(weight)
                

# # 座標の設定
# pos = {}
# for i in range(len(df_pos.columns)):
#     node = df_pos.columns[i]
#     pos[node] = (df_pos[node][0],df_pos[node][1])
    
# # # print(pos)
# # print(G.edges)
# # print(edge_weights)

# # # 描画
# # nx.draw(G, pos, with_labels=True,font_size=16, node_size = 1000, node_color='k', font_color='w', width=edge_weights)

# # # 表示
# # plt.show()


# ### ノック58 ###
# ''' やりたいこと
# 輸送コストを下げられる効率的な輸送ルートを探す
# => 輸送コストを計算する関数 (目的関数) を作成し、輸送ルートの中から目的関数を最小化する組み合わせを探す
# '''
# df_tc = pd.read_csv("./ch6_logistics/trans_cost.csv", index_col="工場")
# df_tc_sample = pd.read_csv("./ch6_logistics/trans_cost.csv")

# # 輸送コスト関数 (輸送量(trans_route) * 輸送コスト(trans_cost) を計算)
# def trans_cost(df_tr, df_tc):
#     cost = 0
#     for i in range(len(df_tc.index)): # 行数
#         # print(i)
#         for j in range(len(df_tr.columns)): # 列数
#             # cost += df_tr.iloc[i][j] * df_tc.iloc[i][j] # 非推奨な書き方
#             cost += df_tr.iloc[i, j] * df_tc.iloc[i, j]
#             # print(df_tr.iloc[i][j])
#             # print(df_tc.iloc[i][j])
#     return cost

# # print(df_tc.index)
# # print(df_tc_sample.index)
# # print(df_tc.columns)
# # print(df_tc_sample.columns)
# # for k in range(len(df_tc_sample.index)):
# #     print(k) # 0, 1, 2

# # print(df_tc.head())
# print(f"総輸送コスト: {trans_cost(df_tr, df_tc)} 万円") # 1493 万円


# ### ノック59 ###
# ''' 制約条件を作る
# 【制約条件】
# - 各倉庫には供給可能な部品数の上限がある
# - 各上場には満たすべき最低限の製品製造料がある
# ※ 倉庫 (供給) => 工場 (需要)
# '''
# df_demand = pd.read_csv("./ch6_logistics/demand.csv")
# df_supply = pd.read_csv("./ch6_logistics/supply.csv")

# # 需要側の制約条件
# for i in range(len(df_demand.columns)): # F1 ~ F4
#     temp_sum = sum(df_tr[df_demand.columns[i]]) # df["列名"] で列を取得 (行を取得したい場合は条件を指定するか、df.iloc[]などを使う)
#     # print(f"{str(df_demand.columns[i])}への輸送量: {str(temp_sum)} + (需要量: {str(df_demand.iloc[0, i])})")
#     # print(df_demand.iloc[0, i]) # F1 ~ F4 じゃなくて数字
#     # print(df_demand[df_demand.columns[i]]) # 数字
#     if temp_sum >= df_demand.iloc[0, i]:
#         print("需要量を満たしています")
#     else:
#         print("需要量を満たしていません。輸送ルートを再計算してください。")

# # 供給側の制約条件
# for i in range(len(df_supply.columns)):
#     temp_sum = sum(df_tr.loc[df_supply.columns[i]])
#     # print(df_tr.loc[df_supply.columns[i]])
#     print(f"{str(df_supply.columns[i])}からの輸送量: {str(temp_sum)} + (供給限界: {str(df_supply.iloc[0, i])})")
#     if temp_sum <= df_supply.iloc[0, i]:
#         print("供給限界の範囲内です")
#     else:
#         print("供給限界を超過しています。輸送ルートを再計算してください。")



# # print(df_tr.loc["W1"])
# # print(df_tr.loc["F1"]) # エラー
# # print(df_tr.loc[:, "F1"]) # no エラー
# # print(df_tr_sample.loc["W1"]) # エラー
# # print(df_tr_sample.loc[df_tr_sample["工場"] == "W1"]) # no エラー

# # print("")
# # print(df_tr["F2"])
# # print("")
# # print(df_tr_sample["F2"])
# # print("")
# # 1行目を取得する場合
# # print(df_tr.iloc[0])


# ### ノック60 ###
# print("")
# df_tr_new = pd.read_csv("./ch6_logistics/trans_route_new.csv", index_col="工場")
# # print(df_tr_new)

# # 総コスト計算
# print(f"総コスト(変更後): {str(trans_cost(df_tr_new, df_tc))}")
# # total_sum = sum(df_tr_new)
# # total_sum = 0
# # # print(total_sum)
# # for factory in list(df_tr_new.columns):
# #     total_sum += sum(df_tr_new[factory])
# # print(total_sum)

# # 制約条件計算関数
# # 需要側
# def condition_demand(df_tr, df_demand):
#     flag = np.zeros(len(df_demand.columns))
#     for i in range(len(df_demand.columns)):
#         temp_sum = sum(df_tr[df_demand.columns[i]])
#         # 需要量を上回っていたらフラグに 1 を設定
#         if (temp_sum >= df_demand.iloc[0, i]):
#             flag[i] = 1
#     return flag

# # 供給側
# def condition_supply(df_tr, df_supply):
#     flag = np.zeros(len(df_supply.columns))
#     for i in range(len(df_supply.columns)):
#         temp_sum = sum(df_tr.loc[df_supply.columns[i]]) # 行指定
#         # temp_sum = sum(df_tr[df_supply.columns[i]]) だと列指定
#         # 供給量を下回っていたらフラグに 1 を設定
#         if (temp_sum <= df_supply.iloc[0, i]):
#             flag[i] = 1
#     return flag

# print(f"需要条件計算結果: {str(condition_demand(df_tr_new, df_demand))}")
# print(f"供給条件計算結果: {str(condition_supply(df_tr_new, df_supply))}")
# print("")
# print(df_tr.loc[df_supply.columns[0]]) # df_tr.loc["W1"] と同じ意味
# print(df_tr.loc["W1"])
# print(df_tr["W1"])
# sample_zeros = np.zeros(5)
# print(sample_zeros.dtype) # float64


##### 第7章 #####
### ノック61 ###
# df_tc = pd.read_csv("./ch6_logistics/trans_cost.csv", index_col="工場")
# df_demand = pd.read_csv("./ch6_logistics/demand.csv")
# df_supply = pd.read_csv("./ch6_logistics/supply.csv")

# # ## 初期設定 ##
# # # 初期設定 #
# # np.random.seed(1)
# # nw = len(df_tc.index) # 3 (0 ~ 2)
# # # print(df_tc.index) # W1 ~ W3
# # nf = len(df_tc.columns) # 4 (0 ~ 3)
# # # print(nw, nf) # 3 4
# # ''' itertools.product
# # - 指定したイテラブル (リスト、タプル など) の全ての組み合わせを列挙する
# # '''
# # pr = list(product(range(nw), range(nf)))
# # # print(pr)


# # ## 数理モデル作成 ##
# # ''' ortoolpy.model_min()
# # - Google OR-Tools を簡単に使用して線形計画法や整数計画問題を解くための関数

# # '''
# # # 最小化を行うモデルを定義
# # m1 = model_min()
# # # print(m1)
# # # 辞書内包表記を使い、複数の変数を生成
# # v1 = {(i,j):LpVariable('v%d_%d'%(i,j),lowBound=0) for i,j in pr}

# # # 目的関数を m1 に定義
# # # m1 += lpSum(df_tc.iloc[i][j]*v1[i,j] for i,j in pr)
# # m1 += lpSum(df_tc.iloc[i,j]*v1[i,j] for i,j in pr)
# # # 制約条件を定義
# # for i in range(nw):
# #     # m1 += lpSum(v1[i,j] for j in range(nf)) <= df_supply.iloc[0][i]
# #     m1 += lpSum(v1[i,j] for j in range(nf)) <= df_supply.iloc[0,i]
# # for j in range(nf):
# #     # m1 += lpSum(v1[i,j] for i in range(nw)) >= df_demand.iloc[0][j]
# #     m1 += lpSum(v1[i,j] for i in range(nw)) >= df_demand.iloc[0,j]
# # m1.solve()


# # ## 辞書内包表記の練習
# # '''
# # - x: x**2 の x には for x の x が入る
# # '''
# # # squares = {x: x**2 for x in range(1, 6)}
# # # print(squares)

# # # 総輸送コスト計算 #
# # df_tr_sol = df_tc.copy()
# # total_cost = 0
# # for k,x in v1.items():
# #     i,j = k[0],k[1]
# #     # df_tr_sol.iloc[i][j] = value(x)
# #     df_tr_sol.iloc[i,j] = value(x)
# #     # total_cost += df_tc.iloc[i][j]*value(x)
# #     total_cost += df_tc.iloc[i,j]*value(x)
    
# # # print(df_tr_sol)
# # # print("総輸送コスト:"+str(total_cost))


# ### ノック62 ###
# df_tr_sol = df_tc.copy()
# df_tr = df_tr_sol.copy() # index_col = "工場"
# # print(df_tr.index)
# df_pos = pd.read_csv("./ch6_logistics/trans_route_pos.csv")

# G = nx.Graph()

# for i in range(len(df_pos.columns)):
#     G.add_node(df_pos.columns[i])

# # 辺の設定 & エッジの重みリスト化
# num_pre = 0
# edge_weights = []
# size = 0.1
# for i in range(len(df_pos.columns)):
#     for j in range(len(df_pos.columns)):
#         if not (i == j):
#             # 辺の追加
#             G.add_edge(df_pos.columns[i], df_pos.columns[j])
#             # エッジの重みの追加
#             if num_pre < len(G.edges):
#                 num_pre = len(G.edges)
#                 weight = 0
#                 # df_pos.columns[i]が F1 ~ F4 かつ df_pos.columns[j]が W1 ~ W3 の場合
#                 if (df_pos.columns[i] in df_tr.columns) and (df_pos.columns[j] in df_tr.index):
#                     if df_tr[df_pos.columns[i]][df_pos.columns[j]]:
#                         weight = df_tr[df_pos.columns[i]][df_pos.columns[j]] * size
#                 # df_pos.columns[j]が F1 ~ F4 かつ df_pos.columns[i]が W1 ~ W3 の場合
#                 elif (df_pos.columns[j] in df_tr.columns) and (df_pos.columns[i] in df_tr.index):
#                     if df_tr[df_pos.columns[j]][df_pos.columns[i]]:
#                         weight = df_tr[df_pos.columns[j]][df_pos.columns[i]] * size
#                 edge_weights.append(weight)

# # print("")
# # print(G.edges) # len = 21
# # print(edge_weights) # len = 21

# # 座標の設定
# pos = {}
# for i in range(len(df_pos.columns)):
#     node = df_pos.columns[i]
#     pos[node] = (df_pos[node][0], df_pos[node][1])

# # # 描画
# # nx.draw(G, pos, with_labels=True, font_size=16, node_size=1000, node_color="k", font_color="w", width=edge_weights)
# # # 表示
# # plt.show()




# ### ノック63 ###
# df_demand = pd.read_csv("./ch6_logistics/demand.csv")
# df_supply = pd.read_csv("./ch6_logistics/supply.csv")

# # 制約条件計算関数
# # 需要側
# def condition_demand(df_tr, df_demand):
#     flag = np.zeros(len(df_demand.columns))
#     for i in range(len(df_demand.columns)):
#         temp_sum = sum(df_tr[df_demand.columns[i]])
#         if (temp_sum >= df_demand.iloc[0, 1]):
#             flag[i] = 1
#     return flag

# # 供給側
# def condition_supply(df_tr, df_supply):
#     flag = np.zeros(len(df_supply.columns))
#     for i in range(len(df_supply.columns)):
#         temp_sum = sum(df_tr.loc[df_supply.columns[i]])
#         if (temp_sum <= df_supply.iloc[0, 1]):
#             flag[i] = 1
#     return flag

# # print("")
# # print(f"需要条件計算結果: {condition_demand(df_tr_sol, df_demand)}")
# # print(f"供給条件計算結果: {condition_supply(df_tr_sol, df_supply)}")



# ### ノック64 生産計画に関するデータを読み込んでみよう ###
# df_material = pd.read_csv("./ch7_logi_optimization/product_plan_material.csv", index_col="製品")
# # print(df_material)
# df_profit = pd.read_csv("./ch7_logi_optimization/product_plan_profit.csv", index_col="製品")
# # print(df_profit)
# df_stock = pd.read_csv("./ch7_logi_optimization/product_plan_stock.csv", index_col="項目")
# # print(df_stock)
# df_plan = pd.read_csv("./ch7_logi_optimization/product_plan.csv", index_col="製品")
# # print(df_plan)

# '''
# 現状、製品1のみ生産しているという設定
# '''

# ### ノック65 ###
# '''
# 利益計算関数を目的関数として設定し、最大化する組み合わせを求める
# '''
# def product_plan(df_profit, df_plan):
#     profit = 0
#     for i in range(len(df_profit.index)):
#         # print(i)
#         for j in range(len(df_plan.columns)):
#             # print(i, j)
#             profit += df_profit.iloc[i, j] * df_plan.iloc[i, j]
#         return profit

# # print(df_profit.iloc[0][0])
# # print(df_profit.iloc[0, 0])
# print(f"総利益: {str(product_plan(df_profit, df_plan))}")



# ### ノック66 ###
# df = df_material.copy()
# inv = df_stock

# # 最大化モデルを準備
# m = model_max()
# # lp: Linear Programming (線形計画法) の略
# v1 = {(i):LpVariable("v%d"%(i), lowBound=0) for i in range(len(df_profit))}
# print("⭐️", v1) # {0: v0, 1: v1}
# # 5.0*v1 + 4.0*v2 ▼
# m += lpSum(df_profit.iloc[i] * v1[i] for i in range(len(df_profit)))

# # 制約条件 (原料ごとの在庫を超えないようにする)
# for i in range(len(df_material.columns)):
#     # 原料1[0]*v0 + 原料1[1]*v0 + ... + 原料3[1]*v1 <= 原料ごとの在庫
#     m += lpSum(df_material.iloc[j, i] * v1[j] for j in range(len(df_profit))) <= df_stock.iloc[:, i]

# # 最適化問題を解く
# m.solve()

# df_plan_sol = df_plan.copy()
# for k, x in v1.items():
#     df_plan_sol.iloc[k] = value(x)
# print(df_plan_sol)
# print(f"総利益: {str(value(m.objective))}")


# ## lpSum() 練習 ##
# # 2x + 4y + 39 = 0 を表現する
# # x = LpVariable("x")
# # y = LpVariable("y")
# # problem = LpProblem("Example", LpMinimize)
# # problem += lpSum([2 * x, 4 * y, 39]) # 2*x + 4*y + 39
# # print(problem)

# ''' 辞書内包表記
# - for 文や if を含む for 文などを1行で簡潔に書ける方法
# '''
# # evens = [2, 4, 8, 10]
# # squared = [evens[i] ** 2 for i in range(len(evens))]
# # print("2乗: ", squared)


# # ## 製品ごとの利益を求める式を書く練習 ##
# # # 利益 = 5 * 製品1 + 4 * 製品2
# # # 製品1, 製品2にあたる変数を作る (x1, x2 みたいな)
# # xs = {i:LpVariable("x%d"%(i+1)) for i in range(2)}
# # print(xs)
# # # print(df_profit["利益"][0])
# # print(df_profit.iloc[0, 0])
# # print(df_profit.iloc[1, 0]) # 2行1列だから [1, 0] はあるけど [0, 1] はない
# # # print(df_profit)
# # # print(len(df_profit))
# # formula = model_max()
# # for i in range(len(df_profit)):
# #     print("i: ", i)
# #     formula += lpSum(df_profit.iloc[i, 0] * xs[j] for j in range(len(df_profit)))

# # print(formula)



# ### ノック67 ###
# # 制約条件計算関数
# def condition_stock(df_plan, df_material, df_stock):
#     flag = np.zeros(len(df_material.columns))
#     for i in range(len(df_material.columns)):
#         temp_sum = 0
#         for j in range(len(df_material.index)):
#             temp_sum = temp_sum + df_material.iloc[j, i] * float(df_plan.iloc[j])
#         if (temp_sum <= float(df_stock.iloc[0, i])):
#             flag[i] = 1
#         # print(f"{df_material.columns[i]} 使用料: {str(temp_sum)}, 在庫: {str(float(df_stock.iloc[0, i]))}")
#     return flag

# # print(df_material.index) # Index(['製品1', '製品2'], dtype='object', name='製品')
# # print(f"制約条件計算結果: {str(condition_stock(df_plan_sol, df_material, df_stock))}")



# ### ノック68 ###
# '''
# 目的関数: 輸送コストと製造コストの和 (最小化する)
# 制約条件: 各商店での販売数が需要数を上回ること
# '''
# 製品 = list("AB")
# 需要地 = list("PQ")
# 工場 = list("XY")
# レーン = (2, 2)

# ## 輸送費表 ##
# tbdi = pd.DataFrame(((j, k) for j in 需要地 for k in 工場), columns=["需要地", "工場"])
# tbdi["輸送費"] = [1, 2, 3, 1]
# # print(tbdi)


# ## 需要表 ##
# tbde = pd.DataFrame(((j, i) for j in 需要地 for i in 製品), columns=["需要地", "製品"])
# # print(tbde)
# tbde["需要"] = [10, 10, 20, 20]


# ## 生産表 ##
# '''
# - np.inf: 無限大を表す
# - zip(): 複数のイテラブルからペアを作成する
#          要素数が異なる場合、最短に合わせる
# '''
# tbfa = pd.DataFrame(((k, l, i, 0, np.inf) for k, nl in zip(工場, レーン) for l in range(nl) for i in 製品), columns=["工場", "レーン", "製品", "下限", "上限"])
# tbfa["生産費"] = [1, np.nan, np.nan, 1, 3, np.nan, 5, 3]
# ''' df.dropna()
# inplace
#     True: 自分自身が変更される
#     False: コピー先が変更される
# '''
# tbfa.dropna(inplace=True)
# tbfa.loc[4, "上限"] = 10
# # print(tbfa)


# # # 内包表記なしパターン
# # sample_data = []
# # for k, nl in zip(工場, レーン):
# #     for l in range(nl):
# #         for i in 製品:
# #             sample_data.append((k, l, i, 0, np.inf))
# # sample_df = pd.DataFrame(sample_data, columns=["工場", "レーン", "製品", "下限", "上限"])
# # print(sample_df)
# # # print(tbde == sample_df) # 値ごとに boolean が入った df が表示される


# _, tbdi2, _ = logistics_network(tbde, tbdi, tbfa)
# # print(tbfa)
# # print(tbdi2)


# ### ノック69 ###
# trans_cost = 0
# for i in range(len(tbdi2.index)):
#     # print(i) # 0 ~ 7
#     # 以下2つは同じ処理
#     # trans_cost += tbdi2["輸送費"].iloc[i] * tbdi2["ValX"].iloc[i]
#     trans_cost += tbdi2.loc[i, "輸送費"] * tbdi2.loc[i, "ValX"]

# # print(f"総輸送コスト: {trans_cost}") # 80
# # print(len(tbdi2.index)) # 8


# ### ノック70 ###
# product_cost = 0
# for i in range(len(tbfa.index)):
#     product_cost += tbfa["生産費"].iloc[i] * tbfa["ValY"].iloc[i]
# print(tbfa)
# print(f"総生産コスト: {product_cost}") # 120



##### 第8章 #####
### ノック71 ###
df_links = pd.read_csv("./ch8/links.csv", index_col="Node")
# print(df_links.head())

G = nx.Graph()
NUM = len(df_links.index) # 20
for i in range(NUM):
    for j in range(NUM):
        node_name = "Node" + str(j)
        if df_links[node_name].iloc[i] == 1:
            G.add_edge(str(i), str(j))
            # print(df_links.index[i], df_links.index[j])

# print(df_links.iloc[0])
''' nx.draw と nx.draw_networkx の違い
nx.draw_networkx は node の位置を指定しなくても自動で調整してくれる & node 名を表示してくれる
'''
# nx.draw_networkx(G, node_color="k", edge_color="k", font_color="w")
# plt.show()


### ノック72 ###
'''
10のつながりのうち、1つ (10%) の確率で口コミが伝播していく様子を可視化する
'''
def determine_link(percent):
    # 0 ~ 1 までの乱数生成
    rand_val = np.random.rand()
    # percent: 0.1 に設定 => 10%の確率で 1 を返す
    if rand_val <= percent:
        return 1
    else:
        return 0

def simulate_percolation(num, list_active, percent_percolation):
    '''
    Args:
        num (int)                      : ループ回数
        list_active (NDArray(float64)) : len == num の 0 配列
        percent_percolation (float)    : 伝播率

    Returns:
        NDArray(float64): list_active の 0 を伝播率に基づいて 1 に変換したもの
    '''
    for i in range(num):
        # list_active: 初回は先頭のみ 1 に設定
        if list_active[i] == 1:
            for j in range(num):
                # 初回: Node0 のみ口コミ伝播計算
                node_name = "Node" + str(j)
                if df_links[node_name].iloc[i] == 1:
                    # Nodexx の i 行目が 1 だったら伝播率に応じて 1 を設定
                    if determine_link(percent_percolation) == 1:
                        # list_active のどの要素も 1 になる可能性がある
                        list_active[j] = 1
    return list_active

percent_percolation = 0.1
T_NUM = 36 # 36ヶ月繰り返す設定
NUM = len(df_links.index) # 20
list_active = np.zeros(NUM) # 20個の 0
list_active[0] = 1
list_timeSeries = []
for t in range(T_NUM):
    list_active = simulate_percolation(NUM, list_active, percent_percolation)
    list_timeSeries.append(list_active.copy()) # 要素数はどんどん増えて次第に 1 が多くなる

# print("⭐️", len(list_timeSeries)) # 36
# print("⭐️", list_timeSeries) # 2次元配列
# print("⭐️", np.array(list_timeSeries).ndim) # 2次元配列
# print("⭐️", np.array([[[1]]]).ndim) # 3次元配列


## アクティブノード可視化 ##
def active_node_coloring(list_active):
    print("t: ", t) # 下で設定する t の値
    # print(list_timeSeries[t])
    list_color = []
    # list_timeSeries: len == 36
    for i in range(len(list_timeSeries[t])):
        if list_timeSeries[t][i] == 1:
            list_color.append("r")
        else:
            list_color.append("k")
    # print(len(list_color))
    return list_color

# t = 0
# nx.draw_networkx(G, font_color="w", node_color=active_node_coloring(list_timeSeries[t]))
# plt.show()

# t = 11
# nx.draw_networkx(G, font_color="w", node_color=active_node_coloring(list_timeSeries[t]))
# plt.show()

# t = 35
# nx.draw_networkx(G, font_color="w", node_color=active_node_coloring(list_timeSeries[t]))
# plt.show()



### ノック73 口コミ数の時系列変化をグラフ化してみよう ###
list_timeSeries_num = []
for i in range(len(list_timeSeries)): # 36
    list_timeSeries_num.append(sum(list_timeSeries[i]))

# plt.plot(list_timeSeries_num)
# plt.show()



### ノック74 ###
'''
口コミによる会員数の変化をシミュレーションする
'''

def simulate_population(num, list_active, percent_percolation, percent_disappearance, df_links):
    '''
    Args:
        num (int)                      : ループ回数
        list_active (NDArray(float64)) : len == num の 0 配列
        percent_percolation (float)    : 伝播率
        percent_disappearance (float)  : 消滅率
        df_links (pd.Dataframe)        : links.csv の内容

    Returns:
        NDArray(float64): list_active
    '''
    ## 拡散 ##
    for i in range(num):
        if list_active[i] == 1:
            for j in range(num):
                if df_links.iloc[i, j] == 1:
                    if determine_link(percent_percolation) == 1:
                        list_active[j] = 1

    ## 消滅 ##
    for i in range(num):
        if determine_link(percent_disappearance) == 1:
            list_active[i] = 0

    return list_active

# print("⭐️", df_links.iloc[0][0])

percent_percolation = 0.1
percent_disappearance = 0.05
T_NUM = 100
NUM = len(df_links.index) # 20
list_active = np.zeros(NUM)
list_active[0] = 1
list_timeSeries = []

for t in range(T_NUM):
    list_active = simulate_population(NUM, list_active, percent_percolation, percent_disappearance, df_links)
    # if t == 20:
    #     print(f"21回目の list_active: {list_active}")
    #     print(f"21回目の list_active.copy(): {list_active.copy()}")

    list_timeSeries.append(list_active.copy())

list_timeSeries_num = []
for i in range(len(list_timeSeries)):
    list_timeSeries_num.append(sum(list_timeSeries[i]))

# plt.plot(list_timeSeries_num)
# plt.show()



### ノック75 パラメータの全体像を、「相図」を見ながら把握しよう ###
# print("相図計算開始")
# T_NUM = 100
# NUM_PhaseDiagram = 20
# # 20 * 20 の 0 配列作成
# phaseDiagram = np.zeros((NUM_PhaseDiagram, NUM_PhaseDiagram))
# for i_p in range(NUM_PhaseDiagram): # 20
#     for i_d in range(NUM_PhaseDiagram): # 20
#         percent_percolation = 0.05 * i_p # 0.05ずつ増える
#         percent_disappearance = 0.05 * i_d # 0.05ずつ増える
#         list_active = np.zeros(NUM) # 100個の 0 配列
#         list_active[0] = 1
#         for t in range(T_NUM):
#             list_active = simulate_population(NUM, list_active, percent_percolation, percent_disappearance, df_links)
#         phaseDiagram[i_p][i_d] = sum(list_active)
# # print(phaseDiagram)

# plt.matshow(phaseDiagram)
# plt.colorbar(shrink=0.8)
# plt.xlabel('percent_disapparence')
# plt.ylabel('percent_percolation')
# plt.xticks(np.arange(0.0, 20.0,5), np.arange(0.0, 1.0, 0.25))
# plt.yticks(np.arange(0.0, 20.0,5), np.arange(0.0, 1.0, 0.25))
# plt.tick_params(bottom=False,
#                 left=False,
#                 right=False,
#                 top=False)
# plt.show()


### ノック76 ###
df_mem_links = pd.read_csv("./ch8/links_members.csv", index_col="Node")
df_mem_info = pd.read_csv("./ch8/info_members.csv", index_col="Node")
# print(df_mem_links.head())


### ノック77 ###
NUM = len(df_mem_links.index)
array_linkNum = np.zeros(NUM)
for i in range(NUM):
    array_linkNum[i] = sum(df_mem_links["Node" + str(i)])

# plt.hist(array_linkNum, bins=10, range=(0, 250))
# plt.show()


### ノック78 ###
# NUM = len(df_mem_info.index)
# T_NUM = len(df_mem_info.columns) - 1
# ## 消滅の確率推定 ##
# count_active = 0
# count_active_to_inactive = 0
# for t in range(NUM):
#     for i in range(NUM):
#         if (df_mem_info.iloc[i][t] == 1):
#             count_active_to_inactive += 1
#             if (df_mem_info.iloc[i][t + 1] == 0):
#                 count_active += 1
# estimated_percent_disappearance = count_active / count_active_to_inactive
# print(estimated_percent_disappearance)

# ## 拡散の確率推定 ##
# count_link = 0
# count_link_to_active = 0
# count_link_temp = 0
# for t in range(T_NUM):
#     df_link_t = df_mem_info[df_mem_info[str(t)]==1]
#     temp_flag_count = np.zeros(NUM)
#     for i in range(len(df_link_t.index)):
#         index_i = int(df_link_t.index[i].replace("Node",""))
#         df_link_temp = df_mem_links[df_mem_links["Node"+str(index_i)]==1]
#         for j in range(len(df_link_temp.index)):
#             index_j = int(df_link_temp.index[j].replace("Node",""))
#             if (df_mem_info.iloc[index_j][t]==0):
#                 if (temp_flag_count[index_j]==0):
#                     count_link += 1
#                 if (df_mem_info.iloc[index_j][t+1]==1):
#                     if (temp_flag_count[index_j]==0):
#                         temp_flag_count[index_j] = 1
#                         count_link_to_active += 1
# estimated_percent_percolation = count_link_to_active/count_link
# print(estimated_percent_percolation)



### ノック79 ###
### ノック80 ###


##### 第9章 #####
img = cv2.imread("./ch9/img/img01.jpg")
# print(img)
height, width = img.shape[:2]
print(width)
print(height)




print("===== End =====")
print("")

