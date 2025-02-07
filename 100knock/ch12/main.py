##### 第12章 #####
### 放課後ノック111 ###

import pandas as pd
import numpy as np
import math
import hashlib
import chardet
import json
import sqlite3

# data = pd.read_excel("12-1.xlsx")
# # print(data)
# outputs = pd.DataFrame(columns=["都道府県", "市区町村", "男性", "女性"])

# prefecture = ""
# city = ""
# man_num = None
# woman_num = None
# woman_added_flag = False
# for i in range(len(data)):
#     tmp_data = data.iloc[i]
#     # print(tmp_data)
#     prefecture_tmp = tmp_data["都道府県"]
#     city_tmp = tmp_data["市区町村"]
#     human_num = tmp_data["人数（男性、女性）"]
#     # print(prefecture)
#     # print(city)
#     # print(prefecture_tmp)
#     # なければ設定
#     if prefecture == "":
#         prefecture = prefecture_tmp
#     elif pd.isna(prefecture_tmp) == False and prefecture_tmp != prefecture:
#         prefecture = prefecture_tmp

#     if city == "":
#         # 初回
#         city = city_tmp
#         man_num = human_num
#     elif pd.isna(city_tmp) == True:
#         # 女性人数を追加
#         woman_num = human_num
#         woman_added_flag = True
#     elif city_tmp != city:
#         city = city_tmp

#     if pd.isna(city_tmp) == False:
#         man_num = human_num
#     # output["都道府県"] = city
#     # print(f"output: \n{output}")


#     # output = pd.Series([prefecture, city, human_num, human_num], index=["都道府県", "市区町村", "男性", "女性"])
#     if woman_added_flag == True:
#         outputs.loc[i] = [prefecture, city, man_num, woman_num]

#     woman_added_flag = False
#     # print("==============")
#     # print(output)
#     # outputs = pd.concat([outputs, output], ignore_index=True)

#     # if city == "":
#     #     '''
#     #     .iloc[] だとコピーを返す場合があるので、データの更新は
#     #     .at[] または .loc[] を使用するのが安全
#     #     '''
#     #     # data.iloc[i]["女性"] = human_num
#     #     data.loc[i - 1, "女性"] = human_num
#     # else:
#     #     # data.iloc[i]["男性"] = human_num
#     #     data.loc[i, "男性"] = human_num

# # print(data)
# # reset_outputs = outputs.index()

# # インデックスを振り直す
# outputs = outputs.reset_index(drop=True)
# # print(outputs)
# # print(reset_outputs)
# # print(outputs.columns)


# ## 模範解答 ##
# '''
# slice[start:stop:step] の書き方で偶数行、奇数行に分割
#     [0::2] は 0始まり、stopなし、2ずつという意味 => 0から始まる偶数 => 配列上は奇数になる
# '''
# men = data['人数（男性、女性）'][0::2]
# women = data['人数（男性、女性）'][1::2]
# # print(men)
# # print(women)

# # 不要なインデックスを削除
# men.reset_index(inplace=True, drop=True)
# women.reset_index(inplace=True, drop=True)

# # 他のカラムの取得 (配列の奇数番目のみ)
# # []のインデックス指定と配列の要素の偶数奇数は反対の関係 ([0] が配列の 1番目)
# output_data = data[0::2].copy() # 偶数に見えて奇数番目をとっている
# output_data.reset_index(inplace=True, drop=True)
# output_data["男性"] = men
# output_data["女性"] = women

# # 不要となったからむ「人数（男性、女性）」を削除
# ''' df.drop()
# Args:
#     labels(単独指定 or リスト指定): 削除するラベル名 (行、列どちらのラベルでもOK)
#     axis(int):
#         0       : 行
#         1       : 列
#         デフォルト: 0
#     inplace(bool):
#         True    : 元 df を変更 (返り値: None)
#         False   : コピー先 df を変更 (返り値: df)
#         デフォルト: False
# '''
# output_data.drop("人数（男性、女性）", axis=1, inplace=True)

# # 欠損している都道府県を設定
# # 直接指定しているのであまり良い方法とは思えない
# ''' .iloc[] と .iat[] の違い

# .iloc[]
#     - 複数の行、列の指定が可能
#     - 比較的処理が遅い
# .iat[]
#     - 単一の要素の指定のみ可能
#     - 比較的処理が速い
# '''
# # print(f"修正前:\n{output_data}\n")
# # print(f"[0, 3] ... 14:\n{output_data.iloc[0, 3]}\n")
# # .iloc[1:3, 0:2] は OK
# # print(f".iloc[1:3, 0:2] ... 14:\n{output_data.iloc[1:3, 0:2]}\n")
# # .iat[1:3, 0:2] はエラーになる
# # print(f".at[1:3, 0:2] ... 14:\n{output_data.iat[1:3, 0:2]}\n")
# output_data.iat[1, 0] = output_data.iat[0, 0]
# output_data.iat[3, 0] = output_data.iat[2, 0]

# # print(output_data)

#########################################

# ### 放課後ノック 112 ###
# people = pd.read_excel("12-2.xlsx")
# print(people.head(10))
# print("============")

# people["役職"].fillna("", inplace=True)
# output_people = people.sort_values("更新日", ascending=True).drop_duplicates(["社員名", "生年月日"], keep="last")
# output_people.reset_index(inplace=True, drop=True)
# # print(unique_people.head(10))
# # print("============")
# print(output_people.head(10))

#########################################

### 放課後ノック 113 ###
# product = pd.read_excel("12-3.xlsx")
# print()
# # print(product.head())

# '''
# - 仕入先 (farmer)
#     - 仕入先
#     - 仕入先TEL
# - 商品 (product)
#     - 商品
#     - 販売単価
# - 入荷 (order)
#     - 商品
#     - 入荷日
# '''

# ''' df.dropna()
# - subset: NaN かどうかチェックするラベル
# - how: ラベルが全て NaN の場合に削除するかいずれかが NaN の場合に削除するか
#     - any: いずれかが NaN の場合削除 (default)
#     - all: 全てが NaN の場合削除
# '''
# product["仕入先"] = product["仕入先"].fillna(product["仕入先"][0])
# product["仕入先TEL"] = product["仕入先TEL"].fillna(product["仕入先TEL"][0])

# farmer = product.drop_duplicates(subset=["仕入先", "仕入先TEL"])
# farmer_index = []

# # ## インデックス設定①
# # for i in range(len(farmer)):
# #     farmer_index.append("F" + str(i+1))
# # farmer.index = farmer_index

# ## インデックス設定②
# # farmer.index = [f"F{i + 1}" for i in range(len(farmer))]
# # farmer["仕入先ID"] = [f"F{i + 1}" for i in range(len(farmer))]
# # 最初の列に仕入先IDを設定
# farmer.insert(0, "仕入先ID", [f"F{i + 1}" for i in range(len(farmer))])

# print()
# print(farmer)

# print()
# print(product)

# item = product[["商品", "販売単価"]]
# unique_item = item.drop_duplicates(subset="商品")
# # unique_item.index = [f"P{i + 1}" for i in range(len(unique_item))]
# unique_item.insert(0, "商品ID", [f"P{i + 1}" for i in range(len(unique_item))])
# print()
# print(unique_item)

# order_data = pd.merge(product, farmer[["仕入先ID", "仕入先"]], on="仕入先", how="left")
# order_data = pd.merge(order_data, unique_item[["商品ID", "商品"]], on="商品", how="left")
# order_data = order_data[["仕入先ID", "商品ID", "入荷日"]]
# print()
# print(order_data)

# # order_data.to_csv("./output/12-3_order.csv", index=False)


#########################################

### 放課後ノック114 ###
# input_prices = pd.read_csv("12-4.csv")
# print()
# print(input_prices.head())
# # print(f"\n{input_prices["金額"].describe().astype("int")}")

# # 第三四分位数
# q3 = input_prices["金額"].quantile(0.75)
# # print(f"\n第三四分位数: {q3}")

# output_prices = input_prices.copy()
# output_prices.loc[output_prices["金額"] > q3, "金額"] = q3
# # print(f"\n{output_prices["金額"].describe()}")

# # print()
# # print(input_prices.head(10))
# # print()
# # print(output_prices.head(10))

#########################################

### 放課後ノック115 ###
### 自前解答 ###
# input_person = pd.read_excel("12-5.xlsx")

# print(input_person[input_person["都道府県"].isna()])

# # 市区町村テーブルを作成
# # columns: "都道府県", "市区町村", "平均年齢"
# city = input_person.copy()

# # 年齢カラムを整数値に変換
# # ※ NaN とかあるとエラーするので後で整形する
# # city["年齢"] = city["年齢"].astype(int)

# # city_age_float = city.groupby("市区町村").mean("年齢")
# # print(city_age_float)
# '''
# .astype(int): 整数型に変換し、小数点以下は切り捨て
# '''
# city_age = city.groupby("市区町村").mean("年齢").astype(int)
# print()
# print(city_age)
# city_age_dict = city_age["年齢"].to_dict()
# print()
# print(city_age_dict)

# unique_city = city.drop_duplicates(subset=["都道府県", "市区町村"], inplace=False)
# print()
# # print(unique_city)

# '''
# set_index("市区町村") で「市区町村」をインデックスに指定
# => ["都道府県"] で「都道府県」カラムを取得 (インデックスとして「市区町村」情報も取れる)

# ※ set_index(): DataFrame or None を返す
# '''
# city_to_prefecture = city.dropna(subset="都道府県").set_index("市区町村")["都道府県"].to_dict()
# print()
# print(city_to_prefecture)

# # print(f"\n === 練習 === \n{city.dropna(subset="都道府県").set_index("市区町村")["都道府県"]}")


# # 都道府県が欠損しているレコードを補完
# ''' lambda
# - 無名関数を定義する書き方
# - 使い方
#     lambda 引数: 処理内容
#                  => A if 条件 else B (三項演算子)
# '''

# ''' df.apply()
# df の各行 or 各列に対して処理を行う

# Args:
#     - f(func): 関数 (lambda 記法で書くことが多い？)
#     - axis(int or str):
#         0 or "index"   : column に関数を適用
#         1 or "columns" : index に関数を適用

# city["都道府県"] には
#     レコードの「都道府県」が NaN であれば
#         レコードの「市区町村」をキーに dict から都道府県を取ってきて入れる
#         （キーで探して値がなければレコードの「都道府県」を入れる）
#     else
#         レコードの「都道府県」を入れる
# '''
# ## 都道府県の欠損値補完
# city["都道府県"] = city.apply(lambda row: city_to_prefecture.get(row["市区町村"], row["都道府県"])
#                              if pd.isna(row["都道府県"]) else row["都道府県"], axis=1)

# ## 年齢の欠損値補完
# # print(city_age_dict.get(city.loc[0, "市区町村"]))
# city["年齢"] = city.apply(lambda row: city_age_dict.get(row["市区町村"], row["年齢"]) if pd.isna(row["年齢"]) else row["年齢"], axis=1)

# ## 年齢カラムを整数値に変換し小数点以下を切り捨て
# # city["年齢"] = city["年齢"].astype(int)
# print()
# print(city)


''' df(dict).get()
- key(str): 取得したいキー名
- default(any): key で値が取れなかった場合に返す値

city.loc[0, "市区町村"] = "H市"
になるので a は "東京" になる
'''
# a = city_to_prefecture.get(city.loc[0, "市区町村"])
# print()
# print(a)

# d = {"name": ["Federer", "Nadal", "Nishikori"], "age": [41, 39, 37]}
# name = d.get("Federer", "Default") # Default
# print()
# print(name)


# nums = pd.DataFrame({
#     "A": [1, 2, 3],
#     "B": [4, 5, 6]
# })
# print()
# print(nums)
# # 各要素を2倍する
# d_nums = nums.apply(lambda x: x * 2)
# print()
# print(d_nums)

# ## lambda + 三項演算子 の練習
# ns = [13, 2, 32, 44, 25, 6, 37, 48]
# nns = lambda x: x - 10 if x >= 30 else x * 2
# print()
# print(nns(29))


# ### 模範解答 ###
# input_data = pd.read_excel("12-5.xlsx")
# output_data = input_data.copy()

# # 都道府県の欠損値を補完
# # 都道府県カラムが NaN の市区町村カラム
# target_div = output_data.loc[output_data["都道府県"].isnull(), "市区町村"]
# # print(target_div)
# ''' & ~ について
# and not との違い

# and: 2つの単純な bool を比較する
# not: 単一の bool を反転する

# &: 要素ごとの論理積 (AND) を行う
# ~: 要素ごとの論理否定 (NOT) を行う

# [使い分け]
# pandas の Series のような配列っぽいデータ型に対しては
# and や not は使えないので & や ~ を使う
# '''
# for division in target_div:
#     # 左辺: output_data の 都道府県が NaN、かつ、回してる division と市区町村が同じレコードの都道府県カラム
#     # 右辺: 「output_data の 市区町村が回してる division と一致、かつ、都道府県が NaN でない」 の unique()[0]
#     ''' pd.Series.unique()
#     指定したカラムのユニークな値を返す
#     引数は取らないので、事前にカラムを指定しておく必要がある
#     '''
#     output_data.loc[
#         (output_data["都道府県"].isnull()) & (output_data["市区町村"] == division),
#         "都道府県"
#     ] = output_data.loc[
#             (output_data["市区町村"] == division) &~ (output_data["都道府県"].isnull()),
#             "都道府県"
#         ].unique()[0]


# # 年齢の欠損値を補完
# target_age_city = output_data.loc[output_data["年齢"].isnull(), "市区町村"]
# print(target_age_city)

# for city in target_age_city:
#     output_data.loc[(output_data["年齢"].isnull()) & (output_data["市区町村"] == city), "年齢"] = math.floor(input_data.loc[input_data["市区町村"] == city, "年齢"].mean())
#     # 市区町村ごとの平均を算出
#     # input_data を使わないとデータを追加するたびに平均値が変わる
#     # m = input_data.loc[input_data["市区町村"] == city, "年齢"].mean()
#     # print(f"「{city}」の平均年齢: {m.astype(int)}")

# # 年齢を整数値(小数点以下切り捨て)に変換
# # output_data["年齢"] = output_data["年齢"].astype(int)
# print(output_data)


# ## & ~ の練習
# ''' & ~
# &: and と同じ
# ~: ビットを反転 (True/False を反転)
# '''
# # score = 80

# # if score >= 60 &~ False:
# #     print(f"点数: {score} (まあまあ)")
# # else:
# #     print("else")


# ## .unique() の練習
# # players_df = pd.DataFrame({"name": ["Federer", "Nadal", "Nishikori", "Wawrinka"], "age": [41, 39, 37, 40], "nation": ["Switzerland", "Serbia", "Japan", "Switzerland"]})
# # unique_nations = players_df["nation"].unique()
# # print()
# # print(unique_nations)



### 放課後ノック116 ###
# ''' データのスクランブル化を行う
# 機密性の高いデータをハッシュ化する
# '''

# input_person_purchase = pd.read_excel("12-6.xlsx")
# # print(input_person_purchase.head())
# output_data = input_person_purchase.copy()

# ''' hashlib
# hashlib.sha256()
#     Args:
#         string(buffer): ハッシュ化する文字列のバイト列
#                         ※ string だとエラーになるので注意
#                         ※ string.encode() すれば OK
# _Hash.hexdigest()
#     ハッシュ値を16進数文字列で返す
# '''

# output_data["氏名"] = output_data["氏名"].apply(lambda row: hashlib.sha256(row.encode()).hexdigest())

# # print(output_data)
# # print()
# # print(output_data.groupby("氏名").sum())


# ## hashlib の練習
# t = "Hello World!"
# h = hashlib.sha256(t.encode())
# d = h.hexdigest()
# print(h)
# print(d)


### 放課後ノック117 ###
# '''
# csv が UTF-8 以外の文字コードで保存されていた場合の対処

# 文字コード判定ライブラリ「chardet」を使用する
# ※ chardet: char detector の略
# '''

# input_sjis = pd.read_csv("12-7-1.csv")
# input_utf16 = pd.read_csv("12-7-2.csv")
# input_eucjp = pd.read_csv("12-7-3.csv")

# ''' with 文
# - Python の文法の1つ
# - 任意の処理の前処理、後処理を自動で行う
#     - ファイルの操作などによく使用される
#         - ファイルを開く
#         - 読み書きする
#         - ファイルを閉じる
# - open()
#     Args:
#         file: ファイル
#         mode: モード
#             "r": read (default)
#             "w": write
#             "rb": read (binary mode)
#                   テキストではなくバイトデータとして扱う
# '''

# df = pd.DataFrame()

# with open("12-7-1.csv", "rb") as file:
#     contents = file.read()
#     enc = chardet.detect(contents)["encoding"]
#     df = pd.concat([df, pd.read_csv("12-7-1.csv", encoding=enc)])

# files = ['12-7-1.csv', '12-7-2.csv', '12-7-3.csv']
# df = pd.DataFrame()

# for file in files:
#   # バイナリでファイルを開く
#   with open(file, mode='rb') as f:
#     contents = f.read()
#     enc = chardet.detect(contents)['encoding']
#     # 判明した文字コードでファイルを読み込む
#     df = pd.concat([df, pd.read_csv(file, encoding=enc)])

# print(df)



### 放課後ノック118 ###
# '''
# 2つの異なるIoTセンサーのログを取得し加工する

# 2つは別のセンサーのため取得周期にズレが生じている
# => データを加工し取得周期のズレを修正する (線形補間を利用)

# 【線形補間 (linear interpolation) とは】
# 前後の値をもとにその間にある値を線形的に補間すること
# ※ 補間: 間を埋めること

# [例]
# (x, y) = (1, 4) と (3, 8) がある時、x が 2 だったら y に 6 を入れる
# '''

# s1 = pd.read_csv("12-8-1.csv")
# s2 = pd.read_csv("12-8-2.csv")
# # print(s1.head())
# # print(s2.head())
# # print(len(s1))
# # print(len(s2))

# # df を結合
# new_s = pd.concat([s1, s2])
# # print(len(new_s))
# # print(new_s.head())
# # time_stamp の昇順でソート
# new_s.sort_values("time_stamp", inplace=True)
# # print(new_s.head())

# # 欠損値を線形補間
# new_s.interpolate(method="linear", inplace=True)

# # 1行目は前のデータがなく補間できないため 0 を入れる
# new_s.iat[0, 1] = 0
# new_s.iat[0, 2] = 0

# print(new_s.head())



# ### 放課後ノック119 ###
# '''
# csv データを JSON形式に変換する
# '''

# input_csv = pd.read_csv("12-9.csv")
# # print(input_csv.head())
# ''' df.to_json()
# df をjsonに変形し出力する

# Args:
#     path_or_buf: 出力するファイルパス
# '''
# # input_csv.to_json("12-9-output.json")

# # JSON形式のファイルを辞書形式で読み込み
# ''' json.load() と json.loads() の違い

# 共通
#     json 形式を Python の辞書形式に変換する
# json.load()
#     第一引数に大体ファイルをとるはず（str は取れない）

# json.loads()
#     第一引数に str を取る
# '''
# # with open("12-9-output.json") as f:
# #     dict_json = json.load(f)

# # print(dict_json)
# # print(dict_json["order_id"]["0"])

# ## json.load(), json.loads() の練習
# json_s = '{"order_id": {"0": 79339111, "1": 18941733, "2": 56217880}}'
# result = json.loads(json_s)
# print(result)



# ### 放課後ノック120 ###
# '''
# - SQLite を構築
# - csv を読み込み SQLite に登録する
# - DB から SQL でデータを抽出し出力
# '''
# # print(f"SQLite3 version: {sqlite3.version}")

# # DB 接続
# conn = sqlite3.connect("example.db")

# # カーソルオブジェクトを作成
# # カーソルオブジェクト: SQLクエリから結果を取得するために必要
# cursor = conn.cursor()

# # テーブルを作成
# cursor.execute('''
#     CREATE TABLE IF NOT EXISTS orders (
#         order_id TEXT PRIMARY KEY,
#         customer_id TEXT,
#         order_accept_date TEXT,
#         total_amount INTEGER
#     );
# ''')

# # 変更を保存
# conn.commit()

# input_orders = pd.read_csv("12-9.csv")
# print(input_orders.head())

# #df を sqlite3 に登録
# ''' df.to_sql()
# df に格納された情報を SQL DB に登録する

# Args:
#     name(str): テーブル名
#     con(str | Connection): sqlite3 connection
#     if_exists
#         'fail' (default) : すでにテーブルがある場合、エラーを発生させる
#         'replace'        : すでにテーブルがある場合、新しい値を登録する前にテーブルを削除する
#         'append'         : すでにテーブルがある場合、既存のテーブルに値を追加する
# '''
# input_orders.to_sql("orders", conn, if_exists="replace", index=None)

# # # テーブルの中身を確認
# # input_orders_check = pd.read_sql("SELECT * FROM orders", conn)
# # print(input_orders_check)


# # DB からデータを取得し、結果を df 型で受け取る
# sql = "SELECT * FROM orders"

# ''' pd.read_sql_query()
# SQLクエリを実行し、結果を df 型に代入する

# Args:
#     sql(str)            : SQLクエリ
#     conn(SQLConnection) : sql connection
# '''
# orders_df = pd.read_sql_query(sql, conn)

# # csv ファイルに出力
# orders_df.to_csv("12-10_output.csv", index=False)


# # 変更を保存し接続を閉じる
# conn.commit()
# conn.close()

