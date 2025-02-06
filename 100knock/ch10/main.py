##### 第10章 #####
### ノック91 ###
import pandas as pd
import matplotlib.pyplot as plt
import MeCab
import numpy as np

survey = pd.read_csv("survey.csv")
# # print(survey.dtypes)
# # print(len(survey))
# # print(survey.head())
# # print(survey.isna().sum())

# # 欠損値を削除
survey = survey.dropna()
# # print(survey.isna().sum())


# ### ノック92 ###
# # 不要な文字を削除
# survey["comment"] = survey["comment"].str.replace("AA", "")
# # survey["comment"] = survey["comment"].str.replace(r"(.+?)", "", regex=True)
# survey["comment"] = survey["comment"].str.replace(r"\(.+?\)", "", regex=True)
# # survey["comment"] = survey["comment"].str.replace(r"（.+?）", "", regex=True)
# survey["comment"] = survey["comment"].str.replace(r"\（.+?\）", "", regex=True)

# # print(survey.head())


# ### ノック93 ###
# # アンケートごとに文字数を数える
# survey["length"] = survey["comment"].str.len()
# print(survey.head())
# print(survey["length"].sum())

# ヒストグラムを作成
# x: 文字数, y: 度数
''' plt.hist()
- x:    X軸に設定するデータ
- bins: 分ける区間(ビン)の数
- range: (min, max) 形式。min 未満または max より大きいデータは無視される
'''
# plt.hist(survey["length"], bins=20, range=(0, 80))
# plt.hist(survey["length"])
# plt.show()


### ノック94: 形態素解析 ###
'''
MeCab を用いて形態素解析を行う
'''
tagger = MeCab.Tagger()
text = "すもももももももものうち"
# words = tagger.parse(text)
words = tagger.parse(text).splitlines()
# print(words)
# words_arr = []
# parts = ["名詞", "動詞"]
# for i in words:
#     # print("=============")
#     # print(i)
#     # print(i.split())

#     # i: 形態素解析後の1単語 (付加情報あり)

#     # ノック94
#     # if i == "EOS": continue
#     # ノック95
#     if i == "EOS" or i == "": continue

#     word_tmp = i.split()[0]

#     # ノック95
#     # "名詞-普通名詞-一般" のような文字列から "名詞" を取得
#     part = i.split()[4].split("-")[0]
#     if not (part in parts): continue

#     words_arr.append(word_tmp)
# # print(words_arr)


# ### ノック96 ###
# all_words = []
# parts = ["名詞"]
# for n in range(len(survey)):
#     text = survey["comment"].iloc[n]
#     # print("==================")
#     # print(text)
#     words = tagger.parse(text).splitlines()
#     words_arr = []
#     for i in words:
#         if i == "EOS" or i == "": continue
#         word_tmp = i.split()[0]
#         if len(i.split()) >= 4:
#             # 品詞
#             part = i.split()[4].split("-")[0]
#             if not (part in parts): continue
#             words_arr.append(word_tmp)
#     # all_words.append にすると、all_words 配列に words_arr 配列が入り 2次元になる
#     # extend は配列を展開して追加するので 1次元のまま
#     all_words.extend(words_arr)
# # print(all_words)


# # for 文の練習
# # l = ["apple", "banana", "orange", "grape"]
# # for i in range(len(l)):
# #     print(i) # 0 1 2 3
# # for i in l:
# #     print(i) # apple banana orange grape


# # 頻出単語上位5つを調べる
# # all_words_df = pd.DataFrame(all_words, columns=["words"])
# # count カラム: 全て1を入れておく
# all_words_df = pd.DataFrame({"words": all_words, "count": len(all_words)* [1]})
# # print(len(all_words_df))


# ''' groupby()
# - 同じ値を持つデータをまとめてそれぞれのデータに対して共通の操作を行う
#   => まとめた後に何か加工するのが前提なのでまとめるだけでは DataFrame 型を返さない

# Args:
#     as_index
#         False: インデックスが renge インデックスになる
#         True もしくは未設定の場合: グループラベルがインデックスになる
# '''
# # all_words_df = all_words_df.groupby("words", as_index=False).sum()
# all_words_df = all_words_df.groupby("words").sum()
# # print(all_words_df[0:10])
# print(all_words_df.head(10))
# # print(df_g_words)
# # print(len(df_g_words))
# # print(df_g_words.sum())
# # print(all_words_df.sort_values("count", ascending=False).head(20))

# # count の降順でソート
# print(all_words_df.sort_values("count", ascending=False).head(20))


# groupby() の練習
# 数値カラムなし
# d = pd.DataFrame({"fruit": ["apple", "banana", "orange", "grape", "banana"]})
# # 数値カラムあり
# d = pd.DataFrame({
#     "fruit": ["apple", "banana", "orange", "grape", "banana"],
#     "price": [100, 120, 250, 400, 150],
#     "num": [340, 135, 405, 439, 534]
# })
# print("===========")
# # print(d)
# # print("")
# print("カラム指定なし mean")
# whole_mean = d.groupby("fruit").mean()
# print(whole_mean)
# print("price 指定mean")
# price_mean = d.groupby("fruit")["price"].mean()
# print(price_mean)
# price_df = pd.DataFrame(price_mean)
# # ["price"].mean() をすると、カラムが price のみになる
# print(f"加工後df: \n{price_df.columns}")


# # どのようにグルーピングされたかを表示
# print(d.groupby("fruit").groups)

# # 平均値を表示
# # mean(), sum() などは数値カラムがない場合は処理を行わない
# print(d.groupby("fruit").mean())

# # 合計値を表示
# print(d.groupby("fruit").sum())


# ### ノック97 ###
# '''
# 今回は名刺だけ取得しているが、言語処理の際に、ノイズのような単語が含まれてしまうことがある
# => 関係のない単語を除去する
#    = 形態素解析の際に関係のない単語を数えないようにする
# '''

# stop_words = ["時"]
# all_words = []
# parts = ["名詞"]

# # ノック98
# satisfaction = []

# for n in range(len(survey)):
#     text = survey["comment"].iloc[n]
#     # print("==================")
#     # print(text)
#     words = tagger.parse(text).splitlines()
#     words_arr = []
#     for i in words:
#         if i == "EOS" or i == "": continue
#         word_tmp = i.split()[0]
#         if len(i.split()) >= 4:
#             # 品詞
#             part = i.split()[4].split("-")[0]
#             if not (part in parts): continue
#             if word_tmp in stop_words: continue
#             words_arr.append(word_tmp)

#             # ノック98
#             satisfaction.append(survey["satisfaction"].iloc[n])

#             # if n <= 10:
#             #     print("=======")
#             #     print(survey["satisfaction"])
#                 # print(survey["satisfaction"].iloc[n])

#     # all_words.append にすると、all_words 配列に words_arr 配列が入り 2次元になる
#     # extend は配列を展開して追加するので 1次元のまま
#     all_words.extend(words_arr)
# # print(all_words)

# # 頻出単語を取得
# '''
# "count": len(all_words) * [1]
#     1 が len(all_words) 個という意味
#     ※ len(all_words) == 100 なら 1 が 100個 (list)

# "count": len(all_words) * 1
#     len(all_words) × 1 という意味
#     ※ len(all_words) == 100 なら 100 (int)
# '''
# # all_words_df = pd.DataFrame({"words": all_words, "count": len(all_words) * [1]})
# # all_words_df = all_words_df.groupby("words").sum()
# # print(all_words_df.sort_values("count", ascending=False).head(20))


# ## int * int, int * [int] の練習
# # i1 = 10 * 5 # 50
# # i2 = 10 * [5] # [5, 5, 5, 5, 5, 5, 5, 5, 5, 5]
# # print(i1)
# # print(i2)


# ### ノック98 ###
# '''
# 「駅前」という単語がアンケートに多く書かれていることが分かったが、
# ポジティブな内容が書かれているのかはわからない
# => 頻出単語と顧客満足度の関係を調べる
# '''
# all_words_df = pd.DataFrame({"words": all_words, "satisfaction": satisfaction, "count": len(all_words) * [1]})
# # all_words_df = all_words_df.groupby("words").sum()
# # print(all_words_df.head(20))

# # words ごとに集計
# # satisfaction: 平均値、count: 合計値
# words_satisfaction = all_words_df.groupby("words").mean()["satisfaction"]
# # 練習
# '''
# all_words_df.groupby("words").mean("satisfaction") はダメな書き方
# => mean() は引数を取らない
# => "satisfaction" だけ平均を求めたいなら以下のように書く

# all_words_df.groupby("words")["satisfaction"].mean()
# '''
# # words_satisfaction2 = all_words_df.groupby("words").mean("satisfaction")
# words_count = all_words_df.groupby("words").sum()["count"]
# # print(f"単語ごとの満足度の平均: \n{words_satisfaction}")
# # print(f"単語ごとの出現回数: \n{words_count}")

# # all_words_df とほぼ同じ
# words_df = pd.concat([words_satisfaction, words_count], axis=1)
# # print(all_words_df.head())
# # print(words_df.head())

# '''
# count が 1 の単語は特定のコメントに引っ張られるので
# count が 3以上のデータに絞って再計算する
# '''

# words_df = words_df.loc[words_df["count"] >= 3]
# # print(words_df.head(10))

# # 顧客満足度の降順・昇順に5件並べる
# print(f"\n昇順: \n{words_df.sort_values("satisfaction").head(5)}")
# print(f"\n降順: \n{words_df.sort_values("satisfaction", ascending=False).head(5)}")


### ノック99 ###
'''
アンケートごとの特徴を表現する
    どの単語が含まれているかのみを特徴にする
    例えば「駅前に若者が集まっている」というコメントは「駅前」「若者」という名詞にフラグを立てる
'''
parts = ["名詞"]
all_words_df = pd.DataFrame()
satisfaction = []
for n in range(len(survey)):
    text = survey["comment"].iloc[n]
    words = tagger.parse(text).splitlines()
    words_df = pd.DataFrame()
    for i in words:
        if i == "EOS" or i == "": continue
        word_tmp = i.split()[0]
        if len(i.split()) >= 4:
            part = i.split()[4].split("-")[0]
            if not (part in parts): continue
            # words_df[もも] に 1 が入るはず
            words_df[word_tmp] = [1]
    ''' pd.merge() と pd.concat() の違い
    - pd.merge()
        - 共通のキーを指定し、結合する (SQL の JOIN のイメージ)
        - デフォルトでは共通の列を基準に結合する

    - pd.conncat()
        - 単純に結合する
        - 行方向、列方向に結合可能 (デフォルトは行方向)
        - SQL の UNION に近い操作
    '''
    # concat なので list.append() に近いイメージ
    all_words_df = pd.concat([all_words_df, words_df], ignore_index=True)
# print(all_words_df.head())

# 欠損値に 0 を代入
all_words_df = all_words_df.fillna(0)
# print(all_words_df.head())


### ノック100 ###
'''
類似アンケートを探してみる
=> 今回は満足度の高かった「子育て」というキーワードを含むコメント
「子育て支援が嬉しい」の類似アンケートを探す
'''

# print(survey["comment"].iloc[2])
target_text = all_words_df.iloc[2]
# print(target_text)

'''
コサイン類似度を用いて類似検索を行う
コサイン類似度
    特徴量同士の角度の近さで類似度を表す
'''

cos_sim = []
for i in range(len(all_words_df)):
    cos_text = all_words_df.iloc[i]
    ''' コサイン類似度 の求め方
    Y = A * B / |A| * |B|

    np.dot(a, b)
        a と b のドット積 (内積)
    np.linalg.norm(c)
        c のノルム (ベクトルの長さ) を計算する
        linalg: linear algebra (線形代数) の略
    '''
    # if i == 1:
    #     print("target_text: \n", target_text)
    cos = np.dot(target_text, cos_text) / (np.linalg.norm(target_text) * np.linalg.norm(cos_text))
    cos_sim.append(cos)
all_words_df["cos_sim"] = cos_sim
print(all_words_df.sort_values("cos_sim", ascending=False).head())

# 類似度が高いコメントを表示
print(survey["comment"].iloc[2])
print(survey["comment"].iloc[24])
print(survey["comment"].iloc[15])
print(survey["comment"].iloc[33])
print(survey["comment"].iloc[50])

# np.linalg.norm() の練習
# ベクトル (3, 4) の長さ => 5 のような計算をしてくれる
# print(np.linalg.norm([3, 4])) # 5.0
