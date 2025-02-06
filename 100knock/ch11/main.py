##### 第11章 深層学習 #####
'''
手書き数字画像データ MNIST を用いて深層学習を学ぶ
'''

from tensorflow.keras import datasets, layers, models
import numpy as np
import cv2

# 学習用データ/検証用データの読み込み
mnist = datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# print(X_train.shape) # (60000, 28, 28) => 60000個で 28 * 28 px
# print(X_test.shape) # (10000, 28, 28) => 10000個で 28 * 28 px

# print(X_train[0].shape) # (28, 28)
# print(X_train[0])

# 画像の表示
# そのままだと小さいので大きくする
resized_img = cv2.resize(X_train[0], dsize=(300, 300))
# 手書き文字画像 (5) を表示
# cv2.imshow("img", X_train[0])
# cv2.imshow("img", resized_img)

''' cv.waitKey(0) の 0 の意味
待機する時間を指定する（ミリ秒）
0 : キーが押されるまで無限に待機する
5000: 5秒待機する
'''
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# 正解を表示
# print(y_train[0]) # 5



## 深層学習用にデータを変換 ##

# データを 0 から 1 の範囲に収めるために 255 で割る
X_train_sc, X_test_sc = X_train / 255.0, X_test / 255.0

# 形状を整える
X_train_sc = X_train_sc.reshape((60000, 28, 28, 1))
X_test_sc = X_test_sc.reshape((10000, 28, 28, 1))


### ノック102: 深層学習モデルを構築しよう ###

## ニューラルネットワークモデルを定義 ##
# 順番にレイヤーを追加するモデルを作成
model1 = models.Sequential()
# (28, 28) の2次元画像を1次元のベクトル(784次元)に変換
model1.add(layers.Flatten(input_shape=(28, 28)))
# 512個のニューロンを持つ全結合層を作成
# 全結合層: レイヤーに渡されたユニットが次の層の全てのユニットと結合される
# 活性化関数に ReLU を使用(勾配消失問題を抑える)
model1.add(layers.Dense(512, activation="relu"))
# 20% のニューロンをランダムに無効化 (過学習を防ぐため)
model1.add(layers.Dropout(0.2))
# 10個のニューロンを持つ全結合層を作成
# softmax を使うことで、各クラスの確率を計算し、合計が 1.0 になる
model1.add(layers.Dense(10, activation="softmax"))
# モデルのサマリーを表示
# print(model1.summary())

## CNN (畳み込みニューラルネットワーク) モデルを定義 ##
model2 = models.Sequential()
# 畳み込み層 (Convolutional Layer) を追加
# フィルター数: 32 (32個の特徴マップを作る)
# カーネルサイズ: (3, 3) (3 * 3 のフィルターを使用)
# 入力サイズ: (28, 28, 1) (28 * 28 のグレースケール画像)
# 1 は チャネル数を意味し、グレースケールの場合は 1 を設定する
model2.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)))
# 最大プーリング層を追加
# 2×2 のフィルター でプーリング
# 画像のサイズを半分に縮小 (28×28 → 14×14)
# 計算量を削減し、重要な特徴を残す。
model2.add(layers.MaxPooling2D((2, 2)))
model2.add(layers.Conv2D(64, (3, 3), activation="relu"))
model2.add(layers.MaxPooling2D((2, 2)))
model2.add(layers.Conv2D(64, (3, 3), activation="relu"))

model2.add(layers.Flatten())
model2.add(layers.Dense(64, activation="relu"))
model2.add(layers.Dense(10, activation="softmax"))
# print(model2.summary())

print(f"model1.layers ⭐️: ", {model1.layers})


# ## 準備したデータを学習させ、深層学習モデルを構築する ##
# '''
# model.compile()
#     ニューラルネットワークモデルにデータを学習させるためのアルゴリズムの指定を行う

# model.fit()
#     モデルの学習を行う
#     epochs:
#         学習を繰り返す回数
#         訓練データ全体を1回モデルに通して学習させることを「1エポック」と呼ぶ
#         epochs=10の場合、訓練データ全体を10回繰り返して学習する
#         値を増やすと学習を行うための計算数が増え、精度が上がる
#         増やすと精度が向上する可能性があるが、計算コストや過学習のリスクの増加をもたらす可能性もある
# '''
# model1.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
# model1.fit(X_train_sc, y_train, epochs=10)

# model2.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
# model2.fit(X_train_sc, y_train, epochs=10)



# ### ノック103: モデルの評価をしてみよう ###

# # モデル1 の正解率の出力
# model1_test_loss, model1_test_acc = model1.evaluate(X_test_sc, y_test)
# print(f"モデル1の正解率: {model1_test_acc}") # 0.9807999730110168

# # モデル2 の正解率の出力
# model2_test_loss, model2_test_acc = model2.evaluate(X_test_sc, y_test)
# print(f"モデル2の正解率: {model2_test_acc}") # 0.9922000169754028


# ### ノック104: モデルを使った予測をしてみよう ###

# # 予測の実行
# predictions = model2.predict(X_train_sc)
# # 予測結果の形状の出力
# print(f"モデル2の予測結果の形状: {predictions.shape}")
# print(f"モデル2の予測結果: {predictions}")
# print(f"モデル2の予測結果[0]: {predictions[0]}")
# # 最も高い確率の数字の出力
# print(f"モデル2の予測結果のうち最も高い確率の数字: {np.argmax(predictions[0])}")
# print(f"モデル2の予測結果のに対応する正解: {y_train[0]}")


### ノック105: 物体検出YOLOを使って人の検出を行なってみよう ###
# ~/Downloads/Python100knocks/yolov3-tf2 で管理
# => Google Colob で実装



