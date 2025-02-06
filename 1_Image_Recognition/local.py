# from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import numpy as np
import matplotlib.pyplot as plt

# グレースケールで読み込む場合、第二引数に 0 を指定する
img = cv2.imread("sample.jpg", 0)

# 縮小
h, w = img.shape[:2]
size = (int(h / 2), int(w / 2))
# size = (200, 100)
# img_half = cv2.resize(img, size)
img_half = cv2.resize(img, (0,0), fx=0.5, fy=0.5)

# 回転
rotate = cv2.getRotationMatrix2D((w/2, h/2), 90, 1)
img_rot = cv2.warpAffine(img, rotate, (w,h))

# グレースケール
# img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# トリミング
cropped_img = img[50:200, 50:200]

# 反転
flipped_img = cv2.flip(img, 0)

# 色を反転
inverted_img = cv2.bitwise_not(img)

# 明るさ調整 (alpha: コントラスト, beta: 明るさ)
bright_img = cv2.convertScaleAbs(img, alpha=1, beta=100)

# コントラスト調整: 明るい部分をより明るく、暗い部分をより暗くする
contrast_img = cv2.convertScaleAbs(img, alpha=2, beta=0)

# # データ拡張(画像の水平移動)
# img = np.expand_dims(img, 0)
# datagen = ImageDataGenerator(width_shift_range=0.2)
# it = datagen.flow(img, batch_size=1)
# shifted_img = it.next()[0].astype("uint8")


# # 画像のリサイズ(アスペクト比保持)
# new_width = 300
# aspect_ratio = new_width / w
# print(aspect_ratio)
# new_height = int(h * aspect_ratio)
# resized_img = cv2.resize(img, (new_width, new_height))


# # グレー画像のヒストグラム表示
# ''' cv2.calcHist
# images: [画像パス]
# channels: [int]
#     チャンネル: 色のこと
#     グレースケールの場合、色が2色しかなく、channels には 0 を指定する
#     BGR の場合、 0: Blue, 1: Green, 2: Red になっている
# mask: Numpy配列 | None
#     指定すると、マスクの該当領域のピクセルだけがヒストグラム計算に使用される
#     マスクを使用しない場合は None を指定する
# histSize: [int]
#     ヒストグラムの各次元のビンの数を指定
#     ※ ビン = 区間 => いくつの区間に分けるかを指定
# ranges: [最大値, 最小値]
#     各チャンネルの値の範囲を指定
#     グレースケールの場合、[0, 256] が全範囲
# hist: Numpy配列 | None
#     計算したヒストグラムを格納するための配列を指定します。
#     省略時は新しい配列が作成されます。
# accumulate: bool
#     True にすると、現在のヒストグラム結果に対して新しい値を累積します。
#     False にすると新しい値で上書きします。
# '''
# img_gray = cv2.imread("sample.jpg", 0)
# hist = cv2.calcHist([img_gray], [0], None, [256], [0, 256])
# plt.plot(hist)
# plt.show()


# # カラー画像のヒストグラム表示
# color = ("b", "g", "r")
# for i, col in enumerate(color):
#     hist = cv2.calcHist([img], [i], None, [256], [0,256])
#     plt.plot(hist, color=col)
# plt.title("Color Histogram")
# plt.xlabel("Pixel Value")
# plt.ylabel("Frequency")
# plt.show()


# 画像のヒストグラム均等化
equalized_img = cv2.equalizeHist(img)
# hist = cv2.calcHist([equalized_img], [0], None, [256], [0, 256])
# plt.title("equalized_hist")
# plt.plot(hist)
# plt.show()
# normal_hist = cv2.calcHist([img], [0], None, [256], [0, 256])
# plt.title("normal_hist")
# plt.plot(normal_hist)
# plt.show()


# 画像の二値化
''' cv2.threshold()
src: 対象画像
thresh: 閾値 (指定された値を境にピクセル値を 0 または 255 に分ける)
maxval: 最大値。閾値を超えたピクセルがこの値に設定されます（白: 255）。
type (int): 二値化の種類を指定します。
　cv2.THRESH_BINARY は、次のルールに従ってピクセル値を設定します：
　　ピクセル値が閾値以下（127 以下）の場合: 0（黒）
　　ピクセル値が閾値より大きい（127 より大きい）場合: 255（白）
'''
ret, binary_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)


# 適応的二値化: 画像の局所的な領域ごとに異なる閾値を計算して二値化を行う
adaptive_binary_img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

# 画像のぼかし（平均値フィルタ）
blurred_img = cv2.blur(img, (10,10))

# ガウシアンフィルタの適用


# 画像を表示
# cv2.imshow("img", img)
cv2.imshow("blurred_img", blurred_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
