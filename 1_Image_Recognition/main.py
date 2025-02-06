import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import copy

print("Start ⭐️")

img = cv2.imread("sample.jpg")
img_g = cv2.imread("sample.jpg", 0)

# cv2.imshow("img", img)
# # 画像のプレビュー上で何かしらのキーが押されるまで待機
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# os.mkdir("./output")

# # 画像の解像度を確認
print("img.shape: ", img.shape)

# # 画像のリサイズの解像度を入力
# size = (300, 300)
# img_resize = cv2.resize(img, size)

# # リサイズ
# print("img_resize.shape: ", img_resize.shape)
# cv2.imshow("resize", img_resize)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# 変数にグレースケール画像とカラー画像を入力
# img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# print(cv2.COLOR_BGR2GRAY)
# print(cv2.COLOR_BGR2HSV)

# # 画像の解像度と色の数を確認
# print("img_gray.shape: ", img_gray.shape)
# print("img_hsv.shape: ", img_hsv.shape)


# ヒストグラムを出力
color_list = ["blue", "green", "red"]

# for i, j in enumerate(color_list):
#     hist = cv2.calcHist([img], [i], None, [256], [0,256])
#     plt.plot(hist, color = j)
    
# hist = cv2.calcHist([img], [0], None, [256], [0,256])
# # グレースケールのヒストグラムを出力
# hist2 = cv2.calcHist([img_gray], [0], None, [256], [0,256])
# plt.plot(hist)
# グラフを表示
# plt.show()

# ヒストグラム平坦化
# equalizeHist() はモノクロ画像しか受け付けない
# img_eq = cv2.equalizeHist(img_gray)
# hist_e = cv2.calcHist([img_eq], [0], None, [256], [0,256])
# plt.plot(hist_e)
# # グラフを表示
# plt.show()


# # γ(ガンマ)変換: 明るさを調整する
# gamma = 1.8
# gamma_cvt = np.zeros((256, 1), dtype=np.uint8)
# for i in range (256):
#     gamma_cvt[i][0] = 255 * (float(i) / 255) ** (1.0 / gamma)

# img_gamma = cv2.LUT(img, gamma_cvt)


# # アフィン変換: 画像のスケーリング(拡大・縮小)、回転、シア(ずらし)などの変形を行う
h, w = img.shape[:2]
# dx, dy = 50, 50
# # 画像を移動
# afn_mat = np.float32([[1, 0, dx], [0, 1, dy]])
# img_afn = cv2.warpAffine(img, afn_mat, (w, h))
# 画像を回転
rot_mat = cv2.getRotationMatrix2D((w/2, h/2), 60, 1)
img_afn2 = cv2.warpAffine(img, rot_mat, (w,h))


# # 2値化: 画像を白と黒の2値に変換する
# threshold = 100
# ret, img_th = cv2.threshold(img_gray, threshold, 255, cv2.THRESH_BINARY)
# # 2値画像のヒストグラムを表示
# hist = cv2.calcHist([img_th], [0], None, [256], [0,256])
# plt.plot(hist)
# plt.show()


# # 大津の2値化: 自動的に最適な2値のしきい値を決定する方法
# ret2, img_o = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
# hist = cv2.calcHist([img], [0], None, [256], [0,256])
# plt.plot(hist)
# plt.show()


# # 適応的2値化: 各画素ごとに異なる閾値を設定する(各画素の周囲の領域を見て閾値を決定する)
# img_ada = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 1)


# # 畳み込み
# ones = np.ones((3, 3))
# print(ones)
# kernel = np.ones((3, 3)) / 9.0
# print(kernel)
# img_ke1 = cv2.filter2D(img, -1, kernel)


# # 画像の平滑化
# # ブラー
# img_blur = cv2.blur(img, (3,3))
# # ガウシアンブラー
# img_ga = cv2.GaussianBlur(img, (9,9), 2)
# # メディアンブラー
# img_me = cv2.medianBlur(img, 5)
# # バイラテラルフィルタ
# img_bi = cv2.bilateralFilter(img, 40, 40, 50)


# # Sobelフィルター
# # Cannyエッジ検出①
# img_sobelx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
# img_sobely = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)


# ラプラシアンフィルター
# img_lap = cv2.Laplacian(img, cv2.CV_32F)


# # ガウシアンブラー・ラプラシアン
# img_blur = cv2.GaussianBlur(img, (21,21), 2)
# img_lap2 = cv2.Laplacian(img_blur, cv2.CV_32F)


# # Cannyエッジ検出
# img_canny1 = cv2.Canny(img, 10, 100)
# img_canny2 = cv2.Canny(img, 80, 200)


# # モルフォロジー変換
# ret, img_th = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY)
# # 膨張・収縮、モルフォロジー変換
# kernel = np.ones((3,3), dtype=np.uint8)
# img_dilate = cv2.dilate(img_th, kernel)
# img_erode = cv2.erode(img_th, kernel)
# img_c = cv2.morphologyEx(img_th, cv2.MORPH_OPEN, kernel)


# Harris 特徴検出: 画像内のコーナーを検出するアルゴリズム
# 画像の小さなウィンドウが移動したときに輝度の変化が大きい点をコーナーとして検出する
# img_harris = copy.deepcopy(img)
# img_dst = cv2.cornerHarris(img_g, 2, 3, 0.05)
# 閾値の設定
# img_harris = [img_dst > 0.05 * img_dst.max()] = [0, 0, 255]


# # ブロブ検出 ※ブロブ: 画像内の連続した領域、同じようなデータの塊
# ret, img_bi = cv2.threshold(img_g, 50, 255, cv2.THRESH_BINARY)
# # ラベリング処理
# nlabels, labelImage, stats, centroids = cv2.connectedComponentsWithStats(img_bi)
# img_blob = copy.deepcopy(img)
# h, w = img_g.shape
# color = [[255,0,0],[0,255,0],[0,0,255],[255,255,0],[100,255,120],[255,30,0],[15,255,130],[0,100,255],[255,255,30],[0,120,120],[255,255,150]]
# for y in range(h):
#     for x in range(w):
#         if labelImage[y,x] > 0:
#             img_blob[y, x] = color[labelImage[y, x] - 1]

# for i in range(1, nlabels):
#     xc = int(centroids[i][0])
#     yc = int(centroids[i][1])
#     font = cv2.FONT_HERSHEY_COMPLEX
#     scale = 1
#     color = (255,255,255)
#     cv2.putText(img_blob, str(stats[i][-1], (xc,yc), font, scale, color))


# 輪郭(エッジ)検出
ret, img_bi = cv2.threshold(img_g, 40, 255, cv2.THRESH_BINARY)
# 輪郭検出、輪郭に線を引く
contours, hierarchy = cv2.findContours(img_bi,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
img_contour = cv2.drawContours(img,contours,-1,(255,255,0),4)




# 画像の表示
# cv2.imshow("img", img)
cv2.imshow("img_bi", img_bi)
cv2.imshow("img_contour", img_contour)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("Done ⭐️")