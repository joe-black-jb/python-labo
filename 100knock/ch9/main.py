import cv2
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
import dlib
import math
import pandas as pd

### ノック81 ###
# img = cv2.imread("img/img01.jpg")
# height, width = img.shape[:2]
# print(width)
# print(height)

# cv2.imshow("img", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


### ノック82 ###
## 情報取得 ##
# cap = cv2.VideoCapture("mov/mov01.avi")
# width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
# height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
# count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
# fps = cap.get(cv2.CAP_PROP_FPS)
# print(width)
# print(height)
# print(count)
# print(fps)

## 映像のフレーム画像化 ##
# num = 0
# num_frame = 100
# list_frame = []
# while(cap.isOpened()):
#     # 処理 (フレームごとに切り出し)
#     ret, frame = cap.read()
#     # 出力 (フレーム画像を書き出し)
#     if ret:
#         '''
#         OpenCV では画像データを BGR の順番で扱い、
#         Matplotlib などは RGB の順番で使うため
#         BGR を RGB に変換
#         '''
#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         list_frame.append(frame_rgb)
#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             break
#         if num > num_frame:
#             break
#     num = num + 1
# print("処理を完了しました")
# cap.release()

# ## フレーム画像をアニメーションに変換 ##
# plt.figure()
# patch = plt.imshow(list_frame[0])
# plt.axis("off")
# def animate(i):
#     patch.set_data(list_frame[i])
# anim = FuncAnimation(plt.gcf(), animate, frames=len(list_frame), interval = 1000/30.0)
# plt.close()

# ## アニメーションを表示 ##
# HTML(anim.to_jshtml())


### ノック83 ###
# num = 0
# count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
# while (cap.isOpened()):
#     ret, frame = cap.read()
#     if ret:
#         filepath = "snapshot/snapshot_" + str(num) + ".jpg"
#         cv2.imwrite(filepath, frame)
#     num = num + 1
#     if num >= count:
#         break
# cap.release()
# cv2.destroyAllWindows()


### ノック84 ###
'''
HOG特徴量
- HOG: Histogram of Oriented Gradients (輝度勾配)
- 微分処理により輝度 (明るさ) の変化率を求める
'''

# 準備 #
# # HOGDescriptor を宣言
# hog = cv2.HOGDescriptor()
# # 人のモデルを与える
# hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
# hogParams = {'winStride': (8, 8), 'padding': (32, 32), 'scale': 1.05, 'hitThreshold':0, 'groupThreshold':5}

# 検出 #
# img = cv2.imread("img/img01.jpg")
# # 読み込んだ画像をモノクロに変換
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# # human に検出された人の位置情報が格納される
# human, r = hog.detectMultiScale(gray, **hogParams)
# if (len(human)>0):
#     for (x, y, w, h) in human:
#         # 画像に四角形を描く関数 rectangle を利用し人の箇所を四角で囲う
#         cv2.rectangle(img, (x, y), (x + w, y + h), (255,255,255), 3)
# cv2.imshow("img", img)
# cv2.imwrite("temp.jpg",img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


### ノック85 ###
# ## 準備 ##
# '''
# - harrcascade_frontalface_alt.xml
#     顔検出のための Haarcascade（ハールカスケード） の学習済みモデルのファイル

# - Haarcascade（ハールカスケード）
#     - cascade: 滝、連続して起こる状態、 etc..
#     - 物体検出のための特徴ベースのアルゴリズム
#         - 物体を検出するための特徴としてHaar-like特徴を使用している。
#             - Haar-like特徴
#                 - 物体の異なる部分に対する明るさの差を表す矩形フィルター
#     - 特に顔検出などのコンピュータビジョンタスクに広く使用されている

# '''
# cascade_file = "haarcascade_frontalface_alt.xml"
# cascade = cv2.CascadeClassifier(cascade_file)

# ## 検出 ##
# img = cv2.imread("img/img02.jpg")
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# ''' cascade.detectMultiScale()
# 画像内の顔を検出する関数

# Args:
#     gray (MatLike): 画像データ
#     minSize (Size): 最小の顔サイズ ... (int, int) の形式でピクセル指定
#                     指定したサイズ以下の顔は検出されない
# '''
# face_list = cascade.detectMultiScale(gray, minSize=(50, 50))

# ## 検出した顔に印をつける ##
# for (x, y, w, h) in face_list:
#     color = (0, 0, 225)
#     pen_w = 3
#     cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness = pen_w)
# cv2.imshow("img", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


### ノック86 ###
# '''
# Dlib というライブラリを使い、画像内の人がどこを向いているのかを検出する
# - Dlib
#     - 顔器官 (フェイス・ランドマーク) と言われる目・鼻・口・輪郭を68の特徴点で表現できる
# '''

# # 準備 #
# predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
# detector = dlib.get_frontal_face_detector()
# # print(f"predictor: {predictor}")
# # print(f"detector: {detector}")

# # # 検出 #
# img = cv2.imread("img/img02.jpg")
# dets = detector(img, 1) # rectangles[[(1061, 116) (1216, 270)]]
# # print(f"dets: {dets}")


# for k, d in enumerate(dets):
#     # k: インデックス, d: dets ... [(1061, 116) (1216, 270)]
#     # print(f"(k, d) = ({k}, {d})")
#     # print(f"d.left(), d.top(), d.right(), d.bottom(): {d.left()}, {d.top()}, {d.right()}, {d.bottom()}")
#     '''
#     d.left()  : 1061
#     d.top()   : 116
#     d.right() : 1216
#     d.bottom(): 270
#     '''
#     shape = predictor(img, d)

#     # 顔領域の表示
#     color_f = (0, 0, 225) # 赤
#     color_l_out = (255, 0, 0) # 青
#     color_l_in = (0, 255, 0) # 緑
#     line_w = 3
#     circle_r = 3
#     fontType = cv2.FONT_HERSHEY_SIMPLEX
#     fontSize = 1
#     ''' cv2.rectangle()
#     - img: 画像
#     - pt1: 左上の頂点 (left, top)
#     - pt2: 右下の頂点 (right, bottom)

#     [pt1, pt2 を指定した四角形描画イメージ]
#     (left, top) で四角形のスタート地点を指定
#     => 右に right 行って下に bottom 行く

#                right
#     start----------------->
#     (left, top)            |
#     |                      |
#     |                      |  bottom
#     |                      v
#     -----------------------

#     - color: (B, G, R) で枠線の色を指定
#     - thickness: 太さを指定 (多分 px)
#     '''
#     cv2.rectangle(img, (d.left(), d.top()), (d.right(), d.bottom()), color_f, line_w)
#     # cv2.rectangle(img, (0, 200), (600, 300), color_f, line_w)
#     # cv2.putText(img, str(k), (d.left(), d.top()), fontType, fontSize, color_f, line_w)

#     # 重心を導出する箱を用意
#     num_of_points_out = 17
#     num_of_points_in = shape.num_parts - num_of_points_out
#     gx_out = 0
#     gy_out = 0
#     gx_in = 0
#     gy_in = 0
#     for shape_point_count in range(shape.num_parts):
#         shape_point = shape.part(shape_point_count)
#         #print("顔器官No.{} 座標位置: ({},{})".format(shape_point_count, shape_point.x, shape_point.y))
#         #器官ごとに描画
#         if shape_point_count<num_of_points_out:
#             cv2.circle(img,(shape_point.x, shape_point.y),circle_r,color_l_out, line_w)
#             gx_out = gx_out + shape_point.x/num_of_points_out
#             gy_out = gy_out + shape_point.y/num_of_points_out
#         else:
#             cv2.circle(img,(shape_point.x, shape_point.y),circle_r,color_l_in, line_w)
#             gx_in = gx_in + shape_point.x/num_of_points_in
#             gy_in = gy_in + shape_point.y/num_of_points_in

#     # 重心位置を描画
#     cv2.circle(img,(int(gx_out), int(gy_out)),circle_r,(0,0,255), line_w)
#     cv2.circle(img,(int(gx_in), int(gy_in)),circle_r,(0,0,0), line_w)

#     # 顔の方位を計算
#     theta = math.asin(2*(gx_in-gx_out)/(d.right()-d.left()))
#     radian = theta*180/math.pi
#     print("顔方位:{} (角度:{}度)".format(theta,radian))

#     # 顔方位を表示
#     if radian<0:
#         textPrefix = "   left "
#     else:
#         textPrefix = "   right "
#     textShow = textPrefix + str(round(abs(radian),1)) + " deg."
#     cv2.putText(img, textShow, (d.left(), d.top()), fontType, fontSize, color_f, line_w)

# cv2.imshow("img", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


### ノック87 ###
# print("タイムラプス生成を開始します")

# ## 映像取得 ##
# cap = cv2.VideoCapture("mov/mov01.avi")
# width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# ## hog宣言 ##
# '''
# - HOG: 画像の勾配方向のヒストグラムを計算し、物体の特徴を捉える手法
# - OpenCV の HOG では、人物 (pedestrian) の検出に特化した事前学習済みの SVM モデルを使用できる
# - SVM (Support Vector Machine)
#     - 機械学習モデルの中でも特に有名なアルゴリズムの一つ
#     - 教師あり学習における分類タスクで主に使用される
# '''
# # HOG 特徴量を計算するためのオブジェクトを作成
# hog = cv2.HOGDescriptor()
# hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
# # finalThreshold => groupThreshold
# hogParams = {"winStride": (8, 8), "padding": (32, 32), "scale": 1.05, "hitThreshold": 0, "groupThreshold": 5}

# ## タイムラプス作成 ##
# # movie_name = "timelapse.avi"
# # fourcc = cv2.VideoWriter_fourcc("X", "V", "I", "D")
# # video = cv2.VideoWriter(movie_name, fourcc, 30, (width, height))

# num = 0

# ## ノック88 で使う変数 ##
# fps = cap.get(cv2.CAP_PROP_FPS) # 30.0
# # print("fps: ", fps)
# list_df = pd.DataFrame(columns=["time", "people"])
# # print(list_df.columns)

# ## ノック89 で使う変数 ##
# list_df2 = pd.DataFrame(columns=["time", "people"])

# # cols = list_df.columns
# # print(cols)
# # print(cols.tolist())
# # print(len(cols))
# # print(len(cols.tolist()))
# # print(["time", "people"])
# print(10 / fps)

# while(cap.isOpened()):
#     ''' cap.read()
#     Returns:
#         ret (bool): フレームを正常に読み込めたかどうか
#         frame (MatLike): 動画のフレーム (画像データ) が NumPy の ndarray として格納されている
#     '''
#     ret, frame = cap.read()
#     if ret:
#         if (num % 10 == 0):
#             gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#             human, r = hog.detectMultiScale(gray, **hogParams)
#             if (len(human) > 0):
#                 for (x, y, w, h) in human:
#                     cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 3)
#             # ノック87
#             # video.write(frame)
#             # ノック88
#             ''' pd.Series()
#             - 1列のベクトルを保持する構造
#             - インデックス (ラベル) 付きの一次元配列
#             - NumPy の ndarray に似ているが、インデックスを持つ
#             - 辞書のようにキー（インデックス）で要素を取得できる
#             '''
#             # tmp_se = pd.Series([num / fps, len(human)], index=list_df.columns)
#             # tmp_se = pd.Series([num / fps, len(human)], index=list_df.columns.tolist())
#             tmp_se = pd.Series([num / fps, len(human)], index=["time", "people"])
#             # df.append() は pandas 2.0.0 以降ではエラーする
#             # => df.concat() を使う
#             # list_df = list_df.append(tmp_se, ignore_index=True)
#             '''
#             axis を設定しないと index 方向に結合 = 行を増やしていく
#             '''
#             list_df = pd.concat([list_df, tmp_se], ignore_index=True)
#             if cv2.waitKey(1) & 0xFF == ord("q"):
#                 break
#     else:
#         break
#     num = num + 1
# # video.release()
# cap.release()
# cv2.destroyAllWindows()
# # print("タイムラプス生成を終了しました")
# print("分析を終了しました")


### ノック88 ###
# plt.plot(list_df["time"], list_df["people"])
# plt.xlabel("time(sec.)")
# plt.ylabel("population")
# plt.ylim(0, 15)
# plt.show()


## **dict の練習
# def sample_function(a, b, c):
#     print(a, b, c)
# params = {"a": 1, "b": 2, "c": 3}
# print("**params: ", **params)
# print(**params)
# print("params: ", params)
# print("**params: ", sample_function(**params))  # これは sample_function(1, 2, 3) と同じ
# print("params: ", sample_function(params))


