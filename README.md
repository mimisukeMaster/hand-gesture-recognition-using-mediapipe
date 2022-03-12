[Japanese/[English](https://github.com/Kazuhito00/hand-gesture-recognition-using-mediapipe/blob/main/README_EN.md)]

---
# hand-gesture-recognition-using-mediapipe
MediaPipe(Python版)を用いて手の姿勢推定を行い、検出したキーポイントを用いて、<br>簡易なMLPでハンドサインとフィンガージェスチャーを認識するサンプルプログラムです。
![mqlrf-s6x16](https://user-images.githubusercontent.com/37477845/102222442-c452cd00-3f26-11eb-93ec-c387c98231be.gif)

本リポジトリは以下の内容を含みます。
* サンプルプログラム
* **全キーポイント版** サンプルプログラム
* ハンドサイン認識モデル(TFLite)
* フィンガージェスチャー認識モデル(TFLite)
* **全キーポイント版** フィンガージェスチャー認識モデル(TFLite)
* ハンドサイン認識用学習データ、および、学習用ノートブック
* フィンガージェスチャー認識用学習データ、および、学習用ノートブック
* **全キーポイント版** フィンガージェスチャー認識用学習データ、および、学習用ノートブック
* フィンガージェスチャー認識用学習データの復元再生プログラム

# Requirements
* mediapipe 0.8.1
* OpenCV 3.4.2 or Later
* Tensorflow 2.3.0 or Later<br>tf-nightly 2.5.0.dev or later (LSTMモデルのTFLiteを作成する場合のみ)
* scikit-learn 0.23.2 or Later (学習時に混同行列を表示したい場合のみ) 
* matplotlib 3.3.2 or Later (学習時に混同行列を表示したい場合のみ)

# Demo
Webカメラを使ったデモの実行方法は以下です。
```bash
python app.py
```

デモ実行時には、以下のオプションが指定可能です。
* --device<br>カメラデバイス番号の指定 (デフォルト：0)
* --width<br>カメラキャプチャ時の横幅 (デフォルト：960)
* --height<br>カメラキャプチャ時の縦幅 (デフォルト：540)
* --use_static_image_mode<br>MediaPipeの推論にstatic_image_modeを利用するか否か (デフォルト：未指定)
* --min_detection_confidence<br>
検出信頼値の閾値 (デフォルト：0.5)
* --min_tracking_confidence<br>
トラッキング信頼値の閾値 (デフォルト：0.5)

全てのキーポイントを追跡するよう改変したデモの実行方法は以下です。
```bash
python app_allkeypoints.py
```
こちらも、`app.py`と同じオプションが指定可能です。

# Directory
<pre>
│  app.py
│  app_allkeypoints.py
│  keypoint_classification.ipynb
│  point_history_classification.ipynb
│  point_history_classification_allkeypoints.ipynb
│  
│  history_log_show.py
│  
├─model
│  ├─keypoint_classifier
│  │  │  keypoint.csv
│  │  │  keypoint_classifier.hdf5
│  │  │  keypoint_classifier.py
│  │  │  keypoint_classifier.tflite
│  │  └─ keypoint_classifier_label.csv
│  │          
│  └─point_history_classifier
│      │  point_history.csv
│      │  point_history_allkeypoints.csv
│      │  point_history_classifier.hdf5
│      │  point_history_classifier_allkeypoints.hdf5
│      │  point_history_classifier.py
│      │  point_history_classifier_allkeypoints.py
│      │  point_history_classifier.tflite
│      │  point_history_classifier_allkeypoints.tflite
│      │  point_history_classifier_label.csv
│      └─ point_history_classifier_label_allkeypoints.csv
│          
└─utils
    └─cvfpscalc.py
</pre>
### app.py
推論用のサンプルプログラムです。<br>また、ハンドサイン認識用の学習データ(キーポイント)、<br>
フィンガージェスチャー認識用の学習データ(人差指の座標履歴)を収集することもできます。

### app_allkeypoints.py
app.pyの拡張版です。app.pyは[ジェスチャー認識用データ収集](#historysave)の際に1点(人差し指先)のみの動きに追従、座標の履歴を保存していましたが、本拡張にとって全21点の動きを収集することができます。<br><br>
 *この拡張プログラムに合わせて、*※*の付いた、以下の座標履歴を保存するcsvファイル、ラベルデータ、学習用ノートブック、サンプルモデルが追加、変更されます。*

### keypoint_classification.ipynb
ハンドサイン認識用のモデル訓練用スクリプトです。

### point_history_classification.ipynb
1点の座標履歴から、フィンガージェスチャーを認識用するモデルを訓練するスクリプトです。

### point_history_classification_allkeypoints.ipynb ※
**全キーポイントの座標履歴から**、フィンガージェスチャーを認識するモデルを訓練するスクリプトです。(上記の拡張版)

### History_log_show.py  ※
`app_allkeypoints`で収集したキーポイント座標データ群を読み込み、ウィンドウ上でリプレイ再生するスクリプトです。

<br>

### model/keypoint_classifier
ハンドサイン認識に関わるファイルを格納するディレクトリです。<br>
以下のファイルが格納されます。
* 学習用データ(keypoint.csv)
* 学習済モデル(keypoint_classifier.tflite)
* ラベルデータ(keypoint_classifier_label.csv)
* 推論用クラス(keypoint_classifier.py)

### model/point_history_classifier
フィンガージェスチャー認識に関わるファイルを格納するディレクトリです。<br>
以下のファイルが格納されます。
* 学習用データ(point_history.csv)
* 学習済モデル(point_history_classifier.tflite)
* ラベルデータ(point_history_classifier_label.csv)
* 推論用クラス(point_history_classifier.py)

以下、**全キーポイント版**のファイルも格納されます
* 学習用データ(point_history_allkeypoints.csv)
* 学習済モデル(point_history_classifier_allkeypoints.tflite)
* ラベルデータ(point_history_classifier_label_allkeypoints.csv)
* 推論用クラス(point_history_classifier_allkeypoints.py)

### utils/cvfpscalc.py
FPS計測用のモジュールです。

# Training
ハンドサイン認識、フィンガージェスチャー認識は、<br>学習データの追加、変更、モデルの再トレーニングが出来ます。

### ハンドサイン認識トレーニング方法
### 1.学習データ収集
「k」を押すと、キーポイントの保存するモードになります（「MODE:Logging Key Point」と表示される）<br>
<img src="https://user-images.githubusercontent.com/37477845/102235423-aa6cb680-3f35-11eb-8ebd-5d823e211447.jpg" width="60%"><br><br>
「0」～「9」を押すと「model/keypoint_classifier/keypoint.csv」に以下のようにキーポイントが追記されます。<br>
1列目：押下した数字(クラスIDとして使用)、2列目以降：キーポイント座標<br>
<img src="https://user-images.githubusercontent.com/37477845/102345725-28d26280-3fe1-11eb-9eeb-8c938e3f625b.png" width="80%"><br><br>
キーポイント座標は以下の前処理を④まで実施したものを保存します。<br>
<img src="https://user-images.githubusercontent.com/37477845/102242918-ed328c80-3f3d-11eb-907c-61ba05678d54.png" width="80%">
<img src="https://user-images.githubusercontent.com/37477845/102244114-418a3c00-3f3f-11eb-8eef-f658e5aa2d0d.png" width="80%"><br><br>
初期状態では、パー(クラスID：0)、グー(クラスID：1)、指差し(クラスID：2)の3種類の学習データが入っています。<br>
必要に応じて3以降を追加したり、csvの既存データを削除して、学習データを用意してください。<br>
<img src="https://user-images.githubusercontent.com/37477845/102348846-d0519400-3fe5-11eb-8789-2e7daec65751.jpg" width="25%">　<img src="https://user-images.githubusercontent.com/37477845/102348855-d2b3ee00-3fe5-11eb-9c6d-b8924092a6d8.jpg" width="25%">　<img src="https://user-images.githubusercontent.com/37477845/102348861-d3e51b00-3fe5-11eb-8b07-adc08a48a760.jpg" width="25%">

### 2.モデル訓練
「[keypoint_classification.ipynb](keypoint_classification.ipynb)」をJupyter Notebookで開いて上から順に実行してください。<br>
学習データのクラス数を変更する場合は「NUM_CLASSES = 3」の値を変更し、<br>「model/keypoint_classifier/keypoint_classifier_label.csv」のラベルを適宜修正してください。<br><br>

#### X.モデル構造
「[keypoint_classification.ipynb](keypoint_classification.ipynb)」で用意しているモデルのイメージは以下です。
<img src="https://user-images.githubusercontent.com/37477845/102246723-69c76a00-3f42-11eb-8a4b-7c6b032b7e71.png" width="50%"><br><br>

### <h2 id=historysave>フィンガージェスチャー認識トレーニング方法</h2>
#### **全キーポイント版の説明は "★" で示しています**<br><br>
### 1.学習データ収集
「h」を押すと、指先座標の履歴を保存するモードになります（「MODE:Logging Point History」と表示される）<br>
<img src="https://user-images.githubusercontent.com/37477845/102249074-4d78fc80-3f45-11eb-9c1b-3eb975798871.jpg" width="60%"><br>
<br>★ 全キーポイント版の`app_allkeypoints.py`の場合では、プログラム実行直後からジェスチャを認識する状態になり、全点が緑色に光りますが、「h」を押して、保存モードに切り替えてください。
___
<br>「0」～「9」を押すと「model/point_history_classifier/point_history.csv」に以下のようにキーポイントが追記されます。<br>
<br>★ 全キーポイント版の場合は「model/point_history_classifier/point_history_allkeypoints.csv」に追記されます。<br>

1列目：押下した数字(クラスIDとして使用)、2列目以降：座標履歴(2×16=32列)
<br>
<img src="https://user-images.githubusercontent.com/37477845/102345850-54ede380-3fe1-11eb-8d04-88e351445898.png" width="80%"><br>
(★`allkeypoints`の場合)<br>
1列目：押下した数字(クラスIDとして使用)、2列目以降：座標履歴(2×21×16=672列)<br>
![allkeypoints_csv_view](https://user-images.githubusercontent.com/81568941/158016809-eedb25ce-f3be-4b70-a30a-879d6c73fcb6.png)
<br>
___
キーポイント座標は以下の前処理を④まで実施したものを保存します。<br>
<img src="https://user-images.githubusercontent.com/37477845/102244148-49e27700-3f3f-11eb-82e2-fc7de42b30fc.png" width="80%"><br>
★全キーポイント版ではTからT-15までの各時系列番号内にXY座標×21点分のデータが含まれています。<br>
___
`app.py`の初期状態では、静止(クラスID：0)、時計回り(クラスID：1)、反時計回り(クラスID：2)、移動(クラスID：4)の<br>4種類の学習データが入っています。<br>
必要に応じて5以降を追加したり、csvの既存データを削除して、学習データを用意してください。<br>
<img src="https://user-images.githubusercontent.com/37477845/102350939-02b0c080-3fe9-11eb-94d8-54a3decdeebc.jpg" width="20%">　<img src="https://user-images.githubusercontent.com/37477845/102350945-05131a80-3fe9-11eb-904c-a1ec573a5c7d.jpg" width="20%">　<img src="https://user-images.githubusercontent.com/37477845/102350951-06444780-3fe9-11eb-98cc-91e352edc23c.jpg" width="20%">　<img src="https://user-images.githubusercontent.com/37477845/102350942-047a8400-3fe9-11eb-9103-dbf383e67bf5.jpg" width="20%"><br>
<br>★全キーポイント版のプログラムの初期状態では、ポーズ(クラスID：0)、クレシェンド(クラスID：1)、フィニッシュ(クラスID：3)の<br>3種類の学習データが入っています。
`app.py`同様、適宜データを追加・削除してください。
(クレシェンド、フィニッシュは、指揮者をイメージした手の動きです)
<br>
--------------- ここにallkeypoints_labelの各ジェスチャ写真 ---------------<br>
<br>
___
### 2-i.モデル訓練
「[point_history_classification.ipynb](point_history_classification.ipynb)」をJupyter Notebookで開いて上から順に実行してください。<br>
学習データのクラス数を変更する場合は「NUM_CLASSES = 4」の値を変更し、<br>「model/point_history_classifier/[point_history_classifier_label.csv](model/point_history_classifier/point_history_classifier_label.csv)」のラベルを適宜修正してください。<br><br>

#### X.モデル構造 i
「[point_history_classification.ipynb](point_history_classification.ipynb)」で用意しているモデルのイメージは以下です。
<img src="https://user-images.githubusercontent.com/37477845/102246771-7481ff00-3f42-11eb-8ddf-9e3cc30c5816.png" width="50%"><br>
「LSTM」を用いたモデルは以下です。<br>使用する際には「use_lstm = False」を「True」に変更してください（要tf-nightly(2020/12/16時点))<br>
<img src="https://user-images.githubusercontent.com/37477845/102246817-8368b180-3f42-11eb-9851-23a7b12467aa.png" width="60%">
### 2-ii ★モデル訓練(全キーポイント版)

 [point_history_classification_allkeypoints.ipynb](point_history_classification_allkeypoints.ipynb)をJupyter Notebookで開いて上から順に実行してください。
学習データのクラス数を変更する場合は「NUM_CLASSES = 3」の値を変更し、「model/point_history_classifier/[point_history_classifier_label_allkeypoints.csv](model/point_history_classifier/point_history_classifier_label_allkeypoints.csv)」のラベルを適宜修正してください。
#### Y.モデル構造 ii
「[point_history_classification_allkeypoints.ipynb](point_history_classification_allkeypoints.ipynb)」で用意しているモデルのイメージは以下です。
<br>
![allkeypoints_h_cla_model_con](https://user-images.githubusercontent.com/81568941/158016911-82556906-71b7-4248-9164-f9eee0268b54.png)

「LSTM」を用いたモデルは以下です。<br>使用する際には「use_lstm = False」を「True」に変更してください。
<br>
![allkeypoints_h_cla_model_lstm_con](https://user-images.githubusercontent.com/81568941/158016928-1866bf3a-8f04-4808-a237-d8c67448e5de.png)
<br>
### モデルをONNXファイルで保存
[point_history_classification_allkeypoints.ipynb](point_history_classification_allkeypoints.ipynb)で学習した結果のモデルを、ONNXファイルで保存できます。<br>
> ONNXファイルにすると、Unity内でBarracuda、mediapipeを用いて、ジェスチャ認識モデルとして使うことができます。<br>
> 詳しくは[Reference](#reference)をご覧ください。


<br>保存処理はノートブックの中に含まれているので、全てのセルを実行すれば生成されます。

<br>

## ジェスチャー再現機能 ([History_log_show.py](History_log_show.py))追加
これは、★、`app_allkeypoints.py`で収集したデータをリプレイ再生するプログラムです。(`app.py`で収集したデータ、ハンドサインには対応していません)<br>

プログラムの実行方法は以下です。
```bash
python History_log_show.py
```
実行時には、以下のオプションが指定可能です。
* --start<br>再生を始めるcsvデータの行番号を指定 (デフォルト：0  (int))
* --end<br>再生を終えるcsvデータの行番号を指定 (デフォルト：データ行列の最終行(int))
* --pause_time<br>一つのジェスチャ(16フレーム)を再生するごとに一瞬停止させるか(デフォルト：False  (bool))<br>※視覚的にジェスチャを1回づつ区切れをつけ分かりやすくするため<br>

１行につきジェスチャの動き１回分のデータがあります

終了する際はEscキーを押して下さい。

### 仕組み
1. ジェスチャデータ([point_history_allkeypoints](model/point_history_classifier/point_history_allkeypoints.csv))と、ジェスチャ認識ラベル([point_history_classifier_label_allkeypoints](model/point_history_classifier/point_history_classifier_label_allkeypoints.csv))読み込み
2. ジェスチャデータを16フレーム(1ジェスチャ分)ごとに分割して成型<br>

3. そのデータを1フレームづつ描画して表示

以下は実行時のサンプルです
<br>
![mymedpip](https://user-images.githubusercontent.com/81568941/158016985-0c5b68c3-b3b0-47cb-b208-6091498eae1c.gif)
<br>
- row_ID<br>現在再生しているジェスチャの行番号
- GestureLabel<br>現在再生しているジェスチャの[ラベル](model/point_history_classifier/point_history_classifier_label_allkeypoints.csv)番号(何のジェスチャをしているか)<br>
=======================================<br>
TODO:
- [ ] `"GestureLabel"`の文字の横に`"???"`と表示される現象


# Application example
以下に応用事例を紹介します。
* [Control DJI Tello drone with Hand gestures](https://towardsdatascience.com/control-dji-tello-drone-with-hand-gestures-b76bd1d4644f)
* [Classifying American Sign Language Alphabets on the OAK-D](https://www.cortic.ca/post/classifying-american-sign-language-alphabets-on-the-oak-d)

# <h1 id=reference>Reference</h1>
* [MediaPipe](https://mediapipe.dev/)
* [Kazuhito00/mediapipe-python-sample](https://github.com/Kazuhito00/mediapipe-python-sample)
<br>
### ONNXファイルをUnityで動かす際の参考URL
- 手の認識<br> [HandPoseBarracuda](https://github.com/keijiro/HandPoseBarracuda)
- Unity内でのONNXファイルを使った分類<br> [Unity Technologies製推論エンジン Barracudaがスゴイという話](https://qiita.com/highno_RQ/items/478e1145f0eb868c0f2e)


# Author
高橋かずひと(https://twitter.com/KzhtTkhs)
 
# License 
hand-gesture-recognition-using-mediapipe is under [Apache v2 license](LICENSE).
