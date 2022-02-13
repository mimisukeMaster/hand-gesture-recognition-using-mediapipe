#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv
import copy
import argparse
import itertools
from collections import Counter
from collections import deque

import cv2 as cv
import numpy as np
import mediapipe as mp

from utils import CvFpsCalc
from model import KeyPointClassifier
from model import PointHistoryClassifier_allkeypoints


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)

    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.7)
    parser.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=int,
                        default=0.5)

    args = parser.parse_args()

    return args


def main():
    # 引数解析 #################################################################
    args = get_args()

    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    use_static_image_mode = args.use_static_image_mode
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    use_brect = True

    # カメラ準備 ###############################################################
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    # モデルロード #############################################################
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=use_static_image_mode,
        max_num_hands=1,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    keypoint_classifier = KeyPointClassifier()

    point_history_classifier = PointHistoryClassifier_allkeypoints()

    # ラベル読み込み ###########################################################
    with open('model/keypoint_classifier/keypoint_classifier_label.csv',
              encoding='utf-8-sig') as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [
            row[0] for row in keypoint_classifier_labels
        ]
    with open(
            'model/point_history_classifier/point_history_classifier_label_allkeypoints.csv',
            encoding='utf-8-sig') as f:
        point_history_classifier_labels = csv.reader(f)
        point_history_classifier_labels = [
            row[0] for row in point_history_classifier_labels
        ]

    # FPS計測モジュール ########################################################
    cvFpsCalc = CvFpsCalc(buffer_len=10)

    # 座標履歴 dequeはPythonのデータ型の一種(リストとか配列とか) maxlenは要素数の最大を指定し超えたら逆側から削除
    history_length = 16
    
    points_histories = deque(maxlen=history_length) 
    # point_historyは二次元の列で直近16個(記録数)の人差し指の座標を保存するもの dequeにdeque列を渡す、そのdeque列の個数はhistory_landmark_list
    # 16回の指の座標の記録分が二次元[x,y]で,それが２１本分
    ''' 2 * 21 * 16 の <list> in <list> in <deque> 多次元配列. deque = ([]) empty i.e.
            deque([
        1個目        [
                        [x0,y0],[x1,y1],[x2,y2], ...[x21,y21] 全指
                    ],
        2個目        [
                        [x0,y0],[x1,y1],[x2,y2], ...[x21,y21]
                    ],
        3個目        [
                        [x0,y0],[x1,y1],[x2,y2], ...[x21,y21]
                    ],
    |    :     ...    
    V   16個目       [
    時系列16記録分         [x0,y0],[x1,y1],[x2,y2], ...[x21,y21]
                    ]
                ])
                
    型は deque([ list[ list[2] ] ])
    '''

    # フィンガージェスチャー履歴 モデル判定結果番号いれる用##########################
    finger_gesture_history = deque(maxlen=history_length)
    
    # history_gesture_historyはモデルがpoint_historyの記録から判定してこれだと
    # returnされたジェスチャのindex番号を直近16回分集めた列(deque)
    # (16回判定したその番号たちがindex番号で(1,1,1,2,2,1,2,3…)みたいになってる)

    #  ########################################################################
    mode = 0 # defaultは0にしておいてlogでreturnされるように

    while True:
        fps = cvFpsCalc.get()

        # キー処理(ESC：終了) #################################################
        key = cv.waitKey(10)
        if key == 27:  # ESC
            break
        number, mode = select_mode(key, mode)#どのキー押されてモードは何か　どこで人さし指だけって決めてるのかはハンドサイン分類のとこ

        # カメラキャプチャ #####################################################
        ret, image = cap.read()
        if not ret:
            break
        image = cv.flip(image, 1)  # ミラー表示
        debug_image = copy.deepcopy(image)

        # 検出実施 #############################################################
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True

        # なんか検出した結果があったら ぐちゃぐちゃやらない限り普通あるんだけど#############
        # ここOutput、またはmodeによっては学習データ取りね#################################
        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                  results.multi_handedness):
                # 外接矩形の計算
                brect = calc_bounding_rect(debug_image, hand_landmarks)
                # ランドマークの計算listはint,int..
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                # 相対座標・正規化座標への変換
                pre_processed_landmark_list = pre_process_landmark(
                    landmark_list)
                
                # 一次元配列として 各本の2座標 * 時系列16 * 全21点 を平滑化して返す,これはxy*本数*時系列
                pre_processed_point_history_list = pre_process_point_history(
                debug_image, points_histories)
                
                
                # 学習データ保存(k h じゃなかったら(mode=0)returnされるから大丈夫)
                logging_csv(number, mode, pre_processed_landmark_list,
                            pre_processed_point_history_list)

                # ハンドサイン分類-keypoint
                hand_sign_id = keypoint_classifier(pre_processed_landmark_list) 
                #モデルから判別key pointでこれはpointerです、てなった前提でpoint_historyの処理が始まる
                
                # 常時point_histories上書き 全点の位置を追加してる　時系列データではない　時系列はこの処理が何回もされることでできる
                points_histories.append(landmark_list)
                '''全キーポイント座標を保存してく(ここ学習データlogとってんじゃないよ）
                # この列からモデルが判定して今何のシーケンスhistoryジェスチャしてるか判定する(indexで返す)
                #シーケンスhistory判定はココのリストの個々の要素から判定してる
                #points_historiesに(全キーポイントの)データ追加してるのはここでのみ,後は[0,0]でデータなし的扱いで代入する箇所ぐらい
                
                # 常時16記録分でどんどん上書きされ過去のから消える
                
                # 緑に光らせるのはここの列を使っているのもある
                
                #--------init Q-----------
                # Point記録してモデル作成する際の、スタートした後、一連の動きの区切れ目の時間はどのくらいなのか => 16個の点をとって学習モデル生成時もそれで生成してる
                #＃ どこをスタートにして１８点とってるのか =>historyは１点をシーケンスで直近16記録分取っておりkeypointは
                #何フレーム撮ってるのかみる
                #### 一本(人差し指)での動き分かれば２１本分のデータも取れるよね
                # 人差し指がどう動いてるか確かめよう どこからとってるのか、どのくらいとってるのか
                #--------- TODO ----------------
                # 21本の点を記録し1*16ではなく21*16の配列でとってモデルを生成(学習)させる
                # つまりPythonこのファイルで21本分とれるように改変し、model生成ipynbの方も21本の動きを学習駅るように改変する
                 # ここのScriptはpoint_historyで人差し指の点一本の16記録分シーケンス列取ってるのでこのpoint_hisotory的なの21こ必要 
                '''
 
                # フィンガージェスチャー分類-history
                finger_gesture_id = 0 #0ならなにも描画されない処理される(下いっても0ということはそもそもモデル使って判別してない)
                point_history_len = len(pre_processed_point_history_list)
                
                #各21本のキーポイント16時系列分集まったら
                if point_history_len == (history_length * 2 * 21):
                    
                    finger_gesture_id = point_history_classifier(
                        pre_processed_point_history_list) 
                        
                    #モデル使って判別, 一行に21本分のデータあるから判別の仕方ipynbで変えないとね
                    #一行1本で index+16数値 だった前とは違うんやで
                    # 結果のシーケンスジェスチャのID返す

                # 直近検出の中で最多のジェスチャーIDを算出finger_gesture_history max 16の列(historyだよ)
                finger_gesture_history.append(finger_gesture_id)
                most_common_fg_id = Counter(
                    finger_gesture_history).most_common() #16こid番号ができて,
                # どんどん上書き古いのから削除だったよね)もっとの頻出の番号を検出
                # 16回モデルで判別してから最多のシーケンスジェスチャを画像のUIとして表示

                # 描画
                debug_image = draw_bounding_rect(use_brect, debug_image, brect)
                debug_image = draw_landmarks(debug_image, landmark_list)
                debug_image = draw_info_text(
                    debug_image,
                    brect,
                    handedness,
                    keypoint_classifier_labels[hand_sign_id],
                    point_history_classifier_labels[most_common_fg_id[0][0]],
                )
                
                # landmark check
                debug_image = draw_landmarklist(debug_image, pre_processed_landmark_list)               

        else:# なんも検出できなかった
            
            # points_e は全キーポイントの座標を取っておく列、21こ([0,0]で空データ),これが並ぶことでpoints_historiesが時系列データに
            for points_e in points_histories: #points_eはlist[(key) list[(pos) ] ]型,ある一時の全座標
                    points_e.append([0, 0]) #各キーポイントを指定して21回Appendされるこのpoints_eは[[x0,y0],[x1,y1],...]の配列
        
        
        #緑に光らせる,加工画像を上書き代入　手が検出されなかった(else通ってきた)なら未加工で返ってくる
        for i in range(21): #各キーポイントに対して, i は21こ
            point_history = []
            for _, points in enumerate(points_histories): #_は16こ
                
                point_history.append(points[i])
                # pointsは各点の座標[[xp0,yp0],[xp1,yp1],...[xp20,yp20]]_のindexは16    
                # points[i]は一点の座標［x,y］
                    
                #pointsのi番目のキーポイントの座標しかappendされないはず
                #point_historyは[[xt0,yt0],[xt1,yt1]]となるはずゆびは一点のはず
            #一本の指について[[xt0,yt0],[xt1,yt1],..[xt15,yt15]]ができてほしい
        
            #seq16ループしてできたlist送って画像生成    
            debug_image = draw_point_history(debug_image, point_history)
        
        #ここでforfor抜けたから全点の緑加工が終わってる
        
        debug_image = draw_info(debug_image, fps, mode, number)

        # 画面反映 #############################################################
        cv.imshow('Hand Gesture Recognition', debug_image)

    cap.release()
    cv.destroyAllWindows()


def select_mode(key, mode):
    number = -1
    if 48 <= key <= 57:  # 0 ~ 9
        number = key - 48
    if key == 110:  # n
        mode = 0
    if key == 107:  # k
        mode = 1
    if key == 104:  # h
        mode = 2
    return number, mode


def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)

    return [x, y, x + w, y + h]


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # キーポイント
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point


def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # 相対座標に変換
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # 1次元リストに変換
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # 正規化
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list


def pre_process_point_history(image, points_histories):
    image_width, image_height = image.shape[1], image.shape[0]

    temp_points_histories = copy.deepcopy(points_histories) #temp_point_historyは３次元配列

    # 相対座標に変換
    base_x, base_y = 0, 0
    for seq, temp_points_history in  enumerate(temp_points_histories):
        
        for index, point in enumerate(temp_points_history):
            if index == 0:
                base_x, base_y = point[0], point[1]

            temp_points_histories[seq][index][0] = (temp_points_histories[seq][index][0] -
                                            base_x) / image_width
            temp_points_histories[seq][index][1] = (temp_points_histories[seq][index][1] -
                                            base_y) / image_height
            
    # 1次元リストに変換 時系列、指、それら含めた3次元配列を１次元に変換, chain~~は2次元のみ対応
    # 2回平滑化して完全な一次元配列に
    temp_points_histories_flater = list(
        itertools.chain.from_iterable(temp_points_histories))
    
    temp_points_histories = list(
        itertools.chain.from_iterable(temp_points_histories_flater))

    return temp_points_histories # 一次元配列として返す


def logging_csv(number, mode, landmark_list, point_history_list):
    if mode == 0: #なんも指定されてない
        pass
    '''if mode == 1 and (0 <= number <= 9):#kモードで番号押されたとき
        csv_path = 'model/keypoint_classifier/keypoint.csv'
        with open(csv_path, 'a', newline="") as f: # 'a'書き込み用に開き、既存ファイルがある場合は末尾に追記 new line は段落の切り方
            writer = csv.writer(f) # csvファイルに書き込むWriterオブジェクトを変数に格納(ファイルのファイルオブジェクトを渡して、csv.writer関数を使って生成)
            writer.writerow([number, *landmark_list]) #書いてる "(番号), (landmark_list,,)"って書かれる
    '''
    if mode == 2 and (0 <= number <= 9):#hモードで番号押されたとき
        csv_path = 'model/point_history_classifier/point_history_allkeypoints.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *point_history_list])
            '''
            上手くいったら
            番号,全点(各点x0,y0,x1,y1,x2,y2..x20,y20)の1時系列目,全点の2時系列目...全点の16時系列目、てなる("[]" で区切られてないよ","のみ)
             1点の追従をずっとcsv記録している
             1つの動作は16キー(16回の連続押して生成した行)で区切っている
             モデル生成するときもそこで区切って学習している
            '''
    return


def draw_landmarks(image, landmark_point):
    # 接続線
    if len(landmark_point) > 0:
        # 親指
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                (255, 255, 255), 2)

        # 人差指
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                (255, 255, 255), 2)

        # 中指
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                (255, 255, 255), 2)

        # 薬指
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                (255, 255, 255), 2)

        # 小指
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                (255, 255, 255), 2)

        # 手の平
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                (255, 255, 255), 2)

    # キーポイント
    for index, landmark in enumerate(landmark_point):
        if index == 0:  # 手首1
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 1:  # 手首2
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 2:  # 親指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 3:  # 親指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 4:  # 親指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 5:  # 人差指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 6:  # 人差指：第2関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 7:  # 人差指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 8:  # 人差指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 9:  # 中指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 10:  # 中指：第2関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 11:  # 中指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 12:  # 中指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 13:  # 薬指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 14:  # 薬指：第2関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 15:  # 薬指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 16:  # 薬指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 17:  # 小指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 18:  # 小指：第2関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 19:  # 小指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 20:  # 小指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)

    return image


def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        # 外接矩形
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                     (0, 0, 0), 1)

    return image


def draw_info_text(image, brect, handedness, hand_sign_text,
                   finger_gesture_text):
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22),
                 (0, 0, 0), -1)

    info_text = handedness.classification[0].label[0:]
    if hand_sign_text != "":
        info_text = info_text + ':' + hand_sign_text
    cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)

    if finger_gesture_text != "":
        cv.putText(image, "Finger Gesture:" + finger_gesture_text, (10, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv.LINE_AA)
        cv.putText(image, "Finger Gesture:" + finger_gesture_text, (10, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2,
                   cv.LINE_AA)

    return image


def draw_point_history(image, point_history):
    for index, point in enumerate(point_history):
        if point[0] != 0 and point[1] != 0:
            cv.circle(image, (point[0], point[1]), 1 + int(index / 2),
                      (152, 251, 152), 2)

    return image


def draw_info(image, fps, mode, number):
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (0, 0, 0), 4, cv.LINE_AA)
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (255, 255, 255), 2, cv.LINE_AA)

    mode_string = ['Logging Key Point', 'Logging Point History']
    if 1 <= mode <= 2:
        cv.putText(image, "MODE:" + mode_string[mode - 1], (10, 90),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                   cv.LINE_AA)
        if 0 <= number <= 9:
            cv.putText(image, "NUM:" + str(number), (10, 110),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                       cv.LINE_AA)
    return image

def draw_landmarklist(image, landmark_list):
    cv.putText(image, " 0 x:" + '{:.3f}'.format(landmark_list[0]) + " y:" + '{:.3f}'.format(landmark_list[1]),
             (10, 120), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv.LINE_AA)
    cv.putText(image, " 0 x:" + '{:.3f}'.format(landmark_list[0]) + " y:" + '{:.3f}'.format(landmark_list[1]),
             (10, 120), cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv.LINE_AA)
    cv.putText(image, " 4 x:" + '{:.3f}'.format(landmark_list[8]) + " y:" + '{:.3f}'.format(landmark_list[9]),
             (10, 150), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv.LINE_AA)
    cv.putText(image, " 4 x:" + '{:.3f}'.format(landmark_list[8]) + " y:" + '{:.3f}'.format(landmark_list[9]),
             (10, 150), cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv.LINE_AA)
    cv.putText(image, "12 x:" + '{:.3f}'.format(landmark_list[24]) + " y:" + '{:.3f}'.format(landmark_list[25]),
             (10, 180), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv.LINE_AA)
    cv.putText(image, "12 x:" + '{:.3f}'.format(landmark_list[24]) + " y:" + '{:.3f}'.format(landmark_list[25]),
             (10, 180), cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv.LINE_AA)
    cv.putText(image, "20 x:" + '{:.3f}'.format(landmark_list[40]) + " y:" + '{:.3f}'.format(landmark_list[41]),
             (10, 210), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv.LINE_AA)
    cv.putText(image, "20 x:" + '{:.3f}'.format(landmark_list[40]) + " y:" + '{:.3f}'.format(landmark_list[41]),
             (10, 210), cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv.LINE_AA)
    return image

if __name__ == '__main__':
    main()
