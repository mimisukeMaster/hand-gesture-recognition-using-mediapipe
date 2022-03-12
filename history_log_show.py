#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cv2 as cv
import numpy as np
import time
import argparse

def get_args(data_end_row):

    # ターミナルから実行する際に、再生の仕方を設定できます
    # --start <int> 再生を始める行を指定できます 
    # --end <int>   再生を終える行を指定できます
    # --pause_time <bool>　1行ごと(1回分の16フレーム分シーケンス)にポーズする時間を入れるか(一瞬止めるか)
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=data_end_row)  
    parser.add_argument("--pause_time", type=bool, default=False)

    args = parser.parse_args()

    return args

# TODO fps指定できるようにする
def main():
    csv_path = 'model/point_history_classifier/point_history_allkeypoints.csv'
    # DATA_RANGE = 672 + 1
    
    csv_classifier_label_path = 'model/point_history_classifier/point_history_classifier_label_allkeypoints.csv'
    
    with open(csv_path, 'r', newline="") as csvfile:
        data = csvfile.readlines()
        
    # ジェスチャのラベルデータを保持
    with open(csv_classifier_label_path, 'r', newline='', encoding="utf_8") as csvlabelfile:
        labeldata = csvlabelfile.readlines()   
         
    # 引数解析
    args = get_args(len(data))
    
    start_row = args.start
    end_row = args.end
    pause = args.pause_time
    
    debug_image = np.zeros((540, 960, 3))

    
    for i, row in enumerate(data):
        if i < start_row or i > end_row:
            continue
        # row は １行分の生データ(string型、, , ,..)
        
        # ==============================================
        ### 実際に使うデータとして
        ### row_data, gesture_label, row_id の前処理をする
        # ==============================================
        
        # ","で長いStringを分割してlistに変換
        row_data = row.split(',')
        
        # 先頭のデータ(label番号)の要素を削除し、値(label番号)をintに変換して取得
        gesture_label = int( row_data.pop(0) ) 
        
        # listの中身をstring配列からfloat配列に変換
        row_data = [float(s) for s in row_data] # row_dataは一次元
        
        # このforループが処理しているデータ行番号を、row_idで保持
        row_id = i + 1
        
        # ==============================================
        
        
        
        # ジェスチャ番号を対応するラベル番号の文字列に変換(gesture_label + 1 の行のジェスチャ名を取得)
        gesture_label_str = labeldata[gesture_label]
        
        # row_data を 16フレームに分割(1行に (キー押した時点にその時までの)１６フレームが全部ある)
        
        row_data_splited = np.array_split(row_data, 16) 
        # raw_data_splitedは2次元配列[array([, , , ])←1フレーム目全点2*21, array([,,,])←2フレーム目全点2*21,...]

        # 各フレームごとにデータ描画
        
        # 16回ループ seqは1次元の 2 * 21 要素の配列 各フレーム(seq)において、瞬間の全点の動きを計算
        for seq in row_data_splited:
            
            # 1次元配列の2*21要素のlistを2要素list * 21要素の2次元listにする
            # [[x0,y0],[x1,y1],[x2,y2],[x3,y3],[x4,y4]...[x20,y20]]にする
            seq  = np.reshape(seq,[21,2]).tolist()    # 外側から個数指定
            
            '''ここでseqは21要素の各xyの２次元配列 print(seq)>>>
                # ex)
                # [[ 0.          0.        ]
                #  [-0.065625   -0.01666667]
                #  [-0.1125     -0.06296296]
                #  [-0.14270833 -0.12777778]
                #  [-0.1625     -0.2       ]
                #  [-0.08333333 -0.21296296]
                #  [-0.1125     -0.32962963]
                #  [-0.12916667 -0.40925926]
                #  [-0.14583333 -0.48148148]
                #  [-0.05625    -0.39074074]
                #  [-0.06770833 -0.48888889]
                #  [-0.078125   -0.57777778]
                #  [ 0.00208333 -0.24444444]
                #  [-0.00208333 -0.39259259]
                #  [-0.00416667 -0.49259259]
                #  [-0.01041667 -0.57962963]
                #  [ 0.046875   -0.22222222]
                #  [ 0.06666667 -0.33703704]
                #  [ 0.08125    -0.41111111]
                #  [ 0.08854167 -0.48518519]]
                '''
            
            # データから位置計算処理
            debug_image = draw_landmarks(debug_image, seq)
            
            # 情報表示
            debug_image = draw_info(debug_image, row_id, gesture_label_str)
            
            # 画面描画
            cv.imshow('CSV data reproduction', debug_image)
            
            # debug_imageリセット
            debug_image = np.zeros((540, 960, 3))
            
            # キー処理(ESC：終了)
            key = cv.waitKey(10)
            if key == 27:  # ESC
                exit()
        
        # 1秒間ポーズ
        if pause: time.sleep(1) 
            
    cv.destroyAllWindows()
            
        
                
    

def draw_landmarks(image, landmark_point):
        
    # 描画に適した値に加工
    # (小さすぎる値の拡大、原点の違いを埋める平行移動,floatをintに変換)
    landmark_point = list((int(x * 800 + 500), int(y * 800 + 500)) for x,y in landmark_point)            
    
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


    
def draw_info(image, rowid, label):
    
    cv.putText(image, "rowID(16frame):" + str(rowid), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (255, 255, 255), 2, cv.LINE_AA)
    
    cv.putText(image, "GestureLabel:" + label, (10, 60), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (255, 255, 255), 2, cv.LINE_AA)

    return image

if __name__ == '__main__':
    main()