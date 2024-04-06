import csv
import copy
import argparse
import itertools
import os
import time
import math

from collections import Counter
from collections import deque
from gtts import gTTS
from playsound import playsound

import cv2 as cv
import numpy as np
import mediapipe as mp
import pyautogui

from utils import CvFpsCalc
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# 模型
from model import KeyPointClassifier_R
from model import KeyPointClassifier_L
from model import DynamicGesturesClassifier


# from model import MouseClassifier

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)  # 摄像头选择
    parser.add_argument("--width", help='cap width', type=int, default=640)
    parser.add_argument("--height", help='cap height', type=int, default=480)

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
    # 数据加载 #################################################################
    args = get_args()

    cap_device = args.device
    cap_width = args.width
    cap_height = args.height
    use_static_image_mode = args.use_static_image_mode
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence
    use_brect = True

    # 相机 #################################################################
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    # 模型加载 #############################################################
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=use_static_image_mode,
        max_num_hands=1,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    keypoint_classifier_R = KeyPointClassifier_R(invalid_value=8, score_th=0.4)  # 静态手势识别模型
    keypoint_classifier_L = KeyPointClassifier_L(invalid_value=8, score_th=0.4)
    # mouse_classifier = MouseClassifier(invalid_value=2, score_th=0.4)
    dynamic_gestures_classifier = DynamicGesturesClassifier()  # 动态手势识别模型

    # 读取标签文件labels ###########################################################
    with open(
            'model/keypoint_classifier/keypoint_classifier_label.csv', encoding='utf-8-sig') as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [
            row[0] for row in keypoint_classifier_labels
        ]
    with open(
            'model/point_history_classifier/point_history_classifier_label.csv', encoding='utf-8-sig') as f:
        point_history_classifier_labels = csv.reader(f)
        point_history_classifier_labels = [
            row[0] for row in point_history_classifier_labels
        ]

        # FPS 帧率显示 ########################################################
        cvFpsCalc = CvFpsCalc(buffer_len=3)

    # 坐标历史记录 用于动态手势记录食指的16帧坐标变换############################
    history_length = 16  # 记录连续16帧的坐标变化
    point_history = deque(maxlen=history_length)  # 记录连续16帧的坐标变化

    # 手指手势历史记录 ################################################
    finger_gesture_history = deque(maxlen=history_length)

    # 靜態手勢最常出現參數初始化
    keypoint_length = 5  # 静态手势识别队列长度
    keypoint_R = deque(maxlen=keypoint_length)
    keypoint_L = deque(maxlen=keypoint_length)

    # result deque
    rest_length = 300
    rest_result = deque(maxlen=rest_length)
    speed_up_count = deque(maxlen=3)

    # 前置准备
    mode = 0  # 模式，初始为0，select_mode来改变
    pyautogui.PAUSE = 0

    # ========= 鼠标前置作業 =========
    wScr, hScr = pyautogui.size()  # 获取屏幕大小
    frameR = 100
    smoothening = 5
    plocX, plocY = 0, 0
    clocX, clocY = 0, 0
    mousespeed = 1.5
    clicktime = time.time()
    # 关闭 滑鼠移至角落保护措施
    pyautogui.FAILSAFE = False
    # == == == == =  == == == == =

    i = 0
    mouseDown = 0

    # ========= 主程序 =========
    while True:
        fps = cvFpsCalc.get()
        # 键盘按键读取，esc退出
        key = cv.waitKey(10)
        if key == 27:  # ESC
            break
        number, mode = select_mode(key, mode)
        # 相机捕捉 #####################################################
        ret, image = cap.read()
        if not ret:
            break
        image = cv.flip(image, 1)  # 镜像显示
        debug_image = copy.deepcopy(image)  # 创建了当前帧的深拷贝

        # 検出実施 检测强制执行 ###########################################
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True


        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):

                # 手部轮廓矩形计算
                brect = calc_bounding_rect(debug_image, hand_landmarks)
                # Landmark 计算
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                # print(landmark_list)

                # 转化为相对坐标，最终正规化
                pre_processed_landmark_list = pre_process_landmark(landmark_list)
                pre_processed_point_history_list = pre_process_point_history(debug_image, point_history)
                # 写入csv文件
                logging_csv(number, mode, pre_processed_landmark_list, pre_processed_point_history_list)

                # 靜態手勢資料預測
                hand_sign_id_R = keypoint_classifier_R(pre_processed_landmark_list)
                hand_sign_id_L = keypoint_classifier_L(pre_processed_landmark_list)


                if handedness.classification[0].label[0:] == 'Left':
                    left_id = hand_sign_id_L

                else:
                    right_id = hand_sign_id_R

                    # 手比one 触发动态手势捕捉
                if right_id == 1 or left_id == 1:
                    point_history.append(landmark_list[8])  # 将食指尖坐标加入point_history队列
                else:
                    point_history.append([0, 0])  # 反之加入0

                # 动态手势识别
                finger_gesture_id = 0
                point_history_len = len(pre_processed_point_history_list)
                if point_history_len == (history_length * 2):
                    finger_gesture_id = dynamic_gestures_classifier(pre_processed_point_history_list)
                # print(finger_gesture_id) # 0 = stop, 1 = clockwise, 2 = counterclockwise, 3 = move,偵測出現的動態手勢

                # 计算动态手勢最常出現id #########################################
                finger_gesture_history.append(finger_gesture_id)
                most_common_fg_id = Counter(finger_gesture_history).most_common()

                # 滑鼠的deque
                # mouse_id_history.append(mouse_id)
                # most_common_ms_id = Counter(mouse_id_history).most_common()
                # print(f'finger_gesture_history = {finger_gesture_history}')
                # print(f'most_common_fg_id = {most_common_fg_id}')

                # 静态手勢最常出現id #########################################
                hand_gesture_id = [right_id, left_id]
                keypoint_R.append(hand_gesture_id[0])
                keypoint_L.append(hand_gesture_id[1])
                # print(keypoint_R) # deque右手的靜態id
                # print(most_common_keypoint_id) # 右手靜態id最大
                if right_id != -1:
                    most_common_keypoint_id = Counter(keypoint_R).most_common()
                else:
                    most_common_keypoint_id = Counter(keypoint_L).most_common()

                # print(f'keypoint = {keypoint}')
                # print(f'most_common_keypoint_id = {most_common_keypoint_id}')

                ###############################################################

                # 绘图
                debug_image = draw_bounding_rect(use_brect, debug_image, brect)  # 手部轮廓矩形
                debug_image = draw_landmarks(debug_image, landmark_list)  # 21个关键点坐标连线
                debug_image = draw_info_text(  # 文字
                    debug_image,
                    brect,
                    handedness,
                    keypoint_classifier_labels[most_common_keypoint_id[0][0]],
                    point_history_classifier_labels[most_common_fg_id[0][0]],
                )
                resttime = time.time()
        else:
            point_history.append([0, 0])

        debug_image = draw_point_history(debug_image, point_history)
        debug_image = draw_info(debug_image, fps, mode, number)

        # 偵測是否有手勢 #########################################

        if left_id + right_id > -2:
            if time.time() - presstime > 1:

                # 模式切换
                # if most_common_ms_id[0][0] == 3 and most_common_ms_id[0][1] == 40:  # Gesture six changes to the different mode
                #     print('Mode has changed')
                #     detect_mode = (detect_mode + 1) % 3
                #     if detect_mode == 0:
                #         what_mode = 'Sleep'
                #         # playsound('rest.mp3', block=False)
                #     if detect_mode == 1:
                what_mode = 'Keyboard'
                #         # playsound('keyboard.mp3', block=False)
                #     if detect_mode == 2:
                #         what_mode = 'Mouse'
                #         # playsound('mouse.mp3', block=False)
                #     print(f'Current mode => {what_mode}')
                detect_mode = 2
                presstime = time.time() + 1

                # 键盘控制
                if detect_mode == 1:
                    if time.time() - presstime_2 > 1:
                        # 靜態手勢控制 ################################################
                        # ppt播放控制 #################################
                        control_keyboard(most_common_keypoint_id, 2, 'N', keyboard_TF=True, print_TF=True)  # 下一张ppt
                        control_keyboard(most_common_keypoint_id, 9, 'P', keyboard_TF=True, print_TF=True)  # 上一张ppt
                        control_keyboard_hotkey(most_common_keypoint_id, 5,
                                         ['shift', 'f5'], keyboard_TF=True, print_TF=True)  # 从当前ppt放映
                        control_keyboard(most_common_keypoint_id, 6, 'esc', keyboard_TF=True, print_TF=True)  # 结束放映
                        # 画笔控制 #################################
                        control_keyboard_hotkey(most_common_keypoint_id, 5,
                                                ['ctrl', 'p'], keyboard_TF=True, print_TF=True)  # 变笔
                        control_keyboard_hotkey(most_common_keypoint_id, 5,
                                                ['ctrl', 'A'], keyboard_TF=True, print_TF=True)  # 变箭头
                        control_keyboard(most_common_keypoint_id, 9, 'E', keyboard_TF=True, print_TF=True)  # 全部擦除
                        # 视频播放 #################################
                        control_keyboard_hotkey(most_common_keypoint_id, 5,
                                                ['alt', 'p'], keyboard_TF=True, print_TF=True)  # 播放/暂停
                        control_keyboard_hotkey(most_common_keypoint_id, 5,
                                                ['alt', 'u'], keyboard_TF=True, print_TF=True)  # 静音

                        presstime_2 = time.time()

                    # 放大
                    if most_common_keypoint_id[0][0] == 0 and most_common_keypoint_id[0][1] == 5:
                        print(i, time.time() - presstime_4)
                        if i == 3 and time.time() - presstime_4 > 0.3:
                            pyautogui.press('+')
                            i = 0
                            presstime_4 = time.time()
                        elif i == 3 and time.time() - presstime_4 > 0.25:
                            pyautogui.press('+')
                            presstime_4 = time.time()
                        elif time.time() - presstime_4 > 1:
                            pyautogui.press('+')
                            i += 1
                            presstime_4 = time.time()
                        # print(i,presstime_4)

                    # 缩小
                    if most_common_keypoint_id[0][0] == 7 and most_common_keypoint_id[0][1] == 5:
                        # print(i, time.time() - presstime_4)
                        if i == 3 and time.time() - presstime_4 > 0.3:
                            pyautogui.press('-')
                            i = 0
                            presstime_4 = time.time()
                        elif i == 3 and time.time() - presstime_4 > 0.25:
                            pyautogui.press('-')
                            presstime_4 = time.time()
                        elif time.time() - presstime_4 > 1:
                            pyautogui.press('- ')
                            i += 1
                            presstime_4 = time.time()

                    # 动态手势控制#################
                    # 音量加
                    if most_common_fg_id[0][0] == 1 and most_common_fg_id[0][1] > 12:
                        if time.time() - presstime_3 > 1.5:
                            volumeControl(volChange=True)
                            print('音量+')
                            presstime_3 = time.time()
                    # 音量减
                    elif most_common_fg_id[0][0] == 2 and most_common_fg_id[0][1] > 12:
                        if time.time() - presstime_3 > 1.5:
                            volumeControl(volChange=False)
                            print('音量-')
                            presstime_3 = time.time()

            if detect_mode == 2:
                fingers = fingersUp(landmark_list)
                # print(fingers)

                x1, y1 = landmark_list[8]
                x3 = np.interp(x1, (50, (cap_width - 50)), (0, wScr))
                y3 = np.interp(y1, (30, (cap_height - 170)), (0, hScr))
                cv.rectangle(debug_image, (50, 30), (cap_width - 50, cap_height - 170),
                             (255, 0, 255), 2)
                clocX = plocX + (x3 - plocX) / smoothening
                clocY = plocY + (y3 - plocY) / smoothening
                plocX, plocY = clocX, clocY

                # if mouse_id == 0:  # Point gesture
                if fingers[1] == 1 and fingers[2] == 0:
                    # print(landmark_list[8]) #index finger
                    # print(landmark_list[12]) #middle finger
                    ##x1, y1 = landmark_list[8]
                    # print(landmark_list)
                    # cv.rectangle(debug_image, (frameR, frameR), (cap_width - frameR, cap_height - frameR),
                    #                            (255, 0, 255), 2)
                    # cv.rectangle(debug_image, (50, 30), (cap_width - 50, cap_height - 170),
                    #              (255, 0, 255), 2)
                    # 座標轉換
                    # x軸: 鏡頭上50~(cap_width - 50)轉至螢幕寬0~wScr
                    # y軸: 鏡頭上30~(cap_height - 170)轉至螢幕長0~hScr
                    # x3 = np.interp(x1, (50, (cap_width - 50)), (0, wScr))
                    # y3 = np.interp(y1, (30, (cap_height - 170)), (0, hScr))
                    # x3 = np.interp(x1, (frameR, (cap_width - frameR)), (0, wScr))
                    # y3 = np.interp(y1, (frameR, (cap_height - frameR)), (0, hScr))
                    # print(x3, y3)

                    # 6. Smoothen Values
                    # clocX = plocX + (x3 - plocX) / smoothening
                    # clocY = plocY + (y3 - plocY) / smoothening
                    # 7. Move Mouse
                    pyautogui.moveTo(clocX, clocY)
                    cv.circle(debug_image, (x1, y1), 15, (255, 0, 255), cv.FILLED)
                    # plocX, plocY = clocX, clocY

                #if mouse_id == 1:
                if fingers[1] == 1 and fingers[2] == 1:
                    length, img, lineInfo = findDistance(landmark_list[8], landmark_list[12], debug_image)

                    # 10. Click mouse if distance short
                    # if time.time() - clicktime > 0.5:
                    #     if length < 40:
                    #         cv.circle(img, (lineInfo[4], lineInfo[5]),
                    #                   15, (0, 255, 0), cv.FILLED)
                    #         #pyautogui.click(clicks=1)
                    #         #pyautogui.dragTo(x3, y3, button='left')
                    #         pyautogui.mouseDown(button='left')
                    #         pyautogui.moveTo(clocX, clocY)
                    #         print('click')
                    #         clicktime = time.time()
                    # plocX, plocY = clocX, clocY
                    if length < 40:
                        cv.circle(img, (lineInfo[4], lineInfo[5]),
                                  15, (0, 255, 0), cv.FILLED)
                        print('mouseDown=' + str(mouseDown))

                        # pyautogui.click(clicks=1)
                        # pyautogui.dragTo(x3, y3, button='left')
                        if mouseDown == 0:
                            pyautogui.mouseDown(button='left')
                            print('mouseDown')
                            mouseDown = 1
                            print(mouseDown)
                        pyautogui.moveTo(clocX, clocY)
                    else:

                        if mouseDown == 1:
                            pyautogui.mouseUp(button='left')
                            mouseDown = 0

                            # pyautogui.mouseUp()
                            print('mouseUp')

                    # if length > 70:
                    #     cv.circle(img, (lineInfo[4], lineInfo[5]),
                    #               15, (0, 255, 0), cv.FILLED)
                    # pyautogui.click(clicks=2)
                    # print('click*2')
                    # clicktime = time.time()

                # if most_common_keypoint_id[0][0] == 5 and most_common_keypoint_id[0][1] == 5:
                #     pyautogui.scroll(20)
                #
                # if most_common_keypoint_id[0][0] == 6 and most_common_keypoint_id[0][1] == 5:
                #     pyautogui.scroll(-20)

                # if left_id == 7 or right_id == 7:
                # if most_common_keypoint_id[0][0] == 0 and most_common_keypoint_id[0][1] == 5:
                #     if time.time() - clicktime > 1:
                #         pyautogui.click(clicks=2)
                #         clicktime = time.time()

                if most_common_keypoint_id[0][0] == 9 and most_common_keypoint_id[0][1] == 5:
                    if time.time() - clicktime > 2:
                        pyautogui.hotkey('alt', 'left')
                        clicktime = time.time()

        cv.putText(debug_image, what_mode, (400, 30), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv.LINE_AA)
        # Screen reflection ###################################JL##########################
        cv.imshow('Hand Gesture Recognition', debug_image)

    cap.release()
    cv.destroyAllWindows()


tipIds = [4, 8, 12, 16, 20]


def fingersUp(landmark_list):
    fingers = []
    # 大拇指
    if landmark_list[tipIds[0]][1] < landmark_list[tipIds[1] - 2][1]:
        fingers.append(1)
    elif landmark_list[tipIds[0]][0] < landmark_list[tipIds[0] - 1][0]:
        fingers.append(1)
    else:
        fingers.append(0)
    # 其余四根手指
    for id in range(1, 5):
        if landmark_list[tipIds[id]][1] < landmark_list[tipIds[id] - 2][1]:
            fingers.append(1)
        else:
            fingers.append(0)
    # print(fingers)
    # 伸出手指数
    totalFingers = fingers.count(1)
    # print(totalFingers)
    return fingers




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

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point


def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list


def pre_process_point_history(image, point_history):
    image_width, image_height = image.shape[1], image.shape[0]

    temp_point_history = copy.deepcopy(point_history)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, point in enumerate(temp_point_history):
        if index == 0:
            base_x, base_y = point[0], point[1]

        temp_point_history[index][0] = (temp_point_history[index][0] - base_x) / image_width
        temp_point_history[index][1] = (temp_point_history[index][1] - base_y) / image_height

    # Convert to a one-dimensional list
    temp_point_history = list(
        itertools.chain.from_iterable(temp_point_history))

    return temp_point_history


def logging_csv(number, mode, landmark_list, point_history_list):
    print(mode)
    if mode == 0:
        pass
    if mode == 1 and (0 <= number <= 9):
        csv_path = 'model/keypoint_classifier/keypoint.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *landmark_list])
    if mode == 2 and (0 <= number <= 9):
        csv_path = 'model/point_history_classifier/point_history.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *point_history_list])
    return


def draw_landmarks(image, landmark_point):
    # 连线
    if len(landmark_point) > 0:
        # 拇指
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]), (255, 255, 255), 2)

        # 食指
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]), (255, 255, 255), 2)

        # 中指
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]), (255, 255, 255), 2)

        # 无名指
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]), (255, 255, 255), 2)

        # 小指
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]), (255, 255, 255), 2)

        # 手掌心
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]), (255, 255, 255), 2)

    # 21个关键点
    for index, landmark in enumerate(landmark_point):
        if index == 0:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 1:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 2:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 3:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 4:
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 5:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 6:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 7:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 8:
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 9:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 10:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 11:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 12:
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 13:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 14:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 15:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 16:
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 17:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 18:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 19:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 20:
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)

    return image


def draw_bounding_rect(use_brect, image, brect):
    if use_brect:

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
        cv.putText(image, "手势:" + finger_gesture_text, (10, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv.LINE_AA)
        cv.putText(image, "手势:" + finger_gesture_text, (10, 60),
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
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv.LINE_AA)
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv.LINE_AA)

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


def control_keyboard(most_common_keypoint_id, select_right_id, command, keyboard_TF=True, print_TF=True,
                     speed_up=False):
    if speed_up == False:
        if most_common_keypoint_id[0][0] == select_right_id and most_common_keypoint_id[0][1] == 5:
            if keyboard_TF:
                pyautogui.press(command)
            if print_TF:
                print(command)

def control_keyboard_hotkey(most_common_keypoint_id, select_right_id, command, keyboard_TF=True, print_TF=True
                     ):

    if most_common_keypoint_id[0][0] == select_right_id and most_common_keypoint_id[0][1] == 5:
        if keyboard_TF:
            pyautogui.hotkey(command)
        if print_TF:
            print(command)


def pick_gesture_command():
    left_number = input('left gesture number :')
    right_number = input('right gesture number :')
    command = input('what command :')
    return int(left_number), int(right_number), command


def pick_number(inputstring):
    keepask = True
    while keepask:
        try:
            number = input(f'{inputstring} :')
            number = int(number)
            if number < -1 or number > 3 or number == 0:
                raise Exception('number is not in range')
        except:
            print('choose again')

        else:
            keepask = False
            # print('choosing nicely')
    return number


def pick_command(inputstring='what command'):
    keepask = True
    while keepask:
        try:
            com = input(f'{inputstring} :')
            com_list = ['\t', '\n', '\r', ' ', '!', '"', '#', '$', '%', '&', "'", '(',
                        ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7',
                        '8', '9', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`',
                        'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o',
                        'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '{', '|', '}', '~',
                        'accept', 'add', 'alt', 'altleft', 'altright', 'apps', 'backspace',
                        'browserback', 'browserfavorites', 'browserforward', 'browserhome',
                        'browserrefresh', 'browsersearch', 'browserstop', 'capslock', 'clear',
                        'convert', 'ctrl', 'ctrlleft', 'ctrlright', 'decimal', 'del', 'delete',
                        'divide', 'down', 'end', 'enter', 'esc', 'escape', 'execute', 'f1', 'f10',
                        'f11', 'f12', 'f13', 'f14', 'f15', 'f16', 'f17', 'f18', 'f19', 'f2', 'f20',
                        'f21', 'f22', 'f23', 'f24', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9',
                        'final', 'fn', 'hanguel', 'hangul', 'hanja', 'help', 'home', 'insert', 'junja',
                        'kana', 'kanji', 'launchapp1', 'launchapp2', 'launchmail',
                        'launchmediaselect', 'left', 'modechange', 'multiply', 'nexttrack',
                        'nonconvert', 'num0', 'num1', 'num2', 'num3', 'num4', 'num5', 'num6',
                        'num7', 'num8', 'num9', 'numlock', 'pagedown', 'pageup', 'pause', 'pgdn',
                        'pgup', 'playpause', 'prevtrack', 'print', 'printscreen', 'prntscrn',
                        'prtsc', 'prtscr', 'return', 'right', 'scrolllock', 'select', 'separator',
                        'shift', 'shiftleft', 'shiftright', 'sleep', 'space', 'stop', 'subtract', 'tab',
                        'up', 'volumedown', 'volumemute', 'volumeup', 'win', 'winleft', 'winright', 'yen',
                        'command', 'option', 'optionleft', 'optionright']
            if com not in com_list:
                raise Exception('number is not in range')
        except:
            print('choose again')

        else:
            keepask = False
            print('choosing nicely')
    return com


def findDistance(p1, p2, img, draw=True, r=15, t=3):
    x1, y1 = p1
    x2, y2 = p2
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

    if draw:
        cv.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
        cv.circle(img, (x1, y1), r, (255, 0, 255), cv.FILLED)
        cv.circle(img, (x2, y2), r, (255, 0, 255), cv.FILLED)
        length = math.hypot(x2 - x1, y2 - y1)
        cv.circle(img, (cx, cy), r, (0, 0, 255), cv.FILLED)

    return length, img, [x1, y1, x2, y2, cx, cy]

def volumeControl(volChange=True):
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = interface.QueryInterface(IAudioEndpointVolume)
    # volume.GetMute()
    nowVol = volume.GetMasterVolumeLevel()  # 获取当前音量
    if volChange:
        nowVol = min(nowVol+4, 0)
    else:
        nowVol = max(nowVol-5, -65)

    volume.SetMasterVolumeLevel(nowVol, None)  # 调节音量




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




if __name__ == '__main__':
    main()
