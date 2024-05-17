# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ui2.ui'
#
# Created by: PyQt5 UI code generator 5.13.2
#
# WARNING! All changes made in this file will be lost!
import cv2
import sys

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QMessageBox, QApplication, QMainWindow

from ui2 import Ui_MainWindow
from appMain import VideoCapture


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        # UI界面

        #self.timer = QTimer(self)
        self.setupUi(self)
        self.CAM_NUM = 0
        self.on_top = False
        #self.cap = cv2.VideoCapture()
        self.background()
        # 在label中播放视频
        #self.init_timer()



    def background(self):
        # 文件选择按钮
        self.openCamPushbutton.clicked.connect(self.start_camera)
        self.closeCamPushbutton.clicked.connect(self.stop_camera)
        self.checkBox.clicked.connect(self.toggle_window_stays_on_top)

        self.openCamPushbutton.setEnabled(True)
        # 初始状态不能关闭摄像头
        self.closeCamPushbutton.setEnabled(True)

    # 打开相机采集视频
    def open_camera(self):
        # 获取选择的设备名称
        index = self.comboBox.currentIndex()
        print(index)
        self.CAM_NUM = index
        # 检测该设备是否能打开
        flag = self.cap.open(self.CAM_NUM)
        print(flag)
        if flag is False:
            QMessageBox.information(self, "警告", "该设备未正常连接", QMessageBox.Ok)
        else:
            # 幕布可以播放
            self.camLabel.setEnabled(True)
            # 打开摄像头按钮不能点击
            self.openCamPushbutton.setEnabled(False)
            # 关闭摄像头按钮可以点击
            self.closeCamPushbutton.setEnabled(True)
        self.timer.start()
        print("beginning！")

    def close_camera(self):
        #self.cap.release()
        self.openCamPushbutton.setEnabled(True)
        self.closeCamPushbutton.setEnabled(False)
        #self.timer.stop()
        self.stop_video_capture()


    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        """更新图像的槽."""
        qt_img = self.convert_cv_qt(cv_img)
        self.camLabel.setAlignment(Qt.AlignCenter)
        self.camLabel.setPixmap(qt_img)

    def convert_cv_qt(self, cv_img):
        """将cv图像转换为QPixmap."""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w = rgb_image.shape[:2]
        #bytes_per_line = ch * w
        convert_to_qt_format = QImage(rgb_image.data, w, h, QImage.Format_RGB888)
        p = QPixmap.fromImage(convert_to_qt_format)
        ratio = max(w / self.camLabel.width(), h / self.camLabel.height())
        p.setDevicePixelRatio(ratio)
        return p

    def start_video_capture(self):
        self.video_capture = VideoCapture()
        self.video_capture.frame_captured.connect(self.update_image)  # 连接信号
        self.video_capture.data_sent.connect(self.receive_data)
        #self.video_capture.hand_sent.connect(self.receive_data)
        self.video_capture.main() # 在合适的时机开始捕获图像



    def receive_data(self, data_sent,hand_sent,hand_sent2,x,y,mode,kylabel0,kylabel00,phlabel0,phlabel00):
        #self.video_capture = VideoCapture()
        self.fpsLabel.setText(str(data_sent))
        self.handLabel.setText(str(hand_sent))#静态手势label
        self.handLabel_2.setText(str(hand_sent2))
        self.xLabel.setText(str(x))
        self.yLabel.setText(str(y))
        if mode == 1:
            self.modeLabel.setText('PPT播放控制模式')
            self.doLabel.setText(str(kylabel0))
            self.doLabel_2.setText(str(phlabel0))
        elif mode == 0:
            self.modeLabel.setText('睡眠模式')
        else:
            self.modeLabel.setText('媒体控制模式')
            self.doLabel.setText(str(kylabel00))
            self.doLabel_2.setText(str(phlabel00))



        # 在这里处理接收到的int值
        #print("接收到的数值：", data_sent,hand_sent)

    def start_camera(self):
        index = self.comboBox.currentIndex()
        self.video_capture.start_capture(camera_index=index)  # 开始捕获
        self.video_capture.main()

    def stop_camera(self):
        self.video_capture.stop_capture()  # 停止捕获

    def closeEvent(self, event):
        # 安全停止摄像头线程
        #self.camera_thread.stop()
        self.video_capture.closeSys()
        #self.stop_camera()
        # 你的其他清理逻辑
        super(MainWindow, self).closeEvent(event)

    def toggle_window_stays_on_top(self):
        if self.on_top:
            # 取消窗口始终置顶
            self.setWindowFlags(self.windowFlags() & ~Qt.WindowStaysOnTopHint)
            self.on_top = False
        else:
            # 设置窗口始终置顶
            self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)
            self.on_top = True

        # 重新显示窗口以使变更生效
        self.show()



if __name__ == '__main__':
    app = QApplication(sys.argv)
    main = MainWindow()
    main.show()
    main.start_video_capture()
    #main.stop_camera()
    sys.exit(app.exec_())






