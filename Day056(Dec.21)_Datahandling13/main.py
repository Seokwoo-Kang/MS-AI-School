# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'ui.ui'
##
## Created by: Qt User Interface Compiler version 6.4.1
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt, QSaveFile)
from PySide6.QtGui import (QAction, QBrush, QColor, QConicalGradient,
    QCursor, QFont, QFontDatabase, QGradient,
    QIcon, QImage, QKeySequence, QLinearGradient,
    QPainter, QPalette, QPixmap, QRadialGradient,
    QTransform, QImageReader)
from PySide6.QtWidgets import (QApplication, QHBoxLayout, QLabel, QMainWindow,
    QMenu, QMenuBar, QPushButton, QSizePolicy,
    QStatusBar, QVBoxLayout, QWidget, QFileDialog, QDialog)

import resource
import sys
import os
from pathlib import Path
import cv2


def get_supported_mime_types():
    result = []    
    for f in QImageReader().supportedMimeTypes():        
        data = f.data()
        # print(data)        
        result.append(data.decode('utf-8'))    
        return result



class Ui_MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi()

        # 이미지 경로를 저장하는 변수 초기화'
        self.img_url = None

        # 마우스 위치 저장 변수
        self.begin = QPoint()
        self.end = QPoint()

        # 그리기 위한 도구함
        self.painter = QPainter()

        # # 저장하기 위한 도구함
        # self.saving = QSaveFile()
        self.save_img = None

        # 브러쉬 설정
        self.br = QBrush(QColor(230, 10, 10, 75))
        # self.painter.setBrush(self.br)

        # 버튼을 눌렀을때만 그릴수 있도록 bool 변수 하나 선언
        self.is_paint = False

        # 사각 박스 좌표 저장 리스트
        self.bbox_list = []

        # pixmap 초기화
        self.pixmap = None

        # pixmap 시작 좌표 초기화
        self.set_p_x = 0
        self.set_p_y = 0

        # 되돌리기 기능 추가(컨트롤 + Z 등록)
        prev_action = QAction(self)
        # prev_action.setShortcut(QKeySequence.Undo)
        prev_action.setShortcut(QKeySequence(Qt.CTRL | Qt.Key_Z))
        prev_action.triggered.connect(self.prev_bbox)
        self.addAction(prev_action)

        # 저장 기능 추가(컨트롤 + S 등록)
        save_action = QAction(self)
        # prev_action.setShortcut(QKeySequence.Undo)
        save_action.setShortcut(QKeySequence(Qt.CTRL | Qt.Key_S))
        # save_action.triggered.connect(self.saving)
        self.addAction(save_action)        



    def setupUi(self):
        if not self.objectName():
            self.setObjectName(u"self")
        self.resize(1280, 800)
        self.setMinimumSize(QSize(1280, 800))
        self.setMaximumSize(QSize(1280, 800))
        self.setSizeIncrement(QSize(1280, 800))
        self.setBaseSize(QSize(1280, 800))


        self.Open = QAction(self)
        self.Open.setObjectName(u"Open")
        self.Open.triggered.connect(self.img_open)
        self.Save = QAction(self)
        self.Save.setObjectName(u"Save")
        # self.Open.triggered.connect(self.saving)
        self.Close = QAction(self)
        self.Close.setObjectName(u"Close")
        self.Close.triggered.connect(QCoreApplication.quit)


        self.centralwidget = QWidget(self)
        self.centralwidget.setObjectName(u"centralwidget")
        self.horizontalLayout = QHBoxLayout(self.centralwidget)
        self.horizontalLayout.setSpacing(0)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_2 = QVBoxLayout()
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.paint = QPushButton(self.centralwidget)
        self.paint.setObjectName(u"paint")
        self.paint.setMinimumSize(QSize(0, 50))
        icon = QIcon()
        icon.addFile(u":/ann/paint.png", QSize(), QIcon.Normal, QIcon.Off)
        self.paint.setIcon(icon)
        self.paint.setIconSize(QSize(25, 25))
        self.paint.clicked.connect(self.clicked_paint)
        self.paint.setShortcut(QCoreApplication.translate("self", u"N", None))

        # 버튼구현은 강의를 다시 보면서 터미널로 ui기반 기본 코드생성을 통해 연결하고 나서 하는걸로
        # self.saving = QPushButton(self.centralwidget)
        # self.saving.setObjectName(u"save")
        # self.saving.setMinimumSize(QSize(35, 35))
        # icon = QIcon()
        # icon.addFile(u":/ann/save.jpeg", QSize(), QIcon.Normal, QIcon.Off)
        # self.saving.setIcon(icon)
        # self.saving.setIconSize(QSize(25, 25))
        # self.saving.clicked.connect(self.click_saving)
        # # self.saving.setShortcut(QCoreApplication.translate("self", u"S", None))

        self.verticalLayout_2.addWidget(self.paint)


        self.horizontalLayout.addLayout(self.verticalLayout_2)

        self.img_label = QLabel(self.centralwidget)
        self.img_label.setObjectName(u"img_label")
        self.img_label.setAlignment(Qt.AlignCenter)

        self.horizontalLayout.addWidget(self.img_label)

        self.horizontalLayout.setStretch(0, 1)
        self.horizontalLayout.setStretch(1, 23)
        self.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(self)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 1280, 30))
        self.menu = QMenu(self.menubar)
        self.menu.setObjectName(u"menu")
        self.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(self)
        self.statusbar.setObjectName(u"statusbar")
        self.setStatusBar(self.statusbar)

        self.menubar.addAction(self.menu.menuAction())
        self.menu.addAction(self.Open)
        self.menu.addAction(self.Save)
        self.menu.addAction(self.Close)

        self.retranslateUi()

        QMetaObject.connectSlotsByName(self)
    # setupUi

    def retranslateUi(self):
        self.setWindowTitle(QCoreApplication.translate("self", u"self", None))
        self.Open.setText(QCoreApplication.translate("self", u"\uc5f4\uae30", None))
#if QT_CONFIG(shortcut)
        self.Open.setShortcut(QCoreApplication.translate("self", u"Ctrl+O", None))
#endif // QT_CONFIG(shortcut)
        self.Save.setText(QCoreApplication.translate("self", u"\uc800\uc7a5", None))
#if QT_CONFIG(shortcut)
        self.Save.setShortcut(QCoreApplication.translate("self", u"Ctrl+S", None))
#endif // QT_CONFIG(shortcut)
        self.Close.setText(QCoreApplication.translate("self", u"\ub2eb\uae30", None))
#if QT_CONFIG(shortcut)
        self.Close.setShortcut(QCoreApplication.translate("self", u"Ctrl+F4", None))
#endif // QT_CONFIG(shortcut)
        self.paint.setText("")
        self.img_label.setText("")
        self.menu.setTitle(QCoreApplication.translate("self", u"\ud30c\uc77c", None))
    # retranslateUi

    def img_open(self):
        open_path = str(Path(__file__).resolve().parent)
        file_dialog = QFileDialog(self, '이미지 열기', open_path)
 
        # _mime_types = get_supported_mime_types()
        # file_dialog.setMimeTypeFilters(_mime_types)
        # print(_mime_types)
 
        # default_mimetype = 'image/png'
        # if default_mimetype in _mime_types:
        #     file_dialog.selectMimeTypeFilter(default_mimetype)

        file_dialog.setMimeTypeFilters(['image/png'])
 
        if file_dialog.exec() == QDialog.Accepted:
            url = file_dialog.selectedUrls()[0]
            self.img_url = url
            # print(url)

            self.pixmap = QPixmap(url.toLocalFile())
            self.pixmap = self.pixmap.scaled(self.img_label.size(), aspectMode= Qt.KeepAspectRatio)
            # self.img_label.setPixmap(self.pixmap)

            l_width = self.img_label.geometry().width()
            l_height = self.img_label.geometry().height()
            l_x = self.img_label.geometry().x()
            l_y = self.img_label.geometry().y()

            # 메뉴바 크기도 고려해야함(실제 0,0 은 메튜바에 가림)
            m_height = self.menubar.geometry().height()

            l_center_x = (l_width//2) + l_x
            l_center_y = (l_height//2) + l_y + m_height

            p_width = self.pixmap.rect().width()
            p_height = self.pixmap.rect().height()

            self.set_p_x = l_center_x - (p_width//2)
            self.set_p_y = l_center_y - (p_height//2)
            # print(self.set_p_x, self.set_p_y)

    # 그리기 버튼 눌렀을 때 이벤트
    def clicked_paint(self):

        if not self.is_paint:
            self.is_paint = True



    def prev_bbox(self):
        # 맨 마지막 것을 제외한 나머지를 다시 그려야함
        if len(self.bbox_list) == 0:
            return
        
        # 새로 그리기 위해서 pixmap을 다시 정의한다.
        self.pixmap = QPixmap(self.img_url.toLocalFile())
        self.pixmap = self.pixmap.scaled(self.img_label.size(), aspectMode=Qt.KeepAspectRatio)

        self.bbox_list = self.bbox_list[:-1]
        # print(self.bbox_list) 결과출력 X

        self.painter.begin(self.pixmap)
        self.painter.setBrush(self.br)

        for qrect in self.bbox_list:
            # print(self.bbox_list) 결과출력 X
            self.painter.drawRect(qrect)

        self.painter.end()
        self.update()


    def paintEvent(self, event):   ## 도화지 그려주는 함수 정의
        if self.pixmap:
            with QPainter(self) as painter:
                painter.drawPixmap(self.set_p_x, self.set_p_y, self.pixmap)

                if self.is_paint:
                    painter.setBrush(self.br)
                    painter.drawRect(QRect(self.begin, self.end))

    def mousePressEvent(self, event):
        self.begin = event.position().toPoint()

        QWidget.mousePressEvent(self, event)

    def mouseMoveEvent(self, event):
        self.end = event.position().toPoint()
        self.update()

        QWidget.mouseMoveEvent(self, event)

    def mouseReleaseEvent(self, event):
        self.end = event.position().toPoint()

        # print(self.is_paint)

        if self.pixmap and self.is_paint:
            self.painter.begin(self.pixmap)
            self.painter.setBrush(self.br)

            begin = list(self.begin.toTuple())
            end = list(self.end.toTuple())

            begin[0] -= self.set_p_x
            begin[1] -= self.set_p_y
            end[0] -= self.set_p_x
            end[1] -= self.set_p_y

            w = end[0] -begin[0]
            h = end[1] -begin[1]

            self.painter.drawRect(QRect(begin[0], begin[1],w,h))
            self.painter.end()

            self.bbox_list.append(QRect(begin[0], begin[1],w,h))

            self.is_paint = False
        
        self.update()
        QWidget.mouseReleaseEvent(self, event)

    def click_saving(self):
        pass

    # cv2나 PIL로 저장하는 방법 스킵(앞선과정에서 bbox 좌표 csv저장하고 이미지 저장하는 과제했음)
    # ./bboxSave.py 참고
    # Ptqt 클래스와 메소드 이용하는 방법 (시도 1)
    # def saving(self, save_event):

    #     self.pixmap = QPixmap(self.img_url.toLocalFile())
    #     self.pixmap = self.pixmap.scaled(self.img_label.size(), aspectMode=Qt.KeepAspectRatio)

    #     self.painter.begin(self.pixmap)
    #     self.painter.setBrush(self.br)

    #     self.bbox_list = self.bbox_list[:]
    #     print(self.bbox_list)

    #     for qrect in self.bbox_list:
    #         self.painter.drawRect(qrect)

    #     self.painter.end()
    #     self.update()        

    #     # self.deleteLater()
    #     # self.saveState()
    #     # self.saveGeometry()

    #     # save_path = str(Path(__file__).resolve().parent)
    #     # file_dialog = QFileDialog(self, '이미지 저장', save_path)
 
    #     # _mime_types = get_supported_mime_types()
    #     # file_dialog.setMimeTypeFilters(_mime_types)
    #     # print(_mime_types)
 
    #     # default_mimetype = 'image/png'
    #     # if default_mimetype in _mime_types:
    #     #     file_dialog.selectMimeTypeFilter(default_mimetype)

    #     # if file_dialog.result() == QDialog.Accepted:
    #     #     url = file_dialog.saveState()
    #     #     self.img_url = url
    #     #     print(url)

    #     # file_dialog.setMimeTypeFilters(['image/png'])


    #     # save_event.accept()
    #     pass

    


    def closeEvent(self, close_event):
        
        self.deleteLater()
        
        close_event.accept()
        # close_event.ignore()    # close 기능 무시



if __name__ == "__main__":
    app = QApplication(sys.argv)
    # app.setStyle('Fusion')
    widget = Ui_MainWindow()
    widget.show()
    sys.exit(app.exec())