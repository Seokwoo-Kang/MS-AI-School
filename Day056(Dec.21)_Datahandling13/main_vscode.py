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
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QAction, QBrush, QColor, QConicalGradient,
    QCursor, QFont, QFontDatabase, QGradient,
    QIcon, QImage, QKeySequence, QLinearGradient,
    QPainter, QPalette, QPixmap, QRadialGradient,
    QTransform)
from PySide6.QtWidgets import (QApplication, QHBoxLayout, QLabel, QMainWindow,
    QMenu, QMenuBar, QPushButton, QSizePolicy,
    QStatusBar, QVBoxLayout, QWidget)
import resource
import sys

class Ui_MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setupUi()

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
        self.Save = QAction(self)
        self.Save.setObjectName(u"Save")
        self.Close = QAction(self)
        self.Close.setObjectName(u"Close")
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

        self.verticalLayout_2.addWidget(self.paint)


        self.horizontalLayout.addLayout(self.verticalLayout_2)

        self.img_label = QLabel(self.centralwidget)
        self.img_label.setObjectName(u"img_label")

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



if __name__ == "__main__":
    app = QApplication(sys.argv)
    # app.setStyle('Fusion')
    widget = Ui_MainWindow()
    widget.show()
    sys.exit(app.exec())