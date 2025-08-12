# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'MSE_Window.ui'
##
## Created by: Qt User Interface Compiler version 6.9.0
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QAbstractButton, QApplication, QDialog, QDialogButtonBox,
    QLabel, QSizePolicy, QVBoxLayout, QWidget)

class Ui_mse_window(object):
    def setupUi(self, mse_window):
        if not mse_window.objectName():
            mse_window.setObjectName(u"mse_window")
        mse_window.resize(530, 604)
        font = QFont()
        font.setPointSize(16)
        mse_window.setFont(font)
        self.verticalLayout = QVBoxLayout(mse_window)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.label = QLabel(mse_window)
        self.label.setObjectName(u"label")

        self.verticalLayout.addWidget(self.label)

        self.label_2 = QLabel(mse_window)
        self.label_2.setObjectName(u"label_2")
        font1 = QFont()
        font1.setPointSize(11)
        self.label_2.setFont(font1)

        self.verticalLayout.addWidget(self.label_2)

        self.dis_lb = QLabel(mse_window)
        self.dis_lb.setObjectName(u"dis_lb")

        self.verticalLayout.addWidget(self.dis_lb)

        self.buttonBox = QDialogButtonBox(mse_window)
        self.buttonBox.setObjectName(u"buttonBox")
        font2 = QFont()
        font2.setPointSize(9)
        self.buttonBox.setFont(font2)
        self.buttonBox.setOrientation(Qt.Orientation.Horizontal)
        self.buttonBox.setStandardButtons(QDialogButtonBox.StandardButton.Cancel|QDialogButtonBox.StandardButton.Ok)

        self.verticalLayout.addWidget(self.buttonBox)


        self.retranslateUi(mse_window)
        self.buttonBox.accepted.connect(mse_window.accept)
        self.buttonBox.rejected.connect(mse_window.reject)

        QMetaObject.connectSlotsByName(mse_window)
    # setupUi

    def retranslateUi(self, mse_window):
        mse_window.setWindowTitle(QCoreApplication.translate("mse_window", u"Dialog", None))
        self.label.setText(QCoreApplication.translate("mse_window", u"MSELoss\u53ef\u89c6\u5316:", None))
        self.label_2.setText(QCoreApplication.translate("mse_window", u"(\u989c\u8272\u8d8a\u7ea2\uff0c\u5dee\u5f02\u8d8a\u5927\uff09", None))
        self.dis_lb.setText("")
    # retranslateUi

