# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'input.ui'
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
from PySide6.QtWidgets import (QApplication, QLabel, QPushButton, QSizePolicy,
    QWidget)

class Ui_Input(object):
    def setupUi(self, Input):
        if not Input.objectName():
            Input.setObjectName(u"Input")
        Input.resize(1099, 669)
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(Input.sizePolicy().hasHeightForWidth())
        Input.setSizePolicy(sizePolicy)
        self.no10 = QLabel(Input)
        self.no10.setObjectName(u"no10")
        self.no10.setGeometry(QRect(880, 480, 111, 21))
        font = QFont()
        font.setPointSize(14)
        self.no10.setFont(font)
        self.start_btn = QPushButton(Input)
        self.start_btn.setObjectName(u"start_btn")
        self.start_btn.setGeometry(QRect(920, 620, 161, 31))
        font1 = QFont()
        font1.setPointSize(18)
        self.start_btn.setFont(font1)
        self.label_4 = QLabel(Input)
        self.label_4.setObjectName(u"label_4")
        self.label_4.setGeometry(QRect(662, 304, 16, 16))
        self.epoch_lb = QLabel(Input)
        self.epoch_lb.setObjectName(u"epoch_lb")
        self.epoch_lb.setGeometry(QRect(1020, 480, 45, 25))
        self.epoch_lb.setFont(font)
        self.no3 = QLabel(Input)
        self.no3.setObjectName(u"no3")
        self.no3.setGeometry(QRect(600, 480, 152, 25))
        self.no3.setFont(font)
        self.no3.setAlignment(Qt.AlignmentFlag.AlignLeading|Qt.AlignmentFlag.AlignLeft|Qt.AlignmentFlag.AlignVCenter)
        self.no5 = QLabel(Input)
        self.no5.setObjectName(u"no5")
        self.no5.setGeometry(QRect(880, 430, 133, 25))
        self.no5.setFont(font)
        self.no5.setAlignment(Qt.AlignmentFlag.AlignLeading|Qt.AlignmentFlag.AlignLeft|Qt.AlignmentFlag.AlignVCenter)
        self.modelb = QLabel(Input)
        self.modelb.setObjectName(u"modelb")
        self.modelb.setGeometry(QRect(770, 530, 331, 25))
        self.modelb.setFont(font)
        self.ratelb = QLabel(Input)
        self.ratelb.setObjectName(u"ratelb")
        self.ratelb.setGeometry(QRect(770, 480, 51, 25))
        self.ratelb.setFont(font)
        self.dlb = QLabel(Input)
        self.dlb.setObjectName(u"dlb")
        self.dlb.setGeometry(QRect(770, 430, 50, 25))
        self.dlb.setFont(font)
        self.pic_lb = QLabel(Input)
        self.pic_lb.setObjectName(u"pic_lb")
        self.pic_lb.setGeometry(QRect(600, 70, 471, 331))
        self.display_lb = QLabel(Input)
        self.display_lb.setObjectName(u"display_lb")
        self.display_lb.setGeometry(QRect(20, 70, 512, 512))
        self.display_lb.setAlignment(Qt.AlignmentFlag.AlignLeading|Qt.AlignmentFlag.AlignLeft|Qt.AlignmentFlag.AlignTop)
        self.label_3 = QLabel(Input)
        self.label_3.setObjectName(u"label_3")
        self.label_3.setGeometry(QRect(600, 530, 95, 25))
        self.label_3.setFont(font)
        self.label = QLabel(Input)
        self.label.setObjectName(u"label")
        self.label.setGeometry(QRect(151, 51, 16, 16))
        self.no2 = QLabel(Input)
        self.no2.setObjectName(u"no2")
        self.no2.setGeometry(QRect(600, 20, 135, 35))
        font2 = QFont()
        font2.setPointSize(20)
        self.no2.setFont(font2)
        self.no4 = QLabel(Input)
        self.no4.setObjectName(u"no4")
        self.no4.setGeometry(QRect(600, 430, 131, 25))
        self.no4.setFont(font)
        self.no4.setAlignment(Qt.AlignmentFlag.AlignLeading|Qt.AlignmentFlag.AlignLeft|Qt.AlignmentFlag.AlignVCenter)
        self.lamblb = QLabel(Input)
        self.lamblb.setObjectName(u"lamblb")
        self.lamblb.setGeometry(QRect(1020, 430, 71, 25))
        self.lamblb.setFont(font)
        self.no1 = QLabel(Input)
        self.no1.setObjectName(u"no1")
        self.no1.setGeometry(QRect(20, 20, 135, 35))
        self.no1.setFont(font2)

        self.retranslateUi(Input)

        QMetaObject.connectSlotsByName(Input)
    # setupUi

    def retranslateUi(self, Input):
        Input.setWindowTitle(QCoreApplication.translate("Input", u"Form", None))
        self.no10.setText(QCoreApplication.translate("Input", u"\u8bad\u7ec3\u8f6e\u6b21\uff1a", None))
        self.start_btn.setText(QCoreApplication.translate("Input", u"\u5f00\u59cb\u751f\u6210", None))
        self.label_4.setText("")
        self.epoch_lb.setText(QCoreApplication.translate("Input", u"1500", None))
        self.no3.setText(QCoreApplication.translate("Input", u"\u91c7\u6837\u7387\uff08\u5fae\u7c73\uff09\uff1a", None))
        self.no5.setText(QCoreApplication.translate("Input", u"\u6ce2\u957f\uff08\u7eb3\u7c73\uff09\uff1a", None))
        self.modelb.setText(QCoreApplication.translate("Input", u"TextLabel", None))
        self.ratelb.setText(QCoreApplication.translate("Input", u"60", None))
        self.dlb.setText(QCoreApplication.translate("Input", u"1.580", None))
        self.pic_lb.setText("")
        self.display_lb.setText("")
        self.label_3.setText(QCoreApplication.translate("Input", u"\u4f7f\u7528\u7b97\u6cd5\uff1a", None))
        self.label.setText("")
        self.no2.setText(QCoreApplication.translate("Input", u"\u53c2\u6570\u9884\u89c8\uff1a", None))
        self.no4.setText(QCoreApplication.translate("Input", u"\u8ddd\u79bbd\uff08\u7c73)   \uff1a", None))
        self.lamblb.setText(QCoreApplication.translate("Input", u"632", None))
        self.no1.setText(QCoreApplication.translate("Input", u"\u8f93\u5165\u9884\u89c8\uff1a", None))
    # retranslateUi

