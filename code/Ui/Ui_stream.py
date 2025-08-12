# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'stream.ui'
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
from PySide6.QtWidgets import (QApplication, QHBoxLayout, QLabel, QSizePolicy,
    QVBoxLayout, QWidget)

class Ui_Stream(object):
    def setupUi(self, Stream):
        if not Stream.objectName():
            Stream.setObjectName(u"Stream")
        Stream.resize(714, 548)
        self.verticalLayout = QVBoxLayout(Stream)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.no1 = QLabel(Stream)
        self.no1.setObjectName(u"no1")
        font = QFont()
        font.setPointSize(16)
        self.no1.setFont(font)

        self.verticalLayout.addWidget(self.no1)

        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.no2 = QLabel(Stream)
        self.no2.setObjectName(u"no2")
        font1 = QFont()
        font1.setPointSize(12)
        self.no2.setFont(font1)

        self.horizontalLayout.addWidget(self.no2)

        self.no3 = QLabel(Stream)
        self.no3.setObjectName(u"no3")
        self.no3.setFont(font1)

        self.horizontalLayout.addWidget(self.no3)


        self.verticalLayout.addLayout(self.horizontalLayout)

        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.input_lb = QLabel(Stream)
        self.input_lb.setObjectName(u"input_lb")

        self.horizontalLayout_2.addWidget(self.input_lb)

        self.rec_light_lb = QLabel(Stream)
        self.rec_light_lb.setObjectName(u"rec_light_lb")

        self.horizontalLayout_2.addWidget(self.rec_light_lb)


        self.verticalLayout.addLayout(self.horizontalLayout_2)

        self.horizontalLayout_3 = QHBoxLayout()
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.no4 = QLabel(Stream)
        self.no4.setObjectName(u"no4")
        self.no4.setFont(font1)

        self.horizontalLayout_3.addWidget(self.no4)

        self.no5 = QLabel(Stream)
        self.no5.setObjectName(u"no5")
        self.no5.setFont(font1)

        self.horizontalLayout_3.addWidget(self.no5)


        self.verticalLayout.addLayout(self.horizontalLayout_3)

        self.horizontalLayout_4 = QHBoxLayout()
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.aper_lb = QLabel(Stream)
        self.aper_lb.setObjectName(u"aper_lb")

        self.horizontalLayout_4.addWidget(self.aper_lb)

        self.rec_lb = QLabel(Stream)
        self.rec_lb.setObjectName(u"rec_lb")

        self.horizontalLayout_4.addWidget(self.rec_lb)


        self.verticalLayout.addLayout(self.horizontalLayout_4)

        self.verticalLayout.setStretch(0, 1)
        self.verticalLayout.setStretch(1, 1)
        self.verticalLayout.setStretch(2, 10)
        self.verticalLayout.setStretch(3, 1)
        self.verticalLayout.setStretch(4, 10)

        self.retranslateUi(Stream)

        QMetaObject.connectSlotsByName(Stream)
    # setupUi

    def retranslateUi(self, Stream):
        Stream.setWindowTitle(QCoreApplication.translate("Stream", u"Form", None))
        self.no1.setText(QCoreApplication.translate("Stream", u"\u6574\u4f53\u6d41\u7a0b", None))
        self.no2.setText(QCoreApplication.translate("Stream", u"1.\u8f93\u5165\u56fe\u6837\uff1a", None))
        self.no3.setText(QCoreApplication.translate("Stream", u"4.\u91cd\u5efa\u884d\u5c04\u56fe\uff08\u7b49\u6bd4\u4f8b\u63d0\u4eae\uff09", None))
        self.input_lb.setText("")
        self.rec_light_lb.setText("")
        self.no4.setText(QCoreApplication.translate("Stream", u"2.\u8f93\u51fa\u5149\u9611", None))
        self.no5.setText(QCoreApplication.translate("Stream", u"3.\u91cd\u5efa\u884d\u5c04\u56fe\uff08\u539f\u59cb\uff09", None))
        self.aper_lb.setText("")
        self.rec_lb.setText("")
    # retranslateUi

