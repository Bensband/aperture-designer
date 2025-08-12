# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'More.ui'
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
from PySide6.QtWidgets import (QApplication, QHBoxLayout, QLabel, QPushButton,
    QSizePolicy, QWidget)

class Ui_more(object):
    def setupUi(self, more):
        if not more.objectName():
            more.setObjectName(u"more")
        more.resize(1170, 853)
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(more.sizePolicy().hasHeightForWidth())
        more.setSizePolicy(sizePolicy)
        self.no1 = QLabel(more)
        self.no1.setObjectName(u"no1")
        self.no1.setGeometry(QRect(10, 10, 201, 41))
        font = QFont()
        font.setPointSize(26)
        self.no1.setFont(font)
        self.label_5 = QLabel(more)
        self.label_5.setObjectName(u"label_5")
        self.label_5.setGeometry(QRect(820, 70, 221, 27))
        font1 = QFont()
        font1.setPointSize(18)
        self.label_5.setFont(font1)
        self.no2 = QLabel(more)
        self.no2.setObjectName(u"no2")
        self.no2.setGeometry(QRect(820, 460, 105, 27))
        self.no2.setFont(font1)
        self.dis_input_lb = QLabel(more)
        self.dis_input_lb.setObjectName(u"dis_input_lb")
        self.dis_input_lb.setGeometry(QRect(20, 100, 331, 331))
        self.layoutWidget = QWidget(more)
        self.layoutWidget.setObjectName(u"layoutWidget")
        self.layoutWidget.setGeometry(QRect(820, 500, 211, 31))
        self.horizontalLayout_2 = QHBoxLayout(self.layoutWidget)
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.loss_btn = QPushButton(self.layoutWidget)
        self.loss_btn.setObjectName(u"loss_btn")
        sizePolicy1 = QSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.loss_btn.sizePolicy().hasHeightForWidth())
        self.loss_btn.setSizePolicy(sizePolicy1)
        self.loss_btn.setMaximumSize(QSize(150, 16777215))
        font2 = QFont()
        font2.setPointSize(12)
        self.loss_btn.setFont(font2)

        self.horizontalLayout_2.addWidget(self.loss_btn)

        self.mse_btn = QPushButton(self.layoutWidget)
        self.mse_btn.setObjectName(u"mse_btn")
        sizePolicy1.setHeightForWidth(self.mse_btn.sizePolicy().hasHeightForWidth())
        self.mse_btn.setSizePolicy(sizePolicy1)
        self.mse_btn.setMaximumSize(QSize(150, 16777215))
        self.mse_btn.setFont(font2)

        self.horizontalLayout_2.addWidget(self.mse_btn)

        self.label_4 = QLabel(more)
        self.label_4.setObjectName(u"label_4")
        self.label_4.setGeometry(QRect(390, 460, 271, 31))
        self.label_4.setFont(font1)
        self.label_3 = QLabel(more)
        self.label_3.setObjectName(u"label_3")
        self.label_3.setGeometry(QRect(20, 460, 241, 31))
        self.label_3.setFont(font1)
        self.label = QLabel(more)
        self.label.setObjectName(u"label")
        self.label.setGeometry(QRect(390, 60, 201, 31))
        self.label.setFont(font1)
        self.label_2 = QLabel(more)
        self.label_2.setObjectName(u"label_2")
        self.label_2.setGeometry(QRect(20, 60, 181, 31))
        self.label_2.setFont(font1)
        self.psnrlb = QLabel(more)
        self.psnrlb.setObjectName(u"psnrlb")
        self.psnrlb.setGeometry(QRect(820, 110, 281, 21))
        font3 = QFont()
        font3.setPointSize(14)
        self.psnrlb.setFont(font3)
        self.ssimlb = QLabel(more)
        self.ssimlb.setObjectName(u"ssimlb")
        self.ssimlb.setGeometry(QRect(820, 140, 291, 31))
        self.ssimlb.setFont(font3)
        self.dis_aper_lb = QLabel(more)
        self.dis_aper_lb.setObjectName(u"dis_aper_lb")
        self.dis_aper_lb.setGeometry(QRect(390, 100, 331, 331))
        self.dis_rec_lb = QLabel(more)
        self.dis_rec_lb.setObjectName(u"dis_rec_lb")
        self.dis_rec_lb.setGeometry(QRect(390, 500, 331, 331))
        self.dis_origin_lb = QLabel(more)
        self.dis_origin_lb.setObjectName(u"dis_origin_lb")
        self.dis_origin_lb.setGeometry(QRect(20, 500, 331, 331))
        sizePolicy2 = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        sizePolicy2.setHorizontalStretch(0)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(self.dis_origin_lb.sizePolicy().hasHeightForWidth())
        self.dis_origin_lb.setSizePolicy(sizePolicy2)

        self.retranslateUi(more)

        QMetaObject.connectSlotsByName(more)
    # setupUi

    def retranslateUi(self, more):
        more.setWindowTitle(QCoreApplication.translate("more", u"More Results", None))
        self.no1.setText(QCoreApplication.translate("more", u"\u6d41\u7a0b\u8bb0\u5f55:", None))
        self.label_5.setText(QCoreApplication.translate("more", u"\u91cd\u5efa\u56fe\u50cf\u8d28\u91cf\u8bc4\u4f30\uff1a", None))
        self.no2.setText(QCoreApplication.translate("more", u"\u66f4\u591a\u9009\u9879\uff1a", None))
        self.dis_input_lb.setText("")
        self.loss_btn.setText(QCoreApplication.translate("more", u"\u8bad\u7ec3\u8be6\u60c5", None))
        self.mse_btn.setText(QCoreApplication.translate("more", u"MSE\u50cf\u7d20\u5dee\u503c", None))
        self.label_4.setText(QCoreApplication.translate("more", u"\u91cd\u5efa\u884d\u5c04\u56fe\u50cf\uff08\u63d0\u4eae/\u6697\uff09", None))
        self.label_3.setText(QCoreApplication.translate("more", u"\u91cd\u5efa\u884d\u5c04\u56fe\u50cf\uff08\u539f\u59cb\uff09", None))
        self.label.setText(QCoreApplication.translate("more", u"\u9884\u6d4b\u63a9\u819c\uff08\u8f93\u51fa\uff09", None))
        self.label_2.setText(QCoreApplication.translate("more", u"\u76ee\u6807\u56fe\u50cf\uff08\u8f93\u5165\uff09", None))
        self.psnrlb.setText("")
        self.ssimlb.setText("")
        self.dis_aper_lb.setText("")
        self.dis_rec_lb.setText("")
        self.dis_origin_lb.setText("")
    # retranslateUi

