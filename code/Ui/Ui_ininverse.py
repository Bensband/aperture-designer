# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'ininverse.ui'
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
from PySide6.QtWidgets import (QApplication, QComboBox, QLabel, QLineEdit,
    QPushButton, QSizePolicy, QSlider, QWidget)

class Ui_ForwDif(object):
    def setupUi(self, ForwDif):
        if not ForwDif.objectName():
            ForwDif.setObjectName(u"ForwDif")
        ForwDif.resize(1420, 610)
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(ForwDif.sizePolicy().hasHeightForWidth())
        ForwDif.setSizePolicy(sizePolicy)
        self.label = QLabel(ForwDif)
        self.label.setObjectName(u"label")
        self.label.setGeometry(QRect(10, 0, 131, 51))
        font = QFont()
        font.setPointSize(18)
        self.label.setFont(font)
        self.label_2 = QLabel(ForwDif)
        self.label_2.setObjectName(u"label_2")
        self.label_2.setGeometry(QRect(840, 45, 111, 61))
        self.label_2.setFont(font)
        self.comboBox = QComboBox(ForwDif)
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.setObjectName(u"comboBox")
        self.comboBox.setGeometry(QRect(130, 10, 261, 31))
        self.comboBox.setStyleSheet(u"font-size: 16px")
        self.label_4 = QLabel(ForwDif)
        self.label_4.setObjectName(u"label_4")
        self.label_4.setGeometry(QRect(840, 100, 391, 291))
        self.horizontalSlider = QSlider(ForwDif)
        self.horizontalSlider.setObjectName(u"horizontalSlider")
        self.horizontalSlider.setGeometry(QRect(1040, 440, 160, 16))
        self.horizontalSlider.setOrientation(Qt.Orientation.Horizontal)
        self.no2 = QLabel(ForwDif)
        self.no2.setObjectName(u"no2")
        self.no2.setGeometry(QRect(850, 440, 81, 20))
        font1 = QFont()
        font1.setPointSize(12)
        self.no2.setFont(font1)
        self.dedit = QLineEdit(ForwDif)
        self.dedit.setObjectName(u"dedit")
        self.dedit.setGeometry(QRect(930, 440, 101, 20))
        self.no3 = QLabel(ForwDif)
        self.no3.setObjectName(u"no3")
        self.no3.setGeometry(QRect(850, 400, 101, 21))
        self.no3.setFont(font1)
        self.rateEdit = QLineEdit(ForwDif)
        self.rateEdit.setObjectName(u"rateEdit")
        self.rateEdit.setGeometry(QRect(990, 480, 101, 20))
        self.horizontalSlider_3 = QSlider(ForwDif)
        self.horizontalSlider_3.setObjectName(u"horizontalSlider_3")
        self.horizontalSlider_3.setGeometry(QRect(1100, 480, 160, 16))
        self.horizontalSlider_3.setOrientation(Qt.Orientation.Horizontal)
        self.no4 = QLabel(ForwDif)
        self.no4.setObjectName(u"no4")
        self.no4.setGeometry(QRect(850, 480, 131, 20))
        self.no4.setFont(font1)
        self.lambEdit = QLineEdit(ForwDif)
        self.lambEdit.setObjectName(u"lambEdit")
        self.lambEdit.setGeometry(QRect(960, 400, 101, 21))
        self.dis_aperlb = QLabel(ForwDif)
        self.dis_aperlb.setObjectName(u"dis_aperlb")
        self.dis_aperlb.setGeometry(QRect(40, 90, 351, 381))
        self.label_5 = QLabel(ForwDif)
        self.label_5.setObjectName(u"label_5")
        self.label_5.setGeometry(QRect(40, 55, 111, 41))
        font2 = QFont()
        font2.setPointSize(16)
        self.label_5.setFont(font2)
        self.label_6 = QLabel(ForwDif)
        self.label_6.setObjectName(u"label_6")
        self.label_6.setGeometry(QRect(430, 55, 111, 41))
        self.label_6.setFont(font2)
        self.dis_diflb = QLabel(ForwDif)
        self.dis_diflb.setObjectName(u"dis_diflb")
        self.dis_diflb.setGeometry(QRect(430, 90, 351, 381))
        self.startbtn = QPushButton(ForwDif)
        self.startbtn.setObjectName(u"startbtn")
        self.startbtn.setGeometry(QRect(1060, 550, 151, 31))
        self.startbtn.setFont(font2)
        self.inputbtn = QPushButton(ForwDif)
        self.inputbtn.setObjectName(u"inputbtn")
        self.inputbtn.setGeometry(QRect(150, 62, 91, 31))
        self.inputbtn.setFont(font1)
        self.save_btn = QPushButton(ForwDif)
        self.save_btn.setObjectName(u"save_btn")
        self.save_btn.setGeometry(QRect(1230, 550, 151, 31))
        self.save_btn.setFont(font2)
        self.resizebtn = QPushButton(ForwDif)
        self.resizebtn.setObjectName(u"resizebtn")
        self.resizebtn.setGeometry(QRect(250, 62, 121, 31))
        self.resizebtn.setFont(font1)

        self.retranslateUi(ForwDif)

        self.comboBox.setCurrentIndex(0)


        QMetaObject.connectSlotsByName(ForwDif)
    # setupUi

    def retranslateUi(self, ForwDif):
        ForwDif.setWindowTitle(QCoreApplication.translate("ForwDif", u"\u6b63\u5411\u884d\u5c04\u6a21\u62df", None))
        self.label.setText(QCoreApplication.translate("ForwDif", u"\u4f7f\u7528\u7b97\u6cd5\uff1a", None))
        self.label_2.setText(QCoreApplication.translate("ForwDif", u"\u521d\u59cb\u6761\u4ef6\uff1a", None))
        self.comboBox.setItemText(0, QCoreApplication.translate("ForwDif", u"\u83f2\u6d85\u5c14\u884d\u5c04\uff08\u632f\u5e45\u578b\uff09", None))
        self.comboBox.setItemText(1, QCoreApplication.translate("ForwDif", u"\u83f2\u6d85\u5c14\u884d\u5c04\uff08\u76f8\u4f4d\u578b\uff09", None))

        self.label_4.setText("")
        self.no2.setText(QCoreApplication.translate("ForwDif", u"d\uff08\u7c73\uff09\uff1a", None))
        self.dedit.setPlaceholderText(QCoreApplication.translate("ForwDif", u"\uff08\u5355\u4f4d\uff1a\u7c73\uff09", None))
        self.no3.setText(QCoreApplication.translate("ForwDif", u"\u6ce2\u957f\uff08\u7eb3\u7c73\uff09\uff1a", None))
        self.rateEdit.setPlaceholderText(QCoreApplication.translate("ForwDif", u"\uff08\u5355\u4f4d\uff1a\u5fae\u7c73\uff09", None))
        self.no4.setText(QCoreApplication.translate("ForwDif", u"\u91c7\u6837\u7387\uff08\u5fae\u7c73\uff09\uff1a", None))
        self.lambEdit.setPlaceholderText(QCoreApplication.translate("ForwDif", u"\uff08\u5355\u4f4d\uff1a\u7eb3\u7c73\uff09", None))
        self.dis_aperlb.setText("")
        self.label_5.setText(QCoreApplication.translate("ForwDif", u"\u8f93\u5165\u63a9\u819c\uff1a", None))
        self.label_6.setText(QCoreApplication.translate("ForwDif", u"\u884d\u5c04\u56fe\u50cf\uff1a", None))
        self.dis_diflb.setText("")
        self.startbtn.setText(QCoreApplication.translate("ForwDif", u"\u5f00\u59cb\u751f\u6210", None))
        self.inputbtn.setText(QCoreApplication.translate("ForwDif", u"\u5bfc\u5165\u56fe\u50cf", None))
        self.save_btn.setText(QCoreApplication.translate("ForwDif", u"\u4fdd\u5b58\u7ed3\u679c", None))
        self.resizebtn.setText(QCoreApplication.translate("ForwDif", u"\u8c03\u6574\u56fe\u7247\u5c3a\u5bf8", None))
    # retranslateUi

