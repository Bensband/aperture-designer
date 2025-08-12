# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'Input_Dialog.ui'
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
    QLabel, QLineEdit, QSizePolicy, QSlider,
    QWidget)

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        if not Dialog.objectName():
            Dialog.setObjectName(u"Dialog")
        Dialog.resize(733, 519)
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(Dialog.sizePolicy().hasHeightForWidth())
        Dialog.setSizePolicy(sizePolicy)
        self.buttonBox = QDialogButtonBox(Dialog)
        self.buttonBox.setObjectName(u"buttonBox")
        self.buttonBox.setGeometry(QRect(570, 480, 156, 24))
        self.buttonBox.setOrientation(Qt.Orientation.Horizontal)
        self.buttonBox.setStandardButtons(QDialogButtonBox.StandardButton.Cancel|QDialogButtonBox.StandardButton.Ok)
        self.label = QLabel(Dialog)
        self.label.setObjectName(u"label")
        self.label.setGeometry(QRect(20, 20, 481, 351))
        self.label.setPixmap(QPixmap(u":/pic/pic2.png"))
        self.dedit = QLineEdit(Dialog)
        self.dedit.setObjectName(u"dedit")
        self.dedit.setGeometry(QRect(170, 410, 101, 20))
        self.no2 = QLabel(Dialog)
        self.no2.setObjectName(u"no2")
        self.no2.setGeometry(QRect(80, 410, 61, 20))
        self.no3 = QLabel(Dialog)
        self.no3.setObjectName(u"no3")
        self.no3.setGeometry(QRect(80, 380, 81, 20))
        self.lambEdit = QLineEdit(Dialog)
        self.lambEdit.setObjectName(u"lambEdit")
        self.lambEdit.setGeometry(QRect(170, 380, 101, 20))
        self.no4 = QLabel(Dialog)
        self.no4.setObjectName(u"no4")
        self.no4.setGeometry(QRect(80, 440, 91, 20))
        self.rateEdit = QLineEdit(Dialog)
        self.rateEdit.setObjectName(u"rateEdit")
        self.rateEdit.setGeometry(QRect(170, 440, 101, 20))
        self.horizontalSlider = QSlider(Dialog)
        self.horizontalSlider.setObjectName(u"horizontalSlider")
        self.horizontalSlider.setGeometry(QRect(280, 410, 160, 16))
        self.horizontalSlider.setOrientation(Qt.Orientation.Horizontal)
        self.horizontalSlider_3 = QSlider(Dialog)
        self.horizontalSlider_3.setObjectName(u"horizontalSlider_3")
        self.horizontalSlider_3.setGeometry(QRect(280, 440, 160, 16))
        self.horizontalSlider_3.setOrientation(Qt.Orientation.Horizontal)

        self.retranslateUi(Dialog)
        self.buttonBox.accepted.connect(Dialog.accept)
        self.buttonBox.rejected.connect(Dialog.reject)

        QMetaObject.connectSlotsByName(Dialog)
    # setupUi

    def retranslateUi(self, Dialog):
        Dialog.setWindowTitle(QCoreApplication.translate("Dialog", u"Dialog", None))
        self.label.setText("")
        self.dedit.setPlaceholderText(QCoreApplication.translate("Dialog", u"\uff08\u5355\u4f4d\uff1a\u7c73\uff09", None))
        self.no2.setText(QCoreApplication.translate("Dialog", u"d\uff08\u7c73\uff09\uff1a", None))
        self.no3.setText(QCoreApplication.translate("Dialog", u"\u6ce2\u957f\uff08\u7eb3\u7c73\uff09\uff1a", None))
        self.lambEdit.setPlaceholderText(QCoreApplication.translate("Dialog", u"\uff08\u5355\u4f4d\uff1a\u7eb3\u7c73\uff09", None))
        self.no4.setText(QCoreApplication.translate("Dialog", u"\u91c7\u6837\u7387\uff08\u5fae\u7c73\uff09\uff1a", None))
        self.rateEdit.setPlaceholderText(QCoreApplication.translate("Dialog", u"\uff08\u5355\u4f4d\uff1a\u5fae\u7c73\uff09", None))
    # retranslateUi

