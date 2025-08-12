# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'rec.ui'
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
from PySide6.QtWidgets import (QApplication, QHBoxLayout, QLabel, QRadioButton,
    QSizePolicy, QVBoxLayout, QWidget)

class Ui_Reconstruct(object):
    def setupUi(self, Reconstruct):
        if not Reconstruct.objectName():
            Reconstruct.setObjectName(u"Reconstruct")
        Reconstruct.resize(714, 548)
        self.verticalLayout = QVBoxLayout(Reconstruct)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.label = QLabel(Reconstruct)
        self.label.setObjectName(u"label")
        font = QFont()
        font.setPointSize(16)
        self.label.setFont(font)

        self.verticalLayout.addWidget(self.label)

        self.display_lb = QLabel(Reconstruct)
        self.display_lb.setObjectName(u"display_lb")
        self.display_lb.setAlignment(Qt.AlignmentFlag.AlignLeading|Qt.AlignmentFlag.AlignLeft|Qt.AlignmentFlag.AlignVCenter)

        self.verticalLayout.addWidget(self.display_lb)

        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.orig_radioButton = QRadioButton(Reconstruct)
        self.orig_radioButton.setObjectName(u"orig_radioButton")

        self.horizontalLayout.addWidget(self.orig_radioButton)

        self.light_radioButton = QRadioButton(Reconstruct)
        self.light_radioButton.setObjectName(u"light_radioButton")

        self.horizontalLayout.addWidget(self.light_radioButton)

        self.label_2 = QLabel(Reconstruct)
        self.label_2.setObjectName(u"label_2")

        self.horizontalLayout.addWidget(self.label_2)


        self.verticalLayout.addLayout(self.horizontalLayout)

        self.verticalLayout.setStretch(0, 1)
        self.verticalLayout.setStretch(1, 20)
        self.verticalLayout.setStretch(2, 2)

        self.retranslateUi(Reconstruct)

        QMetaObject.connectSlotsByName(Reconstruct)
    # setupUi

    def retranslateUi(self, Reconstruct):
        Reconstruct.setWindowTitle(QCoreApplication.translate("Reconstruct", u"Form", None))
        self.label.setText(QCoreApplication.translate("Reconstruct", u"\u91cd\u5efa\u884d\u5c04\u56fe\u6837\uff1a", None))
        self.display_lb.setText("")
        self.orig_radioButton.setText(QCoreApplication.translate("Reconstruct", u"\u539f\u59cb\u56fe\u50cf", None))
        self.light_radioButton.setText(QCoreApplication.translate("Reconstruct", u"\u63d0\u4eae\u56fe\u50cf", None))
        self.label_2.setText("")
    # retranslateUi

