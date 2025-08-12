# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'Loss_window.ui'
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
    QHBoxLayout, QLabel, QSizePolicy, QSpacerItem,
    QVBoxLayout, QWidget)

class Ui_Loss_window(object):
    def setupUi(self, Loss_window):
        if not Loss_window.objectName():
            Loss_window.setObjectName(u"Loss_window")
        Loss_window.resize(400, 300)
        self.verticalLayout = QVBoxLayout(Loss_window)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.label = QLabel(Loss_window)
        self.label.setObjectName(u"label")
        font = QFont()
        font.setPointSize(16)
        self.label.setFont(font)

        self.verticalLayout.addWidget(self.label)

        self.dis_lb = QLabel(Loss_window)
        self.dis_lb.setObjectName(u"dis_lb")

        self.verticalLayout.addWidget(self.dis_lb)

        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.label_3 = QLabel(Loss_window)
        self.label_3.setObjectName(u"label_3")

        self.horizontalLayout.addWidget(self.label_3)

        self.ssmi_lb = QLabel(Loss_window)
        self.ssmi_lb.setObjectName(u"ssmi_lb")

        self.horizontalLayout.addWidget(self.ssmi_lb)

        self.horizontalSpacer = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout.addItem(self.horizontalSpacer)


        self.verticalLayout.addLayout(self.horizontalLayout)

        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.label_2 = QLabel(Loss_window)
        self.label_2.setObjectName(u"label_2")

        self.horizontalLayout_2.addWidget(self.label_2)

        self.psnr_lb = QLabel(Loss_window)
        self.psnr_lb.setObjectName(u"psnr_lb")

        self.horizontalLayout_2.addWidget(self.psnr_lb)

        self.horizontalSpacer_2 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_2.addItem(self.horizontalSpacer_2)


        self.verticalLayout.addLayout(self.horizontalLayout_2)

        self.buttonBox = QDialogButtonBox(Loss_window)
        self.buttonBox.setObjectName(u"buttonBox")
        self.buttonBox.setOrientation(Qt.Orientation.Horizontal)
        self.buttonBox.setStandardButtons(QDialogButtonBox.StandardButton.Cancel|QDialogButtonBox.StandardButton.Ok)

        self.verticalLayout.addWidget(self.buttonBox)


        self.retranslateUi(Loss_window)
        self.buttonBox.accepted.connect(Loss_window.accept)
        self.buttonBox.rejected.connect(Loss_window.reject)

        QMetaObject.connectSlotsByName(Loss_window)
    # setupUi

    def retranslateUi(self, Loss_window):
        Loss_window.setWindowTitle(QCoreApplication.translate("Loss_window", u"Dialog", None))
        self.label.setText(QCoreApplication.translate("Loss_window", u"Loss\u8bb0\u5f55\uff1a", None))
        self.dis_lb.setText("")
        self.label_3.setText(QCoreApplication.translate("Loss_window", u"SSMI\uff1a", None))
        self.ssmi_lb.setText("")
        self.label_2.setText(QCoreApplication.translate("Loss_window", u"PSNR:", None))
        self.psnr_lb.setText(QCoreApplication.translate("Loss_window", u"TextLabel", None))
    # retranslateUi

