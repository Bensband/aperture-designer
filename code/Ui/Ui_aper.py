# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'aper.ui'
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
from PySide6.QtWidgets import (QApplication, QLabel, QSizePolicy, QVBoxLayout,
    QWidget)

class Ui_Aperture(object):
    def setupUi(self, Aperture):
        if not Aperture.objectName():
            Aperture.setObjectName(u"Aperture")
        Aperture.resize(714, 548)
        self.verticalLayout = QVBoxLayout(Aperture)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.no2 = QLabel(Aperture)
        self.no2.setObjectName(u"no2")
        font = QFont()
        font.setPointSize(16)
        self.no2.setFont(font)

        self.verticalLayout.addWidget(self.no2)

        self.display_lb = QLabel(Aperture)
        self.display_lb.setObjectName(u"display_lb")
        self.display_lb.setAlignment(Qt.AlignmentFlag.AlignLeading|Qt.AlignmentFlag.AlignLeft|Qt.AlignmentFlag.AlignVCenter)

        self.verticalLayout.addWidget(self.display_lb)

        self.verticalLayout.setStretch(0, 1)
        self.verticalLayout.setStretch(1, 20)

        self.retranslateUi(Aperture)

        QMetaObject.connectSlotsByName(Aperture)
    # setupUi

    def retranslateUi(self, Aperture):
        Aperture.setWindowTitle(QCoreApplication.translate("Aperture", u"Form", None))
        self.no2.setText(QCoreApplication.translate("Aperture", u"\u5149\u9611\uff1a", None))
        self.display_lb.setText("")
    # retranslateUi

