# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'resizelog.ui'
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
    QLabel, QLineEdit, QSizePolicy, QWidget)

class Ui_Resizelog(object):
    def setupUi(self, Resizelog):
        if not Resizelog.objectName():
            Resizelog.setObjectName(u"Resizelog")
        Resizelog.resize(453, 316)
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(Resizelog.sizePolicy().hasHeightForWidth())
        Resizelog.setSizePolicy(sizePolicy)
        self.buttonBox = QDialogButtonBox(Resizelog)
        self.buttonBox.setObjectName(u"buttonBox")
        self.buttonBox.setGeometry(QRect(100, 280, 341, 32))
        self.buttonBox.setOrientation(Qt.Orientation.Horizontal)
        self.buttonBox.setStandardButtons(QDialogButtonBox.StandardButton.Cancel|QDialogButtonBox.StandardButton.Ok)
        self.label = QLabel(Resizelog)
        self.label.setObjectName(u"label")
        self.label.setGeometry(QRect(20, 20, 401, 91))
        self.label_2 = QLabel(Resizelog)
        self.label_2.setObjectName(u"label_2")
        self.label_2.setGeometry(QRect(140, 145, 31, 31))
        self.label_3 = QLabel(Resizelog)
        self.label_3.setObjectName(u"label_3")
        self.label_3.setGeometry(QRect(140, 194, 31, 31))
        self.widthEdit = QLineEdit(Resizelog)
        self.widthEdit.setObjectName(u"widthEdit")
        self.widthEdit.setGeometry(QRect(170, 150, 121, 21))
        self.heightEdit = QLineEdit(Resizelog)
        self.heightEdit.setObjectName(u"heightEdit")
        self.heightEdit.setGeometry(QRect(170, 200, 121, 20))

        self.retranslateUi(Resizelog)
        self.buttonBox.accepted.connect(Resizelog.accept)
        self.buttonBox.rejected.connect(Resizelog.reject)

        QMetaObject.connectSlotsByName(Resizelog)
    # setupUi

    def retranslateUi(self, Resizelog):
        Resizelog.setWindowTitle(QCoreApplication.translate("Resizelog", u"resize dialog", None))
        self.label.setText(QCoreApplication.translate("Resizelog", u"<html><head/><body><p><span style=\" font-size:12pt; font-weight:700;\">Warning: </span></p><p><span style=\" font-size:12pt;\">Due to interpolation,</span></p><p><span style=\" font-size:12pt;\">slight changes may occur on the original apperture</span></p></body></html>", None))
        self.label_2.setText(QCoreApplication.translate("Resizelog", u"<html><head/><body><p><span style=\" font-size:14pt;\">\u957f\uff1a</span></p></body></html>", None))
        self.label_3.setText(QCoreApplication.translate("Resizelog", u"<html><head/><body><p><span style=\" font-size:14pt;\">\u5bbd\uff1a</span></p></body></html>", None))
    # retranslateUi

