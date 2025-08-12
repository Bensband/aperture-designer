# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'histWindow.ui'
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
from PySide6.QtWidgets import (QApplication, QDialog, QListWidget, QListWidgetItem,
    QSizePolicy, QVBoxLayout, QWidget)

class Ui_HistWindow(object):
    def setupUi(self, HistWindow):
        if not HistWindow.objectName():
            HistWindow.setObjectName(u"HistWindow")
        HistWindow.resize(400, 300)
        self.verticalLayout = QVBoxLayout(HistWindow)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.hist_list = QListWidget(HistWindow)
        self.hist_list.setObjectName(u"hist_list")

        self.verticalLayout.addWidget(self.hist_list)


        self.retranslateUi(HistWindow)

        QMetaObject.connectSlotsByName(HistWindow)
    # setupUi

    def retranslateUi(self, HistWindow):
        HistWindow.setWindowTitle(QCoreApplication.translate("HistWindow", u"Dialog", None))
    # retranslateUi

