# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'train_dialog.ui'
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
from PySide6.QtWidgets import (QApplication, QDialog, QHBoxLayout, QLabel,
    QPushButton, QSizePolicy, QTextEdit, QVBoxLayout,
    QWidget)

class Ui_Training_Diag(object):
    def setupUi(self, Training_Diag):
        if not Training_Diag.objectName():
            Training_Diag.setObjectName(u"Training_Diag")
        Training_Diag.resize(400, 300)
        self.verticalLayout = QVBoxLayout(Training_Diag)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.no1 = QLabel(Training_Diag)
        self.no1.setObjectName(u"no1")

        self.verticalLayout.addWidget(self.no1)

        self.textEdit = QTextEdit(Training_Diag)
        self.textEdit.setObjectName(u"textEdit")

        self.verticalLayout.addWidget(self.textEdit)

        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.no2 = QLabel(Training_Diag)
        self.no2.setObjectName(u"no2")

        self.horizontalLayout.addWidget(self.no2)

        self.stop_btn = QPushButton(Training_Diag)
        self.stop_btn.setObjectName(u"stop_btn")

        self.horizontalLayout.addWidget(self.stop_btn)

        self.horizontalLayout.setStretch(0, 3)
        self.horizontalLayout.setStretch(1, 1)

        self.verticalLayout.addLayout(self.horizontalLayout)


        self.retranslateUi(Training_Diag)

        QMetaObject.connectSlotsByName(Training_Diag)
    # setupUi

    def retranslateUi(self, Training_Diag):
        Training_Diag.setWindowTitle(QCoreApplication.translate("Training_Diag", u"Dialog", None))
        self.no1.setText(QCoreApplication.translate("Training_Diag", u"\u8bad\u7ec3\u4e2d\uff0c\u8bf7\u7a0d\u540e\uff1a", None))
        self.no2.setText("")
        self.stop_btn.setText(QCoreApplication.translate("Training_Diag", u"\u6682\u505c\u5e76\u9000\u51fa", None))
    # retranslateUi

