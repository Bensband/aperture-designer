# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'set_train.ui'
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
    QHBoxLayout, QLabel, QLineEdit, QSizePolicy,
    QVBoxLayout, QWidget)

class Ui_train_settiings(object):
    def setupUi(self, train_settiings):
        if not train_settiings.objectName():
            train_settiings.setObjectName(u"train_settiings")
        train_settiings.resize(266, 238)
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(train_settiings.sizePolicy().hasHeightForWidth())
        train_settiings.setSizePolicy(sizePolicy)
        self.verticalLayout = QVBoxLayout(train_settiings)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.label = QLabel(train_settiings)
        self.label.setObjectName(u"label")

        self.horizontalLayout.addWidget(self.label)

        self.num_epoch_edit = QLineEdit(train_settiings)
        self.num_epoch_edit.setObjectName(u"num_epoch_edit")
        sizePolicy1 = QSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.num_epoch_edit.sizePolicy().hasHeightForWidth())
        self.num_epoch_edit.setSizePolicy(sizePolicy1)
        self.num_epoch_edit.setMaximumSize(QSize(118000, 16777215))

        self.horizontalLayout.addWidget(self.num_epoch_edit)

        self.horizontalLayout.setStretch(0, 1)
        self.horizontalLayout.setStretch(1, 4)

        self.verticalLayout.addLayout(self.horizontalLayout)

        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.label_2 = QLabel(train_settiings)
        self.label_2.setObjectName(u"label_2")
        sizePolicy2 = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        sizePolicy2.setHorizontalStretch(0)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(self.label_2.sizePolicy().hasHeightForWidth())
        self.label_2.setSizePolicy(sizePolicy2)

        self.horizontalLayout_2.addWidget(self.label_2)

        self.pat_edit = QLineEdit(train_settiings)
        self.pat_edit.setObjectName(u"pat_edit")
        sizePolicy3 = QSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Maximum)
        sizePolicy3.setHorizontalStretch(0)
        sizePolicy3.setVerticalStretch(0)
        sizePolicy3.setHeightForWidth(self.pat_edit.sizePolicy().hasHeightForWidth())
        self.pat_edit.setSizePolicy(sizePolicy3)
        self.pat_edit.setMaximumSize(QSize(18000, 16777215))

        self.horizontalLayout_2.addWidget(self.pat_edit)

        self.horizontalLayout_2.setStretch(0, 1)
        self.horizontalLayout_2.setStretch(1, 4)

        self.verticalLayout.addLayout(self.horizontalLayout_2)

        self.horizontalLayout_3 = QHBoxLayout()
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.label_5 = QLabel(train_settiings)
        self.label_5.setObjectName(u"label_5")

        self.horizontalLayout_3.addWidget(self.label_5)

        self.mse_edit = QLineEdit(train_settiings)
        self.mse_edit.setObjectName(u"mse_edit")
        self.mse_edit.setMaximumSize(QSize(18000, 16777215))

        self.horizontalLayout_3.addWidget(self.mse_edit)

        self.horizontalLayout_3.setStretch(0, 1)
        self.horizontalLayout_3.setStretch(1, 4)

        self.verticalLayout.addLayout(self.horizontalLayout_3)

        self.horizontalLayout_4 = QHBoxLayout()
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.label_6 = QLabel(train_settiings)
        self.label_6.setObjectName(u"label_6")

        self.horizontalLayout_4.addWidget(self.label_6)

        self.freq_edit = QLineEdit(train_settiings)
        self.freq_edit.setObjectName(u"freq_edit")
        self.freq_edit.setMaximumSize(QSize(16000, 16777215))

        self.horizontalLayout_4.addWidget(self.freq_edit)


        self.verticalLayout.addLayout(self.horizontalLayout_4)

        self.horizontalLayout_5 = QHBoxLayout()
        self.horizontalLayout_5.setObjectName(u"horizontalLayout_5")
        self.label_4 = QLabel(train_settiings)
        self.label_4.setObjectName(u"label_4")

        self.horizontalLayout_5.addWidget(self.label_4)

        self.l1_edit = QLineEdit(train_settiings)
        self.l1_edit.setObjectName(u"l1_edit")
        self.l1_edit.setMaximumSize(QSize(16000, 16777215))

        self.horizontalLayout_5.addWidget(self.l1_edit)


        self.verticalLayout.addLayout(self.horizontalLayout_5)

        self.horizontalLayout_6 = QHBoxLayout()
        self.horizontalLayout_6.setObjectName(u"horizontalLayout_6")
        self.label_3 = QLabel(train_settiings)
        self.label_3.setObjectName(u"label_3")

        self.horizontalLayout_6.addWidget(self.label_3)

        self.SSMI_edit = QLineEdit(train_settiings)
        self.SSMI_edit.setObjectName(u"SSMI_edit")

        self.horizontalLayout_6.addWidget(self.SSMI_edit)


        self.verticalLayout.addLayout(self.horizontalLayout_6)

        self.horizontalLayout_7 = QHBoxLayout()
        self.horizontalLayout_7.setObjectName(u"horizontalLayout_7")
        self.label_7 = QLabel(train_settiings)
        self.label_7.setObjectName(u"label_7")

        self.horizontalLayout_7.addWidget(self.label_7)

        self.PSNR_edit = QLineEdit(train_settiings)
        self.PSNR_edit.setObjectName(u"PSNR_edit")

        self.horizontalLayout_7.addWidget(self.PSNR_edit)


        self.verticalLayout.addLayout(self.horizontalLayout_7)

        self.buttonBox = QDialogButtonBox(train_settiings)
        self.buttonBox.setObjectName(u"buttonBox")
        self.buttonBox.setOrientation(Qt.Orientation.Horizontal)
        self.buttonBox.setStandardButtons(QDialogButtonBox.StandardButton.Cancel|QDialogButtonBox.StandardButton.Ok)

        self.verticalLayout.addWidget(self.buttonBox)


        self.retranslateUi(train_settiings)
        self.buttonBox.accepted.connect(train_settiings.accept)
        self.buttonBox.rejected.connect(train_settiings.reject)

        QMetaObject.connectSlotsByName(train_settiings)
    # setupUi

    def retranslateUi(self, train_settiings):
        train_settiings.setWindowTitle(QCoreApplication.translate("train_settiings", u"Train Settings", None))
        self.label.setText(QCoreApplication.translate("train_settiings", u"\u8bad\u7ec3\u603b\u8f6e\u6b21\uff1a", None))
        self.label_2.setText(QCoreApplication.translate("train_settiings", u"\u65e9\u505c\u5fcd\u8010\u8f6e\u6b21\uff1a", None))
        self.label_5.setText(QCoreApplication.translate("train_settiings", u"MSELoss\u6743\u91cd\uff1a", None))
        self.label_6.setText(QCoreApplication.translate("train_settiings", u"\u52a0\u6743\u9891\u57df\u635f\u5931\u6743\u91cd\uff1a", None))
        self.label_4.setText(QCoreApplication.translate("train_settiings", u"l1Loss\u6743\u91cd:", None))
        self.label_3.setText(QCoreApplication.translate("train_settiings", u"SSMILoss\u6743\u91cd:", None))
        self.label_7.setText(QCoreApplication.translate("train_settiings", u"PSNRLoss\u6743\u91cd:", None))
    # retranslateUi

