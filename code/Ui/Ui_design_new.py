# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'design_new.ui'
##
## Created by: Qt User Interface Compiler version 6.9.0
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QAction, QBrush, QColor, QConicalGradient,
    QCursor, QFont, QFontDatabase, QGradient,
    QIcon, QImage, QKeySequence, QLinearGradient,
    QPainter, QPalette, QPixmap, QRadialGradient,
    QTransform)
from PySide6.QtWidgets import (QApplication, QGridLayout, QLabel, QMainWindow,
    QMenu, QMenuBar, QSizePolicy, QSlider,
    QSpacerItem, QStatusBar, QTextEdit, QWidget)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(820, 699)
        self.actionload = QAction(MainWindow)
        self.actionload.setObjectName(u"actionload")
        self.actioninitial = QAction(MainWindow)
        self.actioninitial.setObjectName(u"actioninitial")
        self.actiontraining = QAction(MainWindow)
        self.actiontraining.setObjectName(u"actiontraining")
        self.actionhistory = QAction(MainWindow)
        self.actionhistory.setObjectName(u"actionhistory")
        self.actionstart_stimulation = QAction(MainWindow)
        self.actionstart_stimulation.setObjectName(u"actionstart_stimulation")
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.gridLayout = QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName(u"gridLayout")
        self.lambEdit = QLabel(self.centralwidget)
        self.lambEdit.setObjectName(u"lambEdit")

        self.gridLayout.addWidget(self.lambEdit, 5, 0, 1, 1)

        self.dedit = QLabel(self.centralwidget)
        self.dedit.setObjectName(u"dedit")

        self.gridLayout.addWidget(self.dedit, 4, 0, 1, 1)

        self.rec_origin_dis = QLabel(self.centralwidget)
        self.rec_origin_dis.setObjectName(u"rec_origin_dis")
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.rec_origin_dis.sizePolicy().hasHeightForWidth())
        self.rec_origin_dis.setSizePolicy(sizePolicy)
        self.rec_origin_dis.setPixmap(QPixmap(u":/D:/python_works(conda)/pytorch/3_unet/pyside6/proj_4_stacked/test5.jpeg"))

        self.gridLayout.addWidget(self.rec_origin_dis, 4, 3, 2, 1)

        self.log_edit = QTextEdit(self.centralwidget)
        self.log_edit.setObjectName(u"log_edit")
        sizePolicy1 = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.log_edit.sizePolicy().hasHeightForWidth())
        self.log_edit.setSizePolicy(sizePolicy1)
        self.log_edit.setMaximumSize(QSize(16777215, 100))

        self.gridLayout.addWidget(self.log_edit, 7, 0, 1, 5)

        self.rec_dis = QLabel(self.centralwidget)
        self.rec_dis.setObjectName(u"rec_dis")
        self.rec_dis.setPixmap(QPixmap(u":/D:/python_works(conda)/pytorch/3_unet/pyside6/proj_4_stacked/test5.jpeg"))
        self.rec_dis.setScaledContents(False)

        self.gridLayout.addWidget(self.rec_dis, 4, 4, 2, 1)

        self.horizontalSlider = QSlider(self.centralwidget)
        self.horizontalSlider.setObjectName(u"horizontalSlider")
        sizePolicy2 = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        sizePolicy2.setHorizontalStretch(0)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(self.horizontalSlider.sizePolicy().hasHeightForWidth())
        self.horizontalSlider.setSizePolicy(sizePolicy2)
        self.horizontalSlider.setOrientation(Qt.Orientation.Horizontal)

        self.gridLayout.addWidget(self.horizontalSlider, 4, 1, 1, 1)

        self.rateEdit = QLabel(self.centralwidget)
        self.rateEdit.setObjectName(u"rateEdit")

        self.gridLayout.addWidget(self.rateEdit, 5, 1, 1, 1)

        self.init_dis = QLabel(self.centralwidget)
        self.init_dis.setObjectName(u"init_dis")

        self.gridLayout.addWidget(self.init_dis, 1, 0, 3, 2)

        self.label = QLabel(self.centralwidget)
        self.label.setObjectName(u"label")
        font = QFont()
        font.setPointSize(16)
        self.label.setFont(font)

        self.gridLayout.addWidget(self.label, 0, 0, 1, 2)

        self.apper_dis = QLabel(self.centralwidget)
        self.apper_dis.setObjectName(u"apper_dis")
        self.apper_dis.setPixmap(QPixmap(u":/D:/python_works(conda)/pytorch/3_unet/pyside6/proj_4_stacked/test5.jpeg"))

        self.gridLayout.addWidget(self.apper_dis, 1, 3, 2, 1)

        self.label_13 = QLabel(self.centralwidget)
        self.label_13.setObjectName(u"label_13")
        self.label_13.setFont(font)

        self.gridLayout.addWidget(self.label_13, 3, 4, 1, 1)

        self.label_12 = QLabel(self.centralwidget)
        self.label_12.setObjectName(u"label_12")
        self.label_12.setFont(font)

        self.gridLayout.addWidget(self.label_12, 3, 3, 1, 1)

        self.label_11 = QLabel(self.centralwidget)
        self.label_11.setObjectName(u"label_11")
        self.label_11.setFont(font)

        self.gridLayout.addWidget(self.label_11, 0, 4, 1, 1)

        self.label_10 = QLabel(self.centralwidget)
        self.label_10.setObjectName(u"label_10")
        self.label_10.setFont(font)

        self.gridLayout.addWidget(self.label_10, 0, 3, 1, 1)

        self.input_dis = QLabel(self.centralwidget)
        self.input_dis.setObjectName(u"input_dis")
        self.input_dis.setPixmap(QPixmap(u":/D:/python_works(conda)/pytorch/3_unet/pyside6/proj_4_stacked/test5.jpeg"))

        self.gridLayout.addWidget(self.input_dis, 1, 4, 2, 1)

        self.horizontalSpacer = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.gridLayout.addItem(self.horizontalSpacer, 0, 2, 1, 1)

        self.horizontalSpacer_2 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.gridLayout.addItem(self.horizontalSpacer_2, 5, 2, 1, 1)

        self.horizontalSpacer_3 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.gridLayout.addItem(self.horizontalSpacer_3, 4, 2, 1, 1)

        self.horizontalSpacer_4 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.gridLayout.addItem(self.horizontalSpacer_4, 3, 2, 1, 1)

        self.horizontalSpacer_5 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.gridLayout.addItem(self.horizontalSpacer_5, 2, 2, 1, 1)

        self.horizontalSpacer_6 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.gridLayout.addItem(self.horizontalSpacer_6, 1, 2, 1, 1)

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 820, 33))
        self.menu13 = QMenu(self.menubar)
        self.menu13.setObjectName(u"menu13")
        self.menuHelp = QMenu(self.menubar)
        self.menuHelp.setObjectName(u"menuHelp")
        self.menuSettings = QMenu(self.menubar)
        self.menuSettings.setObjectName(u"menuSettings")
        self.menuOptions = QMenu(self.menubar)
        self.menuOptions.setObjectName(u"menuOptions")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.menubar.addAction(self.menu13.menuAction())
        self.menubar.addAction(self.menuSettings.menuAction())
        self.menubar.addAction(self.menuOptions.menuAction())
        self.menubar.addAction(self.menuHelp.menuAction())
        self.menu13.addAction(self.actionload)
        self.menu13.addAction(self.actionhistory)
        self.menuSettings.addAction(self.actioninitial)
        self.menuSettings.addAction(self.actiontraining)
        self.menuOptions.addAction(self.actionstart_stimulation)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"MainWindow", None))
        self.actionload.setText(QCoreApplication.translate("MainWindow", u"load", None))
        self.actioninitial.setText(QCoreApplication.translate("MainWindow", u"initial settings", None))
        self.actiontraining.setText(QCoreApplication.translate("MainWindow", u"training settings", None))
        self.actionhistory.setText(QCoreApplication.translate("MainWindow", u"history", None))
        self.actionstart_stimulation.setText(QCoreApplication.translate("MainWindow", u"start stimulation", None))
        self.lambEdit.setText(QCoreApplication.translate("MainWindow", u"TextLabel", None))
        self.dedit.setText(QCoreApplication.translate("MainWindow", u"TextLabel", None))
        self.rec_origin_dis.setText("")
        self.rec_dis.setText("")
        self.rateEdit.setText(QCoreApplication.translate("MainWindow", u"TextLabel", None))
        self.init_dis.setText("")
        self.label.setText(QCoreApplication.translate("MainWindow", u"\u521d\u59cb\u6761\u4ef6\u9884\u89c8\uff1a", None))
        self.apper_dis.setText("")
        self.label_13.setText(QCoreApplication.translate("MainWindow", u"\u91cd\u5efa\uff08\u63d0\u4eae\uff09", None))
        self.label_12.setText(QCoreApplication.translate("MainWindow", u"\u91cd\u5efa\u56fe\u50cf\uff08\u539f\u59cb\uff09", None))
        self.label_11.setText(QCoreApplication.translate("MainWindow", u"\u9884\u6d4b\u5149\u9611", None))
        self.label_10.setText(QCoreApplication.translate("MainWindow", u"\u76ee\u6807\u56fe\u50cf\uff1a", None))
        self.input_dis.setText("")
        self.menu13.setTitle(QCoreApplication.translate("MainWindow", u"File", None))
        self.menuHelp.setTitle(QCoreApplication.translate("MainWindow", u"Help", None))
        self.menuSettings.setTitle(QCoreApplication.translate("MainWindow", u"Settings", None))
        self.menuOptions.setTitle(QCoreApplication.translate("MainWindow", u"Options", None))
    # retranslateUi

