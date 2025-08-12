# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'design.ui'
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
from PySide6.QtWidgets import (QApplication, QHBoxLayout, QMainWindow, QMenu,
    QMenuBar, QPushButton, QSizePolicy, QStackedWidget,
    QStatusBar, QTextEdit, QVBoxLayout, QWidget)

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
        self.actionhist = QAction(MainWindow)
        self.actionhist.setObjectName(u"actionhist")
        self.actionFresne_amplitude = QAction(MainWindow)
        self.actionFresne_amplitude.setObjectName(u"actionFresne_amplitude")
        self.actionFresnel_phase = QAction(MainWindow)
        self.actionFresnel_phase.setObjectName(u"actionFresnel_phase")
        self.actionFraunhofer_amplitude = QAction(MainWindow)
        self.actionFraunhofer_amplitude.setObjectName(u"actionFraunhofer_amplitude")
        self.actionForward_diffraction_stimulator = QAction(MainWindow)
        self.actionForward_diffraction_stimulator.setObjectName(u"actionForward_diffraction_stimulator")
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.verticalLayout = QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.input_btn = QPushButton(self.centralwidget)
        self.input_btn.setObjectName(u"input_btn")

        self.horizontalLayout.addWidget(self.input_btn)

        self.aper_btn = QPushButton(self.centralwidget)
        self.aper_btn.setObjectName(u"aper_btn")

        self.horizontalLayout.addWidget(self.aper_btn)

        self.rec_btn = QPushButton(self.centralwidget)
        self.rec_btn.setObjectName(u"rec_btn")

        self.horizontalLayout.addWidget(self.rec_btn)

        self.more_btn = QPushButton(self.centralwidget)
        self.more_btn.setObjectName(u"more_btn")

        self.horizontalLayout.addWidget(self.more_btn)


        self.verticalLayout.addLayout(self.horizontalLayout)

        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.stackedWidget = QStackedWidget(self.centralwidget)
        self.stackedWidget.setObjectName(u"stackedWidget")
        

        self.horizontalLayout_2.addWidget(self.stackedWidget)

        self.horizontalLayout_2.setStretch(0, 8)

        self.verticalLayout.addLayout(self.horizontalLayout_2)

        self.log_edit = QTextEdit(self.centralwidget)
        self.log_edit.setObjectName(u"log_edit")
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.log_edit.sizePolicy().hasHeightForWidth())
        self.log_edit.setSizePolicy(sizePolicy)
        self.log_edit.setMaximumSize(QSize(16777215, 70))

        self.verticalLayout.addWidget(self.log_edit)

        self.verticalLayout.setStretch(0, 1)
        self.verticalLayout.setStretch(1, 20)
        self.verticalLayout.setStretch(2, 6)
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 820, 33))
        self.menu13 = QMenu(self.menubar)
        self.menu13.setObjectName(u"menu13")
        self.menuSettings = QMenu(self.menubar)
        self.menuSettings.setObjectName(u"menuSettings")
        self.menuMode = QMenu(self.menubar)
        self.menuMode.setObjectName(u"menuMode")
        self.menuOthers = QMenu(self.menubar)
        self.menuOthers.setObjectName(u"menuOthers")
        MainWindow.setMenuBar(self.menubar)

        self.menubar.addAction(self.menu13.menuAction())
        self.menubar.addAction(self.menuSettings.menuAction())
        self.menubar.addAction(self.menuMode.menuAction())
        self.menubar.addAction(self.menuOthers.menuAction())
        self.menu13.addAction(self.actionload)
        self.menu13.addAction(self.actionhist)
        self.menuSettings.addAction(self.actioninitial)
        self.menuSettings.addAction(self.actiontraining)
        self.menuMode.addAction(self.actionFresne_amplitude)
        self.menuMode.addAction(self.actionFresnel_phase)
        self.menuOthers.addAction(self.actionForward_diffraction_stimulator)

        self.retranslateUi(MainWindow)

        self.stackedWidget.setCurrentIndex(1)


        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"MainWindow", None))
        self.actionload.setText(QCoreApplication.translate("MainWindow", u"load", None))
        self.actioninitial.setText(QCoreApplication.translate("MainWindow", u"initial settings", None))
        self.actiontraining.setText(QCoreApplication.translate("MainWindow", u"training settings", None))
        self.actionhist.setText(QCoreApplication.translate("MainWindow", u"history", None))
        self.actionFresne_amplitude.setText(QCoreApplication.translate("MainWindow", u"Fresnel_amplitude", None))
        self.actionFresnel_phase.setText(QCoreApplication.translate("MainWindow", u"Fresnel_phase", None))
        self.actionFraunhofer_amplitude.setText(QCoreApplication.translate("MainWindow", u"Fraunhofer_amplitude", None))
        self.actionForward_diffraction_stimulator.setText(QCoreApplication.translate("MainWindow", u"Forward diffraction stimulator", None))
        self.input_btn.setText(QCoreApplication.translate("MainWindow", u"\u8f93\u5165\u9884\u89c8", None))
        self.aper_btn.setText(QCoreApplication.translate("MainWindow", u"\u63a9\u819c", None))
        self.rec_btn.setText(QCoreApplication.translate("MainWindow", u"\u91cd\u5efa\u884d\u5c04\u56fe\u6837", None))
        self.more_btn.setText(QCoreApplication.translate("MainWindow", u"\u66f4\u591a\u7ed3\u679c", None))
        self.menu13.setTitle(QCoreApplication.translate("MainWindow", u"File", None))
        self.menuSettings.setTitle(QCoreApplication.translate("MainWindow", u"Settings", None))
        self.menuMode.setTitle(QCoreApplication.translate("MainWindow", u"Mode", None))
        self.menuOthers.setTitle(QCoreApplication.translate("MainWindow", u"Others", None))
    # retranslateUi

