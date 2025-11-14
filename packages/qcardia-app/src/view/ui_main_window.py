# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'main_window.ui'
##
## Created by: Qt User Interface Compiler version 6.8.0
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (
    QCoreApplication,
    QDate,
    QDateTime,
    QLocale,
    QMetaObject,
    QObject,
    QPoint,
    QRect,
    QSize,
    Qt,
    QTime,
    QUrl,
)
from PySide6.QtGui import (
    QBrush,
    QColor,
    QConicalGradient,
    QCursor,
    QFont,
    QFontDatabase,
    QGradient,
    QIcon,
    QImage,
    QKeySequence,
    QLinearGradient,
    QPainter,
    QPalette,
    QPixmap,
    QRadialGradient,
    QTransform,
)
from PySide6.QtWidgets import (
    QApplication,
    QGraphicsView,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMenuBar,
    QPushButton,
    QSizePolicy,
    QStatusBar,
    QTableView,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName("MainWindow")
        MainWindow.resize(824, 641)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.treeWidget = QTreeWidget(self.centralwidget)
        __qtreewidgetitem = QTreeWidgetItem(self.treeWidget)
        __qtreewidgetitem.setFlags(
            Qt.ItemIsDragEnabled
            | Qt.ItemIsDropEnabled
            | Qt.ItemIsUserCheckable
            | Qt.ItemIsEnabled
        )
        self.treeWidget.setObjectName("treeWidget")
        self.treeWidget.setGeometry(QRect(0, 0, 171, 641))
        sizePolicy = QSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.treeWidget.sizePolicy().hasHeightForWidth())
        self.treeWidget.setSizePolicy(sizePolicy)
        self.treeWidget.setAnimated(True)
        self.treeWidget.setExpandsOnDoubleClick(True)
        self.graphicsView = QGraphicsView(self.centralwidget)
        self.graphicsView.setObjectName("graphicsView")
        self.graphicsView.setGeometry(QRect(170, 70, 281, 331))
        self.widget = QWidget(self.centralwidget)
        self.widget.setObjectName("widget")
        self.widget.setGeometry(QRect(720, 180, 89, 146))
        self.verticalLayout_2 = QVBoxLayout(self.widget)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.graphicsView_2 = QGraphicsView(self.centralwidget)
        self.graphicsView_2.setObjectName("graphicsView_2")
        self.graphicsView_2.setGeometry(QRect(460, 70, 281, 331))
        self.label_27 = QLabel(self.centralwidget)
        self.label_27.setObjectName("label_27")
        self.label_27.setGeometry(QRect(260, 20, 376, 36))
        font = QFont()
        QFontDatabase.addApplicationFont("assets/fonts/segoeui.ttf")
        QFontDatabase.addApplicationFont("assets/fonts/segoeuib.ttf")
        font.setFamily("Segoe UI")
        font.setPointSize(30)
        font.setBold(False)
        font.setItalic(True)
        font.setUnderline(False)
        font.setStrikeOut(False)
        self.label_27.setFont(font)
        self.label_27.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.tableView = QTableView(self.centralwidget)
        self.tableView.setObjectName("tableView")
        self.tableView.setGeometry(QRect(520, 410, 221, 161))
        self.load_button = QPushButton(self.centralwidget)
        self.load_button.setObjectName("load_button")
        self.load_button.setGeometry(QRect(750, 190, 65, 32))
        self.seg_button = QPushButton(self.centralwidget)
        self.seg_button.setObjectName("seg_button")
        self.seg_button.setGeometry(QRect(750, 260, 65, 32))
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName("menubar")
        self.menubar.setGeometry(QRect(0, 0, 824, 37))
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)

    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(
            QCoreApplication.translate("MainWindow", "MainWindow", None)
        )
        ___qtreewidgetitem = self.treeWidget.headerItem()
        ___qtreewidgetitem.setText(
            0, QCoreApplication.translate("MainWindow", "Dashboard", None)
        )

        __sortingEnabled = self.treeWidget.isSortingEnabled()
        self.treeWidget.setSortingEnabled(False)
        ___qtreewidgetitem1 = self.treeWidget.topLevelItem(0)
        ___qtreewidgetitem1.setText(
            0, QCoreApplication.translate("MainWindow", "Patient name", None)
        )
        self.treeWidget.setSortingEnabled(__sortingEnabled)

        # if QT_CONFIG(accessibility)
        self.graphicsView.setAccessibleName(
            QCoreApplication.translate("MainWindow", "graphicsView", None)
        )
        # endif // QT_CONFIG(accessibility)
        # if QT_CONFIG(accessibility)
        self.graphicsView_2.setAccessibleName(
            QCoreApplication.translate("MainWindow", "graphicsView", None)
        )
        # endif // QT_CONFIG(accessibility)
        self.label_27.setText(
            QCoreApplication.translate("MainWindow", "q-cardIA", None)
        )
        # if QT_CONFIG(tooltip)
        self.load_button.setToolTip(
            QCoreApplication.translate(
                "MainWindow", "<html><head/><body><p>Tool Tip</p></body></html>", None
            )
        )
        # endif // QT_CONFIG(tooltip)
        # if QT_CONFIG(whatsthis)
        self.load_button.setWhatsThis(
            QCoreApplication.translate(
                "MainWindow", "<html><head/><body><p>Whats this</p></body></html>", None
            )
        )
        # endif // QT_CONFIG(whatsthis)
        # if QT_CONFIG(accessibility)
        self.load_button.setAccessibleName(
            QCoreApplication.translate("MainWindow", "load_or_save_button", None)
        )
        # endif // QT_CONFIG(accessibility)
        self.load_button.setText(QCoreApplication.translate("MainWindow", "Load", None))
        # if QT_CONFIG(accessibility)
        self.seg_button.setAccessibleName(
            QCoreApplication.translate("MainWindow", "load_or_save_button", None)
        )
        # endif // QT_CONFIG(accessibility)
        self.seg_button.setText(
            QCoreApplication.translate("MainWindow", "Segment", None)
        )

    # retranslateUi
