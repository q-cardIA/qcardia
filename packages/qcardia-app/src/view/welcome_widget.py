import logging
import os

from PySide6.QtCore import QSize, Qt, QUrl, Signal
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import (
    QCommonStyle,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSpacerItem,
    QStyle,
    QVBoxLayout,
    QWidget,
)

from view.utils import SoftwareConfigResources


class WelcomeWidget(QWidget):
    """
    Starting page when launching the software.
    """

    about_clicked = Signal()
    community_clicked = Signal()
    help_clicked = Signal()
    issues_clicked = Signal()
    transition_signal = Signal()  # Signal to notify MainWindow

    def __init__(self, parent=None):
        super(WelcomeWidget, self).__init__()
        self.widget_name = "welcome_widget"
        self.setObjectName("welcomeRoot")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.parent = parent
        self.__set_interface()
        self.__set_layout_dimensions()
        self.__set_stylesheets()
        self.__set_connections()

    def __set_interface(self):
        self.__top_title_interface()
        self.__set_right_panel_interface()
        self.__set_left_panel_interface()
        self.__bottom_logo_panel_interface()

        # Use QWidget containers for left and right panels
        self.left_panel = QWidget()
        self.left_panel.setObjectName("welcomeLeftPanel")
        self.left_panel.setAttribute(Qt.WA_StyledBackground, True)
        self.left_panel.setLayout(self.left_panel_layout)
        self.right_panel = QWidget()
        self.right_panel.setObjectName("welcomeRightPanel")
        self.right_panel.setAttribute(Qt.WA_StyledBackground, True)
        self.right_panel.setLayout(self.right_panel_layout)

        self.central_label = QWidget()
        self.central_label.setObjectName("centralContainer")
        self.central_label.setAttribute(Qt.WA_StyledBackground, True)
        self.central_label.setContentsMargins(0, 0, 0, 0)
        self.central_layout = QHBoxLayout()
        self.central_layout.setContentsMargins(0, 0, 0, 0)
        self.central_layout.setSpacing(0)
        self.central_layout.addStretch(1)
        self.central_layout.addWidget(self.left_panel)
        self.central_layout.addWidget(self.right_panel)
        self.central_layout.addStretch(1)
        self.central_label.setLayout(self.central_layout)

        self.layout = QVBoxLayout(self)
        self.layout.addLayout(self.top_logo_panel_layout)
        self.layout.addWidget(self.central_label)
        self.layout.addLayout(self.bottom_logo_panel_layout)

    def __top_title_interface(self):
        self.top_logo_panel_layout = QHBoxLayout()
        self.top_logo_panel_label = QLabel("Welcome to q-cardIA")
        self.top_logo_panel_layout.addStretch(1)
        self.top_logo_panel_layout.addWidget(self.top_logo_panel_label, Qt.AlignCenter)
        self.top_logo_panel_layout.addStretch(1)

    def __bottom_logo_panel_interface(self):
        self.bottom_logo_panel_layout = QHBoxLayout()
        self.bottom_logo_panel_label = QLabel()
        self.bottom_logo_panel_label.setPixmap(
            QPixmap(
                os.path.join(
                    os.path.dirname(os.path.realpath(__file__)),
                    "../../assets/images/q-cardIA.png",
                )
            ).scaled(200, 300, Qt.KeepAspectRatio)
        )
        self.bottom_logo_panel_layout.addStretch(1)
        self.bottom_logo_panel_layout.addWidget(
            self.bottom_logo_panel_label, Qt.AlignRight
        )

    def __set_right_panel_interface(self):
        self.right_panel_layout = QVBoxLayout()
        self.right_panel_layout.setContentsMargins(0, 0, 0, 0)
        self.right_panel_layout.setSpacing(5)
        self.right_panel_title_layout = QHBoxLayout()
        self.right_panel_title_layout.setSpacing(0)
        self.right_panel_title_layout.setContentsMargins(0, 50, 0, 0)
        self.right_panel_title_layout.addStretch(1)
        self.right_panel_title_label = QLabel("For more information")
        self.right_panel_title_layout.addWidget(self.right_panel_title_label)
        self.right_panel_title_layout.addStretch(1)
        self.right_panel_layout.addLayout(self.right_panel_title_layout)
        self.right_panel_top_spaceritem = QSpacerItem(1, 50)
        self.right_panel_layout.addItem(self.right_panel_top_spaceritem)

        self.right_panel_community_layout = QHBoxLayout()
        self.right_panel_community_layout.setSpacing(5)
        self.right_panel_community_layout.setContentsMargins(50, 0, 50, 0)
        self.right_panel_community_pushbutton = QPushButton("Research community")
        self.right_panel_community_pushbutton.setIcon(
            QIcon(
                QPixmap(
                    os.path.join(
                        os.path.dirname(os.path.realpath(__file__)),
                        "../../assets/images/globe-icon.png",
                    )
                )
            )
        )
        self.right_panel_community_layout.addWidget(
            self.right_panel_community_pushbutton
        )
        self.right_panel_layout.addLayout(self.right_panel_community_layout)

        self.right_panel_about_layout = QHBoxLayout()
        self.right_panel_about_layout.setSpacing(5)
        self.right_panel_about_layout.setContentsMargins(50, 30, 50, 0)
        self.right_panel_about_software_pushbutton = QPushButton("About Raidionics")
        self.right_panel_about_software_pushbutton.setIcon(
            QIcon(
                QPixmap(
                    os.path.join(
                        os.path.dirname(os.path.realpath(__file__)),
                        "../../assets/images/circle_question_icon.png",
                    )
                )
            )
        )
        self.right_panel_about_layout.addWidget(
            self.right_panel_about_software_pushbutton
        )
        self.right_panel_layout.addLayout(self.right_panel_about_layout)

        self.right_panel_helpwiki_layout = QHBoxLayout()
        self.right_panel_helpwiki_layout.setSpacing(5)
        self.right_panel_helpwiki_layout.setContentsMargins(50, 30, 50, 0)
        self.right_panel_helpwiki_pushbutton = QPushButton("Tutorials and wiki")
        self.right_panel_helpwiki_pushbutton.setIcon(
            QIcon(
                QPixmap(
                    os.path.join(
                        os.path.dirname(os.path.realpath(__file__)),
                        "../../assets/images/help_wavy_question_icon_blue.png",
                    )
                )
            )
        )
        self.right_panel_helpwiki_layout.addWidget(self.right_panel_helpwiki_pushbutton)
        self.right_panel_layout.addLayout(self.right_panel_helpwiki_layout)

        self.right_panel_issues_layout = QHBoxLayout()
        self.right_panel_issues_layout.setSpacing(5)
        self.right_panel_issues_layout.setContentsMargins(50, 30, 50, 0)
        self.right_panel_issues_pushbutton = QPushButton("Issues and suggestions")
        self.right_panel_issues_pushbutton.setIcon(
            QIcon(
                QPixmap(
                    os.path.join(
                        os.path.dirname(os.path.realpath(__file__)),
                        "../../assets/images/github-icon.png",
                    )
                )
            )
        )
        self.right_panel_issues_layout.addWidget(self.right_panel_issues_pushbutton)
        self.right_panel_layout.addLayout(self.right_panel_issues_layout)

        self.right_panel_layout.addStretch(1)
        # self.right_panel_label is now replaced by self.right_panel QWidget

    def __set_left_panel_interface(self):
        self.left_panel_layout = QVBoxLayout()
        self.left_panel_layout.setContentsMargins(0, 0, 0, 0)
        self.left_panel_layout.setSpacing(5)
        self.left_panel_title_layout = QHBoxLayout()
        self.left_panel_title_layout.setSpacing(0)
        self.left_panel_title_layout.setContentsMargins(0, 50, 0, 0)
        self.left_panel_title_layout.addStretch(1)
        self.left_panel_title_label = QLabel("Start by")
        self.left_panel_title_layout.addWidget(self.left_panel_title_label)
        self.left_panel_title_layout.addStretch(1)
        self.left_panel_layout.addLayout(self.left_panel_title_layout)
        self.left_panel_top_spaceritem = QSpacerItem(1, 50)
        self.left_panel_layout.addItem(self.left_panel_top_spaceritem)

        self.left_panel_single_layout = QHBoxLayout()
        self.left_panel_single_layout.setSpacing(5)
        self.left_panel_single_layout.setContentsMargins(50, 0, 50, 0)
        self.left_panel_single_patient_pushbutton = QPushButton("Single patient mode")
        self.left_panel_single_patient_pushbutton.setIcon(
            QIcon(
                QPixmap(
                    os.path.join(
                        os.path.dirname(os.path.realpath(__file__)),
                        "../../assets/images/patient-icon.png",
                    )
                )
            )
        )
        self.left_panel_single_layout.addWidget(
            self.left_panel_single_patient_pushbutton
        )
        self.left_panel_layout.addLayout(self.left_panel_single_layout)

        self.left_panel_study_layout = QHBoxLayout()
        self.left_panel_study_layout.setSpacing(5)
        self.left_panel_study_layout.setContentsMargins(50, 30, 50, 0)
        self.left_panel_multiple_patients_pushbutton = QPushButton("Batch/study mode")
        self.left_panel_multiple_patients_pushbutton.setIcon(
            QIcon(
                QPixmap(
                    os.path.join(
                        os.path.dirname(os.path.realpath(__file__)),
                        "../../assets/images/study_icon.png",
                    )
                )
            )
        )
        self.left_panel_study_layout.addWidget(
            self.left_panel_multiple_patients_pushbutton
        )
        self.left_panel_layout.addLayout(self.left_panel_study_layout)
        self.left_panel_layout.addStretch(1)
        # self.left_panel_label is now replaced by self.left_panel QWidget
        # self.left_panel_layout = QVBoxLayout()
        # self.left_panel_layout.setContentsMargins(140, 110, 50, 50)
        # self.left_panel_label = QLabel()
        # self.left_panel_label.setLineWidth(0)
        # self.left_panel_label.setFrameShape(QFrame.NoFrame)
        # self.left_panel_top_spaceritem = QSpacerItem(1, 20)
        # self.left_panel_layout.addItem(self.left_panel_top_spaceritem)
        #
        # self.left_panel_startby_label = QLabel()
        # self.left_panel_startby_label.setText("Start by")
        # self.left_panel_layout.addWidget(self.left_panel_startby_label)
        # self.left_panel_spaceritem = QSpacerItem(1, 15)
        # self.left_panel_layout.addItem(self.left_panel_spaceritem)
        # self.left_panel_single_patient_pushbutton = QPushButton()
        # self.left_panel_single_patient_pushbutton_icon = QIcon(QPixmap(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'Images/patient-icon.png')))
        # self.left_panel_single_patient_pushbutton.setIcon(self.left_panel_single_patient_pushbutton_icon)
        # self.left_panel_single_patient_pushbutton.setText("Single patient")
        # self.left_panel_layout.addWidget(self.left_panel_single_patient_pushbutton)
        #
        # self.left_panel_or_layout = QHBoxLayout()
        # self.left_panel_left_line_or_label = QLabel()
        # self.left_panel_left_line_or_label.setFixedSize(QSize(120, 2))
        # self.left_panel_right_line_or_label = QLabel()
        # self.left_panel_right_line_or_label.setFixedSize(QSize(120, 2))
        # self.left_panel_center_or_label = QLabel()
        # self.left_panel_center_or_label.setText("or")
        # self.left_panel_center_or_label.setFixedSize(QSize(15, 40))
        # self.left_panel_or_layout.addWidget(self.left_panel_left_line_or_label)
        # self.left_panel_or_layout.addWidget(self.left_panel_center_or_label)
        # self.left_panel_or_layout.addWidget(self.left_panel_right_line_or_label)
        # self.left_panel_or_layout.addStretch(1)
        # self.left_panel_layout.addLayout(self.left_panel_or_layout)
        #
        # self.left_panel_multiple_patients_pushbutton = QPushButton()
        # self.left_panel_multiple_patients_pushbutton.setText("New study")
        # self.left_panel_multiple_patients_pushbutton_icon = QIcon(QPixmap(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'Images/study_icon.png')))
        # self.left_panel_multiple_patients_pushbutton.setIcon(self.left_panel_multiple_patients_pushbutton_icon)
        # self.left_panel_layout.addWidget(self.left_panel_multiple_patients_pushbutton)
        # self.left_panel_layout.addStretch(1)
        #
        # self.left_panel_label.setLayout(self.left_panel_layout)

    def __set_layout_dimensions(self):
        ################################## WIDGET ######################################
        self.setBaseSize(self.parent.baseSize())
        # Allow central_label to expand
        self.central_label.setMinimumSize(400, 400)
        self.central_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.top_logo_panel_label.setMinimumSize(400, 80)
        self.top_logo_panel_label.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Fixed
        )
        ################################## LOGO PANEL ######################################
        self.bottom_logo_panel_label.setMinimumSize(100, 150)
        self.bottom_logo_panel_label.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Fixed
        )

        ################################## LEFT PANEL ######################################
        self.left_panel_single_patient_pushbutton.setFixedHeight(50)
        self.left_panel_single_patient_pushbutton.setIconSize(QSize(35, 35))
        self.left_panel_multiple_patients_pushbutton.setFixedHeight(50)
        self.left_panel_multiple_patients_pushbutton.setIconSize(QSize(35, 35))
        self.left_panel.setMinimumSize(200, 200)
        self.left_panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        ################################## RIGHT PANEL ######################################
        self.right_panel_community_pushbutton.setFixedHeight(50)
        self.right_panel_community_pushbutton.setIconSize(QSize(35, 35))
        self.right_panel_about_software_pushbutton.setFixedHeight(50)
        self.right_panel_about_software_pushbutton.setIconSize(QSize(35, 35))
        self.right_panel_helpwiki_pushbutton.setFixedHeight(50)
        self.right_panel_helpwiki_pushbutton.setIconSize(QSize(35, 35))
        self.right_panel_issues_pushbutton.setFixedHeight(50)
        self.right_panel_issues_pushbutton.setIconSize(QSize(35, 35))
        self.right_panel.setMinimumSize(200, 200)
        self.right_panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def __set_stylesheets(self):
        software_ss = SoftwareConfigResources.getInstance().stylesheet_components
        self.setStyleSheet(
            "#welcomeRoot { background-color: " + software_ss["Color2"] + "; }"
        )
        self.central_label.setStyleSheet(
            "background-color: " + software_ss["Color2"] + ";"
        )

        self.top_logo_panel_label.setStyleSheet(
            """
        QLabel{
        color: """
            + software_ss["Color7"]
            + """;
        font: 72px;
        font-style: bold;
        margin-top: 40px;
        }"""
        )

        self.right_panel.setStyleSheet(
            "#welcomeRightPanel {\n            min-width: 200px;\n            min-height: 200px;\n            background-color: "
            + software_ss["Color2"]
            + ";\n        }"
        )

        self.right_panel_title_label.setStyleSheet(
            """
        QLabel{
        color: """
            + software_ss["Color7"]
            + """;
        border: 0px;
        font-size: 48px;
        font-style: bold;
        }"""
        )

        self.right_panel_community_pushbutton.setStyleSheet(
            """
        QPushButton{
        color: """
            + software_ss["Color7"]
            + """;
        background-color: """
            + software_ss["Color3"]
            + """;
        font-size: 24px;
        text-align: center;
        border-radius:20px;
        margin-left:5px;
        margin-right:5px;
        font:bold}
        QPushButton:pressed{
        background-color: """
            + software_ss["Color4"]
            + """;
        border-style:inset
        }"""
        )

        self.right_panel_about_software_pushbutton.setStyleSheet(
            """
        QPushButton{
        color: """
            + software_ss["Color7"]
            + """;
        background-color: """
            + software_ss["Color3"]
            + """;
        font-size: 24px;
        text-align: center;
        border-radius:20px;
        margin-left:5px;
        margin-right:5px;
        font:bold}
        QPushButton:pressed{
        background-color: """
            + software_ss["Color4"]
            + """;
        border-style:inset
        }"""
        )

        self.right_panel_helpwiki_pushbutton.setStyleSheet(
            """
        QPushButton{
        color: """
            + software_ss["Color7"]
            + """;
        background-color: """
            + software_ss["Color3"]
            + """;
        font-size: 24px;
        text-align: center;
        border-radius:20px;
        margin-left:5px;
        margin-right:5px;
        font:bold}
        QPushButton:pressed{
        background-color: """
            + software_ss["Color4"]
            + """;
        border-style:inset
        }"""
        )

        self.right_panel_issues_pushbutton.setStyleSheet(
            """
        QPushButton{
        color: """
            + software_ss["Color7"]
            + """;
        background-color: """
            + software_ss["Color3"]
            + """;
        font-size: 24px;
        text-align: center;
        border-radius:20px;
        margin-left:5px;
        margin-right:5px;
        font:bold}
        QPushButton:pressed{
        background-color: """
            + software_ss["Color4"]
            + """;
        border-style:inset
        }"""
        )

        self.left_panel.setStyleSheet(
            "#welcomeLeftPanel {\n            min-width: 200px;\n            min-height: 200px;\n            background-color: "
            + software_ss["Color2"]
            + ";\n        }"
        )

        self.left_panel_title_label.setStyleSheet(
            """
        QLabel{
        color: """
            + software_ss["Color7"]
            + """;
        border: 0px;
        font-size: 48px;
        font-style: bold;
        }"""
        )

        self.left_panel_single_patient_pushbutton.setStyleSheet(
            """
        QPushButton{
        color: """
            + software_ss["Color7"]
            + """;
        background-color: """
            + software_ss["Process"]
            + """;
        font-size: 24px;
        text-align: center;
        border-radius:20px;
        margin-left:5px;
        margin-right:5px;
        font:bold}
        QPushButton:pressed{
        background-color: """
            + software_ss["Process_pressed"]
            + """;
        border-style:inset
        }"""
        )

        self.left_panel_multiple_patients_pushbutton.setStyleSheet(
            """
        QPushButton{
        color: """
            + software_ss["Color7"]
            + """;
        background-color: """
            + software_ss["Process"]
            + """;
        font-size: 24px;
        text-align: center;
        border-radius:20px;
        margin-left:5px;
        margin-right:5px;
        font:bold}
        QPushButton:pressed{
        background-color: """
            + software_ss["Process_pressed"]
            + """;
        border-style:inset
        }"""
        )

    def __set_connections(self):
        self.right_panel_community_pushbutton.clicked.connect(self.community_clicked)
        self.right_panel_about_software_pushbutton.clicked.connect(self.about_clicked)
        self.right_panel_helpwiki_pushbutton.clicked.connect(self.help_clicked)
        self.right_panel_issues_pushbutton.clicked.connect(self.issues_clicked)
        self.left_panel_single_patient_pushbutton.clicked.connect(
            self.emit_transition_signal
        )

    def get_widget_name(self):
        return self.widget_name

    def emit_transition_signal(self):
        self.transition_signal.emit()
