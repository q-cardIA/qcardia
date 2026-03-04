import os

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSpacerItem,
    QVBoxLayout,
    QWidget,
)

from view.utils import SoftwareConfigResources


class WelcomeWidget(QWidget):
    """
    Starting page when launching the software.
    """

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
        self.left_panel.setProperty("role", "welcomeSidePanel")
        self.left_panel.setLayout(self.left_panel_layout)
        self.right_panel = QWidget()
        self.right_panel.setObjectName("welcomeRightPanel")
        self.right_panel.setAttribute(Qt.WA_StyledBackground, True)
        self.right_panel.setProperty("role", "welcomeSidePanel")
        self.right_panel.setLayout(self.right_panel_layout)

        self.central_label = QWidget()
        self.central_label.setObjectName("centralContainer")
        self.central_label.setAttribute(Qt.WA_StyledBackground, True)
        self.central_label.setProperty("role", "welcomeCentral")
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
        self.top_logo_panel_label = QLabel("Quantitative cardiac image analysis")
        self.top_logo_panel_label.setProperty("role", "welcomeHeadline")
        self.top_logo_panel_layout.addStretch(1)
        self.top_logo_panel_layout.addWidget(self.top_logo_panel_label, Qt.AlignCenter)
        self.top_logo_panel_layout.addStretch(1)

    def __bottom_logo_panel_interface(self):
        self.bottom_logo_panel_layout = QHBoxLayout()
        self.bottom_logo_panel_layout.setContentsMargins(0, 30, 30, 30)
        self.bottom_logo_panel_label = QLabel()
        self.bottom_logo_panel_label.setPixmap(
            QPixmap(
                os.path.join(
                    os.path.dirname(os.path.realpath(__file__)),
                    "../../assets/images/q-cardIA.png",
                )
            ).scaled(180, 270, Qt.KeepAspectRatio)
        )
        self.bottom_logo_panel_layout.addStretch(1)
        self.bottom_logo_panel_layout.addWidget(
            self.bottom_logo_panel_label, Qt.AlignCenter
        )
        self.bottom_logo_panel_layout.addStretch(1)

    def __set_right_panel_interface(self):
        self.right_panel_layout = QVBoxLayout()
        self.right_panel_layout.setContentsMargins(0, 0, 0, 0)
        self.right_panel_layout.setSpacing(5)
        self.right_panel_title_layout = QHBoxLayout()
        self.right_panel_title_layout.setSpacing(0)
        self.right_panel_title_layout.setContentsMargins(0, 50, 0, 0)
        self.right_panel_title_layout.addStretch(1)
        self.right_panel_title_label = QLabel("For more information")
        self.right_panel_title_label.setProperty("role", "welcomeSectionTitle")
        self.right_panel_title_layout.addWidget(self.right_panel_title_label)
        self.right_panel_title_layout.addStretch(1)
        self.right_panel_layout.addLayout(self.right_panel_title_layout)
        self.right_panel_top_spaceritem = QSpacerItem(1, 30)
        self.right_panel_layout.addItem(self.right_panel_top_spaceritem)

        self.right_panel_community_layout = QHBoxLayout()
        self.right_panel_community_layout.setSpacing(5)
        self.right_panel_community_layout.setContentsMargins(50, 0, 50, 0)
        self.right_panel_community_pushbutton = QPushButton("  Research community")
        self.right_panel_community_pushbutton.setProperty("role", "welcomeInfoAction")
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
        self.right_panel_about_layout.setContentsMargins(50, 25, 50, 0)
        self.right_panel_about_software_pushbutton = QPushButton("  About q-cardIA")
        self.right_panel_about_software_pushbutton.setProperty(
            "role", "welcomeInfoAction"
        )
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
        self.right_panel_helpwiki_layout.setContentsMargins(50, 25, 50, 0)
        self.right_panel_helpwiki_pushbutton = QPushButton("  Tutorials and wiki")
        self.right_panel_helpwiki_pushbutton.setProperty("role", "welcomeInfoAction")
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
        self.right_panel_issues_layout.setContentsMargins(50, 25, 50, 0)
        self.right_panel_issues_pushbutton = QPushButton("  Issues and suggestions")
        self.right_panel_issues_pushbutton.setProperty("role", "welcomeInfoAction")
        self.right_panel_issues_pushbutton.setIcon(
            QIcon(
                QPixmap(
                    os.path.join(
                        os.path.dirname(os.path.realpath(__file__)),
                        "../../assets/images/github-icon.png",
                    )
                ).scaled(40, 40, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )
        )
        self.right_panel_issues_layout.addWidget(self.right_panel_issues_pushbutton)
        self.right_panel_layout.addLayout(self.right_panel_issues_layout)
        self.right_panel_layout.addStretch(1)

    def __set_left_panel_interface(self):
        self.left_panel_layout = QVBoxLayout()
        self.left_panel_layout.setContentsMargins(0, 0, 0, 0)
        self.left_panel_layout.setSpacing(5)
        self.left_panel_title_layout = QHBoxLayout()
        self.left_panel_title_layout.setSpacing(0)
        self.left_panel_title_layout.setContentsMargins(0, 50, 0, 0)
        self.left_panel_title_layout.addStretch(1)
        self.left_panel_title_label = QLabel("Start by")
        self.left_panel_title_label.setProperty("role", "welcomeSectionTitle")
        self.left_panel_title_layout.addWidget(self.left_panel_title_label)
        self.left_panel_title_layout.addStretch(1)
        self.left_panel_layout.addLayout(self.left_panel_title_layout)
        self.left_panel_top_spaceritem = QSpacerItem(1, 30)
        self.left_panel_layout.addItem(self.left_panel_top_spaceritem)

        self.left_panel_single_layout = QHBoxLayout()
        self.left_panel_single_layout.setSpacing(5)
        self.left_panel_single_layout.setContentsMargins(50, 0, 50, 0)
        self.left_panel_single_patient_pushbutton = QPushButton("  Single patient mode")
        self.left_panel_single_patient_pushbutton.setProperty(
            "role", "welcomePrimaryAction"
        )
        self.left_panel_single_patient_pushbutton.setProperty("variant", "primary")
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
        self.left_panel_layout.addStretch(1)

    def __set_layout_dimensions(self):
        opt = SoftwareConfigResources.getInstance().get_optimal_dimensions()
        target_size = opt
        if self.parent is not None:
            parent_size = self.parent.size()
            if parent_size.width() > 0 and parent_size.height() > 0:
                target_size = parent_size
            else:  # Ensure the main window is at least the optimal dimensions
                self.parent.resize(opt)

        self.setMinimumSize(target_size)
        self.resize(target_size)

        # Central area
        self.central_label.setMinimumSize(400, 550)
        self.central_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Top title
        self.top_logo_panel_label.setMinimumSize(400, 120)
        self.top_logo_panel_label.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Fixed
        )

        # Bottom logo
        self.bottom_logo_panel_label.setMinimumSize(100, 200)
        self.bottom_logo_panel_label.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Fixed
        )

        # Left panel buttons
        self.left_panel_single_patient_pushbutton.setMinimumHeight(75)
        self.left_panel_single_patient_pushbutton.setMinimumWidth(350)
        self.left_panel_single_patient_pushbutton.setIconSize(QSize(40, 40))
        self.left_panel.setMinimumSize(300, 300)
        self.left_panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Right panel buttons
        self.right_panel_community_pushbutton.setMinimumHeight(75)
        self.right_panel_community_pushbutton.setMinimumWidth(350)
        self.right_panel_community_pushbutton.setIconSize(QSize(40, 40))
        self.right_panel_about_software_pushbutton.setMinimumHeight(75)
        self.right_panel_about_software_pushbutton.setMinimumWidth(350)
        self.right_panel_about_software_pushbutton.setIconSize(QSize(40, 40))
        self.right_panel_helpwiki_pushbutton.setMinimumHeight(75)
        self.right_panel_helpwiki_pushbutton.setMinimumWidth(350)
        self.right_panel_helpwiki_pushbutton.setIconSize(QSize(40, 40))
        self.right_panel_issues_pushbutton.setMinimumHeight(75)
        self.right_panel_issues_pushbutton.setMinimumWidth(350)
        self.right_panel_issues_pushbutton.setIconSize(QSize(40, 40))
        self.right_panel.setMinimumSize(300, 300)
        self.right_panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def __set_stylesheets(self):
        self.setProperty("role", "welcomeRoot")

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

    def community_clicked(self):
        """Open research community link in browser."""
        from PySide6.QtCore import QUrl
        from PySide6.QtGui import QDesktopServices

        QDesktopServices.openUrl(QUrl("https://github.com/q-cardIA"))

    def about_clicked(self):
        """Open about/documentation page in browser."""
        from PySide6.QtCore import QUrl
        from PySide6.QtGui import QDesktopServices

        QDesktopServices.openUrl(QUrl("https://github.com/q-cardIA/qcardia"))

    def help_clicked(self):
        """Open tutorials and wiki in browser."""
        from PySide6.QtCore import QUrl
        from PySide6.QtGui import QDesktopServices

        QDesktopServices.openUrl(QUrl("https://github.com/q-cardIA/qcardia/wiki"))

    def issues_clicked(self):
        """Open issues page in browser."""
        from PySide6.QtCore import QUrl
        from PySide6.QtGui import QDesktopServices

        QDesktopServices.openUrl(QUrl("https://github.com/q-cardIA/qcardia/issues"))
