from PySide6.QtCore import QSize


class LayoutParameters:
    def __init__(self) -> None:
        self.spacing = 6  # /3 + /2

        self.large_width = 120  # *2/3
        self.medium_width = (self.large_width * 2 - self.spacing) // 3
        self.small_width = (self.medium_width - self.spacing) // 2

        self.grid_settings_labels_width = 105
        self.transform_button_width = 192
        self.small_label_width = 16

        self.row_height = 28


class SoftwareConfigResources:
    __instance = None

    def __init__(self):
        """Virtually private constructor."""
        if SoftwareConfigResources.__instance != None:
            raise Exception("This class is a singleton!")
        else:
            SoftwareConfigResources.__instance = self
            self.__setup()

    def __setup(self):
        """Setup the instance."""
        self.__set_default_stylesheet_components()
        self.layout_parameters = LayoutParameters()
        self.optimal_dimensions = QSize(1280, 720)

    @staticmethod
    def getInstance():
        """Static access method."""
        if SoftwareConfigResources.__instance == None:
            SoftwareConfigResources()
        return SoftwareConfigResources.__instance

    def __set_default_stylesheet_components(self):
        self.stylesheet_components = {}
        self.stylesheet_components["Color1"] = "rgba(0, 0, 0, 1)"  # Black

        # Dark mode enabled
        self.stylesheet_components["Color2"] = (
            "rgba(44, 49, 60, 1)"  # Main background color (dark)
        )
        self.stylesheet_components["Color3"] = "rgba(55, 61, 75, 1)"  # Dark green/gray
        self.stylesheet_components["Color4"] = (
            "rgba(66, 73, 90, 1)"  # Lighter dark (when pressed)
        )
        self.stylesheet_components["Color5"] = (
            "rgba(35, 39, 48, 1)"  # Dark (standard background)
        )
        self.stylesheet_components["Color6"] = (
            "rgba(50, 55, 68, 1)"  # Lighter dark (when pressed)
        )
        self.stylesheet_components["Color7"] = (
            "rgba(210, 210, 210, 1)"  # Main font color (light gray)
        )

        self.stylesheet_components["White"] = "rgba(255, 255, 255, 1)"  # White
        self.stylesheet_components["Process"] = "rgba(43, 88, 118, 1)"  # Dark blue
        self.stylesheet_components["Process_pressed"] = (
            "rgba(59, 104, 134, 1)"  # Lighter dark blue
        )
        self.stylesheet_components["Import"] = "rgba(73, 99, 171, 1)"  # Greyish blue
        self.stylesheet_components["Import_pressed"] = (
            "rgba(81, 101, 153, 1)"  # Dark greyish blue
        )
        self.stylesheet_components["Data"] = "rgba(60, 80, 120, 1)"  # Dark blue
        self.stylesheet_components["Background_pressed"] = (
            "rgba(0, 120, 230, 1)"  # Bright blue
        )

    def get_optimal_dimensions(self):
        return self.optimal_dimensions
