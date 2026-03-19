from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)


class QuickViewPanelWidget(QFrame):
    """Right-hand panel showing quick-view cardiac metrics after segmentation."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("quickViewPanel")
        self._build()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _build(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        title = QLabel("Quick View")
        title.setProperty("role", "panelTitle")
        layout.addWidget(title)

        divider = QLabel()
        divider.setFixedHeight(2)
        divider.setProperty("role", "divider")
        layout.addWidget(divider)

        self._status_label = QLabel("Run segmentation to see results.")
        self._status_label.setWordWrap(True)
        self._status_label.setAlignment(Qt.AlignTop)
        self._status_label.setProperty("role", "statusHint")
        layout.addWidget(self._status_label)

        # Global function section
        self._global_widget, self._global_rows = self._build_section(
            "Global Function",
            [
                ("LVEDV", "--"),
                ("LVESV", "--"),
                ("LVEF", "--"),
                ("LV Mass", "--"),
                ("RVEDV", "--"),
                ("RVESV", "--"),
                ("RVEF", "--"),
            ],
        )
        self._global_widget.hide()
        layout.addWidget(self._global_widget)

        # LGE section
        self._lge_widget, self._lge_rows = self._build_section(
            "LGE",
            [
                ("Fibrosis Volume", "--"),
                ("Fibrosis %", "--"),
            ],
        )
        self._lge_widget.hide()
        layout.addWidget(self._lge_widget)

        layout.addStretch(1)
        self.setFixedWidth(200)

    def _build_section(self, title, fields):
        widget = QWidget()
        widget.setProperty("role", "resultsSection")
        vlayout = QVBoxLayout(widget)
        vlayout.setContentsMargins(0, 4, 0, 4)
        vlayout.setSpacing(6)

        lbl_title = QLabel(title)
        lbl_title.setProperty("role", "sectionTitle")
        vlayout.addWidget(lbl_title)

        rows = {}
        for key, default in fields:
            row = QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            lbl_key = QLabel(key + ":")
            lbl_key.setProperty("role", "metricKey")
            lbl_val = QLabel(default)
            lbl_val.setProperty("role", "metricValue")
            lbl_val.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            row.addWidget(lbl_key)
            row.addStretch(1)
            row.addWidget(lbl_val)
            vlayout.addLayout(row)
            rows[key] = lbl_val

        return widget, rows

    # ------------------------------------------------------------------
    # Public update API
    # ------------------------------------------------------------------

    def show_cine_results(self):
        """Reveal the global-function section with placeholder values.

        Call this immediately after segmentation completes, before metrics
        are available.
        """
        self._status_label.hide()
        self._global_widget.show()

    def update_cine_metrics(self, metrics):
        """Populate global-function section from a metrics dict.

        Expected keys: lvedv, lvesv, lvef, lv_mass, rvedv, rvesv, rvef.
        Values should be floats (mL / % / g) or None.
        """
        self._status_label.hide()
        self._global_widget.show()

        def ml(v):
            return f"{v:.1f} mL" if v is not None else "--"

        def pct(v):
            return f"{v:.1f} %" if v is not None else "--"

        def g(v):
            return f"{v:.1f} g" if v is not None else "--"

        self._global_rows["LVEDV"].setText(ml(metrics.get("lvedv")))
        self._global_rows["LVESV"].setText(ml(metrics.get("lvesv")))
        self._global_rows["LVEF"].setText(pct(metrics.get("lvef")))
        self._global_rows["LV Mass"].setText(g(metrics.get("lv_mass")))
        self._global_rows["RVEDV"].setText(ml(metrics.get("rvedv")))
        self._global_rows["RVESV"].setText(ml(metrics.get("rvesv")))
        self._global_rows["RVEF"].setText(pct(metrics.get("rvef")))

    def update_lge_metrics(self, metrics):
        """Populate LGE section from a metrics dict.

        Expected keys: fibrosis_volume_ml, fibrosis_pct.
        """
        self._lge_widget.show()

        fib_vol = metrics.get("fibrosis_volume_ml")
        fib_pct = metrics.get("fibrosis_pct")
        self._lge_rows["Fibrosis Volume"].setText(
            f"{fib_vol:.1f} mL" if fib_vol is not None else "--"
        )
        self._lge_rows["Fibrosis %"].setText(
            f"{fib_pct:.1f} %" if fib_pct is not None else "--"
        )

    def reset(self):
        """Clear all metrics and show the status hint."""
        self._global_widget.hide()
        self._lge_widget.hide()
        self._status_label.show()
        for lbl in self._global_rows.values():
            lbl.setText("--")
        for lbl in self._lge_rows.values():
            lbl.setText("--")
