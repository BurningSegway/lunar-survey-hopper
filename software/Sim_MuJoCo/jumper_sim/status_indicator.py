from __future__ import annotations

import tkinter as tk


class RelaxationIndicator:
    def __init__(self) -> None:
        self._root = tk.Tk()
        self._root.title("Jumper state")
        self._root.resizable(False, False)
        self._active: bool | None = None
        self._button = tk.Button(
            self._root,
            state=tk.DISABLED,
            width=24,
            font=("TkDefaultFont", 12, "bold"),
            disabledforeground="white",
            relief=tk.RAISED,
            padx=12,
            pady=8,
        )
        self._button.pack(padx=12, pady=12)
        self._root.protocol("WM_DELETE_WINDOW", self._root.withdraw)
        self.set_active(False)

    def set_active(self, active: bool) -> None:
        if self._active == active:
            return
        self._active = active
        self._button.configure(
            text="RELAXATION ACTIVE" if active else "RELAXATION INACTIVE",
            background="#16803c" if active else "#c62828",
            disabledforeground="white",
        )

    def update(self, active: bool) -> None:
        self.set_active(active)
        self._root.update_idletasks()
        self._root.update()

    def close(self) -> None:
        self._root.destroy()
