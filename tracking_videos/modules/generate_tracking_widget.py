import os
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import tkinter as tk
import matplotlib.pyplot as plt  # type: ignore
import matplotlib.ticker as mtick  # type: ignore

from PIL import Image  # type: ignore
from tkinter import filedialog  # type: ignore
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # type: ignore


class CSVPlotterApp:
    """
    A Tkinter-based application for visualizing cockroach tracking data.
    Allows users to load CSV files, display tracking data, edit positions,
    and navigate through frames using keyboard shortcuts.
    """

    def __init__(self, root):
        """Initialize the application window, UI elements, and event
        bindings."""
        self.root = root
        self.root.title("HoI - Cockroach Tracking")

        # Screen dimensions
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        aspect_ratio = 1920 / 1080

        if screen_width / screen_height > aspect_ratio:
            window_width = int(screen_height * aspect_ratio)
            window_height = screen_height
        else:
            window_width = screen_width
            window_height = int(screen_width / aspect_ratio)

        self.root.geometry(f"{window_width}x{window_height}")

        # Upper Frame (Button's panel)
        self.top_frame = tk.Frame(root, bg="grey")
        self.top_frame.pack(fill=tk.X, side=tk.TOP, anchor="w")

        # Create Load CSV button in upper left corner
        self.upload_button = tk.Button(
            self.top_frame,
            text="Load CSV",
            command=self.load_csv,
            font=("Calibri Light", 16)
        )
        self.upload_button.pack(side=tk.LEFT, padx=5, pady=5)

        # Create Load Background folder button in upper left corner
        self.bg_folder_button = tk.Button(
            self.top_frame,
            text="Select Background Folder",
            command=self.select_background_folder,
            font=("Calibri Light", 16)
        )
        self.bg_folder_button.pack(side=tk.LEFT, padx=5, pady=5)

        # Checkbox Frame Legend for ID visibility
        self.checkbox_frame = tk.Frame(root, bg="lightgrey")
        self.checkbox_frame.pack(fill=tk.X, side=tk.TOP, anchor="w")
        self.checkboxes = {}

        # Time sliding button
        self.time_slider = tk.Scale(
            root,
            from_=0,
            to_=100,
            orient=tk.HORIZONTAL,
            label="Frame",
            length=800,
            resolution=1,
            bg="black",
            fg="white",
            font=("Calibri Light", 14),
            command=self.update_secondary_plots
        )
        self.time_slider.pack()

        # Edit Position Button
        self.edit_position_button = tk.Button(
            self.top_frame,
            text="Edit Positions",
            command=self.toggle_edit_mode,
            font=("Calibri Light", 16),
            bg="orange"
        )
        self.edit_position_button.pack(side=tk.LEFT, padx=5, pady=5)

        # Swap ID Buttons
        self.swap_button_1 = tk.Button(
            self.top_frame,
            text="Swap 0 ↔ 1",
            command=lambda: self.swap_ids(0, 1),
            font=("Calibri Light", 14),
            bg="lightblue"
        )
        self.swap_button_1.pack(side=tk.LEFT, padx=5, pady=5)

        self.swap_button_2 = tk.Button(
            self.top_frame,
            text="Swap 0 ↔ 2",
            command=lambda: self.swap_ids(0, 2),
            font=("Calibri Light", 14),
            bg="lightblue"
        )
        self.swap_button_2.pack(side=tk.LEFT, padx=5, pady=5)

        self.swap_button_3 = tk.Button(
            self.top_frame,
            text="Swap 0 ↔ 3",
            command=lambda: self.swap_ids(0, 3),
            font=("Calibri Light", 14),
            bg="lightblue"
        )
        self.swap_button_3.pack(side=tk.LEFT, padx=5, pady=5)

        self.swap_button_4 = tk.Button(
            self.top_frame,
            text="Swap 1 ↔ 2",
            command=lambda: self.swap_ids(1, 2),
            font=("Calibri Light", 14),
            bg="lightgreen"
        )
        self.swap_button_4.pack(side=tk.LEFT, padx=5, pady=5)

        self.swap_button_5 = tk.Button(
            self.top_frame,
            text="Swap 1 ↔ 3",
            command=lambda: self.swap_ids(1, 3),
            font=("Calibri Light", 14),
            bg="lightgreen"
        )
        self.swap_button_5.pack(side=tk.LEFT, padx=5, pady=5)

        self.swap_button_6 = tk.Button(
            self.top_frame,
            text="Swap 2 ↔ 3",
            command=lambda: self.swap_ids(2, 3),
            font=("Calibri Light", 14),
            bg="lightcoral"
        )
        self.swap_button_6.pack(side=tk.LEFT, padx=5, pady=5)

        # Save CSV Button
        self.save_csv_button = tk.Button(
            self.top_frame,
            text="Save CSV",
            command=self.save_csv,
            font=("Calibri Light", 16)
        )
        self.save_csv_button.pack(side=tk.LEFT, padx=5, pady=5)

        # Main layout frame
        self.main_frame = tk.Frame(root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Right: Small plots
        self.side_frame = tk.Frame(self.main_frame, width=300)
        self.side_frame.pack(side=tk.RIGHT, fill=tk.Y)

        # Three small plots
        self.fig2, (self.ax_x, self.ax_y, self.ax_o) = plt.subplots(
            3,
            1,
            figsize=(5, 10)
        )
        self.fig2.subplots_adjust(hspace=0.4)
        self.canvas2 = FigureCanvasTkAgg(self.fig2, master=self.side_frame)
        self.canvas2.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Initial parameters
        self.data = None
        self.current_time = 0
        self.time_step = 3  # Time step between tracked frames
        self.show_legend = True  # Flag for legend visibility
        self.edit_mode = False  # Edit mode flag
        self.selected_id = None
        self.last_edited_id = None  # Track last edited ID

        # Left frame: Main Canvas
        self.canvas_frame = tk.Frame(self.main_frame, bg="white")
        self.canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Background in the figure
        self.fig, self.ax = plt.subplots(facecolor="white")

        # Default background folder keeping cache
        self.background_folder = ""
        self.background_cache = {}

        # Interactive Canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.canvas_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Connect mouse events
        self.canvas.mpl_connect("button_press_event", self.on_click)
        self.canvas.mpl_connect("motion_notify_event", self.on_drag)
        self.canvas.mpl_connect("button_release_event", self.on_release)

        # Dynamic resize of the app window
        self.root.bind("<Configure>", self.on_resize)

        # Bind keyboard shortcuts for user interaction
        self.root.bind("<Shift_L>", self.toggle_edit_mode)
        self.root.bind("q", lambda event: self.rotate_id(0))  # Rotate ID 0 (Q)
        self.root.bind("w", lambda event: self.rotate_id(1))  # Rotate ID 1 (W)
        self.root.bind("e", lambda event: self.rotate_id(2))  # Rotate ID 2 (E)
        self.root.bind("r", lambda event: self.rotate_id(3))  # Rotate ID 3 (R)
        self.root.bind("<Left>", self.prev_frame)  # Previous Frame (E)
        self.root.bind("<Right>", self.next_frame)  # Next Frame (T)
        self.root.bind("x", lambda event: self.swap_ids(0, 1))  # Swap 0-1 (X)
        self.root.bind("c", lambda event: self.swap_ids(0, 2))  # Swap 0-2 (C)
        self.root.bind("v", lambda event: self.swap_ids(0, 3))  # Swap 0-3 (V)
        self.root.bind("b", lambda event: self.swap_ids(1, 2))  # Swap 1-2 (B)
        self.root.bind("n", lambda event: self.swap_ids(1, 3))  # Swap 1-3 (N)
        self.root.bind("m", lambda event: self.swap_ids(2, 3))  # Swap 2-3 (M)

    def swap_ids(self, from_id, to_id):
        """Swaps two IDs in the dataset from the current time onward."""
        if not self.edit_mode or self.data is None:
            return

        unique_ids = self.data["id"].unique()
        num_ids = len(unique_ids)

        # Ensure the swap meets the constraints
        if (from_id, to_id) == (0, 2) and num_ids < 3:
            return
        if (from_id, to_id) == (1, 2) and num_ids < 3:
            return
        if (from_id, to_id) == (0, 3) and num_ids < 4:
            return
        if (from_id, to_id) == (1, 3) and num_ids < 4:
            return
        if (from_id, to_id) == (2, 3) and num_ids < 4:
            return

        # Perform ID swapping for current time and later frames
        mask = self.data["time"] >= self.current_time
        swap_mask_1 = (self.data["id"] == from_id) & mask
        swap_mask_2 = (self.data["id"] == to_id) & mask

        self.data.loc[swap_mask_1, "id"] = to_id
        self.data.loc[swap_mask_2, "id"] = from_id

        self.update_plot()

    def select_background_folder(self):
        """Opens a dialog for the user to select a background image folder."""
        folder_selected = filedialog.askdirectory()
        if folder_selected:
            self.background_folder = folder_selected
            self.background_cache.clear()  # Clear cache

    def load_csv(self):
        """Opens a file dialog to select and load a CSV file into the
        application."""
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])  # noqa: 501
        if file_path:
            self.data = pd.read_csv(file_path)
            self.data["orientation"] = self.data["corrected_orientation"]
            self.time_slider.config(to=int(self.data["time"].max() / self.time_step))  # noqa: 501
            self.create_checkboxes()
            self.update_plot()

    def create_checkboxes(self):
        """Creates checkboxes to toggle visibility of cockroach IDs in the
        plot."""
        for widget in self.checkbox_frame.winfo_children():
            widget.destroy()

        if self.data is not None:
            unique_ids = self.data["id"].unique()
            self.checkboxes = {}
            for uid in unique_ids:
                var = tk.BooleanVar(value=True)
                chk = tk.Checkbutton(
                    self.checkbox_frame,
                    text=f"Cockroach ID {uid}",
                    variable=var,
                    command=self.update_plot,
                    font=("Calibri Light", 14)
                )
                chk.pack(side=tk.LEFT, padx=2, pady=2)
                self.checkboxes[uid] = var

    def next_frame(self, event):
        """Advances the time slider by one step."""
        current_frame = self.time_slider.get()
        max_frame = self.time_slider["to"]
        if current_frame < max_frame:
            self.time_slider.set(current_frame + 1)
            self.update_plot()

    def prev_frame(self, event):
        """Moves the time slider back by one step."""
        current_frame = self.time_slider.get()
        if current_frame > 0:
            self.time_slider.set(current_frame - 1)
            self.update_plot()

    def update_plot(self):
        """Updates the plot with current cockroach positions and
        orientations in tracked frames."""
        self.ax.clear()
        self.current_time = int(self.time_step * self.time_slider.get())

        if self.data is not None:
            data_time = self.data[self.data["time"] == self.current_time]

            unique_ids = data_time["id"].unique()
            num_colors = len(unique_ids)
            colors = np.linspace(0, 1, num_colors)
            cmap = plt.get_cmap("autumn", num_colors)

            for particle_id, var in self.checkboxes.items():
                if var.get():
                    data_id = data_time[data_time["id"] == particle_id]
                    self.ax.plot(
                        data_id["position_x"].values,
                        data_id["position_y"].values,
                        marker="o",
                        c=cmap(colors[particle_id]),
                        ms=8,
                        ls="",
                        label=r"$p_{{{}}}$".format(particle_id)
                    )  # Centroid
                    self.ax.plot(
                        data_id["weighted_x"].values,
                        data_id["weighted_y"].values,
                        marker="v",
                        c=cmap(colors[particle_id]),
                        ms=8,
                        ls="",
                        label=r"$w_{{{}}}$".format(particle_id)
                    )  # Centroid weighted
                    self.ax.plot(
                        data_id["darkest_x"].values,
                        data_id["darkest_y"].values,
                        marker="x",
                        c=cmap(colors[particle_id]),
                        ms=8,
                        ls="",
                        label=r"$d_{{{}}}$".format(particle_id)
                    )  # Darkest pixel

                    length = 50
                    self.ax.arrow(
                        x=data_id["position_x"].values[0],
                        y=data_id["position_y"].values[0],
                        dx=length * np.sin(data_id["orientation"].values[0]),
                        dy=length * np.cos(data_id["orientation"].values[0]),
                        fc=cmap(colors[particle_id]),
                        ec=cmap(colors[particle_id]),
                        head_width=20,
                        head_length=20,
                        ls="-",
                        label=r"$p_{{{}}}$".format(particle_id)
                    )  # Orientation

        # Add background image
        self.add_background_image()

        # Axis configuration
        self.ax.set_xlabel("Position X", color="white", fontsize=16)
        self.ax.set_ylabel("Position Y", color="white", fontsize=16)
        self.ax.set_title(
            f"Tracking plot at time: {self.current_time}",
            color="black"
        )

        if self.show_legend:
            self.ax.legend(
                facecolor="grey",
                edgecolor="white",
                labelcolor="white",
                bbox_to_anchor=(1.10, 1.00)
            )

        # Fixed limits
        self.ax.set_xlim(0, 1920)
        self.ax.set_ylim(1080, 0)

        # Ticks configuration in the X-axis
        n_x_breaks, n_y_breaks = 20, 20
        self.ax.xaxis.set_major_locator(mtick.MaxNLocator(n_x_breaks))
        self.ax.xaxis.set_minor_locator(mtick.MaxNLocator(5 * n_x_breaks))
        self.ax.yaxis.set_major_locator(mtick.MaxNLocator(n_y_breaks))
        self.ax.yaxis.set_minor_locator(mtick.MaxNLocator(5 * n_y_breaks))
        self.ax.tick_params(axis="x", labelrotation=90, colors="black")
        self.canvas.draw()

        # Update secondary plots
        self.update_secondary_plots()

    def update_secondary_plots(self, event=None):
        """Updates the evolution of cockroaches positions and orientations."""
        if self.data is None or self.selected_id is None:
            return

        delta_t = 100
        mask = (
            (self.data["time"] >= self.current_time - delta_t) &
            (self.data["time"] <= self.current_time + delta_t)
        )
        selected_data = self.data[mask]

        # Clear previous plots
        self.ax_x.clear()
        self.ax_y.clear()
        self.ax_o.clear()

        # Plot new data
        if selected_data is not None:
            unique_ids = selected_data["id"].unique()
            num_colors = len(unique_ids)
            colors = np.linspace(0, 1, num_colors)
            cmap = plt.get_cmap("autumn", num_colors)

            for particle_id, var in self.checkboxes.items():
                if var.get():
                    data_id = selected_data[selected_data["id"] == particle_id]
                    self.ax_x.plot(
                        data_id["time"].values,
                        data_id["position_x"].values,
                        marker="o",
                        c=cmap(colors[particle_id]),
                        ms=4,
                        ls="-",
                        label=r"$p_{{{}}}$".format(particle_id)
                    )  # Centroid X
                    self.ax_y.plot(
                        data_id["time"].values,
                        data_id["position_y"].values,
                        marker="o",
                        c=cmap(colors[particle_id]),
                        ms=4,
                        ls="-",
                        label=r"$p_{{{}}}$".format(particle_id)
                    )  # Centroid Y
                    self.ax_o.plot(
                        data_id["time"].values,
                        data_id["orientation"].values,
                        marker="o",
                        c=cmap(colors[particle_id]),
                        ms=4,
                        ls="-",
                        label=r"$\theta_{{{}}}$".format(particle_id)
                    )  # Orientation

        # Axis configuration
        self.ax_x.set_xlabel("Time", color="black", fontsize=10)
        self.ax_y.set_xlabel("Time", color="black", fontsize=10)
        self.ax_o.set_xlabel("Time", color="black", fontsize=10)
        self.ax_x.set_ylabel("Position X", color="black", fontsize=10)
        self.ax_y.set_ylabel("Position Y", color="black", fontsize=10)
        self.ax_o.set_ylabel("Orientation", color="black", fontsize=10)

        if self.show_legend:
            self.ax.legend(
                facecolor="white",
                edgecolor="white",
                labelcolor="white",
                framealpha=0
            )

        # Fixed time limits
        min_time = max(0, self.current_time - delta_t)
        max_time = min(self.data["time"].max(), self.current_time + delta_t)
        self.ax_x.set_xlim(min_time, max_time)
        self.ax_y.set_xlim(min_time, max_time)
        self.ax_o.set_xlim(min_time, max_time)
        self.ax_o.set_ylim(-0.5 * np.pi, 1.5 * np.pi)

        # Ticks configuration in the X-axis
        n_x_breaks, n_y_breaks = 20, 10
        self.ax_x.xaxis.set_major_locator(mtick.MaxNLocator(n_x_breaks))
        self.ax_x.xaxis.set_minor_locator(mtick.MaxNLocator(5 * n_x_breaks))
        self.ax_x.yaxis.set_major_locator(mtick.MaxNLocator(n_y_breaks))
        self.ax_x.yaxis.set_minor_locator(mtick.MaxNLocator(5 * n_y_breaks))
        self.ax_x.tick_params(axis="x", labelrotation=90, colors="black")

        self.ax_y.xaxis.set_major_locator(mtick.MaxNLocator(n_x_breaks))
        self.ax_y.xaxis.set_minor_locator(mtick.MaxNLocator(5 * n_x_breaks))
        self.ax_y.yaxis.set_major_locator(mtick.MaxNLocator(n_y_breaks))
        self.ax_y.yaxis.set_minor_locator(mtick.MaxNLocator(5 * n_y_breaks))
        self.ax_y.tick_params(axis="x", labelrotation=90, colors="black")

        self.ax_o.xaxis.set_major_locator(mtick.MaxNLocator(n_x_breaks))
        self.ax_o.xaxis.set_minor_locator(mtick.MaxNLocator(5 * n_x_breaks))
        self.ax_o.yaxis.set_major_locator(mtick.MaxNLocator(n_y_breaks))
        self.ax_o.yaxis.set_minor_locator(mtick.MaxNLocator(5 * n_y_breaks))
        self.ax_o.tick_params(axis="x", labelrotation=90, colors="black")

        # Refresh canvas
        self.canvas2.draw()

    def add_background_image(self):
        """Load and adjust the background image in the selected folder."""
        image_filename = f"frame_{self.current_time:06d}.png"
        image_path = os.path.join(self.background_folder, image_filename)
        if image_path not in self.background_cache:
            try:
                img = Image.open(image_path).resize((1920, 1080))
                self.background_cache[image_path] = img
            except Exception as e:
                print("Error loading the image for time {}: {}".format(
                    self.current_time,
                    e
                ))
                return

        self.ax.imshow(
            self.background_cache[image_path],
            aspect="auto",
            extent=[0, 1920, 1080, 0],
            alpha=1.0
        )

    def toggle_edit_mode(self, event=None):
        """Toggle edit mode on/off using Shift key or button."""
        self.edit_mode = not self.edit_mode
        self.edit_position_button.config(
            bg="green" if self.edit_mode else "orange"
        )

    def on_click(self, event):
        """Selects a point for movement or rotation if edit mode is active."""
        if not self.edit_mode or self.data is None:
            return

        # Detect the closest point
        distances = np.sqrt(
            (self.data["position_x"] - event.xdata) ** 2
            + (self.data["position_y"] - event.ydata) ** 2
        )
        closest_index = distances.idxmin()
        self.selected_id = self.data.loc[closest_index, "id"]
        self.update_plot()

    def on_drag(self, event):
        """Moves a selected point while dragging."""
        if not self.edit_mode or self.selected_id is None or event.xdata is None or event.ydata is None:  # noqa: 501
            return

        mask = (
            (self.data["id"] == self.selected_id) &
            (self.data["time"] == self.current_time)
        )
        self.data.loc[
            mask,
            ["position_x", "position_y"]
        ] = [event.xdata, event.ydata]
        self.last_edited_id = self.selected_id  # Store last moved ID
        self.update_plot()

    def on_release(self, event):
        """Disable edit mode when mouse button is released."""
        self.selected_id = None

    def rotate_id(self, cockroach_id):
        """Changes the orientation of the selected ID."""
        # Auxiliary function for angles
        def mod_pi_shifted(x):
            """
            Computes x modulo pi, considering a phase shift of -pi/2.

            Parameters:
            x : float or array-like
                Input value(s) in radians.

            Returns:
            float or array-like
                The modulo result mapped within the range [-pi/2, 3 * pi/2].
            """
            return (x + np.pi/2) % (2 * np.pi) - np.pi/2

        if self.edit_mode and self.data is not None:
            # Ensure the rotation meets the constraints
            if cockroach_id in self.data["id"].unique():
                mask = (
                    (self.data["id"] == cockroach_id) &
                    (self.data["time"] >= self.current_time)
                )
                theta = np.pi / 18  # Rotate by 10°
                self.data.loc[mask, "orientation"] += theta  # noqa: 501
                self.data["orientation"] = self.data["orientation"].apply(mod_pi_shifted)  # noqa: 501

                self.update_plot()

    def save_csv(self):
        """Save the modified data of tracked cockroaches."""
        if self.data is not None:
            file_path = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv")]
            )
            if file_path:
                self.data["corrected_orientation"] = self.data["orientation"]
                self.data.to_csv(file_path, index=False)

    def on_resize(self, event):
        """Resizing of the app window."""
        window_width, window_height = event.width, event.height
        aspect_ratio = 1920 / 1080
        if window_width / window_height > aspect_ratio:
            new_width = int(window_height * aspect_ratio)
            new_height = window_height
        else:
            new_width = window_width
            new_height = int(window_width / aspect_ratio)

        self.fig.set_size_inches(
            new_width / self.fig.dpi,
            new_height / self.fig.dpi
        )
        self.canvas.draw()

    def run(self):
        root.mainloop()


# Deployment of the HoI - Cockroach Tracking
root = tk.Tk()
app = CSVPlotterApp(root)
app.run()
