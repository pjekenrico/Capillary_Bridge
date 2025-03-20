import os, sys, glob, re, matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QVBoxLayout,
    QWidget,
    QLineEdit,
    QLabel,
    QPushButton,
    QHBoxLayout,
    QFileDialog,
    QCheckBox,
)
import numpy as np
from segment_profiles.tools import (
    rotate_and_crop_img,
    most_common_image_format,
)

matplotlib.use("Qt5Agg")

colors = ["blue", "orange"]


class ImageViewer(QMainWindow):
    def __init__(self, folder_path):
        super().__init__()
        self.folder_path = folder_path
        self.extension = ".tiff"
        self.image_files = self.load_image_files()
        self.image_numbers = self.get_image_numbers()
        self.marked_boxes = [(0, 0, 0, 0), (0, 0, 0, 0)]  # Default to two empty boxes
        self.selected_height = None  # To store the selected height
        self.selected_height2 = None  # To store the second selected height
        self.selected_height3 = None  # To store the third selected height
        self.selected_height4 = None  # To store the third selected height
        self.current_image = None
        self.rect_selector = None
        self.current_box_index = 0  # Track which box is being edited
        self.line_selector_cid1 = None  # Store connection ID for the first height
        self.line_selector_cid2 = None  # Store connection ID for the second height
        self.line_selector_cid3 = None  # Store connection ID for the second height
        self.line_selector_cid4 = None  # Store connection ID for the second height
        self.break_index = None  # Store connection ID for the second height
        self.result = {
            "boxes": self.marked_boxes,
            "heights": [
                self.selected_height,
                self.selected_height2,
                self.selected_height3,
            ],
            "bath_height": self.selected_height4,
            "break_index": self.break_index,
        }
        self.angle = 0
        self.flat_top = False
        self.initUI()

    def load_image_files(self):
        image_files = glob.glob(os.path.join(self.folder_path, "*" + self.extension))
        return sorted(image_files)

    def get_image_numbers(self):
        # Extract image numbers from filenames
        image_numbers = []

        for f in self.image_files:
            # Get the filename without the path
            basename = os.path.basename(f)

            # Use regex to extract the numeric part (digits after the last underscore, before the extension)
            match = re.search(r"_(\d+)\.[a-zA-Z0-9]+$", basename)

            if match:
                # Convert the extracted number to an integer
                image_number = int(match.group(1))
                image_numbers.append(image_number)
            else:
                raise ValueError(f"Filename format is incorrect: {basename}")

        return image_numbers

    def load_image(self, image_number):
        # Format the number with leading zeros (e.g., 00001)
        number_str = f"{image_number:05d}"

        # Find the file that matches the pattern imageseriesname_00001.<extension>
        matching_files = [
            f for f in self.image_files if f"_{number_str}" in os.path.basename(f)
        ]

        if not matching_files:
            number_str = f"{image_number:04d}"

            # Find the file that matches the pattern imageseriesname_00001.<extension>
            matching_files = [
                f for f in self.image_files if f"_{number_str}" in os.path.basename(f)
            ]

        if not matching_files:
            number_str = f"{image_number:03d}"

            # Find the file that matches the pattern imageseriesname_00001.<extension>
            matching_files = [
                f for f in self.image_files if f"_{number_str}" in os.path.basename(f)
            ]

        if matching_files:
            # Use the first matching file (in case there are multiple)
            image_path = os.path.join(self.folder_path, matching_files[0])

            # Check if the file exists and load the image
            if os.path.exists(image_path):
                return plt.imread(image_path)

        return None

    def browse_folder(self):
        # Open a file dialog to select a folder
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder")

        # If a folder was selected, update the input field and reload images
        if folder_path:
            self.folder_path = folder_path
            self.folder_input.setText(folder_path)
            self.extension = most_common_image_format(folder_path)
            self.image_files = self.load_image_files()
            self.image_numbers = self.get_image_numbers()

            if self.image_numbers:  # Update the range label if images are found
                self.range_label.setText(
                    f"Available images: {min(self.image_numbers)} to {max(self.image_numbers)}"
                )
                self.update_image()  # Refresh the displayed image
            else:
                self.range_label.setText("No images found in the selected folder.")
                self.ax.cla()  # Clear the current axes
                self.ax.axis("off")  # Hide axes
                self.fig.canvas.draw()

    def initUI(self):
        # Set up the main widget
        widget = QWidget()
        self.setCentralWidget(widget)

        # Set up the layout
        layout = QVBoxLayout()
        widget.setLayout(layout)

        # Add a layout for folder selection
        folder_layout = QHBoxLayout()

        # Add a text input for displaying and editing the folder path
        self.folder_input = QLineEdit(self)
        self.folder_input.setText(self.folder_path)  # Display the default path
        folder_layout.addWidget(self.folder_input)

        # Add a button to browse for a folder
        browse_button = QPushButton("Browse", self)
        browse_button.clicked.connect(self.browse_folder)
        folder_layout.addWidget(browse_button)

        layout.addLayout(folder_layout)

        # Add a label showing the available image range
        if self.image_numbers:
            self.range_label = QLabel(
                f"Available images: {min(self.image_numbers)} to {max(self.image_numbers)}"
            )
        else:
            self.range_label = QLabel("No images found in the selected folder.")
        layout.addWidget(self.range_label)

        # Add a text input for selecting the image number
        self.image_input = QLineEdit(self)
        self.image_input.setPlaceholderText("Enter image number (e.g., 1243)")
        self.image_input.returnPressed.connect(self.update_image)
        layout.addWidget(self.image_input)

        # Add widgets to show and edit the indices of the two selected boxes
        self.box_inputs = []
        for i in range(2):
            box_layout = QHBoxLayout()

            # Add the button to select the box
            box_button = QPushButton(f"Set Box {i+1}", self)
            box_button.clicked.connect(lambda _, idx=i: self.set_box(idx))
            box_layout.addWidget(box_button)

            # Add the label and input field for the box indices
            box_label = QLabel(f"Box {i+1} indices:", self)
            box_input = QLineEdit(self)
            box_input.setPlaceholderText("x1, y1, x2, y2")
            box_input.editingFinished.connect(
                lambda idx=i: self.update_box_from_input(idx)
            )
            self.box_inputs.append(box_input)
            box_layout.addWidget(box_label)
            box_layout.addWidget(box_input)
            layout.addLayout(box_layout)

        # Add a checkbox for Flat Top
        self.flat_top_checkbox = QCheckBox("Flat top", self)
        self.flat_top_checkbox.stateChanged.connect(self.update_flat_top)
        layout.addWidget(self.flat_top_checkbox)

        # Add a button to select the height
        height_layout = QHBoxLayout()
        height_button = QPushButton("Set Height", self)
        height_button.clicked.connect(self.set_height)
        height_layout.addWidget(height_button)

        # Add the label and input field for the selected height
        self.height_input = QLineEdit(self)
        self.height_input.setPlaceholderText("Enter height (y-coordinate)")
        self.height_input.editingFinished.connect(self.update_height_from_input)
        height_layout.addWidget(QLabel("Height:"))
        height_layout.addWidget(self.height_input)
        layout.addLayout(height_layout)

        # Add a button to select the second height
        height_layout2 = QHBoxLayout()
        height_button2 = QPushButton("Set Height 2", self)
        height_button2.clicked.connect(self.set_height2)
        height_layout2.addWidget(height_button2)

        # Add the label and input field for the second selected height
        self.height_input2 = QLineEdit(self)
        self.height_input2.setPlaceholderText("Enter height 2 (y-coordinate)")
        self.height_input2.editingFinished.connect(self.update_height_from_input2)
        height_layout2.addWidget(QLabel("Height 2:"))
        height_layout2.addWidget(self.height_input2)
        layout.addLayout(height_layout2)

        # Add a button to select the second height
        height_layout3 = QHBoxLayout()
        height_button3 = QPushButton("Set Height 3", self)
        height_button3.clicked.connect(self.set_height3)
        height_layout3.addWidget(height_button3)

        # Add the label and input field for the second selected height
        self.height_input3 = QLineEdit(self)
        self.height_input3.setPlaceholderText("Enter height 3 (y-coordinate)")
        self.height_input3.editingFinished.connect(self.update_height_from_input3)
        height_layout3.addWidget(QLabel("Height 3:"))
        height_layout3.addWidget(self.height_input3)
        layout.addLayout(height_layout3)
        
        # Add a button to select the second height
        height_layout4 = QHBoxLayout()
        height_button4 = QPushButton("Set bath height", self)
        height_button4.clicked.connect(self.set_height4)
        height_layout4.addWidget(height_button4)
        
        # Add the label and input field for the second selected height
        self.height_input4 = QLineEdit(self)
        self.height_input4.setPlaceholderText("Enter bath height (y-coordinate)")
        self.height_input4.editingFinished.connect(self.update_height_from_input4)
        height_layout4.addWidget(QLabel("Bath height:"))
        height_layout4.addWidget(self.height_input4)
        layout.addLayout(height_layout4)

        # Add a label and input field for setting the rotation angle
        angle_layout = QHBoxLayout()
        angle_label = QLabel("Rotation Angle (degrees):", self)
        self.angle_input = QLineEdit(self)
        self.angle_input.setPlaceholderText("Enter rotation angle (e.g., 45)")
        self.angle_input.setText(str(self.angle))
        self.angle_input.editingFinished.connect(self.update_angle)
        angle_layout.addWidget(angle_label)
        angle_layout.addWidget(self.angle_input)
        layout.addLayout(angle_layout)
        
        # Add a label and input field for setting the rotation angle
        break_index_layout = QHBoxLayout()
        break_index_label = QLabel("Image number at breakage:", self)
        self.break_index_input = QLineEdit(self)
        self.break_index_input.setPlaceholderText("Enter break image number (e.g., 1045)")
        self.break_index_input.setText(str(self.break_index))
        break_index_layout.addWidget(break_index_label)
        break_index_layout.addWidget(self.break_index_input)
        layout.addLayout(break_index_layout)

        # Add a confirm button to close the figure and continue
        confirm_button = QPushButton("Confirm", self)
        confirm_button.clicked.connect(self.confirm_selection)
        layout.addWidget(confirm_button)

        # Set up the Matplotlib figure and axes
        self.fig, self.ax = plt.subplots(figsize=(5, 5))
        self.ax.set_title("Selected Image")
        self.ax.axis("off")

        # Add the Matplotlib figure to the Qt layout
        layout.addWidget(self.fig.canvas)

        # Set the window title and show the window
        self.setWindowTitle("Single Image Viewer")
        self.show()

    def update_flat_top(self, state):
        self.flat_top = bool(state)

    def update_angle(self):
        try:
            self.angle = float(self.angle_input.text())
            self.update_image()  # Update the image with the new angle
        except ValueError:
            pass

    def update_image(self):
        # Get the entered image number
        try:
            image_number = int(self.image_input.text())
        except ValueError:
            return  # If the input is not a valid number, do nothing

        if image_number < min(self.image_numbers):
            image_number = min(self.image_numbers)
        elif image_number > max(self.image_numbers):
            image_number = max(self.image_numbers)

        if image_number in self.image_numbers:
            self.current_image = self.load_image(image_number)
            if self.current_image is not None:
                # Ensure the image is grayscale
                if (
                    len(self.current_image.shape) == 3
                    and self.current_image.shape[2] == 3
                ):
                    # Convert to grayscale by averaging RGB channels
                    self.current_image = self.current_image.mean(axis=2).astype(
                        np.uint8
                    )

                # Apply rotation and cropping if necessary
                if self.angle != 0:
                    self.current_image = rotate_and_crop_img(
                        self.current_image, self.angle
                    )

                # Update the plot
                self.ax.cla()  # Clear the current axes
                self.ax.imshow(
                    self.current_image, cmap="gray"
                )  # Show the processed image
                self.ax.axis("off")  # Hide axes
                self.fig.tight_layout()

                # Draw existing boxes if any
                self.draw_boxes()
                self.draw_line()

    def draw_boxes(self):

        # Clear all previous patches (rectangles) before drawing new ones
        for patch in reversed(self.ax.patches):
            patch.remove()

        for box, color in zip(self.marked_boxes, colors):
            self.ax.add_patch(
                plt.Rectangle(
                    (box[0], box[1]),
                    box[2] - box[0],
                    box[3] - box[1],
                    edgecolor=color,
                    facecolor="none",
                    linewidth=2,
                )
            )
        self.fig.canvas.draw()

    def start_selector(self):
        # Enable box selection
        if self.rect_selector:
            self.rect_selector.disconnect_events()  # Disconnect previous selector if exists

        self.rect_selector = RectangleSelector(
            self.ax,
            self.onselect,
            useblit=True,
            button=[1],  # Only respond to left mouse button
            minspanx=5,
            minspany=5,  # Minimum size of the box in pixels
            spancoords="pixels",
            interactive=True,
            props={
                "facecolor": colors[self.current_box_index],
                "alpha": 0.5,
            },
        )

    def onselect(self, eclick, erelease):
        # Function to handle rectangle selection
        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)
        box = (x1, y1, x2, y2)

        # Update the box at the current index
        self.marked_boxes[self.current_box_index] = box

        # Update the corresponding input field
        self.box_inputs[self.current_box_index].setText(
            f"{box[0]}, {box[1]}, {box[2]}, {box[3]}"
        )

        # Redraw the rectangles
        self.draw_boxes()

        # Disable the rectangle selector after the selection is made
        if self.rect_selector:
            self.rect_selector.disconnect_events()
            self.rect_selector = None

    def update_box_from_input(self, index):
        # Update the marked_boxes list based on user input in the QLineEdit widget
        try:
            # Parse the input text into a tuple of integers
            values = list(map(int, self.box_inputs[index].text().split(",")))
            if len(values) == 4:
                self.marked_boxes[index] = tuple(values)
                self.draw_boxes()
        except ValueError:
            pass  # Ignore if the input is not valid

    def set_box(self, index):
        # Set the current box index for selection
        self.current_box_index = index
        self.start_selector()  # Start the selection process

    def confirm_selection(self):
        # Close the window and continue with the rest of the code
        self.result = {
            "boxes": self.marked_boxes,
            "heights": sorted(
                [self.selected_height, self.selected_height2, self.selected_height3]
            ),
            "bath_height": self.selected_height4,
            "angle": self.angle,
            "flat_top": self.flat_top,
            "break_index": self.break_index,
        }

        self.close()

    def get_result(self):
        # This method can be called to retrieve the result after the window is closed
        return self.result

    def start_selector(self):
        # Disconnect all previous events
        self.disconnect_all_selectors()

        # Enable box selection
        self.rect_selector = RectangleSelector(
            self.ax,
            self.onselect,
            useblit=True,
            button=[1],  # Only respond to left mouse button
            minspanx=5,
            minspany=5,  # Minimum size of the box in pixels
            spancoords="pixels",
            interactive=True,
            props={
                "facecolor": colors[self.current_box_index],
                "alpha": 0.5,
            },
        )

    def set_height(self):
        # Disconnect all previous events
        self.disconnect_all_selectors()

        # Enable height selection for the first height
        self.line_selector_cid1 = self.fig.canvas.mpl_connect(
            "button_press_event", self.onclick_height1
        )

    def set_height2(self):
        # Disconnect all previous events
        self.disconnect_all_selectors()

        # Enable height selection for the second height
        self.line_selector_cid2 = self.fig.canvas.mpl_connect(
            "button_press_event", self.onclick_height2
        )

    def set_height3(self):
        # Disconnect all previous events
        self.disconnect_all_selectors()

        # Enable height selection for the second height
        self.line_selector_cid3 = self.fig.canvas.mpl_connect(
            "button_press_event", self.onclick_height3
        )
        
    def set_height4(self):
        # Disconnect all previous events
        self.disconnect_all_selectors()

        # Enable height selection for the second height
        self.line_selector_cid4 = self.fig.canvas.mpl_connect(
            "button_press_event", self.onclick_height4
        )

    def disconnect_all_selectors(self):
        # Disconnect all selection events (box and height)
        if self.rect_selector:
            self.rect_selector.disconnect_events()
            self.rect_selector = None
        if self.line_selector_cid1 is not None:
            self.fig.canvas.mpl_disconnect(self.line_selector_cid1)
            self.line_selector_cid1 = None
        if self.line_selector_cid2 is not None:
            self.fig.canvas.mpl_disconnect(self.line_selector_cid2)
            self.line_selector_cid2 = None
        if self.line_selector_cid3 is not None:
            self.fig.canvas.mpl_disconnect(self.line_selector_cid3)
            self.line_selector_cid3 = None
        if self.line_selector_cid4 is not None:
            self.fig.canvas.mpl_disconnect(self.line_selector_cid4)
            self.line_selector_cid4 = None

    def onclick_height1(self, event):
        # Handle mouse click event to set the first height
        if event.inaxes == self.ax:
            self.selected_height = int(event.ydata)
            self.height_input.setText(str(self.selected_height))
            self.draw_line()

    def onclick_height2(self, event):
        # Handle mouse click event to set the second height
        if event.inaxes == self.ax:
            self.selected_height2 = int(event.ydata)
            self.height_input2.setText(str(self.selected_height2))
            self.draw_line()

    def onclick_height3(self, event):
        # Handle mouse click event to set the second height
        if event.inaxes == self.ax:
            self.selected_height3 = int(event.ydata)
            self.height_input3.setText(str(self.selected_height3))
            self.draw_line()
            
    def onclick_height4(self, event):
        # Handle mouse click event to set the second height
        if event.inaxes == self.ax:
            self.selected_height4 = int(event.ydata)
            self.height_input4.setText(str(self.selected_height4))
            self.draw_line()

    def update_height_from_input(self):
        # Update the first selected height based on the input field
        try:
            self.selected_height = int(self.height_input.text())
            self.draw_line()
        except ValueError:
            pass  # Ignore if the input is not valid

    def update_height_from_input2(self):
        # Update the second selected height based on the input field
        try:
            self.selected_height2 = int(self.height_input2.text())
            self.draw_line()
        except ValueError:
            pass  # Ignore if the input is not valid

    def update_height_from_input3(self):
        # Update the second selected height based on the input field
        try:
            self.selected_height3 = int(self.height_input3.text())
            self.draw_line()
        except ValueError:
            pass  # Ignore if the input is not valid
        
    def update_height_from_input4(self):
        # Update the second selected height based on the input field
        try:
            self.selected_height4 = int(self.height_input4.text())
            self.draw_line()
        except ValueError:
            pass  # Ignore if the input is not valid

    def draw_line(self):
        # Remove previous lines if they exist
        for line in self.ax.lines:
            line.remove()

        if self.selected_height is not None:
            self.ax.axhline(self.selected_height, color="red", linewidth=2)
        if self.selected_height2 is not None:
            self.ax.axhline(self.selected_height2, color="red", linewidth=2)
        if self.selected_height3 is not None:
            self.ax.axhline(self.selected_height3, color="red", linewidth=2)
        if self.selected_height4 is not None:
            self.ax.axhline(self.selected_height4, color="green", linewidth=2)
        self.fig.canvas.draw()


def preprocess_images(folder_path: str) -> dict:
    app = QApplication(sys.argv)
    viewer = ImageViewer(folder_path)
    app.exec_()

    # After the window is closed, get the result
    result = viewer.get_result()

    return (
        result["boxes"],
        result["heights"],
        result["bath_height"],
        result["angle"],
        result["break_index"],
        viewer.folder_path,
        viewer.flat_top,
    )
