import matplotlib.pyplot as plt
from scipy.ndimage import label
import ipywidgets as widgets
from jupyter_bbox_widget import BBoxWidget
import io
from PIL import Image
from IPython.display import display, clear_output
import numpy as np
import cv2


def show_image_and_mask(image: np.ndarray, mask: np.ndarray):
    """
    Plot image and its masked version.

    Args:
        image (np.ndarray)
        mask (np.ndarray)
    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(image_rgb)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Masked Image")
    plt.imshow(image_rgb * mask[:, :, np.newaxis].astype(bool))
    plt.axis("off")

    plt.tight_layout()
    plt.show()


class MaskEditor:
    """
    Class to refine the mask generated automatically with SAM2.
    """

    def __init__(self, mask: np.ndarray):
        """
        Args:
            mask (np.ndarray): bool array with mask of the original image
        """
        self.mask = mask
        self.original_mask = mask
        self.add = True

    def _denoise_mask(self, threshold: float):
        """
        Function to clean automatically the mask by removing small segmented areas. The principle is the following:

        1. dimension of the closed True areas in the mask is computed

        2. the areas are placed in increasing order and divided into deciles

        3. small areas are converted into False, so that they wil no be included in the mask anymore. The parameter that defines the
        threshold under which areas are considered "small" is threshold, which refers to the decile of the area distribution.

        Args:
            threshold (float): decile of the True areas under which they will be converted to False
        """
        self.mask = self.original_mask
        if self.mask.dtype != "float32":
            self.mask = self.mask.astype("float32")

        self.mask = cv2.medianBlur(self.mask, 5)
        self.mask = self.mask.astype("bool")
        labeled_mask, num_features = label(self.mask)

        areas = np.array([(labeled_mask == i + 1).sum() for i in range(num_features)])
        quantile_threshold = np.quantile(areas, threshold)

        for i, area in enumerate(areas):
            if area < quantile_threshold:
                self.mask[labeled_mask == i + 1] = False

        plt.imshow(self.mask, cmap="gray", interpolation="none")
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    def auto_denoise(self):
        """
        Function to clean automatically the mask by removing small segmented areas. The principle is the following:

        1. dimension of the closed True areas in the mask is computed

        2. the areas are placed in increasing order and divided into deciles

        3. small areas are converted into False, so that they wil no be included in the mask anymore.
           The parameter that defines the threshold (selected via interactive slider) under which areas
           are considered "small" is threshold, which refers to the decile of the area distribution.
        """

        def _denoise_from_selection(change):
            clear_output(wait=True)
            display(intensity_slider)
            self._denoise_mask(change["new"])

        intensity_slider = widgets.FloatSlider(
            value=0.5,
            min=0.0,
            max=1.0,
            step=0.1,
            description="Denoise level:",
            continuous_update=False,
        )
        intensity_slider.observe(_denoise_from_selection, names="value")
        display(intensity_slider)
        self._denoise_mask(intensity_slider.value)

    def _mask_to_bytes(self, mask: np.ndarray):
        if mask.dtype != "np.uint8":
            mask = (mask * 255).astype(np.uint8)
        mask_img = Image.fromarray(mask)
        buffer = io.BytesIO()
        mask_img.save(buffer, format="PNG")
        return buffer.getvalue()

    def _toggle_widget(self):
        def _on_switch_change(change):
            self.add = change["new"]
            if self.add != self.prev_toggle:
                self.widget.image_bytes = self._mask_to_bytes(self.widget.temp_mask)
                clear_output(wait=True)
                display(self.menu)
                display(self.widget)
                self.widget.mask = self.widget.temp_mask
                self.prev_toggle = self.add

        self.switch = widgets.ToggleButton(
            value=True,
            description="Add-Mode",
            disabled=False,
            button_style="",
            tooltip="Toggle Switch",
            icon="check",
        )
        self.switch.observe(_on_switch_change, names="value")
        return self.switch

    def _save_mask_widget(self):
        def _on_save_button_clicked(b):
            self.mask = self.widget.temp_mask
            clear_output(wait=True)
            print("Mask correctly saved")

        save_button = widgets.Button(
            description="Save",
            disabled=False,
            button_style="",
            tooltip="Click to save",
            icon="save",
        )

        save_button.on_click(_on_save_button_clicked)
        return save_button

    def _menu_widget(self):
        toggle_button = self._toggle_widget()
        save_button = self._save_mask_widget()
        hbox = widgets.HBox([toggle_button, save_button])
        return hbox

    def manual_edit(self):
        """
        Allows to manually remove or add elements from the mask using bounding boxes (Jupyter widget)
        """
        self.prev_toggle = True

        def manual_clean():
            clear_output(wait=True)
            display(self.menu)
            display(self.widget)
            self.widget.temp_mask = self.widget.mask.copy()
            for box in self.widget.bboxes:
                self.widget.temp_mask[
                    box["y"] : (box["y"] + box["height"]),
                    box["x"] : (box["x"] + box["width"]),
                ] = self.add
            plt.imshow(self.widget.temp_mask, cmap="gray", interpolation="none")
            plt.axis("off")
            plt.tight_layout()
            plt.show()

        self.widget = BBoxWidget()
        self.widget.on_submit(manual_clean)
        self.widget.mask = self.mask
        self.widget.temp_mask = self.mask
        self.widget.image_bytes = self._mask_to_bytes(self.mask)
        self.menu = self._menu_widget()
        display(self.menu)
        display(self.widget)
