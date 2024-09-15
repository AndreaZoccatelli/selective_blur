import numpy as np
import cv2
import ipywidgets as widgets
from IPython.display import display, clear_output
import matplotlib.pyplot as plt


class Blender:
    """
    class to blend the blurred image with the masked subject from the original image
    """

    def __init__(
        self,
        original_image: np.ndarray,
        mask: np.ndarray,
        image_blurred: np.ndarray,
    ):
        """
        Args:
            original_image (np.ndarray)
            mask (np.ndarray): bool mask of the original image
            image_blurred (np.ndarray): blurred version of the original image
        """
        if mask.dtype != "bool":
            mask = mask.astype(bool)
        self.mask = mask
        self.original_image = original_image
        self.image_blurred = image_blurred

    def _blend_mask(self, beta: float = 0.5):
        alpha = 1 - beta
        self.image_blended = self.image_blurred.copy()
        self.image_blended[self.mask] = cv2.addWeighted(
            self.image_blurred[self.mask],
            alpha,
            self.original_image[self.mask],
            beta,
            0.0,
        )
        self.final_rgb = cv2.cvtColor(self.image_blended, cv2.COLOR_BGR2RGB)

        plt.imshow(self.final_rgb)
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    def blend(self):
        """
        Blends original image with mask by choosing the weight beta (from 0 to 1, step of 0.1) of mask with a Jupyter widget slider (weight of original image alpha = 1 - beta)
        """

        def _blend_mask_from_selection(change):
            clear_output(wait=True)
            display(intensity_slider)
            self._blend_mask(change["new"])

        intensity_slider = widgets.FloatSlider(
            value=0.5,
            min=0.0,
            max=1.0,
            step=0.1,
            description="Sharpness level:",
            continuous_update=False,
        )
        intensity_slider.observe(_blend_mask_from_selection, names="value")
        display(intensity_slider)
        self._blend_mask(intensity_slider.value)
