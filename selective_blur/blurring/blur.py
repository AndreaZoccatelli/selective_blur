import numpy as np
import cv2 as cv2
import matplotlib.pyplot as plt
from selective_blur.blurring.midasmodel import MidasModel
import ipywidgets as widgets
from IPython.display import display, clear_output


class SimpleBlur:
    """
    Class to apply depth-aware blur to an image using MiDaS model by Intel
    """

    def _get_depth_map(self):
        if self.image is None:
            raise ValueError("Failed to load image.")
        self.depth_map = self.midas.predict_adjust(self.image)
        if self.depth_map is None:
            raise ValueError("Failed to compute depth map.")
        self.depth_scaled = (self.depth_map - self.depth_map.min()) / (
            self.depth_map.max() - self.depth_map.min()
        )

    def __init__(self, image_path: str, model: str, mask: np.ndarray = None):
        """
        Args:
            image_path (str)
            model (str): MiDaS model version, available options are ["MiDaS_small", "DPT_Hybrid", "DPT_Large"]. Defaults to "DPT_Hybrid".
            mask (np.ndarray, optional): mask that isolates a subject of the image. Defaults to None.
        """
        self.image = cv2.imread(image_path)
        self.midas = MidasModel(model)
        self.mask = mask
        self._get_depth_map()

    def show_image(self, image: np.array):
        if image.shape[-1] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.imshow(image)
        plt.axis("off")
        plt.show()

    def show_depth_map(self):
        self.show_image(self.depth_map.astype(int))

    def blur(self, kernel: int):
        """
        Applies gaussian blur based on the depth decile in which the pixel falls into.
        The process is the following:

        1. the pixel depth values obtained with MiDaS are divided into deciles

        2. if user has provided a mask array for the image its average depth value is computed. If
           it is greater or equal to the median overall depth value of the image (subject near to the camer objective)
           the blur will be higher for more distant subjects, if less or equal viceversa

        3. if the mask has not been provided more far objects will be more blurred by default
           (increasing filter size correspond to lower deciles)

        Args:
            kernel (int): the max kernel size dimension for gaussian blur
        """
        masks = []
        start = 0.1
        end = 1
        step = 0.1

        quantiles = np.arange(start, end, step)
        for val in quantiles:
            q = np.quantile(self.depth_scaled, val)
            if val == start:
                masks.append(self.depth_scaled < q)
            else:
                masks.append((self.depth_scaled >= prev) & (self.depth_scaled < q))

            prev = q

        masks.append(self.depth_scaled > prev)

        factors = np.append(quantiles, 1)[::-1]
        if self.mask is not None and np.mean(
            self.depth_scaled[self.mask]
        ) < np.quantile(self.depth_scaled, 0.5):
            factors = factors[
                ::-1
            ]  # if mask is "near" blur is higher for more distant objects (q1 = far, q10 = near)

        self.image_blurred = self.image.copy()
        for i in range(len(masks)):
            size = int(round(kernel * factors[i]))
            if size % 2 == 0:
                size += 1

            self.image_blurred[masks[i]] = cv2.GaussianBlur(
                self.image, (size, size), 0
            )[masks[i]]


class Selector(SimpleBlur):
    """
    Class to interactively select blur level from an image
    """

    def __init__(self, image_path: str, model: str, mask: np.ndarray = None):
        """
        Args:
            image_path (str)
            model (str): MiDaS model version, available options are ["MiDaS_small", "DPT_Hybrid", "DPT_Large"]. Defaults to "DPT_Hybrid".
            mask (np.ndarray, optional): mask that isolates a subject of the image. Defaults to None.
        """
        super().__init__(image_path, model, mask)

    def select_kernel_size(self):
        """
        Select max kernel for gaussian blur, applied with SimpleBlur, with Jupyter widget slider
        """

        def _update_intensity(change):
            kernel_size = change["new"]
            if kernel_size % 2 == 0:
                kernel_size += 1
            clear_output(wait=True)
            display(intensity_slider)
            self.blur(kernel_size)
            self.show_image(self.image_blurred)

        intensity_slider = widgets.IntSlider(
            value=50,
            min=0,
            max=100,
            step=1,
            description="Intensity:",
            continuous_update=False,
        )
        intensity_slider.observe(_update_intensity, names="value")
        display(intensity_slider)
        self.blur(intensity_slider.value)
        self.show_image(self.image_blurred)
