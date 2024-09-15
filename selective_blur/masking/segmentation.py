import cv2
import torch
import io

import numpy as np
import supervision as sv

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from jupyter_bbox_widget import BBoxWidget

import matplotlib.pyplot as plt
from PIL import Image
from IPython.display import display, clear_output


class Segmenter:
    """
    Class to segment an image from user selection in a jupyter notebook leveraging Segment Anything 2
    """

    def _config_cuda(self):
        torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    def _setup_predictor(self, model):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = f"checkpoints/sam2_hiera_{model}.pt"
        config = f"sam2_hiera_{model[0]}.yaml"
        sam2_model = build_sam2(
            config, checkpoint, device=device, apply_postprocessing=False
        )
        self.predictor = SAM2ImagePredictor(sam2_model)

    def __init__(self, image_path: str, model: str = "tiny"):
        """
        Args:
            image_path (str): path to the image that will be segmented
            model (str, optional): SAM2 model version (choice between "tiny", "small", "base_plus", "large"). Defaults to "tiny".

        Raises:
            ValueError: if model is not one of ["tiny", "small", "base_plus", "large"]
        """
        self.image_bgr = cv2.imread(image_path)
        self.image_rgb = cv2.cvtColor(self.image_bgr, cv2.COLOR_BGR2RGB)
        self._config_cuda()
        if model not in ["tiny", "small", "base_plus", "large"]:
            raise ValueError(
                "Available models are ['tiny', 'small', 'base_plus', 'large']"
            )
        self._setup_predictor(model)

    def _image_to_bytes(self):
        img = Image.fromarray(self.image_rgb)
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        self.image_bytes = buffer.getvalue()

    def select_from_image(self):
        """
        Allows to select with a point or a bounding box the subjects that will be segmented from an image (Jupyter Widget).
        After user submits the selection, three candidates masks (np.ndarray with dtype bool), saved in self.masks are displayed
        along with their confidence scores.
        """

        def segmentation():
            nonlocal widget
            clear_output(wait=True)
            display(widget)
            boxes = widget.bboxes
            input_point = np.array([[box["x"], box["y"]] for box in boxes])
            input_label = np.ones(input_point.shape[0])
            self.predictor.set_image(self.image_rgb)
            self.masks, scores, logits = self.predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=True,
            )
            sv.plot_images_grid(
                images=self.masks,
                titles=[f"score: {score:.2f}" for score in scores],
                grid_size=(1, 3),
                size=(12, 12),
            )

        self._image_to_bytes()

        widget = BBoxWidget()
        widget.on_submit(segmentation)
        widget.image_bytes = self.image_bytes
        display(widget)

    def choose_mask(self, mask_number: int):
        """
        Args:
            mask_number (int): choice between the three mask proposals displayed when select_from_image is executed (choices available are 0, 1 or 2)

        Raises:
            ValueError: if mask_number is not 0, 1 or 2
        """
        if mask_number > len(self.masks) - 1:
            raise ValueError("Choice can be either 0, 1 or 2")
        else:
            self.best_mask = self.masks[mask_number]
            self.best_mask = self.best_mask.squeeze()
