import torch
import numpy as np


class MidasModel:
    """
    Class to load Intel MiDaS model and make predictions
    """

    def __init__(self, model: str = "DPT_Hybrid") -> None:
        """
        Args:
            model (str): MiDaS model version, available options are ["MiDaS_small", "DPT_Hybrid", "DPT_Large"]. Defaults to "DPT_Hybrid".

        Raises:
            ValueError: if model is not one of ["MiDaS_small", "DPT_Hybrid", "DPT_Large"]
        """
        if model not in ["MiDaS_small", "DPT_Hybrid", "DPT_Large"]:
            raise ValueError(
                "Available models are ['MiDaS_small', 'DPT_Hybrid', 'DPT_Large']"
            )
        self.midas = torch.hub.load("intel-isl/MiDaS", model)
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.midas.to(self.device)
        transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        self.transform = transforms.small_transform

    def predict_adjust(self, image: np.ndarray) -> np.ndarray:
        """
        Args:
            image (np.ndarray)

        Returns:
            np.ndarray: depth map of the image
        """
        imgbatch = self.transform(image).to(self.device)
        with torch.no_grad():
            prediction = self.midas(imgbatch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=image.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        output = prediction.cpu().numpy()
        return output
