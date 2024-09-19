# auto denoising example

Auto denoising with ```MaskEditor.auto_denoise``` allows to remove small areas selected by SAM2 but that are not part of the subject of interest.

For example, this mask includes also part of the other horse:

![Masked image](example_images/denoising/original_mask.png)

![Original mask](example_images/denoising/max_noise.png)

## Denoising level = 0.5
![Denoised mask 0.5](example_images/denoising/05.png)

## Denoising level = 1
As the denoise level increases, more small parts are removed from the mask. This, as seen with level = 1, can also produce unwanted results (half of the hat is removed from the mask). For a more precise selection, it is better to use ```MaskEditor.manual_edit```.

![Denoised mask 1](example_images/denoising/1.png)

