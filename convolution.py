import cv2
import torch
from torchvision import transforms

convert_tensor = transforms.ToTensor()

image = cv2.imread("1 - original_image.jpg")
image_tensor = convert_tensor(image)

# Filter creation
filter_raw = torch.tensor(
    [
        [
            [1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0],
        ],  # Channel 1: Get the vertical lines
        [
            [-1.0, -1.0, -1.0],
            [1.0, 1.0, 1.0],
            [-1.0, -1.0, -1.0],
        ],  # Channel 2: Get the vertical lines
        [
            [1.0, -1.0, 1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, 1.0, -1.0],
        ],  # Channel 3: Get crosses
    ]
)
filter_raw = filter_raw[None, :]

# With square kernels and equal stride
m = torch.nn.Conv2d(3, 1, 3, stride=1)
m_next = torch.nn.Conv2d(1, 1, 3, stride=1)
m.weight = torch.nn.Parameter(filter_raw)
output_1 = m(image_tensor)
output = m_next(output_1)
out_image_conv_1 = output_1.permute(1, 2, 0).detach().numpy()
out_image_conv_1 = cv2.convertScaleAbs(out_image_conv_1, alpha=(255.0))
cv2.imwrite("conv_1.jpg", out_image_conv_1)
out_image_conv_2 = output.permute(1, 2, 0).detach().numpy()
out_image_conv_2 = cv2.convertScaleAbs(out_image_conv_2, alpha=(255.0))
cv2.imwrite("conv_2.jpg", out_image_conv_2)
