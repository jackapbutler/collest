import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from dataclasses import dataclass
import scipy.ndimage as ndi


@dataclass
class SyntheticMapConfig:
    size: int = 64


def transform_color_map(color_map: dict[int, list[int]]) -> dict[str, list[float]]:
    return {k: [v / 255.0 for v in color_map[k]] for k in color_map.keys()}


_satellite_colour_map = {
    0: [128, 0, 128],  # Water -> Purple
    1: [255, 165, 0],  # Label 1 -> Orange
    2: [0, 255, 0],  # Label 2 -> Green
    3: [0, 0, 255],  # Label 3 -> Blue
    4: [255, 255, 0],  # Road -> Yellow
    5: [0, 255, 255],  # Label 5 -> Cyan
    6: [255, 0, 255],  # Label 6 -> Magenta
    7: [192, 192, 192],  # Label 7 -> Silver
    8: [255, 0, 0],  # Building -> Red
}
satellite_colour_map = transform_color_map(_satellite_colour_map)

_synthetic_colours = {
    0: (70, 130, 180),  # Water
    1: (34, 139, 34),  # Vegetation
    2: (222, 184, 135),  # Farmland
    3: (169, 169, 169),  # Urban/rock
}
synthetic_colours = transform_color_map(_synthetic_colours)


def softmax_to_rgb(
    softmax_map: torch.Tensor, color_mapping: dict[str, list]
) -> np.ndarray:
    """
    Returns:
    - rgb_image (numpy.ndarray): RGB image of shape (H, W, 3).
    """
    _, H, W = softmax_map.shape
    rgb_image = np.zeros((H, W, 3), dtype=np.float32)  # (H, W, 3)

    # Get class indices of the maximum probability for each pixel
    class_indices = softmax_map.argmax(dim=0).cpu().numpy()  # (H, W)

    # Assign RGB colors based on the class indices
    for class_id, color in color_mapping.items():
        rgb_image[class_indices == class_id] = color

    return rgb_image


def rgb_to_input_tensor(
    input_tensor: torch.Tensor, color_mapping: dict[str, list]
) -> torch.Tensor:
    """
    Returns:
        input_onehot_with_mask: (C+1, H, W)
    """
    H, W, _ = input_tensor.shape

    # Extract mask and RGB
    mask = input_tensor[:, :, 3] != 0  # True for masked (unknown) pixels
    known_mask = ~mask

    # Prepare RGB data
    input_rgb_np = input_tensor[:, :, :3].numpy().round(2)
    known_rgb = input_rgb_np[known_mask.numpy()]  # (N, 3)

    # Prepare color array and label lookup
    label_to_color = np.array(list(color_mapping.values()))  # (C, 3)

    # Compute distances between known RGB pixels and color mapping
    dists = np.linalg.norm(
        known_rgb[:, None, :] - label_to_color[None, :, :], axis=2
    )  # (N, C)
    closest_indices = np.argmin(dists, axis=1)  # (N,)

    # Map closest class index to label
    label_indices = np.vectorize(lambda i: list(color_mapping.keys())[i])(
        closest_indices
    )

    # Fill label map
    input_labels = np.zeros((H, W), dtype=np.int64)
    input_labels[known_mask.numpy()] = label_indices

    # Convert to tensor and one-hot encode
    input_labels_tensor = torch.from_numpy(input_labels).long()
    one_hot_input = (
        F.one_hot(input_labels_tensor, num_classes=len(color_mapping))
        .permute(2, 0, 1)
        .float()
    )  # (C, H, W)

    # Prepare mask tensor
    mask_tensor = mask.unsqueeze(0).float()  # (1, H, W)
    uniform_tensor = torch.ones_like(one_hot_input) / len(color_mapping)

    # Blend uniform prior for masked pixels
    one_hot_input = torch.where(mask_tensor.bool(), uniform_tensor, one_hot_input)

    # Combine with mask
    input_with_mask = torch.cat((one_hot_input, mask_tensor), dim=0)  # (C+1, H, W)
    return input_with_mask


def rgb_to_target_tensor(
    target_tensor: torch.Tensor, color_mapping: dict[int, list]
) -> torch.Tensor:
    """
    Returns:
        one_hot_target: (C, H, W) one-hot encoded target tensor.
    """
    H, W, _ = target_tensor.shape
    target_tensor_np = target_tensor.numpy().round(2).reshape(-1, 3)  # (H*W, 3)

    # Build color array and label mapping
    label_to_color = np.array(list(color_mapping.values()))  # (C, 3)

    # Compute distances from each pixel to each class color
    dists = np.linalg.norm(
        target_tensor_np[:, None, :] - label_to_color[None, :, :], axis=2
    )  # (H*W, C)
    closest_indices = np.argmin(dists, axis=1)  # (H*W,)

    # Map closest class index to label
    label_indices = np.vectorize(lambda i: list(color_mapping.keys())[i])(
        closest_indices
    )
    target_labels = label_indices.reshape(H, W)

    # Convert to tensor and one-hot encode
    target_labels_tensor = torch.from_numpy(target_labels).long()
    one_hot_target = (
        F.one_hot(target_labels_tensor, num_classes=len(color_mapping))
        .permute(2, 0, 1)
        .float()
    )

    return one_hot_target


def scale_image(
    image: Image, scale_down: float | None = None, size: tuple[int, int] | None = None
) -> torch.Tensor:
    if scale_down and size:
        raise ValueError("Only one of scale_down or size can be provided.")
    elif scale_down:
        w, h = image.size
        image = image.resize(
            (int(w / scale_down), int(h / scale_down)), resample=Image.NEAREST
        )
    elif size:
        image = image.resize(size, resample=Image.NEAREST)
    else:
        raise ValueError("Either scale_down or size must be provided.")
    return image


def split_image(image: np.ndarray, n_splits: int) -> list[np.ndarray]:
    """Splits the image into n_splits x n_splits smaller images."""
    height, width, _ = image.shape
    split_height = height // n_splits
    split_width = width // n_splits

    return [
        image[
            i * split_height : (i + 1) * split_height,
            j * split_width : (j + 1) * split_width,
        ]
        for i in range(n_splits)
        for j in range(n_splits)
    ]


def load_openearth_seg_image(
    img_path: str,
    scale_down: int | None = None,
    size: tuple[int, int] = None,
    color_map: dict[int, list[int]] = satellite_colour_map,
):
    image = Image.open(img_path)
    image = image.convert("L")

    if size or scale_down:
        image = scale_image(image, scale_down, size)

    image_tensor_label = torch.tensor(np.array(image), dtype=torch.uint8)
    rgb_image = np.full(
        (image_tensor_label.shape[0], image_tensor_label.shape[1], 3),
        [128, 0, 128],
        dtype=np.float32,
    )
    for label, color in color_map.items():
        rgb_image[image_tensor_label == label] = color
    return rgb_image


def load_rgb_image(
    img_path: str, scale_down: int | None = None, size: tuple[int, int] = None
):
    image = Image.open(img_path)
    image = image.convert("RGB")
    if size or scale_down:
        image = scale_image(image, scale_down, size)
    return torch.tensor(np.array(image), dtype=torch.float32) / 255


def gaussian_random_field(shape, scale):
    noise = np.random.randn(*shape)
    kfreq = (
        np.fft.fftfreq(shape[0])[:, None] ** 2 + np.fft.fftfreq(shape[1])[None, :] ** 2
    )
    kfreq[0, 0] = 1e-6
    spectrum = np.exp(-kfreq * (scale**2))
    field = np.fft.ifft2(np.fft.fft2(noise) * np.sqrt(spectrum)).real
    return (field - field.min()) / (field.max() - field.min())


def roughen_boundaries(mask, sigma=1.0, noise_scale=0.15):
    out = mask.copy()
    for k in np.unique(mask):
        region = (mask == k).astype(float)
        blurred = ndi.gaussian_filter(region, sigma=sigma)
        out[(blurred + np.random.normal(0, noise_scale, mask.shape)) > 0.5] = k
    return out


def generate_archipelago(size, num_classes):
    base = gaussian_random_field((size, size), scale=size // 6)
    land = base > 0.5
    labels = np.zeros((size, size), dtype=int)
    labels[land] = np.random.randint(1, num_classes, size=(land.sum()))
    return roughen_boundaries(labels, sigma=2)


def generate_river_delta(size, num_classes):
    # Start with all land (random land classes)
    labels = np.random.randint(1, num_classes, size=(size, size))

    # Create water mask
    water_mask = np.zeros((size, size), dtype=bool)

    # Main river starting from top
    x = size // 2 + np.random.randint(-10, 11)
    y = 0

    # Draw main river
    for step in range(size):
        if 0 <= x < size and y < size:
            # Make river wider as it approaches the delta
            width = max(1, step // 15)
            for dx in range(-width, width + 1):
                if 0 <= x + dx < size:
                    water_mask[x + dx, y] = True

        # River meanders slightly
        x += np.random.choice([-1, 0, 1], p=[0.2, 0.6, 0.2])
        y += 1

        # Add branching in the delta region (bottom 40% of map)
        if y > 0.6 * size and np.random.rand() < 0.3:
            # Create a branch
            branch_x, branch_y = x, y
            max_branch_length = max(5, size - y)
            branch_length = np.random.randint(3, max_branch_length + 1)

            for _ in range(branch_length):
                if 0 <= branch_x < size and branch_y < size:
                    water_mask[branch_x, branch_y] = True

                # Branch direction (slightly random)
                branch_x += np.random.choice([-1, 0, 1], p=[0.3, 0.4, 0.3])
                branch_y += 1

                if branch_y >= size:
                    break

    # Add some smaller distributary channels in delta region
    for _ in range(5):
        start_x = np.random.randint(0, size)
        start_y = int(0.7 * size)

        for step in range(size - start_y):
            curr_y = start_y + step
            if curr_y < size and 0 <= start_x < size:
                water_mask[start_x, curr_y] = True

            start_x += np.random.choice([-1, 0, 1])
            if start_x < 0 or start_x >= size:
                break

    # Dilate water channels slightly to make them more visible
    water_mask = ndi.binary_dilation(water_mask, iterations=1)

    # Apply water to labels
    labels[water_mask] = 0

    # Add land use patterns near water
    # Vegetation near water (within 2-3 pixels)
    vegetation_zone = ndi.binary_dilation(water_mask, iterations=3) & ~water_mask
    labels[vegetation_zone & (np.random.rand(*labels.shape) < 0.7)] = 1

    # Farmland in a band around vegetation
    farmland_zone = (
        ndi.binary_dilation(vegetation_zone, iterations=4)
        & ~vegetation_zone
        & ~water_mask
    )
    labels[farmland_zone & (np.random.rand(*labels.shape) < 0.8)] = 2

    # Roughen boundaries for more realistic appearance
    labels = roughen_boundaries(labels, sigma=1.2)

    return labels


def generate_mosaic_farmland(size, num_classes):
    seeds = np.random.randint(0, size, (25, 2))
    yy, xx = np.meshgrid(np.arange(size), np.arange(size))
    dists = [np.sqrt((xx - s[0]) ** 2 + (yy - s[1]) ** 2) for s in seeds]
    closest = np.argmin(dists, axis=0)
    labels = closest % num_classes
    # Add some river influence
    river_mask = gaussian_random_field((size, size), scale=size // 8) > 0.85
    labels[river_mask] = 0
    # Make farmland adjacent to river
    farmland_mask = ndi.binary_dilation(river_mask, iterations=3)
    labels[farmland_mask & (labels != 0)] = 2
    return roughen_boundaries(labels, sigma=0.8)


def generate_synthetic_satellite_image(
    city_type: str,
    colour_map: dict[str, list[int]],
    config: SyntheticMapConfig = SyntheticMapConfig(),
):
    # --- Map Generation ---
    size = config.size
    if city_type == "archipelago":
        class_map = generate_archipelago(size, num_classes=len(colour_map))
    elif city_type == "river":
        class_map = generate_river_delta(size, num_classes=len(colour_map))
    elif city_type == "farmland":
        class_map = generate_mosaic_farmland(size, num_classes=len(colour_map))
    else:
        raise ValueError(f"Unknown map kind: {city_type}")

    # --- RGB Conversion ---
    rgb_image = np.zeros((size, size, 3), dtype=np.float32)
    for class_id, color in colour_map.items():
        rgb_image[class_map == class_id] = color

    return rgb_image
