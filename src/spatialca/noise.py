import random
import math
import torch
import numpy as np
from .map import synthetic_colours


def _init_image_and_mask(
    image: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    """Initialize working copy of image and create mask."""
    image = image.clone()
    height, width, _ = image.shape
    mask = torch.zeros((height, width), dtype=torch.float32)
    return image, mask, height, width


def _calculate_shape_count(
    height: int, width: int, noise_rate: float, avg_shape_area: float
) -> int:
    """Calculate number of shapes needed based on target area."""
    target_area = noise_rate * height * width
    return int(target_area / avg_shape_area)


def _finalize_image(image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Add mask as additional channel to image."""
    return torch.cat((image, mask.unsqueeze(2)), dim=2)


def add_random_cells(image: torch.Tensor, noise_rate: float = 0.1):
    """Adds randomly placed, entirely black cells to an image."""
    image, mask, height, width = _init_image_and_mask(image)
    n_black_cells = _calculate_shape_count(height, width, noise_rate, 1)
    black_cells = set()

    while len(black_cells) < n_black_cells:
        x = random.randint(0, width - 1)
        y = random.randint(0, height - 1)
        if (x, y) not in black_cells:
            black_cells.add((x, y))
            image[y, x, :] = 0.0
            mask[y, x] = 1.0

    return _finalize_image(image, mask)


def add_random_circles(
    image: torch.Tensor,
    noise_rate: float = 0.1,
    min_radius: int = 2,
    max_radius: int = 5,
):
    """Adds randomly placed black circles to an image."""
    image, mask, height, width = _init_image_and_mask(image)
    avg_radius = (min_radius + max_radius) / 2
    n_circles = _calculate_shape_count(
        height, width, noise_rate, math.pi * avg_radius**2
    )

    for _ in range(n_circles):
        radius = random.randint(min_radius, max_radius)
        center_x = random.randint(radius, width - radius)
        center_y = random.randint(radius, height - radius)

        y_indices, x_indices = torch.meshgrid(
            torch.arange(max(0, center_y - radius), min(height, center_y + radius + 1)),
            torch.arange(max(0, center_x - radius), min(width, center_x + radius + 1)),
            indexing="ij",
        )

        distances = torch.sqrt(
            (x_indices - center_x) ** 2 + (y_indices - center_y) ** 2
        )
        circle_mask = distances <= radius

        image[y_indices[circle_mask], x_indices[circle_mask], :] = 0.0
        mask[y_indices[circle_mask], x_indices[circle_mask]] = 1.0

    return _finalize_image(image, mask)


def add_random_squares(
    image: torch.Tensor,
    noise_rate: float = 0.1,
    min_size: int = 2,
    max_size: int = 5,
    replace_color: torch.Tensor = torch.tensor([0, 0, 0], dtype=torch.float32),
):
    """Adds randomly placed black squares to an image."""
    if any([1 < x for x in replace_color]):
        replace_color = replace_color / 255.0

    image, mask, height, width = _init_image_and_mask(image)
    avg_size = (min_size + max_size) / 2
    n_squares = _calculate_shape_count(height, width, noise_rate, avg_size * avg_size)

    for _ in range(n_squares):
        sq_size = random.randint(min_size, max_size)
        max_x = width - sq_size
        max_y = height - sq_size

        if max_x <= 0 or max_y <= 0:
            continue

        top_left_x = random.randint(0, max_x)
        top_left_y = random.randint(0, max_y)

        y_indices = slice(top_left_y, top_left_y + sq_size)
        x_indices = slice(top_left_x, top_left_x + sq_size)

        image[y_indices, x_indices, :] = replace_color
        mask[y_indices, x_indices] = 1.0

    return _finalize_image(image, mask)


def add_random_lines(
    image: torch.Tensor,
    noise_rate: float = 0.1,
    min_length: int = 10,
    max_length: int = 30,
    min_thickness: int = 1,
    max_thickness: int = 3,
):
    """Adds randomly placed horizontal and vertical black lines to an image."""
    image, mask, height, width = _init_image_and_mask(image)
    avg_length = (min_length + max_length) / 2
    avg_thickness = (min_thickness + max_thickness) / 2
    n_lines = _calculate_shape_count(
        height, width, noise_rate, avg_length * avg_thickness
    )

    for _ in range(n_lines):
        is_horizontal = random.random() < 0.5
        length = random.randint(min_length, max_length)
        thickness = random.randint(min_thickness, max_thickness)

        if is_horizontal:
            max_y, max_x = height - thickness, width - length
            if max_y <= 0 or max_x <= 0:
                continue
            start_y = random.randint(0, max_y)
            start_x = random.randint(0, max_x)
            y_indices = slice(start_y, start_y + thickness)
            x_indices = slice(start_x, start_x + length)
        else:
            max_y, max_x = height - length, width - thickness
            if max_y <= 0 or max_x <= 0:
                continue
            start_y = random.randint(0, max_y)
            start_x = random.randint(0, max_x)
            y_indices = slice(start_y, start_y + length)
            x_indices = slice(start_x, start_x + thickness)

        image[y_indices, x_indices, :] = 0.0
        mask[y_indices, x_indices] = 1.0

    return _finalize_image(image, mask)


def add_building_clusters(
    class_map: np.ndarray, num_clusters: int = 18, max_size: int = 9
):
    H, W, _ = class_map.shape
    for _ in range(num_clusters):
        cx = random.randint(0, W - 1)
        cy = random.randint(0, H - 1)
        size = random.randint(5, max_size)
        class_map[cy : cy + size, cx : cx + size] = synthetic_colours[3]  # Urban class
    return class_map


def add_floods(class_map: np.ndarray, num_patches: int = 6, max_size: int = 12):
    H, W, _ = class_map.shape
    for _ in range(num_patches):
        radius = random.randint(8, max_size)
        center_x = random.randint(radius, W - radius)
        center_y = random.randint(radius, H - radius)
        y_indices, x_indices = torch.meshgrid(
            torch.arange(max(0, center_y - radius), min(H, center_y + radius + 1)),
            torch.arange(max(0, center_x - radius), min(W, center_x + radius + 1)),
            indexing="ij",
        )
        distances = torch.sqrt(
            (x_indices - center_x) ** 2 + (y_indices - center_y) ** 2
        )
        circle_mask = distances <= radius
        class_map[y_indices[circle_mask], x_indices[circle_mask], :] = (
            synthetic_colours[0]
        )
    return class_map
