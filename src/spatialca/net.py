import torch
import torch.nn as nn
import torch.nn.functional as F


class Base(nn.Module):
    def __init__(self):
        super().__init__()

    def save(self, path: str):
        torch.save(self, path)

    @classmethod
    def load(cls, path: str, folder: str = "./models") -> "Base":
        return torch.load(f"{folder}/{path}", weights_only=False)

    def init_state(
        self,
        logits: torch.Tensor,
        unseen_mask: torch.Tensor,
    ) -> torch.Tensor:
        # --- Initialize the full state 'x' for the first run ---
        # Add the observed_mask channel. This is static throughout evolution.
        B, _, H, W = logits.shape
        x = torch.cat([logits, unseen_mask], dim=1)  # [B, C + 1, H, W]
        # Add hidden state channels, initialized to zeros for a fresh start
        if hasattr(self, "num_hidden_channels"):
            hidden_init = torch.zeros(B, self.num_hidden_channels, H, W)
            x = torch.cat([x, hidden_init], dim=1)  # [B, C + 1 + H, H, W]
        return x


class NeuralCA(Base):
    def __init__(
        self,
        num_classes: int,
        perception_channels: int,
        num_hidden_channels: int,
        kernel_size: int,
        full_steps: int,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.perception_channels = perception_channels
        self.num_hidden_channels = num_hidden_channels
        self.state_dim = num_classes + 1 + num_hidden_channels
        self.full_steps = full_steps

        # Perception and update steps
        hidden_dim = 2 * perception_channels
        self.perception = nn.Sequential(
            nn.Conv2d(
                self.state_dim,
                perception_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            ),
            nn.GroupNorm(1, perception_channels),
        )
        self.update = nn.Sequential(
            nn.Conv2d(perception_channels, hidden_dim, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim, self.state_dim, kernel_size=1),
        )
        nn.init.zeros_(self.update[2].weight)
        self.readout = nn.Conv2d(self.state_dim, num_classes, kernel_size=1)
        self.softmax = nn.Softmax(dim=1)

    def forward(
        self,
        logits: torch.Tensor | None = None,  # [B, C, H, W] - partial observed
        unseen_mask: torch.Tensor | None = None,  # [B, 1, H, W] - 1 unseen, 0 seen
        steps: int | None = None,  # Number of CA steps to run
        current_state: torch.Tensor | None = None,  # [B, self.state_dim, H, W]
        **kwargs,
    ):
        # Decide number of steps to run, and initial state
        steps = steps if steps is not None else self.full_steps
        if current_state is None:
            current_state = self.init_state(logits, unseen_mask)

        # Run the CA update for `steps` iterations
        x = current_state
        for _ in range(steps):
            dx = self.perception(x)  # Extract local features
            dx = self.update(dx)  # Apply transformation (CA update)
            x = x + dx  # Update state

        if kwargs.get("return_state"):
            return x
        else:
            x = self.readout(x)
            if kwargs.get("return_probs"):
                x = self.softmax(x)
            return x


class DoubleConv(nn.Module):
    """(Conv -> GN -> ReLU) * 2"""

    def __init__(self, in_channels, out_channels, groups=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(Base):
    def __init__(
        self,
        num_classes: int,
        base_channels: int = 16,
        num_blocks: int = 2,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.depth = num_blocks
        groups = 4

        # Encoder
        self.encoders = nn.ModuleList()
        prev_ch = num_classes + 1
        ch = base_channels
        for _ in range(num_blocks):
            self.encoders.append(DoubleConv(prev_ch, ch, groups))
            prev_ch = ch
            ch *= 2

        # Bottleneck
        self.bottleneck = DoubleConv(prev_ch, ch, groups)

        # Decoder
        self.upsamples = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for _ in range(num_blocks):
            self.upsamples.append(
                nn.ConvTranspose2d(ch, ch // 2, kernel_size=2, stride=2)
            )
            self.decoders.append(DoubleConv(ch, ch // 2, groups))
            ch //= 2

        # Output projection
        self.output_proj = nn.Conv2d(ch, num_classes, kernel_size=1)

    def forward(
        self,
        logits: torch.Tensor,  # [B, num_classes, H, W] - partially observed
        unseen_mask: torch.Tensor,  # [B, 1, H, W] - 1 for unseen, 0 for known
        **kwargs,
    ):
        x = torch.cat([logits, unseen_mask], dim=1)

        # Encoder
        skips = []
        for enc in self.encoders:
            x = enc(x)
            skips.append(x)
            x = F.max_pool2d(x, 2)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        for up, dec, skip in zip(self.upsamples, self.decoders, reversed(skips)):
            x = up(x)
            # Handle size mismatch
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(
                    x, size=skip.shape[-2:], mode="bilinear", align_corners=False
                )
            x = torch.cat([x, skip], dim=1)
            x = dec(x)

        out = self.output_proj(x)
        if kwargs.get("return_probs"):
            out = F.softmax(out, dim=1)
        return out
