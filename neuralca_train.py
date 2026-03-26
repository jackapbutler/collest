from typing import Callable
import os
import json
import torch
import time
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from spatialca import net, map


class MapTrainDataset(Dataset):
    def __init__(self, raw_data: list, colour_map: dict, num_classes: int):
        self.data = raw_data
        self.colour_map = colour_map

        # Precompute inputs/targets once
        self.preprocessed = []
        for sample, target in self.data:
            input_oh = map.rgb_to_input_tensor(sample, self.colour_map)
            target_oh = map.rgb_to_target_tensor(target, self.colour_map)
            self.preprocessed.append(
                (
                    input_oh[:num_classes],  # logits input tensor
                    input_oh[num_classes : num_classes + 1],  # unseen mask
                    target_oh,  # target one-hot tensor
                    target_oh.argmax(dim=0),  # target class indices
                )
            )

    def __len__(self):
        return len(self.preprocessed)

    def __getitem__(self, idx):
        return self.preprocessed[idx]


# OpenEarth data
COLOUR_MAP = map.satellite_colour_map
datasets_to_load = ["openearth_map_train_data_rio"]
train_steps = 2_000
lr = 1e-3
batch_size = 16
log_interval = int(train_steps / 20)
p_channels = 24
unet_p_channels = p_channels - 12
unet_blocks = 3
kernel_size = 3
num_classes = len(COLOUR_MAP)
h_channel = 24
nca_steps = 36

# ## Synthetic data
# COLOUR_MAP = map.synthetic_colours
# datasets_to_load = ["map_train_data_synth_all"]
# train_steps = 2_000
# lr = 1e-3
# batch_size = 28
# log_interval = int(train_steps / 20)
# p_channels = 12
# unet_p_channels = p_channels - 4
# unet_blocks = 2
# kernel_size = 3
# num_classes = len(COLOUR_MAP)
# h_channel = 12
# nca_steps = 18

models = {
    "UNet": lambda: net.UNet(
        num_classes=num_classes,
        base_channels=unet_p_channels,
        num_blocks=unet_blocks,
    ),
    "NeuralCA": lambda: net.NeuralCA(
        num_classes=num_classes,
        perception_channels=p_channels,
        num_hidden_channels=h_channel,
        kernel_size=kernel_size,
        full_steps=nca_steps,
    ),
    "CNN": lambda: net.NeuralCA(
        num_classes=num_classes,
        perception_channels=p_channels,
        num_hidden_channels=h_channel,
        kernel_size=kernel_size,
        full_steps=1,
    ),
    # **{
    #     f"NeuralCA-t={t}": lambda: net.NeuralCA(
    #         num_classes=num_classes,
    #         perception_channels=p_channels,
    #         num_hidden_channels=h_channel,
    #         kernel_size=kernel_size,
    #         full_steps=t,
    #     )
    #     for t in [2, 5, 9, 13, 24, 32]
    # },
}


def pixel_loss(
    logits: torch.Tensor, mask: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    # Compute loss only on unseen pixels
    ## Shape: (B, H, W)
    loss = F.cross_entropy(logits, targets, reduction="none")
    return loss * mask


def load_data(places_to_train: list[str]) -> DataLoader:
    # Load raw data once
    raw_train_data = []
    for data_source in places_to_train:
        with open(f"./data/{data_source}.json", "r") as f:
            place_data = [
                (torch.tensor(sample[0]), torch.tensor(sample[1]))
                for sample in json.load(f)
            ]
        raw_train_data.extend(place_data)

    print(len(raw_train_data), "training samples loaded")
    train_dataset = MapTrainDataset(raw_train_data, COLOUR_MAP, num_classes)
    return DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )


def main(models: dict[str, Callable[[], torch.nn.Module]]):
    """Train models on synthetic data."""
    train_loader = load_data(datasets_to_load)
    place_to_train = "_".join(datasets_to_load)
    models = {f"{name}-{place_to_train}": model for name, model in models.items()}
    for model_name, model_init in models.items():
        n_params = sum(p.numel() for p in model_init().parameters())
        print(f"{model_name}: {n_params} params")

    for model_name, model_init in models.items():
        save_path = f"./models/best_chkpt_{model_name}.pth"
        if os.path.exists(save_path):
            continue

        model = model_init()
        try:
            model = torch.compile(model)
        except Exception as e:
            print(
                f"torch.compile failed for {model_name}: {e}. Proceeding without compilation."
            )
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # Training loop
        n_params = sum(p.numel() for p in model.parameters())
        print(f"| {time.strftime('%H:%M:%S')} | {model_name}: {n_params} params")

        train_losses = []
        data_iter = iter(train_loader)
        for step in range(train_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                # Restart the iterator when we run out of data
                data_iter = iter(train_loader)
                batch = next(data_iter)
            optimizer.zero_grad()
            logits, unseen_mask, _, targets = batch

            # Regular fresh forward pass: (B, C, H, W)
            out_logits = model(
                logits=logits,
                unseen_mask=unseen_mask,
                return_state=False,
                return_probs=False,
            )
            _mask = unseen_mask[:, 0]  # Mask without C
            masked_loss = pixel_loss(out_logits, _mask, targets)  # (B, H, W)
            tr_loss = masked_loss.sum() / _mask.sum()
            sample_losses = masked_loss.sum(dim=(1, 2)) / _mask.sum(dim=(1, 2))

            if step % log_interval == 0:
                _time = time.strftime("%H:%M:%S")
                invisible_loss_mean = masked_loss.sum() / _mask.sum()
                loss_, min_loss, max_loss = (
                    round(invisible_loss_mean.item(), 4),
                    round(masked_loss.min().item(), 4),
                    round(masked_loss.max().item(), 4),
                )
                print(
                    f"| {_time} | S{step + 1} | Loss: {loss_} [{min_loss}, {max_loss}] |"
                )

            # Backward + optimize
            tr_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_losses.append(sample_losses.detach().cpu().numpy().tolist())
            if step == train_steps - 1:
                # Save model and losses
                with open(f"./models/losses_{model_name}.json", "w") as f:
                    json.dump({"loss": train_losses}, f)
                model.save(save_path)


if __name__ == "__main__":
    main(models)
