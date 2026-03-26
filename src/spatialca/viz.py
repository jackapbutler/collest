import matplotlib.pyplot as plt
import numpy as np
import json


def plot_losses(
    loss_files: dict[str, str],
    place_trained: str,
    zoomed_xlim: int = 2000,
    zoomed_ylims: tuple[int, int] = (0, 0.5),
):
    plt.figure(figsize=(30, 6))
    for l_file in loss_files:
        with open(f"./models/{l_file}", "r") as f:
            train_losses_json = json.load(f)

        losses_array = np.array(train_losses_json["loss"])
        mean_losses = losses_array.mean(axis=1)
        l_name = l_file.split("losses_")[1].split(".")[0]
        l_name = l_name.replace(place_trained, "")
        l_name = l_name.replace("-", " ")

        plt.plot(mean_losses, "-o", label=l_name)
        plt.fill_between(
            range(len(mean_losses)),
            losses_array.min(axis=1),
            losses_array.max(axis=1),
            alpha=0.3,
        )
    plt.xlabel("# Steps")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    plt.figure(figsize=(30, 6))
    for l_file in loss_files:
        with open(f"./models/{l_file}", "r") as f:
            train_losses_json = json.load(f)

        losses_array = np.array(train_losses_json["loss"])
        mean_losses = losses_array.mean(axis=1)

        l_name = l_file.split("losses_")[1].split(".")[0]
        l_name = l_name.replace(f"-synth-{place_trained}", "")
        l_name = l_name.replace("-", " ")

        plt.plot(mean_losses, "-o", label=l_name)
        plt.fill_between(
            range(len(mean_losses)),
            losses_array.min(axis=1),
            losses_array.max(axis=1),
            alpha=0.3,
        )
    plt.ylim(*zoomed_ylims)
    plt.xlim(zoomed_xlim, len(mean_losses) - 1)
    plt.xlabel("# Steps")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
