import torch
import random
from loguru import logger
from .map import rgb_to_input_tensor
from .noise import add_random_cells
import numpy as np
import math
import collections


def initialise_new_map(true_map):
    return add_random_cells(true_map, 1.0)


def init_agents(size_x, size_y, num_agents):
    return [
        {
            "pos": (
                random.randint(0, size_x - 1),
                random.randint(0, size_y - 1),
            ),
        }
        for _ in range(num_agents)
    ]


class MultiAgentMapper:
    def __init__(
        self,
        raw_map: torch.Tensor,
        num_agents: int,
        movement_style: str,
        colour_map: dict[int, tuple[int, int, int]],
        agent_view_radius: int = 2,
        agent_attrs: list[dict[tuple[int, int]]] = None,
        valid_moves: list[tuple[int, int]] = None,
    ):
        self.valid_moves = valid_moves
        if not self.valid_moves:
            self.valid_moves = [
                (0, 1),
                (0, -1),
                (1, 0),
                (-1, 0),
                (1, 1),
                (1, -1),
                (-1, 1),
                (-1, -1),
            ]
        self.colour_map = colour_map
        self.map = initialise_new_map(raw_map)
        self.size_x, self.size_y = self.map.shape[0], self.map.shape[1]
        self.raw_map = raw_map
        self.true_map = add_random_cells(raw_map, 0)
        self.prob_true_map = rgb_to_input_tensor(self.true_map, colour_map)[
            : len(colour_map)
        ]

        self.num_agents = num_agents
        self.agents = agent_attrs or init_agents(self.size_x, self.size_y, num_agents)
        self.movement_style = movement_style
        self.view_radius = agent_view_radius

        self.history_maps = []
        self.history_moves = []
        self.history_refineds = collections.defaultdict(list)
        self.history_sigmas = collections.defaultdict(list)

        self.frontier_cells: list[tuple[int, int]] | None = None
        self.frontier_styles = ["frontier", "extrapolator", "error"]
        self.max_dist = math.hypot(self.size_x, self.size_y)
        if "extrapolator" in self.movement_style:
            self.sigma = torch.zeros(self.size_x, self.size_y)
        if "error" == self.movement_style:
            self.cell_mse = torch.zeros(self.size_x, self.size_y)

        # Pre-compute view offsets for get_observation
        self._view_offsets = torch.tensor(
            [
                (dx, dy)
                for dx in range(-self.view_radius, self.view_radius + 1)
                for dy in range(-self.view_radius, self.view_radius + 1)
            ],
            dtype=torch.long,
        )

    def total_unseen_cells(self, x, y) -> int:
        low_x, high_x = (
            max(0, x - self.view_radius),
            min(self.size_x, x + self.view_radius + 1),
        )
        low_y, high_y = (
            max(0, y - self.view_radius),
            min(self.size_y, y + self.view_radius + 1),
        )
        return int(torch.sum(self.map[low_x:high_x, low_y:high_y, 3] == 1))

    def find_frontier_cells(self) -> list[tuple[int, int]]:
        """Find unseen cells adjacent to at least one seen cell (the true frontier)."""
        seen_mask = (self.map[:, :, 3] == 0).numpy()
        unseen_mask = (self.map[:, :, 3] == 1).numpy()

        frontiers = set()
        unseen_coords = np.where(unseen_mask)

        for i in range(len(unseen_coords[0])):
            x, y = unseen_coords[0][i], unseen_coords[1][i]
            for dx, dy in self.valid_moves:
                nx, ny = x + dx, y + dy
                if (
                    0 <= nx < self.size_x
                    and 0 <= ny < self.size_y
                    and seen_mask[nx, ny]
                ):
                    frontiers.add((x, y))
                    break  # Only need one seen neighbor

        return list(frontiers)

    def move_greedy(self, curr_x: int, curr_y: int) -> tuple[int, int]:
        unseen_count = 0
        for dx, dy in self.valid_moves:
            new_x, new_y = (
                max(0, min(self.size_x - 1, curr_x + dx)),
                max(0, min(self.size_y - 1, curr_y + dy)),
            )
            new_unseen = self.total_unseen_cells(new_x, new_y)
            if unseen_count <= new_unseen:
                new_pos = (new_x, new_y)
                unseen_count = new_unseen
        return new_pos

    def move_random(self, curr_x: int, curr_y: int) -> tuple[int, int]:
        dx, dy = random.choice(self.valid_moves)
        new_x, new_y = (
            max(0, min(self.size_x - 1, curr_x + dx)),
            max(0, min(self.size_y - 1, curr_y + dy)),
        )
        return (new_x, new_y)

    def fov_weighting_wrapper(
        self, candidate: tuple[int, int], curr_pos: tuple[int, int]
    ) -> float:
        """Vectorized computation of FOV weighting for a candidate cell."""
        candidate = torch.tensor(candidate, dtype=torch.long)
        curr_pos = torch.tensor(curr_pos, dtype=torch.long)

        # Generate all FOV positions around candidate
        fov_positions = candidate + self._view_offsets  # shape (fov_size, 2)
        fov_positions[:, 0] = fov_positions[:, 0].clamp(0, self.size_x - 1)
        fov_positions[:, 1] = fov_positions[:, 1].clamp(0, self.size_y - 1)

        # Compute distances to current agent
        dx = fov_positions[:, 0] - curr_pos[0]
        dy = fov_positions[:, 1] - curr_pos[1]
        distances = torch.sqrt(dx.float() ** 2 + dy.float() ** 2) / self.max_dist

        # Compute weighting based on movement style
        fov_scores = distances
        if self.movement_style == "extrapolator":
            discount = self.sigma[fov_positions[:, 0], fov_positions[:, 1]]
        elif self.movement_style == "error":
            discount = self.cell_mse[fov_positions[:, 0], fov_positions[:, 1]]
        else:
            discount = 1.0
            if self.movement_style != "frontier":
                raise ValueError(f"Unsupported movement_style {self.movement_style}")

        # Average over all valid FOV cells
        return (fov_scores / discount).mean().item()

    def move_frontier(self, curr_x, curr_y):
        target = (curr_x, curr_y)
        if self.frontier_cells:
            # Select the frontier cell with best FOV weighting
            target = min(
                self.frontier_cells,
                key=lambda f: self.fov_weighting_wrapper(f, (curr_x, curr_y)),
            )
            self.frontier_cells.remove(target)  # ensure unique assignment

        # Move one step toward target
        dx = 1 if target[0] > curr_x else -1 if target[0] < curr_x else 0
        dy = 1 if target[1] > curr_y else -1 if target[1] < curr_y else 0
        new_x = max(0, min(self.size_x - 1, curr_x + dx))
        new_y = max(0, min(self.size_y - 1, curr_y + dy))
        return (new_x, new_y)

    def get_new_agent_pos(self, agent: dict):
        x, y = agent["pos"]
        new_pos = None
        random.shuffle(self.valid_moves)

        if self.movement_style == "greedy":
            new_pos = self.move_greedy(x, y)

        elif self.movement_style in self.frontier_styles:
            new_pos = self.move_frontier(x, y)

        if not new_pos:
            # default to random movement or if all seen cells
            if self.movement_style != "random":
                logger.warning("Random movement default initiated")
            new_pos = self.move_random(x, y)

        return new_pos

    def get_observation(
        self, agent: dict
    ) -> list[tuple[tuple[int, int], torch.Tensor]]:
        agent_view = []
        x, y = agent["pos"]
        for dx, dy in self._view_offsets:
            new_x, new_y = x + dx, y + dy
            if 0 <= new_x < self.size_x and 0 <= new_y < self.size_y:
                agent_view.append(((new_x, new_y), self.true_map[new_x, new_y, :]))
        return agent_view

    def update_map(self, observations):
        # Deduplicate observations to avoid redundant updates
        obs_dict = {}
        for (x, y), obs in observations:
            obs_dict[(x, y)] = obs
        for (x, y), obs in obs_dict.items():
            self.map[x, y] = obs

    def simulate(self, extrapolators: dict[str, torch.nn.Module], steps: int):
        for _ in range(steps):
            observations = []
            if self.movement_style in self.frontier_styles:
                self.frontier_cells = self.find_frontier_cells()

            for agent in self.agents:
                agent["pos"] = self.get_new_agent_pos(agent)
                agent_obs = self.get_observation(agent)
                observations.extend(agent_obs)
            self.update_map(observations)

            # Store raw map before refinement
            self.history_maps.append(self.map.detach().clone())
            self.history_moves.append([agent["pos"] for agent in self.agents])

            # Apply model for map refinement
            for name, model in extrapolators.items():
                _mask = self.map[:, :, 3]
                seen, unseen = _mask == 0, _mask == 1
                input_oh = rgb_to_input_tensor(self.map, self.colour_map)
                samples = input_oh.unsqueeze(0)
                n_cls = model.num_classes

                # Fresh forward pass
                with torch.no_grad():
                    probs = model(
                        logits=samples[:, :n_cls],
                        unseen_mask=samples[:, n_cls : n_cls + 1],
                        return_state=False,
                        return_probs=True,
                    )[0]

                # Per-cell MSE + normalisation
                self.cell_mse = torch.mean((probs - self.prob_true_map) ** 2, dim=0)
                error_min, error_max = self.cell_mse.min(), self.cell_mse.max()
                self.cell_mse = (self.cell_mse - error_min) / (error_max - error_min)
                self.cell_mse[seen] = self.cell_mse.min()
                self.cell_mse = 1 - self.cell_mse

                # Per-cell uncertainty normalisation
                self.sigma = -torch.sum(probs * torch.log(probs + 1e-9), dim=0)
                e_min, e_max = self.sigma.min(), self.sigma.max()
                self.sigma = (self.sigma - e_min) / (e_max - e_min)
                self.sigma = self.sigma.clamp(1e-6, 1.0)
                self.sigma[seen] = self.sigma.min()

                # Create final normalised predictions
                probs_unmasked = input_oh[:n_cls]
                probs_unmasked *= seen[None, :, :]
                probs_unmasked += probs[:n_cls] * unseen[None, :, :]
                logits_with_mask = torch.cat(
                    [probs_unmasked, _mask.unsqueeze(0)], dim=0
                )

                self.history_refineds[name].append(logits_with_mask.detach().half())
                self.history_sigmas[name].append(self.sigma.detach().half())

    def collect_snapshot_data(
        self, rollouts: int = 10, steps: int = 10, interval: int = 1, min_steps: int = 1
    ):
        """Collect map snapshots at different time intervals across multiple rollouts"""
        snapshots = []

        for _ in range(rollouts):
            # Reset map and agents for this rollout
            self.map = initialise_new_map(self.raw_map)
            self.agents = init_agents(self.size_x, self.size_y, self.num_agents)

            for step in range(1, steps + 1):
                # Move agents and collect observations
                observations = []
                if self.movement_style in ["frontier", "extrapolator"]:
                    self.frontier_cells = self.find_frontier_cells()
                for agent in self.agents:
                    agent["pos"] = self.get_new_agent_pos(agent)
                    agent_obs = self.get_observation(agent)
                    observations.extend(agent_obs)

                # Update the map with new observations
                self.update_map(observations)

                # Take a snapshot at the specified interval
                if step and step != steps and min_steps < step and step % interval == 0:
                    snapshots.append((self.map.clone(), self.true_map))

        return snapshots
