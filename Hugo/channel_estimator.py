# channel_estimator.py
import numpy as np

from signal_types import PilotObservations, ChannelEstimate


class ChannelEstimator:
    def estimate(self, pilots: PilotObservations, eps: float = 1e-12) -> ChannelEstimate:
        y = pilots.y_pilots
        x = pilots.x_pilots

        if y.shape != x.shape:
            raise ValueError("y_pilots y x_pilots deben tener la misma forma.")

        h = y / (x + eps)

        metadata = dict(pilots.metadata)
        metadata["estimation"] = "direct_division_ls"
        return ChannelEstimate(
            h_pilots=h,
            pilot_indices=pilots.pilot_indices,
            symbol_indices=pilots.symbol_indices,
            metadata=metadata,
        )