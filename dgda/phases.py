"""Timing phases for the 500-window dynamic graph simulation."""

from dataclasses import dataclass


@dataclass(frozen=True)
class TimingPhase:
    """Describe one inclusive time range in the simulation.

    The dynamic graph is split into human-readable phases so detection
    algorithms know which windows are normal training data, which windows are
    safe validation data, which windows may contain an attack, and which windows
    represent recovery after the attack interval.
    """

    name: str
    start_window: int
    end_window: int
    description: str
    normal_traffic_only: bool
    attack_injection_allowed: bool = False

    def contains(self, window: int) -> bool:
        """Return True when ``window`` falls inside this phase.

        This small helper keeps phase checks readable in the rest of the code.
        The comparison is inclusive because window 1 and window 500 are both
        real snapshots in the simulation.
        """
        return self.start_window <= window <= self.end_window


TIMING_PHASES = (
    TimingPhase(
        name="baseline",
        start_window=1,
        end_window=150,
        description="Training baseline: pure normal traffic only.",
        normal_traffic_only=True,
    ),
    TimingPhase(
        name="pre_attack",
        start_window=151,
        end_window=250,
        description="Pre-attack calibration: normal traffic only.",
        normal_traffic_only=True,
    ),
    TimingPhase(
        name="attack",
        start_window=251,
        end_window=350,
        description="Attack interval: normal traffic plus an optional attack hook.",
        normal_traffic_only=False,
        attack_injection_allowed=True,
    ),
    TimingPhase(
        name="recovery",
        start_window=351,
        end_window=500,
        description="Recovery: normal traffic returns after the attack interval.",
        normal_traffic_only=True,
    ),
)
