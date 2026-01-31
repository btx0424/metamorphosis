from isaaclab.sim.spawners import SpawnerCfg
from isaaclab.sim import schemas
from isaaclab.utils.configclass import configclass
from isaaclab.sim.utils import get_current_stage, find_matching_prim_paths
from typing import Callable

from metamorphosis.builder import QuadrupedBuilder
from metamorphosis.builder import BipedBuilder

def spawn(
    prim_path: str,
    cfg: "ProceduralQuadrupedCfg",
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
):
    stage = get_current_stage()
    builder = QuadrupedBuilder(
        base_length_range=cfg.base_length_range,
        base_width_range=cfg.base_width_range,
        base_height_range=cfg.base_height_range,
        leg_length_range=cfg.leg_length_range,
        calf_length_ratio=cfg.calf_length_ratio,
    )

    root_path, asset_path = prim_path.rsplit("/", 1)
    source_prim_paths = find_matching_prim_paths(root_path)
    prim_paths = [
        f"{source_prim_path}/{asset_path}" for source_prim_path in source_prim_paths
    ]
    for i, prim_path in enumerate(prim_paths):
        param = builder.sample_params(seed=i)
        prim = builder.spawn(stage, prim_path, param)
        schemas.modify_articulation_root_properties(prim_path, cfg.articulation_props)
        if cfg.activate_contact_sensors:
            schemas.activate_contact_sensors(prim_path, stage=stage)
    return prim


@configclass
class ProceduralQuadrupedCfg(SpawnerCfg):
    """Configuration parameters for spawning a procedural quadruped."""

    func: Callable = spawn
    """Function to use for spawning the asset."""

    activate_contact_sensors: bool = True
    """Whether to activate contact sensors for the asset. Defaults to True."""

    articulation_props: schemas.ArticulationRootPropertiesCfg | None = None

    visible: bool = True
    """Whether the spawned asset should be visible."""

    semantic_tags: list[tuple[str, str]] | None = None
    """List of semantic tags to add to the spawned asset."""

    copy_from_source: bool = False
    """Whether to copy the asset from the source prim or inherit it."""

    base_length_range: tuple[float, float] = (0.5, 1.0)
    """Range for the base length."""

    base_width_range: tuple[float, float] = (0.3, 0.4)
    """Range for the base width."""

    base_height_range: tuple[float, float] = (0.15, 0.25)
    """Range for the base height."""

    leg_length_range: tuple[float, float] = (0.4, 0.8)
    """Range for the leg length."""

    calf_length_ratio: tuple[float, float] = (0.9, 1.0)
    """Range for the calf length ratio."""


def spawn_biped(
    prim_path: str,
    cfg: "ProceduralBipedCfg",
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
):
    stage = get_current_stage()
    builder = BipedBuilder(
        torso_height_range=cfg.torso_height_range,
        torso_width_range=cfg.torso_width_range,
        leg_length_total_range=cfg.leg_length_total_range,
        calf_length_ratio=cfg.calf_length_ratio,
    )

    root_path, asset_path = prim_path.rsplit("/", 1)
    source_prim_paths = find_matching_prim_paths(root_path)
    prim_paths = [
        f"{source_prim_path}/{asset_path}" for source_prim_path in source_prim_paths
    ]
    for i, prim_path in enumerate(prim_paths):
        param = builder.sample_params(seed=i)
        prim = builder.spawn(stage, prim_path, param)
        schemas.modify_articulation_root_properties(prim_path, cfg.articulation_props)
        if cfg.activate_contact_sensors:
            schemas.activate_contact_sensors(prim_path, stage=stage)
    return prim


@configclass
class ProceduralBipedCfg(SpawnerCfg):
    """Configuration parameters for spawning a procedural biped."""

    func: Callable = spawn_biped
    """Function to use for spawning the asset."""

    activate_contact_sensors: bool = True
    """Whether to activate contact sensors for the asset. Defaults to True."""

    articulation_props: schemas.ArticulationRootPropertiesCfg | None = None

    visible: bool = True
    """Whether the spawned asset should be visible."""

    semantic_tags: list[tuple[str, str]] | None = None
    """List of semantic tags to add to the spawned asset."""

    copy_from_source: bool = False
    """Whether to copy the asset from the source prim or inherit it."""

    torso_height_range: tuple[float, float] = (0.4, 0.6)
    """Range for the torso height."""

    torso_width_range: tuple[float, float] = (0.2, 0.3)
    """Range for the torso width."""

    leg_length_total_range: tuple[float, float] = (0.6, 1.0)
    """Range for the total leg length."""

    calf_length_ratio: tuple[float, float] = (0.8, 1.0)
    """Range for the calf length ratio (relative to thigh length)."""
