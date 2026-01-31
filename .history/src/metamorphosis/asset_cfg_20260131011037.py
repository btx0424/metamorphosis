from isaaclab.sim.spawners import SpawnerCfg
from isaaclab.sim import schemas
from isaaclab.utils.configclass import configclass
from isaaclab.sim.utils import get_current_stage, find_matching_prim_paths
from typing import Callable

from metamorphosis.builder import QuadrupedBuilder, BipedBuilder, QuadWheelBuilder


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
        pelvis_length_range=cfg.pelvis_length_range,
        pelvis_width_range=cfg.pelvis_width_range,
        pelvis_height_range=cfg.pelvis_height_range,
        waist_height_range=cfg.waist_height_range,
        hip_spacing_range=cfg.hip_spacing_range,
        hip1_length_range=cfg.hip1_length_range,
        hip1_radius_range=cfg.hip1_radius_range,
        hip2_length_range=cfg.hip2_length_range,
        hip2_radius_range=cfg.hip2_radius_range,
        leg_length_range=cfg.leg_length_range,
        shin_ratio_range=cfg.shin_ratio_range,
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
    """Configuration parameters for spawning a procedural biped (legs only)."""

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

    pelvis_length_range: tuple[float, float] = (0.10, 0.16)
    """Range for the pelvis length (box X dimension) in meters."""

    pelvis_width_range: tuple[float, float] = (0.18, 0.26)
    """Range for the pelvis width (box Y dimension) in meters."""

    pelvis_height_range: tuple[float, float] = (0.08, 0.14)
    """Range for the pelvis height (box Z dimension) in meters."""

    waist_height_range: tuple[float, float] = (0.05, 0.08)
    """Range for the waist connection height in meters."""

    hip_spacing_range: tuple[float, float] = (0.16, 0.24)
    """Range for the distance between left and right hips in meters."""

    hip1_length_range: tuple[float, float] = (0.03, 0.06)
    """Range for hip1 segment length in meters."""

    hip1_radius_range: tuple[float, float] = (0.02, 0.04)
    """Range for hip1 segment radius in meters."""

    hip2_length_range: tuple[float, float] = (0.03, 0.06)
    """Range for hip2 segment length in meters."""

    hip2_radius_range: tuple[float, float] = (0.02, 0.04)
    """Range for hip2 segment radius in meters."""

    leg_length_range: tuple[float, float] = (0.5, 0.7)
    """Range for the total leg length in meters."""

    shin_ratio_range: tuple[float, float] = (0.85, 1.15)
    """Range for shin length as a ratio of thigh length."""


def spawn_quadwheel(
    prim_path: str,
    cfg: "ProceduralQuadWheelCfg",
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
):
    stage = get_current_stage()
    builder = QuadWheelBuilder(
        base_length_range=cfg.base_length_range,
        base_width_range=cfg.base_width_range,
        base_height_range=cfg.base_height_range,
        leg_length_range=cfg.leg_length_range,
        calf_length_ratio=cfg.calf_length_ratio,
        wheel_radius_range=cfg.wheel_radius_range,
        wheel_width_range=cfg.wheel_width_range,
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
class ProceduralQuadWheelCfg(SpawnerCfg):
    """Configuration parameters for spawning a procedural quad-wheel robot."""

    func: Callable = spawn_quadwheel
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

    wheel_radius_range: tuple[float, float] = (0.08, 0.15)
    """Range for the wheel radius in meters."""

    wheel_width_range: tuple[float, float] = (0.03, 0.06)
    """Range for the wheel width in meters."""
