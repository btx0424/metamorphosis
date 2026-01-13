from isaaclab.sim.spawners import SpawnerCfg
from isaaclab.sim import schemas
from isaaclab.utils.configclass import configclass
from isaaclab.sim.utils import get_current_stage, find_matching_prim_paths
from typing import Callable

from metamorphosis.quadruped import QuadrupedBuilder


def spawn(
    prim_path: str,
    cfg: "ProceduralQuadrupedCfg",
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
):
    stage = get_current_stage()
    builder = QuadrupedBuilder(hock_joint_prob=cfg.hock_joint_prob, stage=stage)

    root_path, asset_path = prim_path.rsplit("/", 1)
    source_prim_paths = find_matching_prim_paths(root_path)
    prim_paths = [f"{source_prim_path}/{asset_path}" for source_prim_path in source_prim_paths]
    for i, prim_path in enumerate(prim_paths):
        prim = builder.spawn(prim_path, seed=i)
        schemas.modify_articulation_root_properties(prim_path, cfg.articulation_props)
    return prim

@configclass
class ProceduralQuadrupedCfg(SpawnerCfg):
    """Configuration parameters for spawning a procedural quadruped."""

    func: Callable = spawn
    """Function to use for spawning the asset."""
    
    hock_joint_prob: float = 0.5
    """Whether to add a hock joint to the quadruped."""

    articulation_props: schemas.ArticulationRootPropertiesCfg | None = None

    visible: bool = True
    """Whether the spawned asset should be visible."""
    
    semantic_tags: list[tuple[str, str]] | None = None
    """List of semantic tags to add to the spawned asset."""

    copy_from_source: bool = False
    """Whether to copy the asset from the source prim or inherit it."""


