import argparse
import numpy as np
import torch

from isaaclab.app import AppLauncher

# create argparser
parser = argparse.ArgumentParser(description="Example script for metamorphosis.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from metamorphosis.asset_cfg import ProceduralQuadrupedCfg, QuadrupedBuilder


QUADRUPED_CONFIG = ArticulationCfg(
    spawn=ProceduralQuadrupedCfg(
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=1,
        ),
        hock_joint_prob=0.5,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            ".*_hip_joint": 0.0,
            "[L,R]F_hock_joint": 0.0,
            "[L,R]F_thigh_joint": np.pi  / 4,
            "[L,R]H_thigh_joint": 0.0,
            "[L,R]H_hock_joint": 0.0,
            ".*_calf_joint": -np.pi  / 2,
        },
        pos=(0, 0, 1.0),
    ),
    actuators={
        ".*": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            effort_limit_sim=1000.0,
            stiffness=200.0,
            damping=2.0,
            armature=0.01,
            friction=0.01,
        ),
    },
)


class ProceduralQuadrupedSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    """Designs the scene."""
    # jetbot = JETBOT_CONFIG.replace(prim_path="{ENV_REGEX_NS}/Jetbot")
    quadruped = QUADRUPED_CONFIG.replace(prim_path="{ENV_REGEX_NS}/Quadruped")


def main():
    """Main function."""

    # Initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.0, 0.0, 2.5], [-0.5, 0.0, 0.5])
    
    scene_cfg = ProceduralQuadrupedSceneCfg(
        num_envs=args_cli.num_envs,
        env_spacing=2.5,
        replicate_physics=False
    )
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    scene.reset()
    articulation = scene.articulations["quadruped"]
    
    print(articulation.joint_names)
    print(articulation.body_names)
    # print(articulation.data.default_mass)
    print(articulation.root_physx_view.is_homogeneous)
    
    builder = QuadrupedBuilder._instance
    has_hock_joint = torch.tensor(builder.params.has_hock_joint, device=articulation.device)
    print(has_hock_joint)

    hind_hock_joint_ids = articulation.find_joints("[L,R]H_hock_joint")[0]
    hind_thigh_joint_ids = articulation.find_joints("[L,R]H_thigh_joint")[0]
    default_joint_pos = articulation.data.default_joint_pos.clone()
    default_joint_pos[:, hind_thigh_joint_ids] = torch.where(has_hock_joint.unsqueeze(1), - np.pi * 0.2, np.pi / 4)
    default_joint_pos[:, hind_hock_joint_ids] = torch.where(has_hock_joint.unsqueeze(1), np.pi * 0.6, 0)

    root_state = articulation.data.default_root_state.clone()
    root_state[:, :3] += scene.env_origins
    articulation.root_physx_view.set_masses(articulation.data.default_mass.sqrt(), torch.arange(articulation.num_instances))
    articulation.write_root_pose_to_sim(root_state[:, :7])
    articulation.write_joint_position_to_sim(default_joint_pos)

    # Now we are ready!
    print("[INFO]: Setup complete...")

    # Simulate physics
    while simulation_app.is_running():
        # perform step
        articulation.set_joint_position_target(default_joint_pos)
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim.get_physics_dt())


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()