import mujoco
import numpy as np
import random

from typing import NamedTuple, Callable
from metamorphosis.builder.base import BuilderBase
from metamorphosis.utils.mjs_utils import add_capsule_geom_


class BipedParam(NamedTuple):
    """Parameters for a bipedal robot following G1-style design."""
    # Pelvis dimensions (main body)
    pelvis_length: float     # 骨盆长度 (前后)
    pelvis_width: float      # 骨盆宽度 (左右)  
    pelvis_height: float     # 骨盆高度 (上下)
    pelvis_mass: float       # 骨盆质量
    
    # Hip positioning
    hip_offset_x: float      # 髋关节前后偏移
    hip_offset_y: float      # 髋关节左右间距的一半
    hip_offset_z: float      # 髋关节垂直偏移（相对骨盆底部）
    
    # Thigh parameters  
    thigh_length: float      # 大腿长度
    thigh_radius: float      # 大腿粗细
    thigh_mass: float        # 大腿质量
    
    # Shin parameters
    shin_length: float       # 小腿长度
    shin_radius: float       # 小腿粗细
    shin_mass: float         # 小腿质量
    
    # Foot parameters
    foot_length: float       # 脚长
    foot_width: float        # 脚宽
    foot_height: float       # 脚厚度
    foot_mass: float         # 脚质量


class BipedBuilder(BuilderBase):
    """
    Builder for generating procedurally parameterized bipedal robots (legs only).
    
    This class creates bipedal robot models with configurable body dimensions
    and leg proportions. It samples random parameters within specified ranges
    and generates MuJoCo specifications for the robot model.
    
    The biped consists of:
    - A pelvis (base body) with configurable dimensions
    - Two legs, each with:
      - Hip yaw joint (virtual body for rotation around Z)
      - Hip roll joint (virtual body for rotation around X)
      - Hip pitch joint (thigh body, rotation around Y)
      - Knee joint (shin body)
      - Ankle pitch joint (virtual body)
      - Ankle roll joint (foot body)
    
    This design uses "virtual bodies" (bodies with minimal mass and no geometry)
    to achieve multi-DOF joints while respecting the one-joint-per-body constraint
    of the USD converter.
    
    Attributes:
        pelvis_height_range: Tuple of (min, max) for pelvis height in meters.
        pelvis_width_range: Tuple of (min, max) for pelvis width in meters.
        leg_length_range: Tuple of (min, max) for total leg length in meters.
        shin_ratio_range: Tuple of (min, max) for shin/thigh length ratio.
        valid_filter: Optional callable to filter valid parameter combinations.
    
    Example:
        >>> builder = BipedBuilder(
        ...     pelvis_height_range=(0.10, 0.15),
        ...     leg_length_range=(0.6, 1.0)
        ... )
        >>> param = builder.sample_params(seed=42)
        >>> spec = builder.generate_mjspec(param)
    """
    
    def __init__(
        self,
        pelvis_height_range: tuple[float, float] = (0.10, 0.18),
        pelvis_width_range: tuple[float, float] = (0.20, 0.30),
        pelvis_length_range: tuple[float, float] = (0.10, 0.15),
        pelvis_mass_range: tuple[float, float] = (3.5, 5.0),
        leg_length_range: tuple[float, float] = (0.6, 0.9),
        shin_ratio_range: tuple[float, float] = (0.9, 1.1),
        valid_filter: Callable[[BipedParam], bool] = lambda _: True,
    ):
        super().__init__()
        self.pelvis_height_range = pelvis_height_range
        self.pelvis_width_range = pelvis_width_range
        self.pelvis_length_range = pelvis_length_range
        self.leg_length_range = leg_length_range
        self.shin_ratio_range = shin_ratio_range
        self.valid_filter = valid_filter
    
    def sample_params(self, seed: int = -1) -> BipedParam:
        if seed >= 0:
            np.random.seed(seed)
            random.seed(seed)
        
        for _ in range(10):
            # Pelvis
            pelvis_height = random.uniform(*self.pelvis_height_range)
            pelvis_width = random.uniform(*self.pelvis_width_range)
            pelvis_length = random.uniform(*self.pelvis_length_range)
            
            # Hip width (slightly less than pelvis width)
            hip_width = pelvis_width * random.uniform(0.85, 0.95)
            
            # Legs
            total_leg_length = random.uniform(*self.leg_length_range)
            shin_ratio = random.uniform(*self.shin_ratio_range)
            thigh_length = total_leg_length / (1 + shin_ratio)
            shin_length = thigh_length * shin_ratio
            leg_radius = random.uniform(0.03, 0.05)
            
            # Feet - larger for better stability
            foot_length = random.uniform(0.18, 0.28)
            foot_width = random.uniform(0.10, 0.14)
            
            param = BipedParam(
                pelvis_length=pelvis_length,
                pelvis_width=pelvis_width,
                pelvis_height=pelvis_height,
                hip_width=hip_width,
                thigh_length=thigh_length,
                shin_length=shin_length,
                leg_radius=leg_radius,
                foot_length=foot_length,
                foot_width=foot_width,
            )
            
            if self.valid_filter(param):
                break
        else:
            raise ValueError("Failed to sample valid parameters")
        
        return param
    
    def generate_mjspec(self, param: BipedParam) -> mujoco.MjSpec:
        spec = mujoco.MjSpec()
        
        # ============ PELVIS (root body) ============
        pelvis = spec.worldbody.add_body()
        pelvis.name = "pelvis"
        pelvis.mass = 5.0
        pelvis.inertia = [0.5, 0.5, 0.5]
        
        pelvis_geom = pelvis.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX)
        pelvis_geom.size = [param.pelvis_length / 2, param.pelvis_width / 2, param.pelvis_height / 2]
        
        # ============ LEGS ============
        # Each leg uses a chain of bodies to achieve multi-DOF hip and ankle
        # Structure: pelvis -> hip_yaw -> hip_roll -> thigh (hip_pitch) -> shin (knee) -> ankle_pitch -> foot (ankle_roll)
        
        for side, y_sign in [("L", -1), ("R", 1)]:
            # Hip position (at bottom of pelvis)
            hip_pos_y = y_sign * param.hip_width / 2
            hip_pos_z = -param.pelvis_height / 2
            
            # --- Hip Yaw (virtual body, rotation around Z) ---
            hip_yaw_body = pelvis.add_body(name=f"{side}_hip_yaw")
            hip_yaw_body.pos = [0, hip_pos_y, hip_pos_z]
            hip_yaw_body.mass = 0.1  # Minimal mass for virtual body
            hip_yaw_body.inertia = [0.001, 0.001, 0.001]
            
            hip_yaw_joint = hip_yaw_body.add_joint(
                name=f"{side}_hip_yaw_joint",
                type=mujoco.mjtJoint.mjJNT_HINGE
            )
            hip_yaw_joint.axis = [0, 0, 1]
            hip_yaw_joint.range = [-np.pi * 0.3, np.pi * 0.3]
            
            # --- Hip Roll (virtual body, rotation around X) ---
            hip_roll_body = hip_yaw_body.add_body(name=f"{side}_hip_roll")
            hip_roll_body.pos = [0, 0, 0]  # Same position as yaw
            hip_roll_body.mass = 0.1
            hip_roll_body.inertia = [0.001, 0.001, 0.001]
            
            hip_roll_joint = hip_roll_body.add_joint(
                name=f"{side}_hip_roll_joint",
                type=mujoco.mjtJoint.mjJNT_HINGE
            )
            hip_roll_joint.axis = [1, 0, 0]
            hip_roll_joint.range = [-np.pi * 0.4, np.pi * 0.4]
            
            # --- Thigh (with Hip Pitch joint, rotation around Y) ---
            thigh = hip_roll_body.add_body(name=f"{side}_thigh")
            thigh.pos = [0, 0, 0]
            thigh.mass = 2.5
            thigh.inertia = [0.2, 0.2, 0.05]
            
            hip_pitch_joint = thigh.add_joint(
                name=f"{side}_hip_pitch_joint",
                type=mujoco.mjtJoint.mjJNT_HINGE
            )
            hip_pitch_joint.axis = [0, 1, 0]
            hip_pitch_joint.range = [-np.pi * 0.6, np.pi * 0.6]
            
            add_capsule_geom_(
                thigh,
                radius=param.leg_radius,
                fromto=[0, 0, 0, 0, 0, -param.thigh_length]
            )
            
            # --- Shin (with Knee joint) ---
            shin = thigh.add_body(name=f"{side}_shin")
            shin.pos = [0, 0, -param.thigh_length]
            shin.mass = 1.5
            shin.inertia = [0.1, 0.1, 0.02]
            
            knee_joint = shin.add_joint(
                name=f"{side}_knee_joint",
                type=mujoco.mjtJoint.mjJNT_HINGE
            )
            knee_joint.axis = [0, 1, 0]
            knee_joint.range = [0, np.pi * 0.75]  # Knee only bends backward
            
            shin_radius = param.leg_radius * 0.85
            add_capsule_geom_(
                shin,
                radius=shin_radius,
                fromto=[0, 0, 0, 0, 0, -param.shin_length]
            )
            
            # --- Ankle Pitch (virtual body) ---
            # Position below shin, accounting for capsule end cap radius
            ankle_pitch_body = shin.add_body(name=f"{side}_ankle_pitch")
            ankle_pitch_body.pos = [0, 0, -param.shin_length - shin_radius]
            ankle_pitch_body.mass = 0.1
            ankle_pitch_body.inertia = [0.001, 0.001, 0.001]
            
            ankle_pitch_joint = ankle_pitch_body.add_joint(
                name=f"{side}_ankle_pitch_joint",
                type=mujoco.mjtJoint.mjJNT_HINGE
            )
            ankle_pitch_joint.axis = [0, 1, 0]
            ankle_pitch_joint.range = [-np.pi * 0.4, np.pi * 0.4]
            
            # --- Foot (with Ankle Roll joint) ---
            foot = ankle_pitch_body.add_body(name=f"{side}_foot")
            foot.pos = [0, 0, 0]
            foot.mass = 0.4
            foot.inertia = [0.01, 0.01, 0.01]
            
            ankle_roll_joint = foot.add_joint(
                name=f"{side}_ankle_roll_joint",
                type=mujoco.mjtJoint.mjJNT_HINGE
            )
            ankle_roll_joint.axis = [1, 0, 0]
            ankle_roll_joint.range = [-np.pi * 0.25, np.pi * 0.25]
            
            # Foot geometry (box, extends forward)
            foot_height = 0.02
            foot_geom = foot.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX)
            foot_geom.size = [param.foot_length / 2, param.foot_width / 2, foot_height]
            foot_geom.pos = [param.foot_length / 4, 0, -foot_height]
        
        return spec


if __name__ == "__main__":
    builder = BipedBuilder()
    param = builder.sample_params(seed=0)
    print(param)
    spec = builder.generate_mjspec(param)
    print("Generated biped MjSpec successfully!")
