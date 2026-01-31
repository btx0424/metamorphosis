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
        self.pelvis_mass_range = pelvis_mass_range
        self.leg_length_range = leg_length_range
        self.shin_ratio_range = shin_ratio_range
        self.valid_filter = valid_filter
    
    def sample_params(self, seed: int = -1) -> BipedParam:
        if seed >= 0:
            np.random.seed(seed)
            random.seed(seed)
        
        for _ in range(10):
            # Pelvis dimensions and mass
            pelvis_height = random.uniform(*self.pelvis_height_range)
            pelvis_width = random.uniform(*self.pelvis_width_range)
            pelvis_length = random.uniform(*self.pelvis_length_range)
            pelvis_mass = random.uniform(*self.pelvis_mass_range)
            
            # Hip positioning (G1-style offsets)
            hip_offset_x = random.uniform(0.0, 0.02)  # Slight forward offset
            hip_offset_y = pelvis_width * random.uniform(0.35, 0.45)  # Hip spacing
            hip_offset_z = random.uniform(-0.12, -0.08)  # Below pelvis center
            
            # Leg dimensions
            total_leg_length = random.uniform(*self.leg_length_range)
            shin_ratio = random.uniform(*self.shin_ratio_range)
            thigh_length = total_leg_length / (1 + shin_ratio)
            shin_length = thigh_length * shin_ratio
            
            # Leg masses and radii (proportional to length)
            thigh_radius = random.uniform(0.035, 0.055)
            thigh_mass = thigh_length * random.uniform(1.8, 2.5)  # Mass proportional to length
            
            shin_radius = thigh_radius * random.uniform(0.75, 0.95)
            shin_mass = shin_length * random.uniform(1.5, 2.2)
            
            # Foot parameters (larger for stability)
            foot_length = random.uniform(0.20, 0.30)
            foot_width = random.uniform(0.08, 0.12)
            foot_height = random.uniform(0.025, 0.035)
            foot_mass = random.uniform(0.3, 0.6)
            
            param = BipedParam(
                pelvis_length=pelvis_length,
                pelvis_width=pelvis_width,
                pelvis_height=pelvis_height,
                pelvis_mass=pelvis_mass,
                hip_offset_x=hip_offset_x,
                hip_offset_y=hip_offset_y,
                hip_offset_z=hip_offset_z,
                thigh_length=thigh_length,
                thigh_radius=thigh_radius,
                thigh_mass=thigh_mass,
                shin_length=shin_length,
                shin_radius=shin_radius,
                shin_mass=shin_mass,
                foot_length=foot_length,
                foot_width=foot_width,
                foot_height=foot_height,
                foot_mass=foot_mass,
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
        pelvis.mass = param.pelvis_mass
        # Inertia proportional to dimensions and mass
        ixx = param.pelvis_mass * (param.pelvis_width**2 + param.pelvis_height**2) / 12
        iyy = param.pelvis_mass * (param.pelvis_length**2 + param.pelvis_height**2) / 12
        izz = param.pelvis_mass * (param.pelvis_length**2 + param.pelvis_width**2) / 12
        pelvis.inertia = [ixx, iyy, izz]
        
        pelvis_geom = pelvis.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX)
        pelvis_geom.size = [param.pelvis_length / 2, param.pelvis_width / 2, param.pelvis_height / 2]
        
        # ============ LEGS (G1-style joint structure) ============
        # Each leg: pelvis -> hip_pitch -> hip_roll -> hip_yaw -> thigh -> knee -> shin -> ankle_pitch -> ankle_roll -> foot
        
        for side, y_sign in [("left", 1), ("right", -1)]:
            # Hip position
            hip_pos = [param.hip_offset_x, y_sign * param.hip_offset_y, param.hip_offset_z]
            
            # --- Hip Pitch Link ---
            hip_pitch_link = pelvis.add_body(name=f"{side}_hip_pitch_link")
            hip_pitch_link.pos = hip_pos
            hip_pitch_link.mass = 1.35  # G1-style mass
            hip_pitch_link.inertia = [0.0018, 0.0015, 0.0012]
            
            hip_pitch_joint = hip_pitch_link.add_joint(
                name=f"{side}_hip_pitch_joint",
                type=mujoco.mjtJoint.mjJNT_HINGE
            )
            hip_pitch_joint.axis = [0, 1, 0]  # Y-axis
            hip_pitch_joint.range = [-2.5307, 2.8798]  # G1 ranges
            hip_pitch_joint.armature = 0.01017752004
            
            # --- Hip Roll Link ---
            hip_roll_offset = [0, 0.052 * y_sign, -0.030465]  # G1-style offset
            hip_roll_link = hip_pitch_link.add_body(name=f"{side}_hip_roll_link")
            hip_roll_link.pos = hip_roll_offset
            hip_roll_link.mass = 1.52
            hip_roll_link.inertia = [0.0025, 0.0024, 0.0015]
            
            hip_roll_joint = hip_roll_link.add_joint(
                name=f"{side}_hip_roll_joint",
                type=mujoco.mjtJoint.mjJNT_HINGE
            )
            hip_roll_joint.axis = [1, 0, 0]  # X-axis
            # Different ranges for left/right to match G1
            if side == "left":
                hip_roll_joint.range = [-0.5236, 2.9671]
            else:
                hip_roll_joint.range = [-2.9671, 0.5236]
            hip_roll_joint.armature = 0.025101925
            
            # Hip collision geometry
            hip_geom = hip_roll_link.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE)
            hip_geom.size = [0.06, 0, 0]  # [radius, 0, 0] for capsule
            hip_geom.fromto = [0.02, 0, 0, 0.02, 0, -0.08]
            
            # --- Hip Yaw Link (Thigh) ---
            hip_yaw_offset = [0.025, 0, -0.12412]  # G1-style offset
            hip_yaw_link = hip_roll_link.add_body(name=f"{side}_hip_yaw_link")
            hip_yaw_link.pos = hip_yaw_offset
            hip_yaw_link.mass = param.thigh_mass
            # Thigh inertia based on dimensions
            thigh_ixx = param.thigh_mass * (3 * param.thigh_radius**2 + param.thigh_length**2) / 12
            thigh_iyy = param.thigh_mass * param.thigh_radius**2 / 2
            thigh_izz = thigh_ixx
            hip_yaw_link.inertia = [thigh_ixx, thigh_iyy, thigh_izz]
            
            hip_yaw_joint = hip_yaw_link.add_joint(
                name=f"{side}_hip_yaw_joint",
                type=mujoco.mjtJoint.mjJNT_HINGE
            )
            hip_yaw_joint.axis = [0, 0, 1]  # Z-axis
            hip_yaw_joint.range = [-2.7576, 2.7576]
            hip_yaw_joint.armature = 0.01017752004
            
            # Thigh collision geometry
            thigh_geom = hip_yaw_link.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE)
            thigh_geom.size = [param.thigh_radius, 0, 0]
            thigh_geom.fromto = [0, 0, -0.03, -0.06, 0, -param.thigh_length]
            
            # --- Knee Link (Shin) ---
            knee_offset = [-0.078273, 0.0021489 * y_sign, -param.thigh_length]
            knee_link = hip_yaw_link.add_body(name=f"{side}_knee_link")
            knee_link.pos = knee_offset
            knee_link.mass = param.shin_mass
            # Shin inertia
            shin_ixx = param.shin_mass * (3 * param.shin_radius**2 + param.shin_length**2) / 12
            shin_iyy = param.shin_mass * param.shin_radius**2 / 2
            shin_izz = shin_ixx
            knee_link.inertia = [shin_ixx, shin_iyy, shin_izz]
            
            knee_joint = knee_link.add_joint(
                name=f"{side}_knee_joint",
                type=mujoco.mjtJoint.mjJNT_HINGE
            )
            knee_joint.axis = [0, 1, 0]  # Y-axis
            knee_joint.range = [-0.087267, 2.8798]
            knee_joint.armature = 0.025101925
            
            # Shin collision geometry
            shin_geom = knee_link.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE)
            shin_geom.size = [param.shin_radius]
            shin_geom.fromto = [0.01, 0, 0, 0.01, 0, -param.shin_length]
            
            # --- Ankle Pitch Link ---
            ankle_pitch_offset = [0, -9.4445e-05 * y_sign, -param.shin_length]
            ankle_pitch_link = knee_link.add_body(name=f"{side}_ankle_pitch_link")
            ankle_pitch_link.pos = ankle_pitch_offset
            ankle_pitch_link.mass = 0.52
            ankle_pitch_link.inertia = [0.00067, 0.00067, 0.00027]
            
            ankle_pitch_joint = ankle_pitch_link.add_joint(
                name=f"{side}_ankle_pitch_joint",
                type=mujoco.mjtJoint.mjJNT_HINGE
            )
            ankle_pitch_joint.axis = [0, 1, 0]  # Y-axis
            ankle_pitch_joint.range = [-0.87267, 0.5236]
            ankle_pitch_joint.armature = 0.00721945
            
            # --- Ankle Roll Link (Foot) ---
            ankle_roll_offset = [0, 0, -0.045001]
            ankle_roll_link = ankle_pitch_link.add_body(name=f"{side}_ankle_roll_link")
            ankle_roll_link.pos = ankle_roll_offset
            ankle_roll_link.mass = param.foot_mass
            # Foot inertia
            foot_ixx = param.foot_mass * (param.foot_width**2 + param.foot_height**2) / 12
            foot_iyy = param.foot_mass * (param.foot_length**2 + param.foot_height**2) / 12
            foot_izz = param.foot_mass * (param.foot_length**2 + param.foot_width**2) / 12
            ankle_roll_link.inertia = [foot_ixx, foot_iyy, foot_izz]
            
            ankle_roll_joint = ankle_roll_link.add_joint(
                name=f"{side}_ankle_roll_joint",
                type=mujoco.mjtJoint.mjJNT_HINGE
            )
            ankle_roll_joint.axis = [1, 0, 0]  # X-axis
            ankle_roll_joint.range = [-0.2618, 0.2618]
            ankle_roll_joint.armature = 0.00721945
            
            # Foot geometry (box extending forward)
            foot_geom = ankle_roll_link.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX)
            foot_geom.size = [param.foot_length / 2, param.foot_width / 2, param.foot_height / 2]
            foot_geom.pos = [param.foot_length / 4, 0, -param.foot_height / 2]
        
        return spec


if __name__ == "__main__":
    builder = BipedBuilder()
    param = builder.sample_params(seed=0)
    print(param)
    spec = builder.generate_mjspec(param)
    print("Generated biped MjSpec successfully!")
