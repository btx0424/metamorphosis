import mujoco
import numpy as np
import random

from typing import NamedTuple, Callable
from metamorphosis.builder.base import BuilderBase
from metamorphosis.utils.mjs_utils import add_capsule_geom_


class BipedParam(NamedTuple):
    """Parameters for a bipedal robot with box pelvis and standard humanoid hip structure."""
    # Pelvis (main body - box)
    pelvis_length: float     # 长方体骨盆长度 (X方向)
    pelvis_width: float      # 长方体骨盆宽度 (Y方向)
    pelvis_height: float     # 长方体骨盆高度 (Z方向)
    pelvis_mass: float       # 骨盆质量
    
    # Waist connection
    waist_height: float      # 腰部连接高度
    waist_radius: float      # 腰部半径（小圆柱）
    
    # Hip positioning and dimensions
    hip_spacing: float       # 左右髋关节间距
    
    # Hip1 parameters (first hip segment)
    hip1_length: float       # 髋关节1长度
    hip1_radius: float       # 髋关节1半径
    hip1_mass: float         # 髋关节1质量
    
    # Hip2 parameters (second hip segment)
    hip2_length: float       # 髋关节2长度
    hip2_radius: float       # 髋关节2半径
    hip2_mass: float         # 髋关节2质量
    
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
    
    The biped uses a standard humanoid structure:
    - A pelvis (base body) with configurable dimensions
    - Two legs, each with:
      - Waist connection geometry (cylinder)
      - Hip joint 1 (pitch) -> Hip1 body (capsule)
      - Hip joint 2 (roll) -> Hip2 body (capsule) 
      - Hip joint 3 (yaw) -> Thigh body (capsule)
      - Knee joint (pitch) -> Shin/Calf body (capsule)
      - Ankle pitch joint (virtual body)
      - Ankle roll joint -> Foot body (box)
    
    This design follows: waist geo -> hip joint1 -> hip1 -> hip joint2 -> hip2 -> 
    hip joint3 -> thigh -> knee joint -> calf -> ankle joints -> foot
    
    Attributes:
        pelvis_height_range: Tuple of (min, max) for pelvis height in meters.
        pelvis_width_range: Tuple of (min, max) for pelvis width in meters.
        leg_length_range: Tuple of (min, max) for total leg length in meters.
        shin_ratio_range: Tuple of (min, max) for shin/thigh length ratio.
        hip1_length_range: Tuple of (min, max) for hip1 segment length.
        hip2_length_range: Tuple of (min, max) for hip2 segment length.
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
        pelvis_length_range: tuple[float, float] = (0.10, 0.16),
        pelvis_width_range: tuple[float, float] = (0.18, 0.26),
        pelvis_height_range: tuple[float, float] = (0.08, 2.0),
        waist_height_range: tuple[float, float] = (0.05, 0.08),
        hip_spacing_range: tuple[float, float] = (0.16, 0.24),
        hip1_length_range: tuple[float, float] = (0.03, 0.06),  # Hip1 length range
        hip1_radius_range: tuple[float, float] = (0.02, 0.04),  # Hip1 radius range
        hip2_length_range: tuple[float, float] = (0.03, 0.06),  # Hip2 length range
        hip2_radius_range: tuple[float, float] = (0.02, 0.04),  # Hip2 radius range
        leg_length_range: tuple[float, float] = (0.5, 0.7),
        shin_ratio_range: tuple[float, float] = (0.85, 1.15),
        valid_filter: Callable[[BipedParam], bool] = lambda _: True,
    ):
        super().__init__()
        self.pelvis_length_range = pelvis_length_range
        self.pelvis_width_range = pelvis_width_range
        self.pelvis_height_range = pelvis_height_range
        self.waist_height_range = waist_height_range
        self.hip_spacing_range = hip_spacing_range
        self.hip1_length_range = hip1_length_range
        self.hip1_radius_range = hip1_radius_range
        self.hip2_length_range = hip2_length_range
        self.hip2_radius_range = hip2_radius_range
        self.leg_length_range = leg_length_range
        self.shin_ratio_range = shin_ratio_range
        self.valid_filter = valid_filter
    
    def sample_params(self, seed: int = -1) -> BipedParam:
        if seed >= 0:
            np.random.seed(seed)
            random.seed(seed)
        
        for _ in range(10):
            # Pelvis (box)
            pelvis_length = random.uniform(*self.pelvis_length_range)
            pelvis_width = random.uniform(*self.pelvis_width_range)
            pelvis_height = random.uniform(*self.pelvis_height_range)
            pelvis_mass = random.uniform(3.0, 5.0)
            
            # Waist connection
            waist_height = random.uniform(*self.waist_height_range)
            waist_radius = min(pelvis_width, pelvis_length) * random.uniform(0.2, 0.4)
            
            # Hip spacing and hip segment dimensions
            hip_spacing = random.uniform(*self.hip_spacing_range)
            
            # Hip1 dimensions (first hip segment)
            hip1_length = random.uniform(*self.hip1_length_range)
            hip1_radius = random.uniform(*self.hip1_radius_range)
            hip1_mass = random.uniform(0.4, 0.8)  # Hip1 mass
            
            # Hip2 dimensions (second hip segment)
            hip2_length = random.uniform(*self.hip2_length_range)
            hip2_radius = random.uniform(*self.hip2_radius_range)
            hip2_mass = random.uniform(0.4, 0.8)  # Hip2 mass
            
            # Leg dimensions
            total_leg_length = random.uniform(*self.leg_length_range)
            shin_ratio = random.uniform(*self.shin_ratio_range)
            thigh_length = total_leg_length / (1 + shin_ratio)
            shin_length = thigh_length * shin_ratio
            
            # Leg masses and radii (proportional to length)
            thigh_radius = random.uniform(0.025, 0.040)
            thigh_mass = thigh_length * random.uniform(1.5, 2.2)
            
            shin_radius = thigh_radius * random.uniform(0.75, 0.95)
            shin_mass = shin_length * random.uniform(1.2, 1.8)
            
            # Foot parameters
            foot_length = random.uniform(0.18, 0.25)
            foot_width = random.uniform(0.06, 0.10)
            foot_height = random.uniform(0.02, 0.03)
            foot_mass = random.uniform(0.2, 0.5)
            
            param = BipedParam(
                pelvis_length=pelvis_length,
                pelvis_width=pelvis_width,
                pelvis_height=pelvis_height,
                pelvis_mass=pelvis_mass,
                waist_height=waist_height,
                waist_radius=waist_radius,
                hip_spacing=hip_spacing,
                hip1_length=hip1_length,
                hip1_radius=hip1_radius,
                hip1_mass=hip1_mass,
                hip2_length=hip2_length,
                hip2_radius=hip2_radius,
                hip2_mass=hip2_mass,
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
        
        # ============ Torso ============
        torso_link = spec.worldbody.add_body()
        torso_link.name = "torso_link"
        torso_link.mass = param.pelvis_mass
        # Box inertia
        ixx = param.pelvis_mass * (param.pelvis_width**2 + param.pelvis_height**2) / 12
        iyy = param.pelvis_mass * (param.pelvis_length**2 + param.pelvis_height**2) / 12
        izz = param.pelvis_mass * (param.pelvis_length**2 + param.pelvis_width**2) / 12
        torso_link.inertia = [ixx, iyy, izz]
        
        # Pelvis geometry - box
        pelvis_geom = torso_link.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX)
        pelvis_geom.size = [param.pelvis_length/2, param.pelvis_width/2, param.pelvis_height/2]
        pelvis_geom.rgba = [0.75, 0.75, 0.75, 1.0]
        
        # ============ WAIST CONNECTION ============
        # Small cylindrical waist below pelvis
        pelvis = torso_link.add_body(name="pelvis")
        pelvis.pos = [0, 0, -(param.pelvis_height/2 + param.waist_height/2)]
        pelvis.mass = 0.5  # Small mass for connection
        pelvis.inertia = [0.01, 0.01, 0.01]
        
        # Waist geometry - small cylinder
        waist_geom = pelvis.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER)
        waist_geom.size = [param.waist_radius, param.waist_height/2, 0]
        waist_geom.rgba = [0.65, 0.65, 0.85, 1.0]
        
        # ============ LEFT AND RIGHT LEGS (6 DOF each) ============
        for side, y_sign in [("left", 1), ("right", -1)]:
            
                # --- Hip Pitch Joint Body (joint: hip_pitch_joint, link: hip_pitch_link) ---
            hip_pitch_pos = [0, y_sign * param.hip_spacing/2, -(param.waist_height/2)]
            
            hip_pitch_joint = pelvis.add_body(name=f"{side}_hip_pitch_joint")
            hip_pitch_joint.pos = hip_pitch_pos
            hip_pitch_joint.mass = param.hip1_mass
            # Cylinder inertia for hip1
            hip1_ixx = param.hip1_mass * (3 * param.hip1_radius**2 + param.hip1_length**2) / 12
            hip1_iyy = param.hip1_mass * param.hip1_radius**2 / 2
            hip1_izz = hip1_ixx
            hip_pitch_joint.inertia = [hip1_ixx, hip1_iyy, hip1_izz]
            
            hip_pitch_joint_joint = hip_pitch_joint.add_joint(
                name=f"{side}_hip_pitch_joint",
                type=mujoco.mjtJoint.mjJNT_HINGE
            )
            hip_pitch_joint_joint.axis = [1, 0, 0]  # X-axis (roll)
            hip_pitch_joint_joint.range = [-2.5307, 2.8798]
            hip_pitch_joint_joint.armature = 0.01017752004
            
            # Hip1 geometry - capsule (cylinder)
            hip_pitch_link = hip_pitch_joint.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE)
            hip_pitch_link.name = f"{side}_hip_pitch_link"
            hip_pitch_link.size = [param.hip1_radius, 0, 0]
            hip_pitch_link.fromto = [0, 0, 0, 0, 0, -param.hip1_length]
            hip_pitch_link.rgba = [0.85, 0.45, 0.45, 1.0]
            
                # --- Hip Roll Joint Body (joint: hip_roll_joint, link: hip_roll_link) ---
            hip_roll_joint = hip_pitch_joint.add_body(name=f"{side}_hip_roll_joint")
            hip_roll_joint.pos = [0, 0, -(param.hip1_length + param.hip1_radius + param.hip2_radius)]  # Tangent to hip_pitch_link
            hip_roll_joint.mass = param.hip2_mass
            # Cylinder inertia for hip2
            hip2_ixx = param.hip2_mass * (3 * param.hip2_radius**2 + param.hip2_length**2) / 12
            hip2_iyy = param.hip2_mass * param.hip2_radius**2 / 2
            hip2_izz = hip2_ixx
            hip_roll_joint.inertia = [hip2_ixx, hip2_iyy, hip2_izz]
            
            hip_roll_joint_joint = hip_roll_joint.add_joint(
                name=f"{side}_hip_roll_joint",
                type=mujoco.mjtJoint.mjJNT_HINGE
            )
            hip_roll_joint_joint.axis = [1, 0, 0]  # X-axis (roll)
            # Different ranges for left/right legs
            if side == "left":
                hip_roll_joint_joint.range = [-0.5236, 2.9671]
            else:
                hip_roll_joint_joint.range = [-2.9671, 0.5236]
            hip_roll_joint_joint.armature = 0.025101925
            
            # Hip2 geometry - capsule (cylinder)
            hip_roll_link = hip_roll_joint.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE)
            hip_roll_link.name = f"{side}_hip_roll_link"
            hip_roll_link.size = [param.hip2_radius, 0, 0]
            hip_roll_link.fromto = [0, 0, 0, 0, 0, -param.hip2_length]
            hip_roll_link.rgba = [0.45, 0.75, 0.45, 1.0]
            
                # --- Hip Yaw Joint Body (joint: hip_yaw_hoint, link: hip_yaw_link) ---
            hip_yaw_link = hip_roll_joint.add_body(name=f"{side}_hip_yaw_link")
            hip_yaw_link.pos = [0, 0, -(param.hip2_length + param.hip2_radius + param.thigh_radius)]  # Tangent to hip_roll_link
            hip_yaw_link.mass = param.thigh_mass
            # Cylinder inertia
            thigh_ixx = param.thigh_mass * (3 * param.thigh_radius**2 + param.thigh_length**2) / 12
            thigh_iyy = param.thigh_mass * param.thigh_radius**2 / 2
            thigh_izz = thigh_ixx
            hip_yaw_link.inertia = [thigh_ixx, thigh_iyy, thigh_izz]
            
            hip_yaw_hoint = hip_yaw_link.add_joint(
                name=f"{side}_hip_yaw_hoint",
                type=mujoco.mjtJoint.mjJNT_HINGE
            )
            hip_yaw_hoint.axis = [0, 0, 1]  # Z-axis (yaw)
            hip_yaw_hoint.range = [-2.7576, 2.7576]
            hip_yaw_hoint.armature = 0.01017752004
            
            # Thigh geometry - capsule
            thigh_geom = hip_yaw_link.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE)
            thigh_geom.size = [param.thigh_radius, 0, 0]
            thigh_geom.fromto = [0, 0, 0, 0, 0, -param.thigh_length]
            thigh_geom.rgba = [0.45, 0.45, 0.85, 1.0]
            
            # --- KNEE JOINT (pitch only) ---
            knee_link = hip_yaw_link.add_body(name=f"{side}_knee_link")
            knee_link.pos = [0, 0, -(param.thigh_length + param.thigh_radius + param.shin_radius)]  # Tangent to thigh
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
            knee_joint.axis = [0, 1, 0]  # Y-axis (pitch)
            knee_joint.range = [-0.087267, 2.8798]
            knee_joint.armature = 0.025101925
            
            # Calf geometry - capsule
            knee_link_geom = knee_link.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE)
            knee_link_geom.name = f"{side}_knee_link"
            knee_link_geom.size = [param.shin_radius, 0, 0]
            knee_link_geom.fromto = [0, 0, 0, 0, 0, -param.shin_length]
            knee_link_geom.rgba = [0.85, 0.75, 0.35, 1.0]
            
            # --- ANKLE JOINT 1: PITCH ---
            ankle_pitch_radius = min(param.foot_width, param.foot_height) / 2

            ankle_pitch_link = knee_link.add_body(name=f"{side}_ankle_pitch_link")
            ankle_pitch_link.pos = [0, 0, -(param.shin_length + param.shin_radius + ankle_pitch_radius)]  # Tangent to knee_link
            ankle_pitch_link.mass = 0.3  # Small intermediate mass
            ankle_pitch_link.inertia = [0.005, 0.005, 0.005]

            ankle_pitch_geom = ankle_pitch_link.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE)
            ankle_pitch_geom.size = [ankle_pitch_radius, 0, 0]
            ankle_pitch_geom.rgba = [0.55, 0.85, 0.85, 1.0]
            
            ankle_pitch_joint = ankle_pitch_link.add_joint(
                name=f"{side}_ankle_pitch_joint",
                type=mujoco.mjtJoint.mjJNT_HINGE
            )
            ankle_pitch_joint.axis = [0, 1, 0]  # Y-axis (pitch)
            ankle_pitch_joint.range = [-0.87267, 0.5236]
            ankle_pitch_joint.armature = 0.00721945
            
            # --- ANKLE JOINT 2: ROLL (FOOT) ---
            ankle_roll_link = ankle_pitch_link.add_body(name=f"{side}_ankle_roll_link")
            ankle_roll_link.pos = [0, 0, -(ankle_pitch_radius + param.foot_height/2)]  # Tangent to ankle_pitch_link
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
            ankle_roll_joint.axis = [1, 0, 0]  # X-axis (roll)
            ankle_roll_joint.range = [-0.2618, 0.2618]
            ankle_roll_joint.armature = 0.00721945
            
            # Foot geometry - box extending forward
            foot_geom = ankle_roll_link.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX)
            foot_geom.size = [param.foot_length/2, param.foot_width/2, param.foot_height/2]
            foot_geom.pos = [param.foot_length/4, 0, -param.foot_height/2]
            foot_geom.rgba = [0.85, 0.55, 0.85, 1.0]
        
        return spec


if __name__ == "__main__":
    builder = BipedBuilder()
    param = builder.sample_params(seed=0)
    print(param)
    spec = builder.generate_mjspec(param)
    print("Generated biped MjSpec successfully!")
