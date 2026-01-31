import mujoco
import numpy as np
import random

from typing import NamedTuple, Callable
from metamorphosis.builder.base import BuilderBase
from metamorphosis.utils.mjs_utils import add_capsule_geom_


class BipedParam(NamedTuple):
    """Parameters for a bipedal robot with tall box pelvis and 3-segment capsule hips."""
    # Pelvis (tall rectangular body)
    pelvis_length: float     # 长方体骨盆长度 (X方向)
    pelvis_width: float      # 长方体骨盆宽度 (Y方向)
    pelvis_height: float     # 长方体骨盆高度 (Z方向，要足够高)
    pelvis_mass: float       # 骨盆质量
    
    # Hip positioning
    hip_offset_x: float      # 髋关节前后偏移
    hip_offset_y: float      # 髋关节左右间距的一半
    hip_offset_z: float      # 髋关节垂直偏移（相对骨盆底部）
    
    # Hip capsule segments (三段长胶囊首尾相连)
    hip_capsule_radius: float # 髋关节胶囊半径
    hip_capsule_length: float # 每个髋关节胶囊长度
    hip_capsule_mass: float   # 每个髋关节胶囊质量
    
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
        pelvis_length_range: tuple[float, float] = (0.12, 0.20),
        pelvis_width_range: tuple[float, float] = (0.18, 0.28),
        pelvis_height_range: tuple[float, float] = (0.25, 0.40),  # 更高的pelvis
        hip_offset_range: tuple[float, float] = (0.08, 0.15),   # Hip偏移范围
        hip_capsule_length_range: tuple[float, float] = (0.08, 0.12),  # 长胶囊
        hip_capsule_radius_range: tuple[float, float] = (0.025, 0.040), 
        leg_length_range: tuple[float, float] = (0.5, 0.7),
        shin_ratio_range: tuple[float, float] = (0.85, 1.15),
        valid_filter: Callable[[BipedParam], bool] = lambda _: True,
    ):
        super().__init__()
        self.pelvis_length_range = pelvis_length_range
        self.pelvis_width_range = pelvis_width_range
        self.pelvis_height_range = pelvis_height_range
        self.hip_offset_range = hip_offset_range
        self.hip_capsule_length_range = hip_capsule_length_range
        self.hip_capsule_radius_range = hip_capsule_radius_range
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
            
            # Hip spacing and capsule dimensions
            hip_spacing = random.uniform(*self.hip_spacing_range)
            hip_capsule_radius = random.uniform(*self.hip_capsule_range)
            hip_capsule_length = random.uniform(*self.hip_capsule_range)
            hip_mass = random.uniform(0.6, 1.2)  # Each hip capsule
            
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
            
    def sample_params(self, seed: int = -1) -> BipedParam:
        if seed >= 0:
            np.random.seed(seed)
            random.seed(seed)
        
        for _ in range(10):
            # Pelvis dimensions (tall rectangular body)
            pelvis_length = random.uniform(*self.pelvis_length_range)
            pelvis_width = random.uniform(*self.pelvis_width_range)
            pelvis_height = random.uniform(*self.pelvis_height_range)  # 足够高
            pelvis_mass = random.uniform(4.0, 7.0)  # 基于尺寸
            
            # Hip positioning
            hip_offset_x = random.uniform(0.0, 0.02)  # 前后偏移
            hip_offset_y = random.uniform(*self.hip_offset_range)  # 左右间距的一半
            hip_offset_z = -pelvis_height / 2 + random.uniform(0.02, 0.08)  # 在pelvis底部附近
            
            # Hip capsule segments (三段长胶囊)
            hip_capsule_radius = random.uniform(*self.hip_capsule_radius_range)
            hip_capsule_length = random.uniform(*self.hip_capsule_length_range)  # 长胶囊
            hip_capsule_mass = random.uniform(0.8, 1.5)  # 每段胶囊
            
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
                hip_offset_x=hip_offset_x,
                hip_offset_y=hip_offset_y,
                hip_offset_z=hip_offset_z,
                hip_capsule_radius=hip_capsule_radius,
                hip_capsule_length=hip_capsule_length,
                hip_capsule_mass=hip_capsule_mass,
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
        
        # ============ PELVIS (root body - tall rectangular box) ============
        pelvis = spec.worldbody.add_body()
        pelvis.name = "pelvis"
        pelvis.mass = param.pelvis_mass
        # Box inertia
        ixx = param.pelvis_mass * (param.pelvis_width**2 + param.pelvis_height**2) / 12
        iyy = param.pelvis_mass * (param.pelvis_length**2 + param.pelvis_height**2) / 12
        izz = param.pelvis_mass * (param.pelvis_length**2 + param.pelvis_width**2) / 12
        pelvis.inertia = [ixx, iyy, izz]
        
        # Tall pelvis geometry
        pelvis_geom = pelvis.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX)
        pelvis_geom.size = [param.pelvis_length / 2, param.pelvis_width / 2, param.pelvis_height / 2]
        
        # ============ LEGS (6 DOF each: 3段胶囊hip + knee + 2段ankle) ============
        for side, y_sign in [("left", 1), ("right", -1)]:
            
            # Hip 起始位置
            hip_pos = [param.hip_offset_x, y_sign * param.hip_offset_y, param.hip_offset_z]
            
            # --- Hip Segment 1: PITCH (第一段长胶囊, Y轴旋转) ---
            hip_pitch = pelvis.add_body(name=f"{side}_hip_pitch")
            hip_pitch.pos = hip_pos
            hip_pitch.mass = param.hip_capsule_mass
            # 胶囊惯性
            capsule_ixx = param.hip_capsule_mass * (3 * param.hip_capsule_radius**2 + param.hip_capsule_length**2) / 12
            capsule_iyy = param.hip_capsule_mass * param.hip_capsule_radius**2 / 2
            hip_pitch.inertia = [capsule_ixx, capsule_iyy, capsule_ixx]
            
            hip_pitch_joint = hip_pitch.add_joint(
                name=f"{side}_hip_pitch_joint",
                type=mujoco.mjtJoint.mjJNT_HINGE
            )
            hip_pitch_joint.axis = [0, 1, 0]  # Y-axis (pitch)
            hip_pitch_joint.range = [-2.5307, 2.8798]
            hip_pitch_joint.armature = 0.01017752004
            
            # 第一段胶囊几何体 (沿Z轴向下)
            hip_pitch_geom = hip_pitch.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE)
            hip_pitch_geom.size = [param.hip_capsule_radius, 0, 0]
            hip_pitch_geom.fromto = [0, 0, 0, 0, 0, -param.hip_capsule_length]
            
            # --- Hip Segment 2: ROLL (第二段长胶囊, X轴旋转) ---
            hip_roll = hip_pitch.add_body(name=f"{side}_hip_roll")
            hip_roll.pos = [0, 0, -param.hip_capsule_length]  # 连接到第一段末端
            hip_roll.mass = param.hip_capsule_mass
            hip_roll.inertia = [capsule_ixx, capsule_iyy, capsule_ixx]
            
            hip_roll_joint = hip_roll.add_joint(
                name=f"{side}_hip_roll_joint",
                type=mujoco.mjtJoint.mjJNT_HINGE
            )
            hip_roll_joint.axis = [1, 0, 0]  # X-axis (roll)
            # 左右腿不同范围
            if side == "left":
                hip_roll_joint.range = [-0.5236, 2.9671]
            else:
                hip_roll_joint.range = [-2.9671, 0.5236]
            hip_roll_joint.armature = 0.025101925
            
            # 第二段胶囊几何体 (沿Z轴向下)
            hip_roll_geom = hip_roll.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE)
            hip_roll_geom.size = [param.hip_capsule_radius, 0, 0]
            hip_roll_geom.fromto = [0, 0, 0, 0, 0, -param.hip_capsule_length]
            
            # --- Hip Segment 3: YAW (第三段长胶囊, Z轴旋转) ---
            hip_yaw = hip_roll.add_body(name=f"{side}_hip_yaw")
            hip_yaw.pos = [0, 0, -param.hip_capsule_length]  # 连接到第二段末端
            hip_yaw.mass = param.hip_capsule_mass
            hip_yaw.inertia = [capsule_ixx, capsule_iyy, capsule_ixx]
            
            hip_yaw_joint = hip_yaw.add_joint(
                name=f"{side}_hip_yaw_joint",
                type=mujoco.mjtJoint.mjJNT_HINGE
            )
            hip_yaw_joint.axis = [0, 0, 1]  # Z-axis (yaw)
            hip_yaw_joint.range = [-2.7576, 2.7576]
            hip_yaw_joint.armature = 0.01017752004
            
            # 第三段胶囊几何体 (沿Z轴向下)
            hip_yaw_geom = hip_yaw.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE)
            hip_yaw_geom.size = [param.hip_capsule_radius, 0, 0]
            hip_yaw_geom.fromto = [0, 0, 0, 0, 0, -param.hip_capsule_length]
            
            # --- THIGH (连接到第三段hip末端) ---
            thigh = hip_yaw.add_body(name=f"{side}_thigh")
            thigh.pos = [0, 0, -param.hip_capsule_length]  # 连接到第三段末端
            thigh.mass = param.thigh_mass
            # 大腿惯性
            thigh_ixx = param.thigh_mass * (3 * param.thigh_radius**2 + param.thigh_length**2) / 12
            thigh_iyy = param.thigh_mass * param.thigh_radius**2 / 2
            thigh.inertia = [thigh_ixx, thigh_iyy, thigh_ixx]
            
            # 大腿几何体
            thigh_geom = thigh.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE)
            thigh_geom.size = [param.thigh_radius, 0, 0]
            thigh_geom.fromto = [0, 0, 0, 0, 0, -param.thigh_length]
            
            # --- KNEE JOINT (膝关节 - pitch only) ---
            knee = thigh.add_body(name=f"{side}_knee")
            knee.pos = [0, 0, -param.thigh_length]
            knee.mass = param.shin_mass
            # 小腿惯性
            shin_ixx = param.shin_mass * (3 * param.shin_radius**2 + param.shin_length**2) / 12
            shin_iyy = param.shin_mass * param.shin_radius**2 / 2
            knee.inertia = [shin_ixx, shin_iyy, shin_ixx]
            
            knee_joint = knee.add_joint(
                name=f"{side}_knee_joint",
                type=mujoco.mjtJoint.mjJNT_HINGE
            )
            knee_joint.axis = [0, 1, 0]  # Y-axis (pitch)
            knee_joint.range = [-0.087267, 2.8798]
            knee_joint.armature = 0.025101925
            
            # 小腿几何体
            shin_geom = knee.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE)
            shin_geom.size = [param.shin_radius, 0, 0]
            shin_geom.fromto = [0, 0, 0, 0, 0, -param.shin_length]
            
            # --- ANKLE JOINT 1: PITCH ---
            ankle_pitch = knee.add_body(name=f"{side}_ankle_pitch")
            ankle_pitch.pos = [0, 0, -param.shin_length]
            ankle_pitch.mass = 0.3  # 小的中间质量
            ankle_pitch.inertia = [0.005, 0.005, 0.005]
            
            ankle_pitch_joint = ankle_pitch.add_joint(
                name=f"{side}_ankle_pitch_joint",
                type=mujoco.mjtJoint.mjJNT_HINGE
            )
            ankle_pitch_joint.axis = [0, 1, 0]  # Y-axis (pitch)
            ankle_pitch_joint.range = [-0.87267, 0.5236]
            ankle_pitch_joint.armature = 0.00721945
            
            # --- ANKLE JOINT 2: ROLL (FOOT) ---
            foot = ankle_pitch.add_body(name=f"{side}_foot")
            foot.pos = [0, 0, 0]  # 同位置
            foot.mass = param.foot_mass
            # 脚惯性
            foot_ixx = param.foot_mass * (param.foot_width**2 + param.foot_height**2) / 12
            foot_iyy = param.foot_mass * (param.foot_length**2 + param.foot_height**2) / 12
            foot_izz = param.foot_mass * (param.foot_length**2 + param.foot_width**2) / 12
            foot.inertia = [foot_ixx, foot_iyy, foot_izz]
            
            ankle_roll_joint = foot.add_joint(
                name=f"{side}_ankle_roll_joint",
                type=mujoco.mjtJoint.mjJNT_HINGE
            )
            ankle_roll_joint.axis = [1, 0, 0]  # X-axis (roll)
            ankle_roll_joint.range = [-0.2618, 0.2618]
            ankle_roll_joint.armature = 0.00721945
            
            # 脚几何体 - 向前延伸的盒子
            foot_geom = foot.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX)
            foot_geom.size = [param.foot_length/2, param.foot_width/2, param.foot_height/2]
            foot_geom.pos = [param.foot_length/4, 0, -param.foot_height/2]
        
        return spec
        # Box inertia
        ixx = param.pelvis_mass * (param.pelvis_width**2 + param.pelvis_height**2) / 12
        iyy = param.pelvis_mass * (param.pelvis_length**2 + param.pelvis_height**2) / 12
        izz = param.pelvis_mass * (param.pelvis_length**2 + param.pelvis_width**2) / 12
        pelvis.inertia = [ixx, iyy, izz]
        
        # Pelvis geometry - box
        pelvis_geom = pelvis.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX)
        pelvis_geom.size = [param.pelvis_length/2, param.pelvis_width/2, param.pelvis_height/2]
        
        # ============ WAIST CONNECTION ============
        # Small cylindrical waist below pelvis
        waist = pelvis.add_body(name="waist")
        waist.pos = [0, 0, -(param.pelvis_height/2 + param.waist_height/2)]
        waist.mass = 0.5  # Small mass for connection
        waist.inertia = [0.01, 0.01, 0.01]
        
        # Waist geometry - small cylinder
        waist_geom = waist.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER)
        waist_geom.size = [param.waist_radius, param.waist_height/2, 0]
        
        # ============ LEFT AND RIGHT LEGS (6 DOF each) ============
        for side, y_sign in [("left", 1), ("right", -1)]:
            
            # --- Hip Joint 1: PITCH (rotation around Y-axis) - First Capsule ---
            hip_pitch_pos = [0, y_sign * param.hip_spacing/2, -(param.waist_height/2)]
            
            hip_pitch = waist.add_body(name=f"{side}_hip_pitch")
            hip_pitch.pos = hip_pitch_pos
            hip_pitch.mass = param.hip_mass
            # Capsule inertia (simplified)
            cap_ixx = param.hip_mass * (3 * param.hip_capsule_radius**2 + param.hip_capsule_length**2) / 12
            cap_iyy = param.hip_mass * param.hip_capsule_radius**2 / 2
            cap_izz = cap_ixx
            hip_pitch.inertia = [cap_ixx, cap_iyy, cap_izz]
            
            hip_pitch_joint = hip_pitch.add_joint(
                name=f"{side}_hip_pitch_joint",
                type=mujoco.mjtJoint.mjJNT_HINGE
            )
            hip_pitch_joint.axis = [0, 1, 0]  # Y-axis (pitch)
            hip_pitch_joint.range = [-2.5307, 2.8798]
            hip_pitch_joint.armature = 0.01017752004
            
            # Hip pitch capsule geometry (oriented along X for pitch movement)
            hip_pitch_geom = hip_pitch.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE)
            hip_pitch_geom.size = [param.hip_capsule_radius, 0, 0]
            hip_pitch_geom.fromto = [-param.hip_capsule_length/2, 0, 0, param.hip_capsule_length/2, 0, 0]
            
            # --- Hip Joint 2: ROLL (rotation around X-axis) - Second Capsule ---
            hip_roll = hip_pitch.add_body(name=f"{side}_hip_roll")
            hip_roll.pos = [0, 0, -param.hip_capsule_length/2]  # Offset downward
            hip_roll.mass = param.hip_mass
            hip_roll.inertia = [cap_ixx, cap_iyy, cap_izz]
            
            hip_roll_joint = hip_roll.add_joint(
                name=f"{side}_hip_roll_joint",
                type=mujoco.mjtJoint.mjJNT_HINGE
            )
            hip_roll_joint.axis = [1, 0, 0]  # X-axis (roll)
            # Different ranges for left/right legs
            if side == "left":
                hip_roll_joint.range = [-0.5236, 2.9671]
            else:
                hip_roll_joint.range = [-2.9671, 0.5236]
            hip_roll_joint.armature = 0.025101925
            
            # Hip roll capsule geometry (oriented along Y for roll movement)
            hip_roll_geom = hip_roll.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE)
            hip_roll_geom.size = [param.hip_capsule_radius, 0, 0]
            hip_roll_geom.fromto = [0, -param.hip_capsule_length/2, 0, 0, param.hip_capsule_length/2, 0]
            
            # --- Hip Joint 3: YAW (rotation around Z-axis) - Third Capsule ---
            hip_yaw = hip_roll.add_body(name=f"{side}_hip_yaw")
            hip_yaw.pos = [0, 0, -param.hip_capsule_length/2]  # Offset downward
            hip_yaw.mass = param.hip_mass  
            hip_yaw.inertia = [cap_ixx, cap_iyy, cap_izz]
            
            hip_yaw_joint = hip_yaw.add_joint(
                name=f"{side}_hip_yaw_joint",
                type=mujoco.mjtJoint.mjJNT_HINGE
            )
            hip_yaw_joint.axis = [0, 0, 1]  # Z-axis (yaw)
            hip_yaw_joint.range = [-2.7576, 2.7576]
            hip_yaw_joint.armature = 0.01017752004
            
            # Hip yaw capsule geometry (oriented along Z for yaw movement)
            hip_yaw_geom = hip_yaw.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE)
            hip_yaw_geom.size = [param.hip_capsule_radius, 0, 0]
            hip_yaw_geom.fromto = [0, 0, -param.hip_capsule_length/2, 0, 0, param.hip_capsule_length/2]
            
            # --- THIGH (connected to hip yaw) ---
            thigh = hip_yaw.add_body(name=f"{side}_thigh")
            thigh.pos = [0, 0, -(param.hip_capsule_length/2)]  # Below hip yaw
            thigh.mass = param.thigh_mass
            # Cylinder inertia
            thigh_ixx = param.thigh_mass * (3 * param.thigh_radius**2 + param.thigh_length**2) / 12
            thigh_iyy = param.thigh_mass * param.thigh_radius**2 / 2
            thigh_izz = thigh_ixx
            thigh.inertia = [thigh_ixx, thigh_iyy, thigh_izz]
            
            # Thigh geometry - capsule
            thigh_geom = thigh.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE)
            thigh_geom.size = [param.thigh_radius, 0, 0]
            thigh_geom.fromto = [0, 0, 0, 0, 0, -param.thigh_length]
            
            # --- KNEE JOINT (pitch only) ---
            knee = thigh.add_body(name=f"{side}_knee")
            knee.pos = [0, 0, -param.thigh_length]
            knee.mass = param.shin_mass
            # Shin inertia
            shin_ixx = param.shin_mass * (3 * param.shin_radius**2 + param.shin_length**2) / 12
            shin_iyy = param.shin_mass * param.shin_radius**2 / 2
            shin_izz = shin_ixx
            knee.inertia = [shin_ixx, shin_iyy, shin_izz]
            
            knee_joint = knee.add_joint(
                name=f"{side}_knee_joint",
                type=mujoco.mjtJoint.mjJNT_HINGE
            )
            knee_joint.axis = [0, 1, 0]  # Y-axis (pitch)
            knee_joint.range = [-0.087267, 2.8798]
            knee_joint.armature = 0.025101925
            
            # Shin geometry - capsule
            shin_geom = knee.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE)
            shin_geom.size = [param.shin_radius, 0, 0]
            shin_geom.fromto = [0, 0, 0, 0, 0, -param.shin_length]
            
            # --- ANKLE JOINT 1: PITCH ---
            ankle_pitch = knee.add_body(name=f"{side}_ankle_pitch")
            ankle_pitch.pos = [0, 0, -param.shin_length]
            ankle_pitch.mass = 0.3  # Small intermediate mass
            ankle_pitch.inertia = [0.005, 0.005, 0.005]
            
            ankle_pitch_joint = ankle_pitch.add_joint(
                name=f"{side}_ankle_pitch_joint",
                type=mujoco.mjtJoint.mjJNT_HINGE
            )
            ankle_pitch_joint.axis = [0, 1, 0]  # Y-axis (pitch)
            ankle_pitch_joint.range = [-0.87267, 0.5236]
            ankle_pitch_joint.armature = 0.00721945
            
            # --- ANKLE JOINT 2: ROLL (FOOT) ---
            foot = ankle_pitch.add_body(name=f"{side}_foot")
            foot.pos = [0, 0, 0]  # Same position
            foot.mass = param.foot_mass
            # Foot inertia
            foot_ixx = param.foot_mass * (param.foot_width**2 + param.foot_height**2) / 12
            foot_iyy = param.foot_mass * (param.foot_length**2 + param.foot_height**2) / 12
            foot_izz = param.foot_mass * (param.foot_length**2 + param.foot_width**2) / 12
            foot.inertia = [foot_ixx, foot_iyy, foot_izz]
            
            ankle_roll_joint = foot.add_joint(
                name=f"{side}_ankle_roll_joint",
                type=mujoco.mjtJoint.mjJNT_HINGE
            )
            ankle_roll_joint.axis = [1, 0, 0]  # X-axis (roll)
            ankle_roll_joint.range = [-0.2618, 0.2618]
            ankle_roll_joint.armature = 0.00721945
            
            # Foot geometry - box extending forward
            foot_geom = foot.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX)
            foot_geom.size = [param.foot_length/2, param.foot_width/2, param.foot_height/2]
            foot_geom.pos = [param.foot_length/4, 0, -param.foot_height/2]
        
        return spec
        pelvis.inertia = [pelvis_inertia, pelvis_inertia, pelvis_inertia]
        
        # Pelvis geometry - sphere
        pelvis_geom = pelvis.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE)
        pelvis_geom.size = [param.pelvis_radius, 0, 0]
        
        # ============ WAIST CONNECTION ============
        # Small cylindrical waist below pelvis
        waist = pelvis.add_body(name="waist")
        waist.pos = [0, 0, -(param.pelvis_radius + param.waist_height/2)]
        waist.mass = 0.5  # Small mass for connection
        waist.inertia = [0.01, 0.01, 0.01]
        
        # Waist geometry - small cylinder
        waist_geom = waist.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER)
        waist_geom.size = [param.waist_radius, param.waist_height/2, 0]
        
        # ============ LEFT AND RIGHT LEGS (6 DOF each) ============
        for side, y_sign in [("left", 1), ("right", -1)]:
            
            # --- Hip Joint 1: PITCH (rotation around Y-axis) ---
            hip_pitch_pos = [0, y_sign * param.hip_spacing/2, -(param.waist_height/2)]
            
            hip_pitch = waist.add_body(name=f"{side}_hip_pitch")
            hip_pitch.pos = hip_pitch_pos
            hip_pitch.mass = param.hip_mass
            # Ellipsoid inertia (simplified)
            hip_ixx = param.hip_mass * (param.hip_width**2 + param.hip_height**2) / 5
            hip_iyy = param.hip_mass * (param.hip_length**2 + param.hip_height**2) / 5  
            hip_izz = param.hip_mass * (param.hip_length**2 + param.hip_width**2) / 5
            hip_pitch.inertia = [hip_ixx, hip_iyy, hip_izz]
            
            hip_pitch_joint = hip_pitch.add_joint(
                name=f"{side}_hip_pitch_joint",
                type=mujoco.mjtJoint.mjJNT_HINGE
            )
            hip_pitch_joint.axis = [0, 1, 0]  # Y-axis (pitch)
            hip_pitch_joint.range = [-2.5307, 2.8798]
            hip_pitch_joint.armature = 0.01017752004
            
            # Hip ellipsoid geometry -> use capsule instead (USD compatible)
            hip_pitch_geom = hip_pitch.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE)
            hip_pitch_geom.size = [param.hip_width/4, 0, 0]  # Use width as radius
            hip_pitch_geom.fromto = [-param.hip_length/2, 0, 0, param.hip_length/2, 0, 0]  # Length along X
            
            # --- Hip Joint 2: ROLL (rotation around X-axis) ---
            hip_roll = hip_pitch.add_body(name=f"{side}_hip_roll")
            hip_roll.pos = [0, 0, 0]  # Same position as pitch
            hip_roll.mass = param.hip_mass
            hip_roll.inertia = [hip_ixx, hip_iyy, hip_izz]
            
            hip_roll_joint = hip_roll.add_joint(
                name=f"{side}_hip_roll_joint",
                type=mujoco.mjtJoint.mjJNT_HINGE
            )
            hip_roll_joint.axis = [1, 0, 0]  # X-axis (roll)
            # Different ranges for left/right legs
            if side == "left":
                hip_roll_joint.range = [-0.5236, 2.9671]
            else:
                hip_roll_joint.range = [-2.9671, 0.5236]
            hip_roll_joint.armature = 0.025101925
            
            # Hip ellipsoid geometry -> use capsule instead (USD compatible)
            hip_roll_geom = hip_roll.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE)
            hip_roll_geom.size = [param.hip_width/4, 0, 0]  # Use width as radius
            hip_roll_geom.fromto = [0, -param.hip_length/2, 0, 0, param.hip_length/2, 0]  # Length along Y
            
            # --- Hip Joint 3: YAW (rotation around Z-axis) ---
            hip_yaw = hip_roll.add_body(name=f"{side}_hip_yaw")
            hip_yaw.pos = [0, 0, 0]  # Same position
            hip_yaw.mass = param.hip_mass  
            hip_yaw.inertia = [hip_ixx, hip_iyy, hip_izz]
            
            hip_yaw_joint = hip_yaw.add_joint(
                name=f"{side}_hip_yaw_joint",
                type=mujoco.mjtJoint.mjJNT_HINGE
            )
            hip_yaw_joint.axis = [0, 0, 1]  # Z-axis (yaw)
            hip_yaw_joint.range = [-2.7576, 2.7576]
            hip_yaw_joint.armature = 0.01017752004
            
            # Hip ellipsoid geometry -> use capsule instead (USD compatible)  
            hip_yaw_geom = hip_yaw.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE)
            hip_yaw_geom.size = [param.hip_width/4, 0, 0]  # Use width as radius
            hip_yaw_geom.fromto = [0, 0, -param.hip_height/2, 0, 0, param.hip_height/2]  # Length along Z
            
            # --- THIGH (connected to hip yaw) ---
            thigh = hip_yaw.add_body(name=f"{side}_thigh")
            thigh.pos = [0, 0, -(param.hip_height/2)]  # Below hip
            thigh.mass = param.thigh_mass
            # Cylinder inertia
            thigh_ixx = param.thigh_mass * (3 * param.thigh_radius**2 + param.thigh_length**2) / 12
            thigh_iyy = param.thigh_mass * param.thigh_radius**2 / 2
            thigh_izz = thigh_ixx
            thigh.inertia = [thigh_ixx, thigh_iyy, thigh_izz]
            
            # Thigh geometry - capsule
            thigh_geom = thigh.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE)
            thigh_geom.size = [param.thigh_radius, 0, 0]
            thigh_geom.fromto = [0, 0, 0, 0, 0, -param.thigh_length]
            
            # --- KNEE JOINT (pitch only) ---
            knee = thigh.add_body(name=f"{side}_knee")
            knee.pos = [0, 0, -param.thigh_length]
            knee.mass = param.shin_mass
            # Shin inertia
            shin_ixx = param.shin_mass * (3 * param.shin_radius**2 + param.shin_length**2) / 12
            shin_iyy = param.shin_mass * param.shin_radius**2 / 2
            shin_izz = shin_ixx
            knee.inertia = [shin_ixx, shin_iyy, shin_izz]
            
            knee_joint = knee.add_joint(
                name=f"{side}_knee_joint",
                type=mujoco.mjtJoint.mjJNT_HINGE
            )
            knee_joint.axis = [0, 1, 0]  # Y-axis (pitch)
            knee_joint.range = [-0.087267, 2.8798]
            knee_joint.armature = 0.025101925
            
            # Shin geometry - capsule
            shin_geom = knee.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE)
            shin_geom.size = [param.shin_radius, 0, 0]
            shin_geom.fromto = [0, 0, 0, 0, 0, -param.shin_length]
            
            # --- ANKLE JOINT 1: PITCH ---
            ankle_pitch = knee.add_body(name=f"{side}_ankle_pitch")
            ankle_pitch.pos = [0, 0, -param.shin_length]
            ankle_pitch.mass = 0.3  # Small intermediate mass
            ankle_pitch.inertia = [0.005, 0.005, 0.005]
            
            ankle_pitch_joint = ankle_pitch.add_joint(
                name=f"{side}_ankle_pitch_joint",
                type=mujoco.mjtJoint.mjJNT_HINGE
            )
            ankle_pitch_joint.axis = [0, 1, 0]  # Y-axis (pitch)
            ankle_pitch_joint.range = [-0.87267, 0.5236]
            ankle_pitch_joint.armature = 0.00721945
            
            # --- ANKLE JOINT 2: ROLL (FOOT) ---
            foot = ankle_pitch.add_body(name=f"{side}_foot")
            foot.pos = [0, 0, 0]  # Same position
            foot.mass = param.foot_mass
            # Foot inertia
            foot_ixx = param.foot_mass * (param.foot_width**2 + param.foot_height**2) / 12
            foot_iyy = param.foot_mass * (param.foot_length**2 + param.foot_height**2) / 12
            foot_izz = param.foot_mass * (param.foot_length**2 + param.foot_width**2) / 12
            foot.inertia = [foot_ixx, foot_iyy, foot_izz]
            
            ankle_roll_joint = foot.add_joint(
                name=f"{side}_ankle_roll_joint",
                type=mujoco.mjtJoint.mjJNT_HINGE
            )
            ankle_roll_joint.axis = [1, 0, 0]  # X-axis (roll)
            ankle_roll_joint.range = [-0.2618, 0.2618]
            ankle_roll_joint.armature = 0.00721945
            
            # Foot geometry - box extending forward
            foot_geom = foot.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX)
            foot_geom.size = [param.foot_length/2, param.foot_width/2, param.foot_height/2]
            foot_geom.pos = [param.foot_length/4, 0, -param.foot_height/2]
        
        return spec


if __name__ == "__main__":
    builder = BipedBuilder()
    param = builder.sample_params(seed=0)
    print(param)
    spec = builder.generate_mjspec(param)
    print("Generated biped MjSpec successfully!")
