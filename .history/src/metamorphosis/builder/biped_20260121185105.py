import mujoco
import numpy as np
import random

from typing import NamedTuple, Callable
from metamorphosis.builder.base import BuilderBase
from metamorphosis.utils.mjs_utils import add_capsule_geom_


class BipedParam(NamedTuple):
    torso_height: float
    torso_width: float
    torso_depth: float
    thigh_length: float
    calf_length: float
    thigh_radius: float
    foot_length: float


class BipedBuilder(BuilderBase):
    """
    针对双足机器人的随机生成器。
    结构包含：1个躯干 (Torso)，2条腿。
    每条腿包含：髋关节 (Hip, 2-DOF), 大腿 (Thigh), 小腿 (Calf), 足部 (Foot)。
    """
    
    def __init__(
        self,
        torso_height_range: tuple[float, float] = (0.4, 0.6),
        torso_width_range: tuple[float, float] = (0.2, 0.3),
        leg_length_total_range: tuple[float, float] = (0.6, 1.0), # 腿部总长范围
        calf_length_ratio: tuple[float, float] = (0.8, 1.0),      # 小腿相对于大腿的比例
        valid_filter: Callable[[BipedParam], bool] = lambda _: True,
    ):
        super().__init__()
        self.torso_height_range = torso_height_range
        self.torso_width_range = torso_width_range
        self.leg_length_total_range = leg_length_total_range
        self.calf_length_ratio = calf_length_ratio
        self.valid_filter = valid_filter
    
    def sample_params(self, seed: int = -1) -> BipedParam:
        if seed >= 0:
            np.random.seed(seed)
            random.seed(seed)
        
        for _ in range(10):
            torso_height = random.uniform(*self.torso_height_range)
            torso_width = random.uniform(*self.torso_width_range)
            torso_depth = torso_width * 0.6
            
            total_leg_len = random.uniform(*self.leg_length_total_range)
            ratio = random.uniform(*self.calf_length_ratio)
            # thigh + thigh * ratio = total_leg_len
            thigh_length = total_leg_len / (1 + ratio)
            calf_length = thigh_length * ratio
            
            thigh_radius = random.uniform(0.03, 0.045)
            foot_length = thigh_length * 0.3

            param = BipedParam(
                torso_height=torso_height,
                torso_width=torso_width,
                torso_depth=torso_depth,
                thigh_length=thigh_length,
                calf_length=calf_length,
                thigh_radius=thigh_radius,
                foot_length=foot_length
            )
            if self.valid_filter(param):
                break
        else:
            raise ValueError("Failed to sample valid parameters")
        return param
    
    def generate_mjspec(self, param: BipedParam) -> mujoco.MjSpec:
        thigh_radius = param.thigh_radius
        calf_radius = param.thigh_radius * 0.8
        
        spec = mujoco.MjSpec()
        # 躯干 (Torso)
        torso_body = spec.worldbody.add_body(name="torso")
        torso_body.mass = 2.0
        torso_geom = torso_body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX)
        torso_geom.size = [param.torso_depth/2, param.torso_width/2, param.torso_height/2]
        
        # 左右两条腿
        for name, side_sign in [("L", -1), ("R", 1)]:
            # 髋部节点 (Hip - Roll/Abduction)
            hip_body = torso_body.add_body(name=f"{name}_hip")
            hip_body.pos = [0, (param.torso_width/2) * side_sign, -param.torso_height/2]
            
            # 髋关节1: Roll (前后摆动方向的垂直轴)
            hip_roll = hip_body.add_joint(name=f"{name}_hip_roll", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[1, 0, 0])
            hip_roll.range = [-0.5, 0.5]
            
            # 大腿 (Thigh - Pitch)
            thigh_body = hip_body.add_body(name=f"{name}_thigh")
            thigh_body.pos = [0, 0, 0]
            thigh_pitch = thigh_body.add_joint(name=f"{name}_thigh_pitch", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 1, 0])
            thigh_pitch.range = [-1.5, 1.5]
            add_capsule_geom_(thigh_body, radius=thigh_radius, fromto=[0, 0, 0, 0, 0, -param.thigh_length])
            
            # 小腿 (Calf)
            calf_body = thigh_body.add_body(name=f"{name}_calf")
            calf_body.pos = [0, 0, -param.thigh_length]
            knee_joint = calf_body.add_joint(name=f"{name}_knee", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 1, 0])
            knee_joint.range = [0, 2.0] # 典型的膝盖弯曲范围
            add_capsule_geom_(calf_body, radius=calf_radius, fromto=[0, 0, 0, 0, 0, -param.calf_length])
            
            # 足部 (Foot)
            foot_body = calf_body.add_body(name=f"{name}_foot")
            foot_body.pos = [0, 0, -param.calf_length]
            # 这里简化为一个长方体足部，增加稳定性
            foot_geom = foot_body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX)
            foot_geom.size = [param.foot_length/2, thigh_radius * 1.2, 0.02]
            foot_geom.pos = [param.foot_length/4, 0, -0.01] # 脚掌向前延伸
            
        return spec


if __name__ == "__main__":
    builder = BipedBuilder()
    param = builder.sample_params()
    print("Generated Biped Params:", param)
    spec = builder.generate_mjspec(param)
    # model = spec.compile() # 如果需要编译测试