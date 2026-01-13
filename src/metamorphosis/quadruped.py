import mujoco
import numpy as np
import random

from pxr import Usd, UsdGeom, UsdPhysics, Gf
from typing import NamedTuple, Optional
from .usd_utils import (
    add_default_transform_,
    create_capsule,
    create_revolute_joint,
    create_fixed_joint,
)


def add_leg_(
    parent_body: mujoco.MjsBody,
    name: str,
    pos: list[float],
    radius: float,
    length: float,
):
    body = parent_body.add_body(name=name, pos=pos)
    body.mass = 1.0
    body.inertia = [1.0, 1.0, 1.0]
    geom = body.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE)
    geom.size = [radius, 0, 0.]
    geom.fromto = [0, 0, 0, 0, 0, -length]
    return body


class QuadrupedParams(NamedTuple):
    base_dimensions: list[float]
    leg_lengths: list[float]
    has_hock_joint: list[bool]


class QuadrupedBuilder:
    _instance = None
    _initialized = False

    def __new__(cls, hock_joint_prob: float = 0.5, stage: Optional[Usd.Stage] = None):
        if cls._instance is None:
            cls._instance = super(QuadrupedBuilder, cls).__new__(cls)
        return cls._instance

    @staticmethod
    def get_instance() -> "QuadrupedBuilder":
        return QuadrupedBuilder._instance

    def __init__(
        self,
        hock_joint_prob: float = 0.5,
        stage: Optional[Usd.Stage] = None
    ):
        if not QuadrupedBuilder._initialized:
            self.hock_joint_prob = hock_joint_prob
            self.stage = stage
            QuadrupedBuilder._initialized = True
        
        self.params = QuadrupedParams(
            base_dimensions=[],
            leg_lengths=[],
            has_hock_joint=[]
        )

    def spawn(self, prim_path: str, seed: int=-1, stage: Optional[Usd.Stage] = None):
        if stage is None:
            stage = self.stage
        assert stage is not None

        if seed >= 0:
            np.random.seed(seed)
            random.seed(seed)

        BODY_LENGTH = random.uniform(0.5, 1.0)
        BODY_WIDTH = random.uniform(0.3, 0.4)
        BODY_HEIGHT= random.uniform(0.15, 0.25)

        leg_length = random.uniform(BODY_LENGTH * 0.4, BODY_LENGTH * 0.8)
        thigh_radius = random.uniform(0.03, 0.05)
        calf_radius = thigh_radius * 0.8
        foot_radius = thigh_radius * 0.9

        spec = mujoco.MjSpec()
        base_body = spec.worldbody.add_body()
        base_body.name = "base"
        base_body.mass = 5.0
        base_body.inertia = [1.0, 1.0, 1.0]
        trunk_geom = base_body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[BODY_LENGTH/2, BODY_WIDTH/2, BODY_HEIGHT/2])
        head_geom = base_body.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE)
        head_x = BODY_LENGTH/2 + BODY_HEIGHT* 0.5
        head_z = BODY_HEIGHT* 0.1
        head_geom.size = [0.1, 0, 0.]
        head_geom.fromto = [head_x, -BODY_WIDTH*0.4, head_z, head_x, BODY_WIDTH*0.4, head_z]

        HOCK_JOINT = random.random() < self.hock_joint_prob

        for name, pos, (x, y), hock_joint in [
            ("LF", [.5 * BODY_LENGTH, -.5 * BODY_WIDTH, 0], (1, -1), False),
            ("RF", [.5 * BODY_LENGTH, .5 * BODY_WIDTH, 0], (1, 1), False),
            ("LH", [-.5 * BODY_LENGTH, -.5 * BODY_WIDTH, 0], (-1, -1), HOCK_JOINT),
            ("RH", [-.5 * BODY_LENGTH, .5 * BODY_WIDTH, 0], (-1, 1), HOCK_JOINT)
        ]:
            hip_body = base_body.add_body(name=f"{name}_hip", pos=pos)
            hip_body.mass = 1.0
            hip_body.inertia = [1.0, 1.0, 1.0]
            joint = hip_body.add_joint(name=f"{name}_hip_joint", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[1, 0, 0])
            joint.range = [-np.pi * 0.6, +np.pi * 0.6]
            geom = hip_body.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE)
            geom.size = [thigh_radius * 1.8, 0, 0.]
            geom.fromto = [-0.1, 0, 0, 0.1, 0, 0]

            upper_thigh_body = hip_body.add_body(name=f"{name}_upper_thigh", pos=[0.1 * x, 0.1 * y, 0])
            upper_thigh_body.mass = 1.0
            upper_thigh_body.inertia = [1.0, 1.0, 1.0]
            joint = upper_thigh_body.add_joint(name=f"{name}_thigh_joint", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 1, 0])
            joint.range = [-np.pi * 0.9, np.pi * 0.9]
            geom = upper_thigh_body.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE)
            
            if hock_joint:
                UPPER_THIGH_LENGTH = leg_length * 0.6
                LOWER_THIGH_LENGTH = leg_length * 0.6
                CALF_LENGTH = leg_length * 0.9

                geom.size = [thigh_radius * 1.3, 0, 0.]
                geom.fromto = [0, 0, 0, 0, 0, -UPPER_THIGH_LENGTH]
                
                lower_thigh_body = add_leg_(
                    upper_thigh_body,
                    name=f"{name}_lower_thigh",
                    pos=[0, 0, -UPPER_THIGH_LENGTH],
                    radius=thigh_radius,
                    length=LOWER_THIGH_LENGTH
                )
                joint = lower_thigh_body.add_joint(type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 1, 0])
                joint.name = f"{name}_hock_joint"
                joint.range = [-np.pi * 0.8, np.pi * 0.8]
            else:
                UPPER_THIGH_LENGTH = 0
                LOWER_THIGH_LENGTH = leg_length
                CALF_LENGTH = leg_length

                geom.size = [thigh_radius * 1.3, 0, 0.]
                geom.fromto = [0, 0, 0.1, 0, 0, -0.1]

                lower_thigh_body = add_leg_(
                    upper_thigh_body,
                    name=f"{name}_lower_thigh",
                    pos=[0, 0, 0],
                    radius=thigh_radius,
                    length=LOWER_THIGH_LENGTH
                )
                joint = lower_thigh_body.add_joint(type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 0, 1])
                joint.name = f"{name}_hock_joint"
                joint.range = [-np.pi * 0.3, np.pi * 0.3]

            calf_body = add_leg_(
                lower_thigh_body,
                name=f"{name}_calf",
                pos=[0, 0, -LOWER_THIGH_LENGTH],
                radius=calf_radius,
                length=CALF_LENGTH
            )
            joint = calf_body.add_joint(name=f"{name}_calf_joint", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 1, 0])
            joint.range = [-np.pi, 0]

            feet_body = calf_body.add_body(name=f"{name}_foot", pos=[0, 0, -CALF_LENGTH])
            feet_body.mass = 1.0
            feet_body.inertia = [1.0, 1.0, 1.0]
            geom = feet_body.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE)
            geom.size = [foot_radius, 0., 0.]
        
        self.params.base_dimensions.append([BODY_LENGTH, BODY_WIDTH, BODY_HEIGHT])
        self.params.leg_lengths.append([UPPER_THIGH_LENGTH, LOWER_THIGH_LENGTH, CALF_LENGTH])
        self.params.has_hock_joint.append(HOCK_JOINT)

        return self._from_mjspec(stage, prim_path, spec)
    
    def _from_mjspec(self, stage: Usd.Stage, prim_path: str, spec: mujoco.MjSpec) -> Usd.Prim:
        mjmodel = spec.compile()
        mjdata = mujoco.MjData(mjmodel)
        mujoco.mj_forward(mjmodel, mjdata)

        root_prim = UsdGeom.Xform.Define(stage, prim_path).GetPrim()
        # stage.SetDefaultPrim(root_prim)
        UsdPhysics.ArticulationRootAPI.Apply(root_prim)

        prim_dict = {}
        for mjbody in spec.worldbody.find_all("body"):
            xform = UsdGeom.Xform.Define(stage, f"{prim_path}/{mjbody.name}")
            xform_prim = xform.GetPrim()
            for i, geom in enumerate(mjbody.geoms):
                geom_path = f"{xform.GetPath()}/collision_{i}"
                match geom.type:
                    case mujoco.mjtGeom.mjGEOM_BOX:
                        cube = UsdGeom.Cube.Define(stage, geom_path)
                        cube.CreateSizeAttr(2.0)
                        add_default_transform_(cube.GetPrim())
                        cube.GetPrim().GetAttribute("xformOp:scale").Set(Gf.Vec3f(geom.size[0], geom.size[1], geom.size[2]))
                    case mujoco.mjtGeom.mjGEOM_CAPSULE:
                        capsule = create_capsule(stage, geom_path, geom.size[0], np.array(geom.fromto))
                    case mujoco.mjtGeom.mjGEOM_SPHERE:
                        sphere = UsdGeom.Sphere.Define(stage, geom_path)
                        add_default_transform_(sphere.GetPrim())
                        sphere.CreateRadiusAttr(geom.size[0])
                    case _:
                        raise ValueError(f"Unsupported geometry type: {geom.type}")
            add_default_transform_(xform_prim)
            xform_prim.GetAttribute("xformOp:translate").Set(Gf.Vec3f(*mjdata.xpos[mjbody.id]))
            xform_prim.GetAttribute("xformOp:orient").Set(Gf.Quatf(*mjdata.xquat[mjbody.id]))
            UsdPhysics.CollisionAPI.Apply(xform_prim)
            UsdPhysics.RigidBodyAPI.Apply(xform_prim)

            prim_dict[mjbody.id] = xform_prim
            joints = mjbody.joints
            if mjbody.parent.id > 0:
                parent_prim = prim_dict[mjbody.parent.id]
                if len(joints):
                    assert len(joints) == 1, "Only one joint is supported."
                    joint = joints[0]
                    joint_path = f"{parent_prim.GetPath()}/{joint.name}"
                    joint_range = joint.range / np.pi * 180
                    match joint.type:
                        case mujoco.mjtJoint.mjJNT_HINGE:
                            axis = ["X", "Y", "Z"][np.argmax(np.abs(joint.axis))]
                            joint = create_revolute_joint(stage, joint_path, parent_prim, xform_prim, axis)
                            joint.CreateLowerLimitAttr(joint_range[0])
                            joint.CreateUpperLimitAttr(joint_range[1])
                        case _:
                            raise ValueError(f"Unsupported joint type: {joint.type}")
                else:
                    joint_path = f"{parent_prim.GetPath()}/{mjbody.name}_joint"
                    create_fixed_joint(stage, joint_path, parent_prim, xform_prim)
        return root_prim

