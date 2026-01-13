import mujoco
import mujoco.viewer
import itertools
import numpy as np
import random

from pxr import Usd, UsdGeom, UsdPhysics, Sdf, Gf
from typing import NamedTuple
try:
    from pxr import PhysxSchema
except ImportError:
    PhysxSchema = None
from scipy.spatial.transform import Rotation as R
from typing import Optional


def add_default_transform_(prim: Usd.Prim):
    vec3_dtype, vec3_cls = Sdf.ValueTypeNames.Float3, Gf.Vec3f
    quat_dtype, quat_cls = Sdf.ValueTypeNames.Quatf, Gf.Quatf
    order = ["xformOp:translate", "xformOp:orient", "xformOp:scale"]
    # add xform ops [scale, orient, translate]
    prim.CreateAttribute("xformOp:scale", vec3_dtype, False).Set(vec3_cls(1.0, 1.0, 1.0))
    prim.CreateAttribute("xformOp:orient", quat_dtype, False).Set(quat_cls(1.0, 0.0, 0.0, 0.0))
    prim.CreateAttribute("xformOp:translate", vec3_dtype, False).Set(vec3_cls(0.0, 0.0, 0.0))
    prim.CreateAttribute("xformOpOrder", Sdf.ValueTypeNames.TokenArray, False).Set(order)


def create_capsule(stage: Usd.Stage, path: str, radius: float, fromto: np.ndarray):
    capsule = UsdGeom.Capsule.Define(stage, path)
    add_default_transform_(capsule.GetPrim())
    direction = fromto[3:] - fromto[:3]
    length = np.linalg.norm(direction)
    direction = direction / length
    axis = np.cross(direction, [0, 0, 1])
    angle = np.arccos(np.dot(direction, [0, 0, 1]))
    translation = (fromto[:3] + fromto[3:]) / 2
    orient = R.from_rotvec(angle * axis).as_quat(scalar_first=True)
    capsule.CreateAxisAttr("Z")
    capsule.CreateRadiusAttr(radius)
    capsule.CreateHeightAttr(length)
    capsule.GetPrim().GetAttribute("xformOp:translate").Set(Gf.Vec3f(*translation))
    capsule.GetPrim().GetAttribute("xformOp:orient").Set(Gf.Quatf(*orient))
    return capsule


def create_fixed_joint(stage: Usd.Stage, path: str, body_0: Usd.Prim, body_1: Usd.Prim):
    joint = UsdPhysics.FixedJoint.Define(stage, path)
    joint.CreateBody0Rel().SetTargets([body_0.GetPath()])
    joint.CreateBody1Rel().SetTargets([body_1.GetPath()])
    xfCache = UsdGeom.XformCache()
    body_0_pose = xfCache.GetLocalToWorldTransform(body_0)
    body_1_pose = xfCache.GetLocalToWorldTransform(body_1)
    rel_pose = body_1_pose * body_0_pose.GetInverse()
    rel_pose = rel_pose.RemoveScaleShear()
    pos1 = Gf.Vec3f(rel_pose.ExtractTranslation())
    rot1 = Gf.Quatf(rel_pose.ExtractRotationQuat())
    joint.CreateLocalPos0Attr().Set(pos1)
    joint.CreateLocalRot0Attr().Set(rot1)
    return joint


def create_revolute_joint(stage: Usd.Stage, path: str, body_0: Usd.Prim, body_1: Usd.Prim, axis: str = "Z"):
    assert axis in ["X", "Y", "Z"], f"Invalid axis: {axis}"
    joint = UsdPhysics.RevoluteJoint.Define(stage, path)
    joint.CreateBody0Rel().SetTargets([body_0.GetPath()])
    joint.CreateBody1Rel().SetTargets([body_1.GetPath()])
    joint.CreateAxisAttr(axis)
    xfCache = UsdGeom.XformCache()
    body_0_pose = xfCache.GetLocalToWorldTransform(body_0)
    body_1_pose = xfCache.GetLocalToWorldTransform(body_1)
    rel_pose = body_1_pose * body_0_pose.GetInverse()
    rel_pose = rel_pose.RemoveScaleShear()
    pos1 = Gf.Vec3f(rel_pose.ExtractTranslation())
    rot1 = Gf.Quatf(rel_pose.ExtractRotationQuat())
    joint.CreateLocalPos0Attr().Set(pos1)
    joint.CreateLocalRot0Attr().Set(rot1)

    prim = joint.GetPrim()
    # check if prim has joint drive applied on it
    usd_drive_api = UsdPhysics.DriveAPI(prim, "angular")
    if not usd_drive_api:
        usd_drive_api = UsdPhysics.DriveAPI.Apply(prim, "angular")
    # check if prim has Physx joint drive applied on it
    if PhysxSchema is not None:
        physx_joint_api = PhysxSchema.PhysxJointAPI(prim)
        if not physx_joint_api:
            physx_joint_api = PhysxSchema.PhysxJointAPI.Apply(prim)
    return joint


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

    def spawn(self, prim_path: str, seed: int=0, stage: Optional[Usd.Stage] = None):
        if stage is None:
            stage = self.stage
        assert stage is not None

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
                
                lower_thigh_body = upper_thigh_body.add_body(pos=[0, 0, -UPPER_THIGH_LENGTH])
                lower_thigh_body.name = f"{name}_lower_thigh"
                lower_thigh_body.mass = 1.0
                lower_thigh_body.inertia = [1.0, 1.0, 1.0]
                joint = lower_thigh_body.add_joint(type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 1, 0])
                joint.name = f"{name}_hock_joint"
                joint.range = [-np.pi * 0.8, np.pi * 0.8]
                geom = lower_thigh_body.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE)
                geom.size = [thigh_radius, 0, 0.]
                geom.fromto = [0, 0, 0, 0, 0, -LOWER_THIGH_LENGTH]
            else:
                UPPER_THIGH_LENGTH = 0
                LOWER_THIGH_LENGTH = leg_length
                CALF_LENGTH = leg_length

                geom.size = [thigh_radius * 1.3, 0, 0.]
                geom.fromto = [0, 0, 0.1, 0, 0, -0.1]

                lower_thigh_body = upper_thigh_body.add_body(name=f"{name}_lower_thigh")
                lower_thigh_body.mass = 1.0
                lower_thigh_body.inertia = [1.0, 1.0, 1.0]
                joint = lower_thigh_body.add_joint(type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 0, 1])
                joint.name = f"{name}_hock_joint"
                joint.range = [-np.pi * 0.3, np.pi * 0.3]
                geom = lower_thigh_body.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE)
                geom.size = [thigh_radius, 0, 0.]
                geom.fromto = [0, 0, 0, 0, 0, -LOWER_THIGH_LENGTH]

            calf_body = lower_thigh_body.add_body(name=f"{name}_calf", pos=[0, 0, -LOWER_THIGH_LENGTH])
            calf_body.mass = 1.0
            calf_body.inertia = [1.0, 1.0, 1.0]
            joint = calf_body.add_joint(name=f"{name}_calf_joint", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 1, 0])
            joint.range = [-np.pi, 0]
            geom = calf_body.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE)
            geom.size = [calf_radius, 0, 0.]
            geom.fromto = [0, 0, 0, 0, 0, -CALF_LENGTH]

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

