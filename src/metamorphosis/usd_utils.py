from pxr import Usd, UsdGeom, UsdPhysics, Sdf, Gf
from scipy.spatial.transform import Rotation as R
import numpy as np
try:
    from pxr import PhysxSchema
except ImportError:
    PhysxSchema = None


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