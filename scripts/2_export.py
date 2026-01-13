from metamorphosis.quadruped import QuadrupedBuilder
from pxr import Usd, UsdGeom, UsdPhysics, Sdf, Gf

def make_stage(path="quadruped.usda", meters=1.0, up="Z"):
    stage = Usd.Stage.CreateNew(path)
    UsdGeom.SetStageMetersPerUnit(stage, meters)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z if up == "Z" else UsdGeom.Tokens.y)
    # Physics scene
    phys_scene = UsdPhysics.Scene.Define(stage, "/World/physicsScene")
    phys_scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0, 0, -1))
    phys_scene.CreateGravityMagnitudeAttr().Set(9.81 if meters == 1.0 else 981.0)
    return stage

stage = make_stage()
builder = QuadrupedBuilder(hock_joint=True, stage=stage)
prim = builder.spawn(prim_path="/Robot", seed=0)
stage.SetDefaultPrim(prim)
stage.GetRootLayer().Save()
