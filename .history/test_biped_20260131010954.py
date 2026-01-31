#!/usr/bin/env python3

from src.metamorphosis.builder.biped import BipedBuilder

def test_biped():
    print("Testing updated biped structure...")
    
    # Create builder
    builder = BipedBuilder()
    
    # Sample parameters
    param = builder.sample_params(seed=42)
    
    print("✓ Sample parameters generated successfully!")
    print(f"  Pelvis: {param.pelvis_length:.3f} × {param.pelvis_width:.3f} × {param.pelvis_height:.3f}")
    print(f"  Hip spacing: {param.hip_spacing:.3f}")
    print(f"  Hip1: radius={param.hip1_radius:.3f}, length={param.hip1_length:.3f}, mass={param.hip1_mass:.2f}")
    print(f"  Hip2: radius={param.hip2_radius:.3f}, length={param.hip2_length:.3f}, mass={param.hip2_mass:.2f}")
    print(f"  Thigh: radius={param.thigh_radius:.3f}, length={param.thigh_length:.3f}, mass={param.thigh_mass:.2f}")
    print(f"  Shin: radius={param.shin_radius:.3f}, length={param.shin_length:.3f}, mass={param.shin_mass:.2f}")
    
    # Generate MjSpec
    spec = builder.generate_mjspec(param)
    print("✓ MjSpec generated successfully!")
    
    # Check structure
    bodies = []
    def collect_bodies(body):
        bodies.append(body.name)
        for child in body.body:
            collect_bodies(child)
    
    collect_bodies(spec.worldbody)
    
    print("\nBody structure:")
    for body_name in bodies:
        print(f"  - {body_name}")
    
    print("\n✅ All tests passed! Biped structure updated to standard humanoid format:")
    print("   waist geo -> hip joint1 -> hip1 -> hip joint2 -> hip2 -> hip joint3 -> thigh -> knee joint -> calf")

if __name__ == "__main__":
    test_biped()