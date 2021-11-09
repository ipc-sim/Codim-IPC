import sys
sys.path.insert(0, "../../Python")
import Drivers
from JGSL import *

if __name__ == "__main__":
    sim = Drivers.FEMDiscreteShellBase("double", 3)

    algI = 1
    if len(sys.argv) > 1:
        algI = int(sys.argv[1])

    clothI = 0
    if len(sys.argv) > 2:
        clothI = int(sys.argv[2])

    size = '88K'
    if len(sys.argv) > 3:
        size = sys.argv[3]
    
    membEMult = 0.1
    if len(sys.argv) > 4:
        membEMult = float(sys.argv[4])
    
    bendEMult = 0.1
    if len(sys.argv) > 5:
        bendEMult = float(sys.argv[5])

    offset = 1.5e-3
    if len(sys.argv) > 6:
        offset = float(sys.argv[6])

    frameAmt = 1000
    if len(sys.argv) > 7:
        frameAmt = int(sys.argv[7])


    sim.add_shell_3D("input/cylinder" + size + ".obj", Vector3d(0, 0, 0), \
        Vector3d(0, 0, 0), Vector3d(0, 0, 0), 0)

    # v, rotCenter, rotAxis, angVelDeg
    sim.set_DBC(Vector3d(-0.1, -0.1, -0.1), Vector3d(1e-5, 1.1, 1.1), 
        Vector3d(0.005, 0, 0), Vector3d(0, 0, 0), Vector3d(1, 0, 0), -72)
    sim.set_DBC(Vector3d(1-1e-5, -0.1, -0.1), Vector3d(1.1, 1.1, 1.1), 
        Vector3d(-0.005, 0, 0), Vector3d(0, 0, 0), Vector3d(1, 0, 0), 72)
    
    sim.gravity = Vector3d(0, 0, 0)

    sim.dt = 0.04
    sim.frame_dt = 0.04
    sim.frame_num = frameAmt
    sim.withCollision = True

    # density, E, nu, thickness, initial displacement case
    if algI == 0:
        # iso
        sim.initialize(sim.cloth_density_iso[clothI], sim.cloth_Ebase_iso[clothI] * membEMult,
            sim.cloth_nubase_iso[clothI], sim.cloth_thickness_iso[clothI], 0)
        sim.bendingStiffMult = bendEMult / membEMult
        sim.kappa_s = Vector2d(1e3, 0)
        sim.s = Vector2d(sim.cloth_SL_iso[clothI], 0)
    elif algI == 1:
        # iso, no strain limit
        sim.initialize(sim.cloth_density_iso[clothI], sim.cloth_Ebase_iso[clothI] * membEMult,
            sim.cloth_nubase_iso[clothI], sim.cloth_thickness_iso[clothI], 0)
        sim.bendingStiffMult = bendEMult / membEMult
        sim.kappa_s = Vector2d(0, 0)
    elif algI == 2:
        # aniso
        sim.initialize(sim.cloth_density[clothI], sim.cloth_Ebase[clothI], # actually only affect bending
            0, sim.cloth_thickness[clothI], 0)
        sim.bendingStiffMult = bendEMult
        sim.fiberStiffMult = sim.cloth_weftWarpMult[clothI] * membEMult
        sim.inextLimit = sim.cloth_inextLimit[clothI]
        sim.kappa_s = Vector2d(0, 0)
    # elif algI == 6:
        # coupled strain limiting
    # elif algI == 7:
        # split strain limiting

    sim.initialize_OIPC(1e-3, offset)

    # sim.load_frame('output/cylinder_twist/1_0_88K_0.1_0.1_1.5e-3_500/shell500.obj')
    # sim.load_velocity('output/cylinder_twist/1_0_88K_0.1_0.1_1.5e-3_500/', 500, sim.frame_dt)

    sim.run()
