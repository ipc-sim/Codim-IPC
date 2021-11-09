import sys
sys.path.insert(0, "../../Python")
import Drivers
from JGSL import *

if __name__ == "__main__":
    sim = Drivers.FEMDiscreteShellBase("double", 3)

    algI = 0
    if len(sys.argv) > 1:
        algI = int(sys.argv[1])

    clothI = 0
    if len(sys.argv) > 2:
        clothI = int(sys.argv[2])

    size = '26K'
    if len(sys.argv) > 3:
        size = sys.argv[3]
    
    membEMult = 0.05
    if len(sys.argv) > 4:
        membEMult = float(sys.argv[4])
    
    bendEMult = 1
    if len(sys.argv) > 5:
        bendEMult = float(sys.argv[5])

    sim.mu = 0.4
    
    
    sim.add_shell_3D("input/skirt16x64.obj", Vector3d(0, 0, 0), \
        Vector3d(0, 0, 0), Vector3d(1, 0, 0), 180)
    sim.set_DBC(Vector3d(-0.1, -0.1, -0.1), Vector3d(1.1, 1.1, 1.1), 
        Vector3d(0, 0, 0), Vector3d(0, 0, 0), Vector3d(0, 1, 0), 0)
    
    sim.add_shell_3D("input/tet0.1.obj", Vector3d(0, 0.5, 0), \
        Vector3d(0, 0, 0), Vector3d(0, 0, 0), 0)
    sim.set_DBC(Vector3d(-0.1, 0.5, -0.1), Vector3d(1.1, 1.1, 1.1), 
        Vector3d(0, -0.3, 0), Vector3d(0, 0, 0), Vector3d(0, 1, 0), 0)
    
    sim.add_shell_3D("input/square" + size + ".obj", Vector3d(0, 0.2, 0), \
        Vector3d(0, 0, 0), Vector3d(0, 1, 0), 0)
    sim.add_shell_3D("input/square" + size + ".obj", Vector3d(0, 0.25, 0), \
        Vector3d(0, 0, 0), Vector3d(0, 1, 0), 45)
    sim.add_shell_3D("input/square" + size + ".obj", Vector3d(0, 0.3, 0), \
        Vector3d(0, 0, 0), Vector3d(0, 1, 0), 90)
    
    sim.dt = 0.04
    sim.frame_dt = 0.04
    sim.frame_num = 80
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

    sim.initialize_OIPC(1e-3, 0)

    sim.run()
