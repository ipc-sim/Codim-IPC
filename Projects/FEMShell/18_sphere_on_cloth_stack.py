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

    size = '8K'
    if len(sys.argv) > 3:
        size = sys.argv[3]
    
    membEMult = 0.01
    if len(sys.argv) > 4:
        membEMult = float(sys.argv[4])
    
    bendEMult = 1
    if len(sys.argv) > 5:
        bendEMult = float(sys.argv[5])
    
    N = 10
    if len(sys.argv) > 6:
        N = int(sys.argv[6])

    thickness = 1e-3
    if len(sys.argv) > 7:
        thickness = float(sys.argv[7])

    sim.mu = 0.1


    sim.add_shell_3D("input/square21.obj", Vector3d(0, 0, 0), \
        Vector3d(0, 0, 0), Vector3d(0, 0, 0), 0)

    # v, rotCenter, rotAxis, angVelDeg
    sim.set_DBC(Vector3d(-0.1, -0.1, -0.1), Vector3d(1.1, 1.1, 1.1), 
        Vector3d(0, 0, 0), Vector3d(0, 0, 0), Vector3d(0, 0, 0), 0)
    
    for i in range(N):
        sim.add_shell_3D("input/square" + size + ".obj", Vector3d(0, 0.1 * (i + 1), 0), \
            Vector3d(0, 0, 0), Vector3d(0, 1, 0), (i + 1) * 90.0 / (N + 1))
        
    sim.add_object_3D("../FEM/input/sphere1K.vtk", Vector3d(0, 1 + 0.15 + 0.1 * (N + 1), 0), \
        Vector3d(0, 0, 0), Vector3d(1, 0, 0), 0, Vector3d(0.3, 0.3, 0.3))


    sim.dt = 0.02
    sim.frame_dt = 0.02
    sim.frame_num = 140
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

    sim.initialize_added_objects(Vector3d(0, 0, 0), 1000, 1e4, 0.4)

    sim.initialize_OIPC(thickness, 0)

    sim.run()
