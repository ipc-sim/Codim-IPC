import sys
sys.path.insert(0, "../../Python")
import Drivers
from JGSL import *

if __name__ == "__main__":
    sim = Drivers.FEMDiscreteShellBase("double", 3)

    SL = 1.01
    if len(sys.argv) > 1:
        SL = float(sys.argv[1])

    clothI = 0
    if len(sys.argv) > 2:
        clothI = int(sys.argv[2])

    size = '2K'
    if len(sys.argv) > 3:
        size = sys.argv[3]
    
    membEMult = 0.01
    if len(sys.argv) > 4:
        membEMult = float(sys.argv[4])
    
    bendEMult = 1
    if len(sys.argv) > 5:
        bendEMult = float(sys.argv[5])
    

    sim.add_shell_3D("input/square" + size + ".obj", Vector3d(0, 0, 0), \
        Vector3d(0, 0, 0), Vector3d(0, 0, 0), 0)

    # v, rotCenter, rotAxis, angVelDeg
    sim.set_DBC(Vector3d(-0.1, -0.1, -0.1), Vector3d(1e-3, 1.1, 1e-3), 
        Vector3d(0, 0, 0), Vector3d(0, 0, 0), Vector3d(1, 0, 0), 0)
    sim.set_DBC(Vector3d(1-1e-3, -0.1, 1-1e-3), Vector3d(1.1, 1.1, 1.1), 
        Vector3d(0, 0, 0), Vector3d(0, 0, 0), Vector3d(1, 0, 0), 0)
    
    sim.dt = 0.04
    sim.frame_dt = 0.04
    sim.frame_num = 100
    sim.withCollision = True


    # density, E, nu, thickness, initial displacement case
    sim.initialize(sim.cloth_density_iso[clothI], sim.cloth_Ebase_iso[clothI] * membEMult,
        sim.cloth_nubase_iso[clothI], sim.cloth_thickness_iso[clothI], 0)
    sim.bendingStiffMult = bendEMult / membEMult
    sim.kappa_s = Vector2d(1e3, 0)
    sim.s = Vector2d(SL, 0)
    
    sim.initialize_OIPC(1e-3, 0)

    sim.run()
