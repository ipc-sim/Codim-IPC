import sys
sys.path.insert(0, "../../Python")
import Drivers
from JGSL import *

if __name__ == "__main__":
    sim = Drivers.FEMDiscreteShellBase("double", 3)

    size = '200'
    if len(sys.argv) > 1:
        size = sys.argv[1]

    N = 25
    if len(sys.argv) > 2:
        N = int(sys.argv[2])

    sim.mu = 0.4

    
    sim.add_shell_with_scale_3D("input/bowl.obj", Vector3d(0, 0, 0), Vector3d(0.7, 0.7, 0.7), \
        Vector3d(0, 0, 0), Vector3d(0, 0, 0), 0)
    # v, rotCenter, rotAxis, angVelDeg
    sim.set_DBC(Vector3d(-0.1, -0.1, -0.1), Vector3d(1.1, 1.1, 1.1), 
        Vector3d(0, 0, 0), Vector3d(0, 0, 0), Vector3d(0, 0, 0), 0)
    
    # rod:
    step = 0.08 / (N - 1)
    for i in range(N):
        for j in range(N):
            sim.make_and_add_rod_3D(0.4, int(size), Vector3d(-0.04 + step * i, 0.2, -0.04 + step * j), \
                Vector3d(0, 0, 0), Vector3d(0, 0, 1), 90, Vector3d(1, 1, 1)) # can add some random tilt
    
    sim.dt = 0.02
    sim.frame_dt = 0.02
    sim.frame_num = 200
    sim.withCollision = True
    sim.epsv2 = 1e-10

    sim.initialize(sim.cloth_density[0], sim.cloth_Ebase[0], 0.4, \
            sim.cloth_thickness[0], 0)
    
    sim.initialize_rod(160, 1e3, 1, 1.5e-3)
    
    sim.initialize_OIPC(5e-4, 1e-3) # can change offset

    sim.run()
