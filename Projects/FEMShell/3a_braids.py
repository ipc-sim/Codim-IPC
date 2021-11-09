import sys
sys.path.insert(0, "../../Python")
import math
import Drivers
from JGSL import *

if __name__ == "__main__":
    sim = Drivers.FEMDiscreteShellBase("double", 3)

    size = '60'
    if len(sys.argv) > 1:
        size = sys.argv[1]

    N = 30
    if len(sys.argv) > 2:
        N = int(sys.argv[2])

    sim.mu = 0.1
    sim.epsv2 = 1e-8

    # rod:    
    step = 3e-4
    width = step * (N - 1)
    for j in range(N):
        iMax = int(2 * math.sqrt((float(N - 1) / 2) ** 2 - (j - float(N - 1) / 2) ** 2))
        xWidth = step * iMax
        for i in range(iMax):
            sim.make_and_add_rod_3D(0.15, int(size), Vector3d(-width / 2 - step * 5 -xWidth / 2 + step * i, 0, -width / 2 + step * j), \
                Vector3d(0, 0, 0), Vector3d(0, 0, 1), -90, Vector3d(1, 1, 1))
            sim.make_and_add_rod_3D(0.15, int(size), Vector3d(width / 2 + step * 5 -xWidth / 2 + step * i, 0, -width / 2 + step * j), \
                Vector3d(0, 0, 0), Vector3d(0, 0, 1), -90, Vector3d(1, 1, 1))
    sim.set_DBC(Vector3d(-0.1, 1 - 1.0 / (int(size) - 2), -0.1), Vector3d(1.1, 1.1, 1.1), 
        Vector3d(0, -0.001, 0), Vector3d(0, 0, 0), Vector3d(0, 1, 0), -270)
    sim.set_DBC(Vector3d(-0.1, -0.1, -0.1), Vector3d(1.1, 1.0 / (int(size) - 2), 1.1), 
        Vector3d(0, 0.001, 0), Vector3d(0, 0, 0), Vector3d(0, 1, 0), 270)

    sim.dt = 0.01
    sim.frame_dt = 0.01
    sim.frame_num = 600
    sim.withCollision = True

    sim.MDBC_tmax = 3
    sim.DBCPopBackTStart = 3
    sim.DBCPopBackTEnd = 1e10
    sim.DBCPopBackStep = 1
    sim.DBCPopBackAmt = 1

    sim.initialize(sim.cloth_density[0], sim.cloth_Ebase[0], 0.4, \
            sim.cloth_thickness[0], 0)
    sim.initialize_rod(1300, 4e8, 0.01, 8e-5)
    
    sim.initialize_OIPC(1e-4, 8e-5)

    sim.run()
