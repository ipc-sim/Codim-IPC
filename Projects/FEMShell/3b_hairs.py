import sys
sys.path.insert(0, "../../Python")
import random
import math
import Drivers
from JGSL import *

if __name__ == "__main__":
    sim = Drivers.FEMDiscreteShellBase("double", 3)

    size = '50'
    if len(sys.argv) > 1:
        size = sys.argv[1]

    N = 35
    if len(sys.argv) > 2:
        N = int(sys.argv[2])

    sim.mu = 0.2
    sim.epsv2 = 1e-8
    
    # rod:
    rodCounter = 0
    step = 4e-4
    width = step * (N - 1)
    for j in range(N):
        iMax = int(2 * math.sqrt((float(N - 1) / 2) ** 2 - (j - float(N - 1) / 2) ** 2))
        xWidth = step * iMax
        for i in range(iMax):
            sim.make_and_add_rod_3D(0.2, int(size), Vector3d(-xWidth / 2 + step * i, -width / 2 + step * j, 0), \
                Vector3d(0, 0, 0), Vector3d(0, 1, 0), 90, Vector3d(1, 1, 1))
            rodCounter = rodCounter + 1
    sim.set_DBC(Vector3d(-0.1, -0.1, -0.1), Vector3d(1.1, 1.1, 1e-3), 
        Vector3d(0, 0, 0), Vector3d(0, 0, 0), Vector3d(1, 0, 0), 0)
    sim.set_DBC(Vector3d(-0.1, -0.1, 1-1e-3), Vector3d(1.1, 1.1, 1.1), 
        Vector3d(0, 0, 0), Vector3d(0, 0, 0), Vector3d(1, 0, 0), 0)
    
    for j in range(N):
        iMax = int(2 * math.sqrt((float(N - 1) / 2) ** 2 - (j - float(N - 1) / 2) ** 2))
        xWidth = step * iMax
        for i in range(iMax):
            dl = random.random() * 0.2 * 0.2
            sim.make_and_add_rod_3D(0.2 - dl, int(size), Vector3d(0.053 + dl / 2, width + step -xWidth / 2 + step * i, -width / 2 + step * j), \
                Vector3d(0, 0, 0), Vector3d(0, 1, 0), 0, Vector3d(1, 1, 1))
            rodCounter = rodCounter + 1
    sim.set_DBC(Vector3d(1 - 1.0 / (int(size) - 2), -0.1, -0.1), Vector3d(1.1, 1.1, 1.1), 
        Vector3d(0, 0, 0), Vector3d(0, 0, 0), Vector3d(1, 0, 0), 0)

    counter = sim.add_shell_with_scale_3D('input/sphere1K.obj', Vector3d(0.1, -0.1, 0), Vector3d(0.066, 0.066, 0.066), 
        Vector3d(0, 0, 0), Vector3d(1, 0, 0), 0)
    sim.set_DBC_with_range(Vector3d(-0.1, -0.1, -0.1), Vector3d(1.1, 1.1, 1.1), 
        Vector3d(0, 0, 0), Vector3d(0, 0, 0), Vector3d(1, 0, 0), 0, counter)

    muComp = []
    for i in range(rodCounter + 1):
        for j in range(rodCounter + 1):
            if i == rodCounter or j == rodCounter:
                muComp.append(0.3)
            else:
                muComp.append(0.2)
    sim.muComp = StdVectorXd(muComp)

    sim.dt = 0.01
    sim.frame_dt = 0.01
    sim.frame_num = 300
    sim.withCollision = True

    sim.initialize(sim.cloth_density[0], sim.cloth_Ebase[0], 0.4, \
            sim.cloth_thickness[0], 0)
    sim.initialize_rod(1300, 4e8, 0.01, 8e-5)
    
    sim.initialize_OIPC(1e-4, 8e-5)

    sim.run()
