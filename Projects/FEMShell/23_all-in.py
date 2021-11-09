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
    
    bendEMult = 0.1
    if len(sys.argv) > 5:
        bendEMult = float(sys.argv[5])

    rodSize = '50'
    if len(sys.argv) > 6:
        rodSize = sys.argv[6]    
    
    # rod:
    N = 20
    for i in range(N):
        step = 1.0 / N
        sim.make_and_add_rod_3D(1, int(rodSize), Vector3d(0, 0, -0.5 + step / 2 + step * i), \
            Vector3d(0, 0, 0), Vector3d(1, 0, 0), 0, Vector3d(1, 1, 1))
        up = i % 2
        if up == 1:
            sim.make_and_add_rod_3D(1, int(rodSize), Vector3d(-0.5 + step / 2 + step * i, 1e-3, 0), \
                Vector3d(0, 0, 0), Vector3d(0, 1, 0), 90, Vector3d(1, 1, 1))
        else:
            sim.make_and_add_rod_3D(1, int(rodSize), Vector3d(-0.5 + step / 2 + step * i, -1e-3, 0), \
                Vector3d(0, 0, 0), Vector3d(0, 1, 0), 90, Vector3d(1, 1, 1))
    sim.set_DBC(Vector3d(-0.1, -0.1, -0.1), Vector3d(1e-3, 1.1, 1.1), 
        Vector3d(0, 0, 0), Vector3d(0, 0, 0), Vector3d(1, 0, 0), 0)
    sim.set_DBC(Vector3d(1-1e-3, -0.1, -0.1), Vector3d(1.1, 1.1, 1.1), 
        Vector3d(0, 0, 0), Vector3d(0, 0, 0), Vector3d(1, 0, 0), 0)
    sim.set_DBC(Vector3d(-0.1, -0.1, -0.1), Vector3d(1.1, 1.1, 1e-3), 
        Vector3d(0, 0, 0), Vector3d(0, 0, 0), Vector3d(1, 0, 0), 0)
    sim.set_DBC(Vector3d(-0.1, -0.1, 1-1e-3), Vector3d(1.1, 1.1, 1.1), 
        Vector3d(0, 0, 0), Vector3d(0, 0, 0), Vector3d(1, 0, 0), 0)

    # floor
    sim.add_shell_with_scale_3D("input/square21.obj", Vector3d(0, -0.5, 0), Vector3d(8, 8, 8),\
        Vector3d(0, 0, 0), Vector3d(1, 0, 0), 0)
    sim.set_DBC(Vector3d(-0.1, -0.1, -0.1), Vector3d(1.1, 1e-3, 1.1), 
        Vector3d(0, 0, 0), Vector3d(0, 0, 0), Vector3d(1, 0, 0), 0)

    # sand
    N = 8
    H = 100
    step = 4e-3
    counter = sim.add_particle_3D(Vector3d(step * (N - 1), step * (H - 1), step * (N - 1)), Vector3i(N, H, N), Vector3d(1.25e-3, 1.25e-3, 1.25e-3), \
        Vector3d(0, step * (H - 1) / 2 + 1.01, -0.1), Vector3d(0, 0, 0), Vector3d(1, 0, 0), 0, Vector3d(1, 1, 1))
    sim.set_DBC_with_range(Vector3d(-0.1, -0.1, -0.1), Vector3d(1.1, 1.1, 1.1), 
        Vector3d(0, 0, 0), Vector3d(0, 0, 0), Vector3d(1, 0, 0), 0, counter)

    # elastic body
    sim.add_object_3D("../FEM/input/Armadillo13K.vtk", Vector3d(0, 0.8, 0), \
        Vector3d(0, 0, 0), Vector3d(1, 0, 0), -90, Vector3d(0.003, 0.003, 0.003))

    # cloth
    counter = sim.add_shell_with_scale_3D("input/square" + size + ".obj", Vector3d(0, 1, 0), Vector3d(0.8, 0.8, 0.8),\
        Vector3d(0, 0, 0), Vector3d(1, 0, 0), 0)
    sim.set_DBC_with_range(Vector3d(-0.1, -0.1, -0.1), Vector3d(1.1, 1.1, 1.1), 
        Vector3d(0, 0, 0), Vector3d(0, 0, 0), Vector3d(1, 0, 0), 0, counter)
    
    sim.muComp = StdVectorXd([0, 0.2, 0.2,  0.2, 0, 0.2,  0.2, 0.2, 0.2])
    # sim.epsv2 = 1e-8

    sim.dt = 0.02
    sim.frame_dt = 0.02
    sim.frame_num = 700
    sim.withCollision = True

    sim.DBCPopBackTStart = 2
    sim.DBCPopBackTEnd = 1e10
    sim.DBCPopBackStep = 1
    sim.DBCPopBackAmt = 2
    sim.DBCPopBackBatch = int(2 / sim.dt)

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
    
    sim.initialize_added_objects(Vector3d(0, 0, 0), 1000, 1e5, 0.4)
    sim.initialize_rod(1000, 1e8, 1, 2e-3)
    sim.initialize_particle(1600, 1e7, 2e-3)

    sim.initialize_OIPC(1e-3, 5e-4)

    sim.run()
