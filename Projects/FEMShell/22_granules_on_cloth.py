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
    
    membEMult = 0.05
    if len(sys.argv) > 4:
        membEMult = float(sys.argv[4])
    
    bendEMult = 0.1
    if len(sys.argv) > 5:
        bendEMult = float(sys.argv[5])

    H = 500
    if len(sys.argv) > 6:
        H = int(sys.argv[6])
    
    N = 10
    if len(sys.argv) > 7:
        N = int(sys.argv[7])

    sim.mu = 0.05
    if len(sys.argv) > 8:
        sim.mu = float(sys.argv[8])

    # sim.fricIterAmt = 3
    # sim.epsv2 = 1e-8

    # particle
    step = 4e-3
    sim.add_particle_3D(Vector3d(step * (N - 1), step * (H - 1), step * (N - 1)), Vector3i(N, H, N), Vector3d(5e-4, 5e-4, 5e-4), \
        Vector3d(0, step * (H - 1) / 2 + step, 0), Vector3d(0, 0, 0), Vector3d(1, 0, 0), 0, Vector3d(1, 1, 1))
    
    # cloth
    sim.add_shell_with_scale_3D("input/square" + size + ".obj", Vector3d(0, 0, 0), Vector3d(0.7, 0.7, 0.7), \
        Vector3d(0, 0, 0), Vector3d(1, 0, 0), 0)
    # fix the 4 corner of the cloth
    sim.set_DBC(Vector3d(-0.1, -0.1, -0.1), Vector3d(1e-3, 1.1, 1e-3), 
        Vector3d(0, 0, 0), Vector3d(0, 0, 0), Vector3d(1, 0, 0), 0)
    sim.set_DBC(Vector3d(-0.1, -0.1, 1-1e-3), Vector3d(1e-3, 1.1, 1.1), 
        Vector3d(0, 0, 0), Vector3d(0, 0, 0), Vector3d(1, 0, 0), 0)
    sim.set_DBC(Vector3d(1-1e-3, -0.1, -0.1), Vector3d(1.1, 1.1, 1e-3), 
        Vector3d(0, 0, 0), Vector3d(0, 0, 0), Vector3d(1, 0, 0), 0)
    sim.set_DBC(Vector3d(1-1e-3, -0.1, 1-1e-3), Vector3d(1.1, 1.1, 1.1), 
        Vector3d(0, 0, 0), Vector3d(0, 0, 0), Vector3d(1, 0, 0), 0)

    sim.add_shell_with_scale_3D("input/square21.obj", Vector3d(0, -0.3, 0), Vector3d(8, 8, 8), \
        Vector3d(0, 0, 0), Vector3d(1, 0, 0), 0)
    sim.set_DBC(Vector3d(-0.1, -0.1, -0.1), Vector3d(1.1, 1e-3, 1.1), 
        Vector3d(0, 0, 0), Vector3d(0, 0, 0), Vector3d(1, 0, 0), 0)

    sim.muComp = StdVectorXd([0, sim.mu, sim.mu,  sim.mu, 0, 0,  sim.mu, 0, 0])

    sim.dt = 0.01
    sim.frame_dt = 0.01
    sim.frame_num = 1000
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
        sim.initialize(sim.cloth_density_iso[clothI], membEMult,
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
    
    sim.initialize_particle(1600, 1e7, 2e-3)

    sim.initialize_OIPC(1e-3, 2e-3)

    sim.run()
