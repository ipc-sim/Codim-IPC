import sys
sys.path.insert(0, "../../Python")
import Drivers
from JGSL import *

if __name__ == "__main__":
    sim = Drivers.FEMDiscreteShellBase("double", 3)

    algI = 1
    if len(sys.argv) > 1:
        algI = int(sys.argv[1])

    clothI = 4
    if len(sys.argv) > 2:
        clothI = int(sys.argv[2])

    size = '15x7'
    if len(sys.argv) > 3:
        size = sys.argv[3]
    
    membEMult = 1
    if len(sys.argv) > 4:
        membEMult = float(sys.argv[4])
    
    bendEMult = 1
    if len(sys.argv) > 5:
        bendEMult = float(sys.argv[5])


    sim.add_shell_3D("input/square21.obj", Vector3d(0, 0, 0), \
        Vector3d(0, 0, 0), Vector3d(0, 0, 0), 0)
    # v, rotCenter, rotAxis, angVelDeg
    sim.set_DBC(Vector3d(-0.1, -0.1, -0.1), Vector3d(1.1, 1.1, 1.1), 
        Vector3d(0, 0, 0), Vector3d(0, 0, 0), Vector3d(0, 0, 0), 0)

    counter_lHandle = sim.add_shell_3D("input/card4.obj", Vector3d(-0.09, 0, 0), \
        Vector3d(0, 0, 0), Vector3d(0, 0, 1), -90)
    # v, rotCenter, rotAxis, angVelDeg
    sim.set_DBC2_with_range(Vector3d(-0.1, -0.1, -0.1), Vector3d(1.1, 1.1, 1.1), 
        Vector3d(0.04, 0, 0), Vector3d(0, 0, 0), Vector3d(0, 0, 0), 0, counter_lHandle)
    counter_rHandle = sim.add_shell_3D("input/card4.obj", Vector3d(0.09, 0, 0), \
        Vector3d(0, 0, 0), Vector3d(0, 0, 1), 90)
    # v, rotCenter, rotAxis, angVelDeg
    sim.set_DBC2_with_range(Vector3d(-0.1, -0.1, -0.1), Vector3d(1.1, 1.1, 1.1), 
        Vector3d(-0.04, 0, 0), Vector3d(0, 0, 0), Vector3d(0, 0, 0), 0, counter_rHandle)

    
    
    N = 27
    counter = []
    # # grab cards bottom
    # for i in range(N):
    #     counter.append(sim.add_shell_3D("input/card" + size + ".obj", Vector3d(-0.05 - 0.22e-3 * i, 0.032 + 0.22e-3 * i, 0), \
    #         Vector3d(0, 0, 0), Vector3d(0, 0, 1), 45))
    #     sim.set_DBC_with_range(Vector3d(-0.1, -0.1, -0.1), Vector3d(1e-5, 1.1, 1.1), 
    #         Vector3d(0, 0, 0), Vector3d(0, 0, 0), Vector3d(1, 0, 0), 0, counter[-1])
    # for i in range(N):
    #     counter.append(sim.add_shell_3D("input/card" + size + ".obj", Vector3d(0.05 + 0.22e-3 * i, 0.032 + 0.22e-3 * i, 0), \
    #         Vector3d(0, 0, 0), Vector3d(0, 0, 1), -45))
    #     sim.set_DBC_with_range(Vector3d(1-1e-5, -0.1, -0.1), Vector3d(1.1, 1.1, 1.1), 
    #         Vector3d(0, 0, 0), Vector3d(0, 0, 0), Vector3d(1, 0, 0), 0, counter[-1])

    # # grab cards top
    # for i in range(N):
    #     sim.set_DBC_with_range(Vector3d(1-1e-5, -0.1, -0.1), Vector3d(1.1, 1.1, 1.1), 
    #         Vector3d(0, 0, 0), Vector3d(0, 0, 0), Vector3d(1, 0, 0), 0, counter[N - 1 - i])
    #     sim.set_DBC_with_range(Vector3d(-0.1, -0.1, -0.1), Vector3d(1e-5, 1.1, 1.1), 
    #         Vector3d(0, 0, 0), Vector3d(0, 0, 0), Vector3d(1, 0, 0), 0, counter[2 * N - 1 - i])
    # grab cards bottom
    for j in range(N):
        i = N - 1 - j
        counter.append(sim.add_shell_3D("input/card" + size + ".obj", Vector3d(-0.05 - 0.22e-3 * i, 0.032 + 0.22e-3 * i, 0), \
            Vector3d(0, 0, 0), Vector3d(0, 0, 1), 45))
        # left bottom
        sim.set_DBC_with_range(Vector3d(-0.1, -0.1, -0.1), Vector3d(1e-5, 1.1, 1.1), 
            Vector3d(5e-3, 0, 0), Vector3d(0, 0, 0), Vector3d(1, 0, 0), 0, counter[-1])
        # left top
        sim.set_DBC_with_range(Vector3d(1-1e-5, -0.1, -0.1), Vector3d(1.1, 1.1, 1.1), 
            Vector3d(0, -5e-3, 0), Vector3d(0, 0, 0), Vector3d(1, 0, 0), 0, counter[-1])
        counter.append(sim.add_shell_3D("input/card" + size + ".obj", Vector3d(0.05 + 0.22e-3 * i, 0.032 + 0.22e-3 * i, 0), \
            Vector3d(0, 0, 0), Vector3d(0, 0, 1), -45))
        # right bottom
        sim.set_DBC_with_range(Vector3d(1-1e-5, -0.1, -0.1), Vector3d(1.1, 1.1, 1.1), 
            Vector3d(-5e-3, 0, 0), Vector3d(0, 0, 0), Vector3d(1, 0, 0), 0, counter[-1])
        # right top
        sim.set_DBC_with_range(Vector3d(-0.1, -0.1, -0.1), Vector3d(1e-5, 1.1, 1.1), 
            Vector3d(0, -5e-3, 0), Vector3d(0, 0, 0), Vector3d(1, 0, 0), 0, counter[-1])

    sim.dt = 0.02
    sim.frame_dt = 0.02 #TODO: smaller dt for slow motion
    sim.frame_num = 300
    sim.withCollision = True
    sim.epsv2 = 1e-9
    sim.MDBC_tmax = 1
    sim.DBCPopBackTStart = 1
    sim.DBCPopBackTEnd = 1e10
    sim.DBCPopBackStep = 2
    sim.DBCPopBackAmt = 54
    sim.PNTol = 5e-4

    sim.MDBC_tmax2 = 5
    sim.MDBC_tmin2 = 3
    sim.MDBC_period2 = 1

    muComp = [] # 1 + 54 rows and cols
    for i in range(57):
        for j in range(57):
            if i < 3 or j < 3:
                muComp.append(0.1) # card-floor
            else:
                muComp.append(0.02) # card-card
    sim.muComp = StdVectorXd(muComp)

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

    sim.initialize_OIPC(2e-4, 1e-4)

    sim.run()
