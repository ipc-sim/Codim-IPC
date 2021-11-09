import sys
sys.path.insert(0, "../../Python")
import Drivers
from JGSL import *

if __name__ == "__main__":
    sim = Drivers.FEMDiscreteShellBase("double", 3)

    algI = 0
    if len(sys.argv) > 1:
        algI = int(sys.argv[1])

    clothI = 2
    if len(sys.argv) > 2:
        clothI = int(sys.argv[2])

    garmentName = 'multilayer'
    if len(sys.argv) > 3:
        garmentName = sys.argv[3]
    
    membEMult = 0.01
    if len(sys.argv) > 4:
        membEMult = float(sys.argv[4])
    
    bendEMult = 0.1
    if len(sys.argv) > 5:
        bendEMult = float(sys.argv[5])

    seqName = 'Kick'
    if len(sys.argv) > 6:
        seqName = sys.argv[6]

    sim.frame_num = 145
    if len(sys.argv) > 7:
        sim.frame_num = int(sys.argv[7])
    
    # load garment rest shape
    sim.add_garment_3D("input/" + garmentName + "/stage.obj", Vector3d(0, 0, 0), Vector3d(1, 1, 1),\
        Vector3d(0, 0, 0), Vector3d(1, 0, 0), -90)

    sim.seqDBC = sim.compNodeRange[-1]
    sim.add_mannequin("input/wm2_15k.obj", Vector3d(0, 0, 0), Vector3d(1, 1, 1),\
        Vector3d(0, 0, 0), Vector3d(1, 0, 0), -90)
    sim.seqDBCPath = "input/" + seqName

    sim.dt = 0.04
    sim.frame_dt = 0.04
    sim.withCollision = True
    sim.k_stitch = 1000
    sim.mu = 0.2


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

    # load draped garment as initial config
    sim.load_frame("input/" + garmentName + "/drape.obj") 

    sim.initialize_OIPC(1e-3, 0)

    sim.run()
