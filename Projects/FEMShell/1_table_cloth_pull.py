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

    size = 'default'
    if len(sys.argv) > 3:
        size = sys.argv[3]
    
    membEMult = 0.1
    if len(sys.argv) > 4:
        membEMult = float(sys.argv[4])
    
    bendEMult = 1
    if len(sys.argv) > 5:
        bendEMult = float(sys.argv[5])

    v = -1
    if len(sys.argv) > 6:
        v = -float(sys.argv[6])


    # fixed floor
    floorCounter = sim.add_shell_with_scale_3D("input/tableFloor.obj", Vector3d(-0.5, 0, 0), Vector3d(1, 1, 1),\
        Vector3d(0, 0, 0), Vector3d(0, 0, 0), 0)
    # v, rotCenter, rotAxis, angVelDeg
    sim.set_DBC_with_range(Vector3d(-0.1, -0.1, -0.1), Vector3d(1.1, 1.1, 1.1), 
        Vector3d(0, 0, 0), Vector3d(0, 0, 0), Vector3d(0, 0, 0), 0, floorCounter)

    # cloth
    counter = sim.add_shell_3D("input/tableCloth.obj", Vector3d(0, 0, 0), \
        Vector3d(0, 0, 0), Vector3d(0, 1, 0), 0)
    sim.set_DBC_with_range(Vector3d(-0.1, -0.1, 0.2), Vector3d(1e-1, 1.1, 0.8), 
        Vector3d(v, 0, 0), Vector3d(0, 0, 0), Vector3d(1, 0, 0), 0, counter)

    # geometries
    sim.add_object_3D("../FEM/input/butterPlate.vtk", Vector3d(0, 0, 0), \
        Vector3d(0, 0, 0), Vector3d(1, 0, 0), 0, Vector3d(1, 1, 1))
    butterCounter = sim.add_object_3D("../FEM/input/butter.vtk", Vector3d(0, 0.001, 0), \
        Vector3d(0, 0, 0), Vector3d(1, 0, 0), 0, Vector3d(1, 1, 1))
    sim.add_object_3D("../FEM/input/fork.vtk", Vector3d(0, 0, 0), \
        Vector3d(0, 0, 0), Vector3d(1, 0, 0), 0, Vector3d(1, 1, 1))
    meatCounter = sim.add_object_3D("../FEM/input/meat.vtk", Vector3d(0, 0, 0), \
        Vector3d(0, 0, 0), Vector3d(1, 0, 0), 0, Vector3d(1, 1, 1))
    sim.add_object_3D("../FEM/input/meatPlate.vtk", Vector3d(0, 0, 0), \
        Vector3d(0, 0, 0), Vector3d(1, 0, 0), 0, Vector3d(1, 1, 1))
    sim.add_object_3D("../FEM/input/knife.vtk", Vector3d(0, 0, 0), \
        Vector3d(0, 0, 0), Vector3d(1, 0, 0), 0, Vector3d(1, 1, 1))
    sim.add_object_3D("../FEM/input/glass.vtk", Vector3d(0, 0, 0), \
        Vector3d(0, 0, 0), Vector3d(1, 0, 0), 0, Vector3d(1, 1, 1))
    sim.add_object_3D("../FEM/input/bottle.vtk", Vector3d(0, 0, 0), \
        Vector3d(0, 0, 0), Vector3d(1, 0, 0), 0, Vector3d(1, 1, 1))
    # fixed table
    tableCounter = sim.add_shell_3D("input/table.obj", Vector3d(0, 0, 0), \
        Vector3d(0, 0, 0), Vector3d(1, 0, 0), 0)
    sim.set_DBC_with_range(Vector3d(-0.1, -0.1, -0.1), Vector3d(1.1, 1.1, 1.1), 
        Vector3d(0, 0, 0), Vector3d(0, 0, 0), Vector3d(1, 0, 0), 0, tableCounter)

    
    sim.mu = 0.2

    sim.dt = 0.01
    sim.frame_dt = 0.01
    sim.frame_num = 300
    sim.withCollision = True
    sim.MDBC_tmin = 1
    sim.MDBC_tmax = 1 + 1 / abs(v)
    # sim.MDBC_tmin = 1e10
    # sim.MDBC_tmin = -1
    # sim.MDBC_tmax = 1

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

    sim.initialize_added_objects(Vector3d(0, 0, 0), 2000, 1e9, 0.4)
    sim.adjust_material(butterCounter, 870.0 / 2000, 0.01)
    sim.adjust_material(meatCounter, 0.5, 0.005)
    
    sim.initialize_OIPC(1e-3, 0)

    # sim.load_frame('/home/minchen/Desktop/JGSL/Projects/FEMShell/output/tableCloth_drag_fancy/0_0_defaultSize_0.1_1_4_ACCDv3_NH/50.obj')
    # sim.load_velocity('/home/minchen/Desktop/JGSL/Projects/FEMShell/output/tableCloth_drag_fancy/0_0_defaultSize_0.1_1_4_ACCDv3_NH', 50, sim.dt)
    # sim.load_frame('/home/minchen/Desktop/JGSL/Projects/FEMShell/output/tableCloth_drag_fancy/0_0_defaultSize_0.1_1_0.5_ACCDv3_NH_dt0.01/300.obj')
    # sim.load_velocity('/home/minchen/Desktop/JGSL/Projects/FEMShell/output/tableCloth_drag_fancy/0_0_defaultSize_0.1_1_0.5_ACCDv3_NH_dt0.01', 300, sim.dt)

    sim.run()
