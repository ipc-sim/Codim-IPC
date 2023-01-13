import sys
import os
sys.path.insert(0, "../../Python")
import Drivers
from JGSL import *

sys.path.insert(0, "../../Python")
import Drivers
from JGSL import *


def run(elasticity=1e5):
    algI = 0
    clothI = 0
    membEMult = 0.01
    bendEMult = 1
    sim = Drivers.FEMDiscreteShellBase("double", 3)

    sim.mu = 0.1  # set the coefficient of friction between the objects

    # =====================EXPERIMENTAL SETUP===============================
    # Add a 3d model that represents Floor (Bottom Plane)
    floorPlane = sim.add_shell_with_scale_3D("input/square21.obj", Vector3d(0, 0, 0), Vector3d(4, 4, 4), \
                                             Vector3d(0, 0, 0), Vector3d(0, 0, 0), 0)

    # Set the Dirichlet Boundary Condition (floor remains stationary, ie, no velocity and no rotation)
    sim.set_DBC(Vector3d(-0.1, -0.1, -0.1), Vector3d(1.1, 1.1, 1.1),
                Vector3d(0, 0, 0), Vector3d(0, 0, 0), Vector3d(0, 0, 0), 0)

    # Add a Sphere and place it on top of the bottom plane (Floor)
    sphere = sim.add_object_3D("../FEM/input/sphere1K.vtk", Vector3d(0, 0.5, 0), \
                               Vector3d(0, 0, 0), Vector3d(1, 0, 0), 0, Vector3d(1, 1, 1))

    # Add the top plane and place it above the sphere
    ceilingPlane = sim.add_object_3D("../FEM/input/cube.vtk", Vector3d(-1, 1.6, -1), \
                                     Vector3d(0, 0, 0), Vector3d(1, 0, 0), 0, Vector3d(2, .1, 2))

    #=====================EXPERIMENTAL SETUP END===============================

    sim.dt = 0.04               #time stepping delta
    sim.frame_dt = 0.04         #frame delta corresponding to time steppping delta
    sim.frame_num = 150         #Number of frames to simulate
    sim.withCollision = True    #Simulate collision (So that deformation effect is seen)

    # density, E, nu, thickness, initial displacement case
    if algI == 0:
        # isotrophic object simulation
        sim.initialize(sim.cloth_density_iso[clothI], sim.cloth_Ebase_iso[clothI] * membEMult,
            sim.cloth_nubase_iso[clothI], sim.cloth_thickness_iso[clothI], 0)
        sim.bendingStiffMult = bendEMult / membEMult
        sim.kappa_s = Vector2d(1e3, 0)
        sim.s = Vector2d(sim.cloth_SL_iso[clothI], 0)
    elif algI == 1:
        # isotrophic object simulation with no strain limit
        sim.initialize(sim.cloth_density_iso[clothI], sim.cloth_Ebase_iso[clothI] * membEMult,
            sim.cloth_nubase_iso[clothI], sim.cloth_thickness_iso[clothI], 0)
        sim.bendingStiffMult = bendEMult / membEMult
        sim.kappa_s = Vector2d(0, 0)
    elif algI == 2:
        # anisotrophic object simulation
        sim.initialize(sim.cloth_density[clothI], sim.cloth_Ebase[clothI], # actually only affect bending
            0, sim.cloth_thickness[clothI], 0)
        sim.bendingStiffMult = bendEMult
        sim.fiberStiffMult = sim.cloth_weftWarpMult[clothI] * membEMult
        sim.inextLimit = sim.cloth_inextLimit[clothI]
        sim.kappa_s = Vector2d(0, 0)

    #Initilaize the experimental setup for all objects (2 planes and 1 sphere)
    #Density = 1000  
    #Young's Modulus (For elasticity) = 1e5 (Value closer to one means more rigid body ,so the object doesn't deform)
    #Poission's Ratio = 0.4 (We are only applying parallel force and the shear force is only at the end of simulation when plane slides off. So, this shouldn't matter much)
    sim.initialize_added_objects(Vector3d(0, 0, 0), 1000,s 1e5, 0.4)

    #We make the Upper plane more dense (2 times) and more rigid (10000 times) so that it doesn't deform and the impact on sphere is more 
    sim.adjust_material(ceilingPlane, 2, 1e4)

    #Set Elasticity of sphere based on user passed parameter 
    sim.adjust_material(sphere, 1, elasticity / 1e5)

    sim.initialize_OIPC(1e-3, 0)

    sim.run()
                                     
    #=====================Running the simulations===============================
