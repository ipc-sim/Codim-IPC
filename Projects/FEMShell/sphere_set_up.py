import sys
import os

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