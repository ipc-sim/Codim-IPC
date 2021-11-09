import sys
sys.path.insert(0, "../../build")
import os
try:
    os.mkdir("output")
except OSError:
    pass

from JGSL import *
from .SimulationBase import SimulationBase


class FEMSimulationBase(SimulationBase):
    def __init__(self, precision, dim, model="FCR"):
        super().__init__(precision, dim)
        self.X0 = Storage.V2dStorage() if self.dim == 2 else Storage.V3dStorage()
        self.X = Storage.V2dStorage() if self.dim == 2 else Storage.V3dStorage()
        self.Elem = Storage.V3iStorage() if self.dim == 2 else Storage.V4iStorage()
        self.nodeAttr = Storage.V2dV2dV2dSdStorage() if self.dim == 2 else Storage.V3dV3dV3dSdStorage()
        self.elemAttr = Storage.M2dM2dSdStorage() if self.dim == 2 else Storage.M3dM3dSdStorage()
        self.model = model
        Set_Parameter("Elasticity_model", self.model)
        self.elasticity = FIXED_COROTATED_2.Create() if self.dim == 2 else FIXED_COROTATED_3.Create() #TODO: different material switch
        self.DBC = Storage.V3dStorage() if self.dim == 2 else Storage.V4dStorage()
        self.DBCMotion = Storage.V2iV2dV2dV2dSdStorage() if self.dim == 2 else Storage.V2iV3dV3dV3dSdStorage()
        self.gravity = self.Vec(0, -9.81) if self.dim == 2 else self.Vec(0, -9.81, 0)
        if self.dim == 3:
            self.TriVI2TetVI = Storage.SiStorage() #TODO: together?
            self.Tri = Storage.V3iStorage()
        self.PNIterCountSU = [0] * 5
        self.PNIterCount = 0
        self.PNTol = 1e-2
        self.meshCounter = Vector4i()
        self.withCollision = True
        self.useNewton = True
        self.dHat2 = 1e-6
        self.EIPC = False
        self.kappa = Vector3d(1e4, 0, 0) #1e9 for 2d fracture
        self.mu = 0
        self.epsv2 = 1e-6
        self.staticSolve = False
        self.withShapeMatching = False
        self.enableFracture = False
        self.debrisAmt = -1
        self.strengthenFactor = 2
        self.edge_dupV = StdVectorArray4i() if self.dim == 2 else StdVectorArray6i()
        self.isFracture_edge = StdVectorXi()
        self.incTriV_edge = StdVectorVector4i()
        self.incTriRestDist2_edge = StdVectorXd()
        self.fractureRatio2 = 2
        self.finalV2old = StdVectorXi()
        self.rho0 = 0.0
        Set_Parameter("Zero_velocity", True)

    def set_object(self, filePath, velocity, p_density, E, nu):
        self.add_object(filePath, self.Vec(0), self.Vec(0), self.Vec(0), 0, self.Vec(1))
        self.initialize_added_objects(velocity, p_density, E, nu)

    def add_object(self, filePath, translate, rotCenter, rotAxis, rotDeg, scale):
        if self.dim == 2:
            self.meshCounter = MeshIO.Read_TriMesh_Obj(filePath, self.X0, self.Elem)
        else:
            self.meshCounter = MeshIO.Read_TetMesh_Vtk(filePath, self.X0, self.Elem)
        MeshIO.Transform_Points(translate, rotCenter, rotAxis, rotDeg, scale, self.meshCounter, self.X0)
    
    #TODO: objects with different density, E, and nu
    def initialize_added_objects(self, velocity, p_density, E, nu):
        if self.dim == 3:
            MeshIO.Find_Surface_TriMesh(self.X0, self.Elem, self.TriVI2TetVI, self.Tri)
        MeshIO.Append_Attribute(self.X0, self.X)
        vol = Storage.SdStorage()
        FEM.Compute_Vol_And_Inv_Basis(self.X0, self.Elem, vol, self.elemAttr)
        if self.dim == 2:
            FIXED_COROTATED_2.Append_All_FEM(self.elasticity, self.meshCounter, vol, E, nu)
        else:
            FIXED_COROTATED_3.Append_All_FEM(self.elasticity, self.meshCounter, vol, E, nu)
        FEM.Compute_Mass_And_Init_Velocity(self.X0, self.Elem, vol, p_density, velocity, self.nodeAttr)
        self.rho0 = p_density
        if self.enableFracture:
            FEM.Fracture.Initialize_Fracture(self.X0, self.Elem, self.debrisAmt, self.fractureRatio2, self.strengthenFactor, self.edge_dupV, \
                self.isFracture_edge, self.incTriV_edge, self.incTriRestDist2_edge)
        if self.EIPC:
            FEM.TimeStepper.ImplicitEuler.Initialize_Elastic_IPC(self.X0, self.Elem, self.dt, E, nu, self.dHat2, self.kappa)

    def initialize_OIPC(self, dHat2, stiffMult = 1):
        self.EIPC = False
        self.dHat2 = FEM.DiscreteShell.Initialize_OIPC_VM(dHat2, self.nodeAttr, self.kappa, stiffMult)

    def set_DBC(self, DBCRangeMin, DBCRangeMax, v, rotCenter, rotAxis, angVelDeg):
        FEM.Init_Dirichlet(self.X0, DBCRangeMin, DBCRangeMax,  v, rotCenter, rotAxis, angVelDeg, self.DBC, self.DBCMotion, Vector4i(0, 0, 1000000000, -1))

    #TODO: def add_object for multi-object system

    def advance_one_time_step(self, dt):
        #TODO: self.tol
        if self.current_frame < self.frame_num:
            FEM.Step_Dirichlet(self.DBCMotion, dt, self.DBC)
        else:
            FEM.Reset_Dirichlet(self.X, self.DBC)
            self.dt = 1
            Set_Parameter("Terminate", True)
        # FEM.TimeStepper.ImplicitEuler.Check_Gradient_FEM(self.Elem, self.DBC, \
        #     self.gravity, self.dt, self.X, self.X0, self.nodeAttr, self.elemAttr, self.elasticity)

        if self.useNewton:
            if self.EIPC:
                self.PNIterCount += FEM.TimeStepper.ImplicitEuler.Advance_One_Step_EIPC(self.Elem, self.DBC, \
                    self.gravity, self.dt, self.PNTol, self.withCollision, self.dHat2, self.kappa, self.mu, self.epsv2, \
                    self.staticSolve, self.withShapeMatching, self.output_folder, self.current_frame, self.X, self.X0, self.nodeAttr, self.elemAttr, self.elasticity)
            else:
                self.PNIterCount += FEM.TimeStepper.ImplicitEuler.Advance_One_Step(self.Elem, self.DBC, \
                    self.gravity, self.dt, self.PNTol, self.withCollision, self.dHat2, self.kappa, self.mu, self.epsv2, \
                    self.staticSolve, self.withShapeMatching, self.output_folder, self.current_frame, self.X, self.X0, self.nodeAttr, self.elemAttr, self.elasticity)
        else:
            self.PNIterCount += FEM.TimeStepper.ImplicitEuler.Advance_One_Step_SU(self.Elem, self.DBC, \
                self.gravity, self.dt, self.PNTol, self.withCollision, self.dHat2, self.kappa, self.mu, self.epsv2, \
                self.output_folder, self.current_frame, self.X, self.X0, self.nodeAttr, self.elemAttr, self.elasticity)

        print("Total PN iteration count: ", self.PNIterCount, "\n")
        if self.enableFracture:
            if FEM.Fracture.Edge_Fracture(self.X, self.Elem, self.incTriV_edge, self.incTriRestDist2_edge, \
                self.fractureRatio2, self.isFracture_edge):
                print("edge fractured")
                FEM.Fracture.Node_Fracture(self.edge_dupV, self.isFracture_edge, \
                    self.X, self.Elem, self.finalV2old)
                print("node fractured")
                FEM.Fracture.Update_Fracture(self.Elem, self.rho0, self.finalV2old, \
                    self.X0, self.nodeAttr, self.DBC, self.DBCMotion, self.withCollision, self.dHat2, self.X)
                if self.dim == 3:
                    MeshIO.Find_Surface_TriMesh(self.X0, self.Elem, self.TriVI2TetVI, self.Tri)
                print("fracture updated")

    def write(self, frame_idx):
        MeshIO.Write_TriMesh_Obj(self.X, self.Elem, self.output_folder + str(frame_idx) + ".obj") if self.dim == 2 \
            else MeshIO.Write_Surface_TriMesh_Obj(self.X, self.TriVI2TetVI, self.Tri, \
                self.output_folder + str(frame_idx) + ".obj")
    
    def write_com(self):
        FEM.TimeStepper.ImplicitEuler.Write_COM(self.X, self.nodeAttr, self.output_folder + "com.txt")