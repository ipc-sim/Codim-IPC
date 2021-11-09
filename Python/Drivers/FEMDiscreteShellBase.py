import sys
sys.path.insert(0, "../../build")
import os
try:
    os.mkdir("output")
except OSError:
    pass
import math

from JGSL import *
from .SimulationBase import SimulationBase


class FEMDiscreteShellBase(SimulationBase):
    def __init__(self, precision, dim):
        super().__init__(precision, dim)
        self.X0 = Storage.V2dStorage() if self.dim == 2 else Storage.V3dStorage()
        self.X = Storage.V2dStorage() if self.dim == 2 else Storage.V3dStorage()
        self.X_stage = Storage.V2dStorage() if self.dim == 2 else Storage.V3dStorage()
        self.Elem = Storage.V2iStorage() if self.dim == 2 else Storage.V3iStorage()
        self.segs = StdVectorVector2i()
        self.outputSeg = False
        self.nodeAttr = Storage.V2dV2dV2dSdStorage() if self.dim == 2 else Storage.V3dV3dV3dSdStorage()
        self.massMatrix = CSR_MATRIX_D()
        self.elemAttr = Storage.M2dM2dSdStorage()
        self.elasticity = FIXED_COROTATED_2.Create() #TODO: different material switch
        self.DBC = Storage.V3dStorage() if self.dim == 2 else Storage.V4dStorage()
        self.DBCMotion = Storage.V2iV2dV2dV2dSdStorage() if self.dim == 2 else Storage.V2iV3dV3dV3dSdStorage()
        self.gravity = self.Vec(0, -9.81) if self.dim == 2 else self.Vec(0, -9.81, 0)
        self.bodyForce = StdVectorXd()
        self.edge2tri = StdMapPairiToi()
        self.edgeStencil = StdVectorVector4i()
        self.edgeInfo = StdVectorVector3d()
        self.thickness = 0 #TODO different thickness
        self.bendingStiffMult = 1
        self.fiberStiffMult = Vector4d(0, 0, 0, 0)
        self.inextLimit = Vector3d(0, 0, 0)
        self.s = Vector2d(1.01, 0)
        self.sHat = Vector2d(1, 1)
        self.kappa_s = Vector2d(0, 0)
        self.withCollision = False
        self.PNIterCount = 0
        self.PNTol = 1e-3
        self.dHat2 = 1e-6
        self.kappa = Vector3d(1e5, 0, 0)
        self.mu = 0
        self.epsv2 = 1e-6
        self.fricIterAmt = 1
        self.compNodeRange = StdVectorXi()
        self.muComp = StdVectorXd()
        self.staticSolve = False
        self.t = 0.0
        self.MDBC_tmax = 1e10
        self.MDBC_tmin = -1
        self.MDBC_period = 1e10
        self.MDBC_periodCounter = 1

        # 100% cotton, 100% wool, 95% wool 5% lycra, 100% polyester (PES), paper
        self.cloth_density_iso = [472.641509, 413.380282, 543.292683, 653.174603, 800]
        self.cloth_thickness_iso = [0.318e-3, 0.568e-3, 0.328e-3, 0.252e-3, 0.3e-3]
        self.cloth_Ebase_iso = [0.821e6, 0.170e6, 0.076e6, 0.478e6, 3e9] # for memb: 0.1x, 0.2, 1x, 0.1x, 2e4 ~ 1e5, for bending: 5e4 ~ 1e6
        self.cloth_membEMult_iso = [0.1, 0.2, 1, 0.1, 1]
        self.cloth_nubase_iso = [0.243, 0.277, 0.071, 0.381, 0.3]
        self.cloth_SL_iso = [1.0608, 1.085, 1.134, 1.0646, 1.005]
        # self.cloth_SL_iso = [1.0608, 1.085, 1.253, 1.0646, 1.005] # argus
        # self.cloth_weftWarpMult = [Vector2d(38.6577344702, 14.1473812424),\
        #     Vector2d(127.588235294, 46.9352941176), Vector2d(2.28947368421, 10.1973684211),\
        #     Vector2d(9.7782426778, 23.4225941423), Vector2d(0, 0)]
        # self.cloth_stiffRatio = [39.6577344702, 128.588235294, 11.1973684211, 24.4225941423, 1]

        # cotton, wool, canvas, silk, denim
        self.cloth_density = [103.6, 480.6, 294, 83, 400]
        self.cloth_thickness = [0.18e-3, 1.28e-3, 0.53e-3, 0.18e-3, 0.66e-3]
        self.cloth_Ebase = [1.076e6, 0.371e6, 2.009e6, 0.57e6, 2.448e6]
        self.cloth_weftWarpMult = [Vector4d(15.557e6, 7.779e6, 25.004e6, 1.076e6),\
            Vector4d(2.29e6, 1.145e6, 2.219e6, 0.371e6),\
            Vector4d(5.366e6, 2.683e6, 19.804e6, 2.009e6),\
            Vector4d(4.3e6, 4.971e6, 9.942e6, 0.57e6),\
            Vector4d(4.793e6, 4.515e6, 9.029e6, 2.448e6)]
        self.cloth_inextLimit = [Vector3d(0.14, 0.14, 0.063),\
            Vector3d(0.5, 0.62, 0.12),\
            Vector3d(0.11, 0.067, 0.059),\
            Vector3d(0.41, 0.34, 0.11),\
            Vector3d(0.28, 0.28, 0.05)]
        self.withVol = False
        self.tet = Storage.V4iStorage()
        self.tetAttr = Storage.M3dM3dSdStorage()
        self.tetElasticity = FIXED_COROTATED_3.Create() #TODO: different material switch
        self.TriVI2TetVI = Storage.SiStorage() #TODO: together?
        self.Tri = Storage.V3iStorage()
        self.outputRod = False
        self.rod = StdVectorVector2i()
        self.rodInfo = StdVectorVector3d()
        self.rodHinge = StdVectorVector3i()
        self.rodHingeInfo = StdVectorVector3d()
        self.discrete_particle = StdVectorXi()
        self.elasticIPC = False
        self.split = False

        self.stitchInfo = StdVectorVector3i()
        self.stitchRatio = StdVectorXd()
        self.k_stitch = 10

        self.seqDBC = -1
        self.seqDBCPath = ''
        self.curFrameNum = 0

        self.DBCPopBackCounter = 0
        self.DBCPopBackFrameCounter = 0
        self.DBCPopBackAmt = 0
        self.DBCPopBackTStart = 0
        self.DBCPopBackTEnd = -1
        self.DBCPopBackStep = 1
        self.DBCPopBackBatch = 1

        self.DBCMotion2 = Storage.V2iV2dV2dV2dSdStorage() if self.dim == 2 else Storage.V2iV3dV3dV3dSdStorage()
        self.MDBC_tmax2 = -1
        self.MDBC_tmin2 = 0
        self.MDBC_period2 = 1e10
        self.MDBC_periodCounter2 = 1

        self.lv_fn = -1
        self.shell_density = 1000
        self.shell_E = 1e5
        self.shell_nu = 0.4
        self.shell_thickness = 0.001

        self.flow = False
        self.normalFlowMag = 0

        self.scaleX = 1
        self.scaleY = 1
        self.scaleZ = 1
        self.scaleXTarget = 1
        self.scaleYTarget = 1
        self.scaleZTarget = 1
        self.scaleXMultStep = 1
        self.scaleYMultStep = 1
        self.scaleZMultStep = 1
        self.zeroVel = False

    def add_shell_3D(self, filePath, translate, rotCenter, rotAxis, rotDeg): # 3D
        return FEM.DiscreteShell.Add_Shell(filePath, translate, Vector3d(1, 1, 1), rotCenter, rotAxis, rotDeg, self.X, self.Elem, self.compNodeRange)
    
    def add_shell_with_scale_3D(self, filePath, translate, scale, rotCenter, rotAxis, rotDeg):
        return FEM.DiscreteShell.Add_Shell(filePath, translate, scale, rotCenter, rotAxis, rotDeg, self.X, self.Elem, self.compNodeRange)
    
    def add_garment_3D(self, filePath, translate, scale, rotCenter, rotAxis, rotDeg):
        FEM.DiscreteShell.Add_Garment(filePath, translate, scale, rotCenter, rotAxis, rotDeg, \
            self.X, self.X_stage, self.Elem, self.stitchInfo, self.stitchRatio, self.compNodeRange)

    def add_mannequin(self, filePath, translate, scale, rotCenter, rotAxis, rotDeg):
        meshCounter = FEM.DiscreteShell.Add_Shell(filePath, translate, scale, rotCenter, rotAxis, rotDeg, self.X, self.Elem, self.compNodeRange)
        MeshIO.Transform_Points(Vector3d(1e6, 0, 0), Vector3d(0, 0, 0), Vector3d(1, 0, 0), 0, Vector3d(1, 1, 1), meshCounter, self.X)
        self.set_DBC(Vector3d(0.5, -1, -1), Vector3d(1.1, 1.1, 1.1), Vector3d(0, 0, 0), Vector3d(0, 0, 0), Vector3d(1, 0, 0), 0)
        MeshIO.Transform_Points(Vector3d(-1e6, 0, 0), Vector3d(0, 0, 0), Vector3d(1, 0, 0), 0, Vector3d(1, 1, 1), meshCounter, self.X)
        FEM.Reset_Dirichlet(self.X, self.DBC)
    
    def add_seg_3D(self, filePath, translate, rotCenter, rotAxis, rotDeg, scale):
        meshCounter = MeshIO.Read_SegMesh_Seg(filePath, self.X, self.segs)
        MeshIO.Transform_Points(translate, rotCenter, rotAxis, rotDeg, scale, meshCounter, self.X)
        self.outputSeg = True
    
    def add_rod_3D(self, filePath, translate, rotCenter, rotAxis, rotDeg, scale):
        meshCounter = MeshIO.Read_SegMesh_Seg(filePath, self.X, self.rod)
        MeshIO.Transform_Points(translate, rotCenter, rotAxis, rotDeg, scale, meshCounter, self.X)
        self.outputRod = True

    def make_and_add_rod_3D(self, rLen, nSeg, translate, rotCenter, rotAxis, rotDeg, scale):
        meshCounter = FEM.DiscreteShell.Make_Rod(rLen, nSeg, self.X, self.rod)
        MeshIO.Transform_Points(translate, rotCenter, rotAxis, rotDeg, scale, meshCounter, self.X)
        self.outputRod = True
    
    def make_and_add_rod_net_3D(self, rLen, nSeg, midPointAmt, translate, rotCenter, rotAxis, rotDeg, scale):
        meshCounter = FEM.DiscreteShell.Make_Rod_Net(rLen, nSeg, midPointAmt, self.X, self.rod)
        MeshIO.Transform_Points(translate, rotCenter, rotAxis, rotDeg, scale, meshCounter, self.X)
        self.outputRod = True

    def add_particle_3D(self, boxLen, num, randScale, translate, rotCenter, rotAxis, rotDeg, scale):
        meshCounter = FEM.DiscreteShell.Add_Discrete_Particles(boxLen, num, randScale, self.X, self.discrete_particle, self.compNodeRange)
        MeshIO.Transform_Points(translate, rotCenter, rotAxis, rotDeg, scale, meshCounter, self.X)
        return meshCounter
    
    def add_object_3D(self, filePath, translate, rotCenter, rotAxis, rotDeg, scale):
        self.withVol = True
        meshCounter = MeshIO.Read_TetMesh_Vtk(filePath, self.X, self.tet)
        MeshIO.Transform_Points(translate, rotCenter, rotAxis, rotDeg, scale, meshCounter, self.X)
        return meshCounter

    def initialize(self, p_density, E, nu, thickness, caseI):
        MeshIO.Append_Attribute(self.X, self.X0)
        self.shell_density = p_density
        self.shell_E = E
        self.shell_nu = nu
        self.shell_thickness = thickness
        self.thickness = thickness # later used as offset
        if caseI != 0:
            self.gravity = self.Vec(0, 0) if self.dim == 2 else self.Vec(0, 0, 0)
        self.dHat2 = FEM.DiscreteShell.Initialize_Shell_Hinge_EIPC(p_density, E, nu, thickness, self.dt, self.dHat2, self.X, self.Elem, self.segs, \
            self.edge2tri, self.edgeStencil, self.edgeInfo, self.nodeAttr, self.massMatrix, self.gravity, self.bodyForce, \
            self.elemAttr, self.elasticity, self.kappa)
        if self.flow:
            FEM.Boundary_Dirichlet(self.X, self.Elem, self.DBC) # fix boundary nodes

    def reinitialize_argus(self, filePath):
        FEM.DiscreteShell.Update_Material_With_Tex_Shell(filePath, 0, 0, self.shell_density, self.shell_thickness, \
            self.edge2tri, self.edgeStencil, self.edgeInfo, self.nodeAttr, self.massMatrix, self.gravity, self.bodyForce, \
            self.elemAttr, self.elasticity)
    
    def initialize_garment(self):
        FEM.DiscreteShell.Initialize_Garment(self.X, self.X_stage)

    #TODO: objects with different density, E, and nu
    def initialize_added_objects(self, velocity, p_density, E, nu):
        MeshIO.Find_Surface_TriMesh(self.X, self.tet, self.TriVI2TetVI, self.Tri)
        vol = Storage.SdStorage()
        FEM.Compute_Vol_And_Inv_Basis(self.X, self.tet, vol, self.tetAttr)
        FIXED_COROTATED_3.All_Append_FEM(self.tetElasticity, vol, E, nu)
        FEM.Compute_Mass_And_Init_Velocity_NoAlloc(self.X, self.tet, vol, p_density, velocity, self.nodeAttr)
        FEM.Augment_Mass_Matrix_And_Body_Force(self.X, self.tet, vol, p_density, self.gravity, self.massMatrix, self.bodyForce)
        #TODO: calculate per-primitive vol: FEM.TimeStepper.ImplicitEuler.Initialize_Elastic_IPC(self.X0, self.Elem, self.dt, E, nu, self.dHat2, self.kappa)
    
    def adjust_material(self, meshCounter, densityMult, YoungsMult):
        FEM.DiscreteShell.Adjust_Material(meshCounter, densityMult, YoungsMult, self.nodeAttr, self.massMatrix, self.bodyForce, self.tetElasticity)

    def initialize_rod(self, p_density, E, bendStiffMult, thickness):
        self.dHat2 = FEM.DiscreteShell.Initialize_Discrete_Rod(self.X, self.rod, p_density, E, thickness, \
            self.gravity, self.bodyForce, self.nodeAttr, self.massMatrix, self.rodInfo, \
            self.rodHinge, self.rodHingeInfo, bendStiffMult, self.dt, self.dHat2, self.kappa)

    def initialize_particle(self, p_density, E, thickness):
        FEM.DiscreteShell.Initialize_Discrete_Particle(self.X, self.discrete_particle, \
            p_density, E, thickness, self.gravity, self.bodyForce, self.nodeAttr, self.massMatrix)
    
    def initialize_EIPC(self, E, nu, thickness, h):
        self.dHat2 = FEM.DiscreteShell.Initialize_EIPC(E, nu, thickness, h, self.massMatrix, self.kappa)
        self.elasticIPC = True
    
    def initialize_OIPC(self, thickness, offset, stiffMult = 1):
        self.dHat2 = FEM.DiscreteShell.Initialize_OIPC(0.0, 0.0, thickness, 0.0, self.massMatrix, self.kappa, stiffMult)
        self.elasticIPC = False
        self.thickness = offset

    def load_frame(self, filePath):
        newX = Storage.V2dStorage() if self.dim == 2 else Storage.V3dStorage()
        newElem = Storage.V2iStorage() if self.dim == 2 else Storage.V3iStorage()
        MeshIO.Read_TriMesh_Obj(filePath, newX, newElem)
        self.X = newX
        # self.Elem = newElem
        FEM.Reset_Dirichlet(self.X, self.DBC)
    
    def load_velocity(self, folderPath, lastFrame, h):
        MeshIO.Load_Velocity(folderPath, lastFrame, h, self.nodeAttr)

    def set_DBC(self, DBCRangeMin, DBCRangeMax, v, rotCenter, rotAxis, angVelDeg):
        FEM.Init_Dirichlet(self.X, DBCRangeMin, DBCRangeMax,  v, rotCenter, rotAxis, angVelDeg, self.DBC, self.DBCMotion, Vector4i(0, 0, 1000000000, -1))

    def set_DBC_with_range(self, DBCRangeMin, DBCRangeMax, v, rotCenter, rotAxis, angVelDeg, vIndRange):
        FEM.Init_Dirichlet(self.X, DBCRangeMin, DBCRangeMax,  v, rotCenter, rotAxis, angVelDeg, self.DBC, self.DBCMotion, vIndRange)

    def set_DBC2_with_range(self, DBCRangeMin, DBCRangeMax, v, rotCenter, rotAxis, angVelDeg, vIndRange):
        FEM.Init_Dirichlet(self.X, DBCRangeMin, DBCRangeMax,  v, rotCenter, rotAxis, angVelDeg, self.DBC, self.DBCMotion2, vIndRange)

    def magnify_body_force(self, DBCRangeMin, DBCRangeMax, magnifyFactor):
        FEM.Magnify_Body_Force(self.X, DBCRangeMin, DBCRangeMax, magnifyFactor, self.bodyForce)

    def advance_one_time_step(self, dt):
        #TODO: self.tol
        if self.normalFlowMag != 0:
            # FEM.DiscreteShell.Initialize_Shell_Hinge_EIPC(self.shell_density, self.shell_E, self.shell_nu, self.shell_thickness, self.dt, self.dHat2, self.X, self.Elem, self.segs, \
            #     self.edge2tri, self.edgeStencil, self.edgeInfo, self.nodeAttr, self.massMatrix, self.gravity, self.bodyForce, \
            #     self.elemAttr, self.elasticity, self.kappa)
            FEM.DiscreteShell.Update_Normal_Flow_Neumann(self.X, self.Elem, self.massMatrix, self.normalFlowMag, self.bodyForce)
            # self.initialize_OIPC(math.sqrt(self.dHat2), 0)
        if self.t < self.MDBC_tmax and self.t > self.MDBC_tmin:
            if self.seqDBC < 0:
                if self.t - self.MDBC_tmin >= self.MDBC_period * self.MDBC_periodCounter:
                    self.MDBC_periodCounter += 1
                    FEM.Turn_Dirichlet(self.DBCMotion)
                FEM.Step_Dirichlet(self.DBCMotion, dt, self.DBC)
            else: # load mesh
                # FEM.Load_Dirichlet(self.seqDBCPath + '/Kyra_DVStandClubbing_' + str(self.curFrameNum).zfill(4) + '.obj', self.seqDBC, Vector3d(0, 0, 0), self.DBC) # argus
                # FEM.Load_Dirichlet(self.seqDBCPath + '/' + str(self.curFrameNum) + '.obj', self.seqDBC, Vector3d(0, -0.75, 0), self.DBC)
                FEM.Load_Dirichlet(self.seqDBCPath + '/shell' + str(self.curFrameNum) + '.obj', self.seqDBC, Vector3d(0, -0.75, 0), self.DBC)
                self.curFrameNum = self.curFrameNum + 1
        if self.DBCPopBackCounter < self.DBCPopBackAmt and self.t < self.DBCPopBackTEnd and self.t > self.DBCPopBackTStart:
            if self.DBCPopBackFrameCounter % self.DBCPopBackBatch == 0:
                for i in range(self.DBCPopBackStep):
                    FEM.Pop_Back_Dirichlet(self.DBCMotion, self.DBC)
                self.DBCPopBackCounter = self.DBCPopBackCounter + 1
            self.DBCPopBackFrameCounter = self.DBCPopBackFrameCounter + 1
        if self.t < self.MDBC_tmax2 and self.t > self.MDBC_tmin2:
            if self.t - self.MDBC_tmin2 >= self.MDBC_period2 * self.MDBC_periodCounter2:
                self.MDBC_periodCounter2 += 1
                FEM.Turn_Dirichlet(self.DBCMotion2)
            FEM.Step_Dirichlet(self.DBCMotion2, dt, self.DBC)
        if self.lv_fn >= 0:
            # load next frame target as rest shape
            newX = Storage.V2dStorage() if self.dim == 2 else Storage.V3dStorage()
            newElem = Storage.V2iStorage() if self.dim == 2 else Storage.V3iStorage()
            MeshIO.Read_TriMesh_Obj(self.seqDBCPath + '/' + str(self.lv_fn) + '.obj', newX, newElem)
            # reset rest shape info:
            FEM.DiscreteShell.Initialize_Shell_Hinge_EIPC(self.shell_density, self.shell_E, self.shell_nu, self.shell_thickness, self.dt, self.dHat2, newX, self.Elem, self.segs, \
                self.edge2tri, self.edgeStencil, self.edgeInfo, self.nodeAttr, self.massMatrix, self.gravity, self.bodyForce, \
                self.elemAttr, self.elasticity, self.kappa)
            self.initialize_OIPC(math.sqrt(self.dHat2), self.thickness)
            # reset velocity to point to next frame target
            MeshIO.Load_Velocity_X0(self.seqDBCPath, self.lv_fn, self.dt, self.X, self.nodeAttr) # calc dx based on x_cur
            FEM.Load_Dirichlet(self.seqDBCPath + '/' + str(self.lv_fn) + '.obj', 0, Vector3d(0, 0, 0), self.DBC)
            self.lv_fn = self.lv_fn + 1
        if self.elasticIPC:
            if self.split:
                self.PNIterCount = self.PNIterCount + FEM.DiscreteShell.Advance_One_Step_SIE_Hinge_EIPC(self.Elem, self.segs, self.DBC, \
                    self.edge2tri, self.edgeStencil, self.edgeInfo, \
                    self.thickness, self.bendingStiffMult, self.fiberStiffMult, self.inextLimit, self.s, self.sHat, self.kappa_s, \
                    self.bodyForce, self.dt, self.PNTol, self.withCollision, self.dHat2, self.kappa, self.mu, self.epsv2, self.fricIterAmt, \
                    self.compNodeRange, self.muComp, self.staticSolve, \
                    self.X, self.nodeAttr, self.massMatrix, self.elemAttr, self.elasticity, \
                    self.tet, self.tetAttr, self.tetElasticity, self.rod, self.rodInfo, \
                    self.rodHinge, self.rodHingeInfo, self.discrete_particle, self.output_folder)
            else:
                self.PNIterCount = self.PNIterCount + FEM.DiscreteShell.Advance_One_Step_IE_Hinge_EIPC(self.Elem, self.segs, self.DBC, \
                    self.edge2tri, self.edgeStencil, self.edgeInfo, \
                    self.thickness, self.bendingStiffMult, self.fiberStiffMult, self.inextLimit, self.s, self.sHat, self.kappa_s, \
                    self.bodyForce, self.dt, self.PNTol, self.withCollision, self.dHat2, self.kappa, self.mu, self.epsv2, self.fricIterAmt, \
                    self.compNodeRange, self.muComp, self.staticSolve, \
                    self.X, self.nodeAttr, self.massMatrix, self.elemAttr, self.elasticity, \
                    self.tet, self.tetAttr, self.tetElasticity, self.rod, self.rodInfo, \
                    self.rodHinge, self.rodHingeInfo, self.stitchInfo, self.stitchRatio, self.k_stitch,\
                    self.discrete_particle, self.output_folder)
        else:
            if self.flow:
                self.PNIterCount = self.PNIterCount + FEM.DiscreteShell.Advance_One_Step_IE_Flow(self.Elem, self.segs, self.DBC, \
                        self.edge2tri, self.edgeStencil, self.edgeInfo, \
                        self.thickness, self.bendingStiffMult, self.fiberStiffMult, self.inextLimit, self.s, self.sHat, self.kappa_s, \
                        self.bodyForce, self.dt, self.PNTol, self.withCollision, self.dHat2, self.kappa, self.mu, self.epsv2, self.fricIterAmt, \
                        self.compNodeRange, self.muComp, self.staticSolve, \
                        self.X, self.nodeAttr, self.massMatrix, self.elemAttr, self.elasticity, \
                        self.tet, self.tetAttr, self.tetElasticity, self.rod, self.rodInfo, \
                        self.rodHinge, self.rodHingeInfo, self.stitchInfo, self.stitchRatio, self.k_stitch,\
                        self.discrete_particle, self.output_folder)
            else:
                if self.split:
                    self.PNIterCount = self.PNIterCount + FEM.DiscreteShell.Advance_One_Step_SIE_Hinge(self.Elem, self.segs, self.DBC, \
                        self.edge2tri, self.edgeStencil, self.edgeInfo, \
                        self.thickness, self.bendingStiffMult, self.fiberStiffMult, self.inextLimit, self.s, self.sHat, self.kappa_s, \
                        self.bodyForce, self.dt, self.PNTol, self.withCollision, self.dHat2, self.kappa, self.mu, self.epsv2, self.fricIterAmt, \
                        self.compNodeRange, self.muComp, self.staticSolve, \
                        self.X, self.nodeAttr, self.massMatrix, self.elemAttr, self.elasticity, \
                        self.tet, self.tetAttr, self.tetElasticity, self.rod, self.rodInfo, \
                        self.rodHinge, self.rodHingeInfo, self.discrete_particle, self.output_folder)
                else:
                    self.PNIterCount = self.PNIterCount + FEM.DiscreteShell.Advance_One_Step_IE_Hinge(self.Elem, self.segs, self.DBC, \
                        self.edge2tri, self.edgeStencil, self.edgeInfo, \
                        self.thickness, self.bendingStiffMult, self.fiberStiffMult, self.inextLimit, self.s, self.sHat, self.kappa_s, \
                        self.bodyForce, self.dt, self.PNTol, self.withCollision, self.dHat2, self.kappa, self.mu, self.epsv2, self.fricIterAmt, \
                        self.compNodeRange, self.muComp, self.staticSolve, \
                        self.X, self.nodeAttr, self.massMatrix, self.elemAttr, self.elasticity, \
                        self.tet, self.tetAttr, self.tetElasticity, self.rod, self.rodInfo, \
                        self.rodHinge, self.rodHingeInfo, self.stitchInfo, self.stitchRatio, self.k_stitch,\
                        self.discrete_particle, self.output_folder)
        self.t += dt
        print("Total PN iteration count: ", self.PNIterCount, "\n")
        # self.load_velocity('/Users/minchen/Desktop/JGSL/Projects/FEMShell/output/shrink/', 1, dt)
        if self.scaleXMultStep != 1 or self.scaleYMultStep != 1 or self.scaleZMultStep != 1:
            self.scaleX *= self.scaleXMultStep
            self.scaleY *= self.scaleYMultStep
            self.scaleZ *= self.scaleZMultStep
            if (self.scaleXMultStep > 1 and self.scaleX > self.scaleXTarget) or (self.scaleXMultStep < 1 and self.scaleX < self.scaleXTarget):
                self.scaleX = self.scaleXTarget
            if (self.scaleYMultStep > 1 and self.scaleY > self.scaleYTarget) or (self.scaleYMultStep < 1 and self.scaleY < self.scaleYTarget):
                self.scaleY = self.scaleYTarget
            if (self.scaleZMultStep > 1 and self.scaleZ > self.scaleZTarget) or (self.scaleZMultStep < 1 and self.scaleZ < self.scaleZTarget):
                self.scaleZ = self.scaleZTarget
            FEM.Update_Inv_Basis(self.X0, self.tet, self.tetAttr, self.scaleX, self.scaleY, self.scaleZ)
        if self.zeroVel:
            MeshIO.Zero_Velocity(self.nodeAttr)

    def write(self, frame_idx):
        MeshIO.Write_TriMesh_Obj(self.X, self.Elem, self.output_folder + "shell" + str(frame_idx) + ".obj")
        if self.outputSeg:
            MeshIO.Write_SegMesh_Obj(self.X, self.segs, self.output_folder + "seg" + str(frame_idx) + ".obj")
        if self.outputRod:
            MeshIO.Write_SegMesh_Obj(self.X, self.rod, self.output_folder + "rod" + str(frame_idx) + ".obj")
        if self.withVol:
            MeshIO.Write_Surface_TriMesh_Obj(self.X, self.TriVI2TetVI, self.Tri, \
                    self.output_folder + "vol" + str(frame_idx) + ".obj")
