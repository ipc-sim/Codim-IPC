#pragma once

#include <Math/VECTOR.h>
#include <Utils/STDCONTAINER.h>
#include <set>
#include <sstream>
#include <fstream>
#include <FEM/DATA_TYPE.h>
#include <Grid/SPATIAL_HASH.h>

namespace py = pybind11;
namespace JGSL {

//#####################################################################
// Function Read_TriMesh_Obj
//#####################################################################
template <class T, int dim>
VECTOR<int, 4> Read_TriMesh_Obj(const std::string& filePath, 
    MESH_NODE<T, dim>& X, MESH_ELEM<2>& triangles)
{
    std::ifstream is(filePath);
    if (!is.is_open()) {
        puts((filePath + " not found!").c_str());
        return VECTOR<int, 4>(-1, -1, -1, -1);
    }

    std::string line;
    VECTOR<T, dim> position;
    VECTOR<int, 3> tri;
    VECTOR<int, 4> counter(X.size, triangles.size, 0, 0);
    while (std::getline(is, line)) {
        std::stringstream ss(line);
        if (line[0] == 'v' && line[1] == ' ') {
            ss.ignore();
            for (size_t i = 0; i < dim; i++)
                ss >> position(i);
            X.Append(position);
        }
        else if (line[0] == 'f') {
            int cnt = 0;
            int length = line.size();
            for (int i = 0; i < 3; ++i) {
                while (cnt < length && (line[cnt] < '0' || line[cnt] > '9'))
                    cnt++;
                int index = 0;
                while (cnt < length && '0' <= line[cnt] && line[cnt] <= '9') {
                    index = index * 10 + line[cnt] - '0';
                    cnt++;
                }
                tri(i) = index - 1;
                while (cnt < length && line[cnt] != ' ')
                    cnt++;
            }

            for (int i = 0; i < 3; ++i) {
                tri[i] += counter[0];
            }
            triangles.Append(tri);

            while (cnt < length && (line[cnt] < '0' || line[cnt] > '9'))
                cnt++;
            if (cnt < length) {
                // quad
                int index = 0;
                while (cnt < length && '0' <= line[cnt] && line[cnt] <= '9') {
                    index = index * 10 + line[cnt] - '0';
                    cnt++;
                }
                triangles.Append(VECTOR<int, 3>(tri[0], tri[2], index - 1 + counter[0]));
            }
        }
    }

    is.close();

    counter(2) = X.size;
    counter(3) = triangles.size;

    return counter;
}

template <class T, int dim>
VECTOR<int, 4> Read_TriMesh_Tex_Obj(const std::string& filePath, 
    MESH_NODE<T, dim>& X, MESH_NODE<T, dim>& X_tex, MESH_ELEM<2>& triangles, MESH_ELEM<2>& triangles_tex,
    std::vector<VECTOR<int, 3>>& stitchNodes, std::vector<T>& stitchRatio)
{
    std::ifstream is(filePath);
    if (!is.is_open()) {
        puts((filePath + " not found!").c_str());
        exit(-1);
    }

    std::string line;
    VECTOR<T, dim> position;
    VECTOR<int, 3> tri, tri_tex;
    VECTOR<int, 4> counter(X.size, triangles.size, 0, 0);
    int texVStartInd = X_tex.size;
    while (std::getline(is, line)) {
        std::stringstream ss(line);
        if (line[0] == 'v' && line[1] == ' ') {
            ss.ignore();
            for (size_t i = 0; i < dim; i++)
                ss >> position(i);
            X.Append(position);
        }
        else if (line[0] == 'v' && line[1] == 't') {
            ss.ignore(2);
            for (size_t i = 0; i < 2; i++)
                ss >> position(i);
            position[2] = 0;
            X_tex.Append(position);
        }
        else if (line[0] == 'f') {
            int cnt = 0;
            int length = line.size();
            bool texIndDiff = false;
            for (int i = 0; i < 3; ++i) {
                while (cnt < length && (line[cnt] < '0' || line[cnt] > '9'))
                    cnt++;
                int index = 0;
                while (cnt < length && '0' <= line[cnt] && line[cnt] <= '9') {
                    index = index * 10 + line[cnt] - '0';
                    cnt++;
                }
                tri(i) = index - 1;
                while (cnt < length && line[cnt] != ' ' && line[cnt] != '/')
                    cnt++;
                
                if(line[cnt] == '/') {
                    cnt++;
                    if (line[cnt] != '/') {
                        // texture face
                        texIndDiff = true;
                        int index = 0;
                        while (cnt < length && '0' <= line[cnt] && line[cnt] <= '9') {
                            index = index * 10 + line[cnt] - '0';
                            cnt++;
                        }
                        tri_tex(i) = index - 1;
                    }

                    while (cnt < length && line[cnt] != ' ')
                        cnt++;
                }
            }
            for (int i = 0; i < 3; ++i) {
                tri[i] += counter[0];
            }
            triangles.Append(tri);

            if (texIndDiff) {
                for (int i = 0; i < 3; ++i) {
                    tri_tex[i] += texVStartInd;
                }
                triangles_tex.Append(tri_tex);
            }
            else {
                triangles_tex.Append(tri);
            }
        }
        else if (line[0] == 's' && line[1] == 't' && line[2] == 'i' &&
            line[3] == 't' && line[4] == 'c' && line[5] == 'h') 
        {
            std::string bypass;
            ss >> bypass;
            stitchNodes.resize(stitchNodes.size() + 1);
            ss >> stitchNodes.back()[0] >> stitchNodes.back()[1] >> stitchNodes.back()[2];
            stitchRatio.resize(stitchRatio.size() + 1);
            ss >> stitchRatio.back();
        }
    }

    is.close();

    counter(2) = X.size;
    counter(3) = triangles.size;

    return counter;
}

//#####################################################################
// Function Write_TriMesh_Obj
//#####################################################################
template <class T, int dim>
void Write_TriMesh_Obj(MESH_NODE<T, dim>& nodes,
    MESH_ELEM<2>& faces, const std::string& filename)
{
    TIMER_FLAG("Write_TriMesh_Obj");

    FILE *file = fopen(filename.c_str(), "w");
    if (!file) {
        puts("failed to create file");
        exit(-1);
    }
    nodes.Each([&](int id, auto data) {
        auto &[x] = data;
        if constexpr (dim == 2) {
            fprintf(file, "v %le %le 0\n", x(0), x(1));
        }
        else if (dim == 3) {
            fprintf(file, "v %le %le %le\n", x(0), x(1), x(2));
        }
        else {
            puts("invalid node coordinate dimension");
            exit(-1);
        }
    });
    faces.Each([&](int id, auto data) {
        auto &[t] = data;
        fprintf(file, "f %d %d %d\n", t(0) + 1, t(1) + 1, t(2) + 1);
    });
    fclose(file);
}

template <class T, int dim>
void Write_TriMesh_Tex_Obj(
    MESH_NODE<T, dim>& nodes, MESH_NODE<T, dim>& nodes_tex,
    MESH_ELEM<2>& faces, MESH_ELEM<2>& faces_tex, 
    const std::string& filename)
{
    TIMER_FLAG("Write_TriMesh_Obj");

    FILE *file = fopen(filename.c_str(), "w");
    if (!file) {
        puts("failed to create file");
        exit(-1);
    }
    nodes.Each([&](int id, auto data) {
        auto &[x] = data;
        if constexpr (dim == 2) {
            fprintf(file, "v %le %le 0\n", x(0), x(1));
        }
        else if (dim == 3) {
            fprintf(file, "v %le %le %le\n", x(0), x(1), x(2));
        }
        else {
            puts("invalid node coordinate dimension");
            exit(-1);
        }
    });
    nodes_tex.Each([&](int id, auto data) {
        auto &[x] = data;
        fprintf(file, "vt %le %le\n", x(0), x(1));
    });
    faces.Each([&](int id, auto data) {
        auto &[t] = data;
        const VECTOR<int, 3>& t_tex = std::get<0>(faces_tex.Get_Unchecked(id));
        fprintf(file, "f %d/%d %d/%d %d/%d\n", t(0) + 1, t_tex(0) + 1,
            t(1) + 1, t_tex(1) + 1, t(2) + 1, t_tex(2) + 1);
    });
    fclose(file);
}

template <class T, int dim>
void Write_TriMesh_Region_Obj(MESH_NODE<T, dim>& nodes,
    MESH_ELEM<2>& faces, const std::string& filename, std::vector<int>& subset)
{
    TIMER_FLAG("Write_TriMesh_Region_Obj");
    std::vector<int> back(nodes.size, -1);
    MESH_NODE<T, dim> filtered_nodes(nodes.size);
    for (int i = 0; i < subset.size(); ++i) {
        back[subset[i]] = i;
        filtered_nodes.Insert(i, std::get<0>(nodes.Get_Unchecked(subset[i])));
    }
    MESH_ELEM<2> filtered_faces(faces.size);
    int cnt = 0;
    faces.Each([&](int id, auto data) {
        auto [face] = data;
        bool flag = true;
        for (int d = 0; d <= 2; ++d)
            flag &= (back[face(d)] >= 0);
        if (flag) filtered_faces.Insert(cnt++, VECTOR<int, 3>(back[face(0)], back[face(1)], back[face(2)]));
    });
    Write_TriMesh_Obj(filtered_nodes, filtered_faces, filename);
}

template <class T>
VECTOR<int, 4> Read_TetMesh_Vtk(const std::string& filePath, MESH_NODE<T, 3>& X,
    MESH_ELEM<3>& indices)
{
    std::ifstream in(filePath);
    if (!in.is_open()) {
        puts((filePath + " not found!").c_str());
        exit(-1);
    }

    auto initial_X_size = X.size;
    auto initial_indices_size = indices.size;

    std::string line;
    VECTOR<T, 3> position;

    bool reading_points = false;
    bool reading_tets = false;
    size_t n_points = 0;
    size_t n_tets = 0;

    while (std::getline(in, line)) {
        std::stringstream ss(line);
        if (line.size() == (size_t)(0)) {
        }
        else if (line.substr(0, 6) == "POINTS") {
            reading_points = true;
            reading_tets = false;
            ss.ignore(128, ' '); // ignore "POINTS"
            ss >> n_points;
        }
        else if (line.substr(0, 5) == "CELLS") {
            reading_points = false;
            reading_tets = true;
            ss.ignore(128, ' '); // ignore "CELLS"
            ss >> n_tets;
        }
        else if (line.substr(0, 10) == "CELL_TYPES") {
            reading_points = false;
            reading_tets = false;
        }
        else if (reading_points) {
            for (size_t i = 0; i < 3; i++)
                ss >> position(i);
            X.Append(position);
        }
        else if (reading_tets) {
            ss.ignore(128, ' '); // ignore "4"
            VECTOR<int, 4> tet;
            for (size_t i = 0; i < 4; i++) {
                ss >> tet(i);
                tet(i) += initial_X_size;
            }
            indices.Append(tet);
        }
    }
    in.close();

    assert((n_points == X.size - initial_X_size) && "vtk read X count doesn't match.");
    assert(((size_t)n_tets == indices.size - initial_indices_size) && "vtk read element count doesn't match.");

    return VECTOR<int, 4>(initial_X_size, initial_indices_size, X.size, indices.size);
}

template <class T>
VECTOR<int, 4> Read_TetMesh_Mesh(const std::string& filePath, 
    MESH_NODE<T, 3>& X, MESH_ELEM<3>& indices)
{
    VECTOR<int, 4> meshCounter;
    meshCounter[0] = X.size;
    meshCounter[1] = indices.size;
    meshCounter[2] = meshCounter[3] = -1;

    std::ifstream in(filePath);
    if (in.is_open()) {
        std::string line;
        while (std::getline(in, line)) {
            std::stringstream ss(line);
            std::string keyword;
            ss >> keyword;
            if (keyword == "Vertices") {
                int vAmt;
                std::getline(in, line);
                std::stringstream ss(line);
                ss >> vAmt;
                X.Reserve(X.size + vAmt);
                for (int vI = 0; vI < vAmt; ++vI) {
                    std::getline(in, line);
                    std::stringstream ss(line);
                    VECTOR<T, 3> x;
                    ss >> x[0] >> x[1] >> x[2];
                    X.Append(x);
                }
            }
            else if (keyword == "Tetrahedra") {
                int tAmt;
                std::getline(in, line);
                std::stringstream ss(line);
                ss >> tAmt;
                indices.Reserve(indices.size + tAmt);
                for (int tI = 0; tI < tAmt; ++tI) {
                    std::getline(in, line);
                    std::stringstream ss(line);
                    VECTOR<int, 4> tVInd;
                    for (int i = 0; i < 4; ++i) {
                        ss >> tVInd[i];
                        --tVInd[i];
                    }
                    indices.Append(tVInd);
                }
            }
        }

        in.close();

        meshCounter[2] = X.size;
        meshCounter[3] = indices.size;
    }
    else {
        printf("cannot open %s\n", filePath.c_str());
        exit(-1);
    }

    return meshCounter;
}

template <class T>
void Write_TetMesh_Vtk(MESH_NODE<T, 3>& X,
    MESH_ELEM<3>& indices, const std::string& filePath)
{
    assert((X.size() != 0) && "The X array for writing tetmesh vtk is empty.");
    assert((indices.size() != (size_t)0) && "The tet mesh data structure for writing tetmesh vtk is empty.");

    std::ofstream os(filePath);
    if (!os.is_open()) {
        puts("failed to create file");
        exit(-1);
    }

    os << "# vtk DataFile Version 2.0\n";
    os << "Unstructured Grid\n";
    os << "ASCII\n";
    os << "DATASET UNSTRUCTURED_GRID\n";

    os << "POINTS " << X.size << " ";
    if (std::is_same<T, float>::value)
        os << "float\n";
    else
        os << "double\n";


    X.Each([&](int id, auto data) {
        auto &[x] = data;
        os << x(0) << " " << x(1) << " " << x(2) << "\n";
    });

    os << std::endl;

    os << "CELLS " << indices.size << " " << 5 * indices.size << "\n";
    indices.Each([&](int id, auto data) {
        auto &[m] = data;
        os << 4 << " " << m(0) << " " << m(1) << " " << m(2) << " " << m(3) << "\n";
    });
    os << std::endl;

    os << "CELL_TYPES " << indices.size << "\n";
    for (size_t i = 0; i < indices.size; i++) {
        os << 10 << std::endl;
    }

    os.close();
}

class Triplet {
public:
    int key[3];

    Triplet(const int* p_key)
    {
        key[0] = p_key[0];
        key[1] = p_key[1];
        key[2] = p_key[2];
    }
    Triplet(int key0, int key1, int key2)
    {
        key[0] = key0;
        key[1] = key1;
        key[2] = key2;
    }

    bool operator<(const Triplet& right) const
    {
        if (key[0] < right.key[0]) {
            return true;
        }
        else if (key[0] == right.key[0]) {
            if (key[1] < right.key[1]) {
                return true;
            }
            else if (key[1] == right.key[1]) {
                if (key[2] < right.key[2]) {
                    return true;
                }
            }
        }
        return false;
    }
};
template <class T, bool mapTriVInd = true>
void Find_Surface_TriMesh(
    BASE_STORAGE<VECTOR<T, 3>>& X,
    BASE_STORAGE<VECTOR<int, 4>>& Tet,
    BASE_STORAGE<int>& TriVI2TetVI, BASE_STORAGE<VECTOR<int, 3>>& Tri)
{
    // indexing triangle faces
    std::map<Triplet, int> tri2Tet;
    Tet.Each([&](int id, auto data) {
        auto& [elemVInd] = data;
        tri2Tet[Triplet(elemVInd(0), elemVInd(2), elemVInd(1))] = id;
        tri2Tet[Triplet(elemVInd(0), elemVInd(3), elemVInd(2))] = id;
        tri2Tet[Triplet(elemVInd(0), elemVInd(1), elemVInd(3))] = id;
        tri2Tet[Triplet(elemVInd(1), elemVInd(2), elemVInd(3))] = id;
    });

    //TODO: parallelize
    // extract surface triangles
    // TODO: provide clear
    if (Tri.size) Tri = std::move(BASE_STORAGE<VECTOR<int, 3>>());
    for (const auto& triI : tri2Tet) {
        const int* triVInd = triI.first.key;
        // find dual triangle with reversed indices:
        auto finder = tri2Tet.find(Triplet(triVInd[2], triVInd[1], triVInd[0]));
        if (finder == tri2Tet.end()) {
            finder = tri2Tet.find(Triplet(triVInd[1], triVInd[0], triVInd[2]));
            if (finder == tri2Tet.end()) {
                finder = tri2Tet.find(Triplet(triVInd[0], triVInd[2], triVInd[1]));
                if (finder == tri2Tet.end()) {
                    Tri.Append(VECTOR<int, 3>(triVInd[0], triVInd[1], triVInd[2]));
                }
            }
        }
    }

    // find surface nodes
    std::vector<bool> isSurfNode(X.size, false);
    for (int i = 0; i < Tri.size; ++i) {
        auto [t] = Tri.Get(i).value();
        isSurfNode[t(0)] = isSurfNode[t(1)] = isSurfNode[t(2)] = true;
    }

    // map surface nodes
    std::vector<int> TetVI2TriVI(X.size, -1);
    // TODO: provide clear
    if (TriVI2TetVI.size) TriVI2TetVI = std::move(BASE_STORAGE<int>());
    for (int i = 0; i < isSurfNode.size(); ++i) {
        if (isSurfNode[i]) {
            TetVI2TriVI[i] = TriVI2TetVI.size;
            TriVI2TetVI.Append(i);
        }
    }
    
    if constexpr (mapTriVInd) {
        for (int i = 0; i < Tri.size; ++i) {
            auto [t] = Tri.Get(i).value();
            Tri.update(i, VECTOR<int, 3>(TetVI2TriVI[t(0)], TetVI2TriVI[t(1)], TetVI2TriVI[t(2)]));
        }
    }
}

template <class T>
void Write_Surface_TriMesh_Obj(BASE_STORAGE<VECTOR<T, 3>>& nodes,
    BASE_STORAGE<int>& TriVI2TetVI, BASE_STORAGE<VECTOR<int, 3>>& faces,
    const std::string& filename)
{
    TIMER_FLAG("Write_Surface_TriMesh_Obj");

    FILE *file = fopen(filename.c_str(), "w");
    if (!file) {
        puts("failed to create file");
        exit(-1);
    }
    TriVI2TetVI.Each([&](int id, auto data) {
        auto [i] = data;
        auto [node] = nodes.Get(i).value();
        fprintf(file, "v %.12le %.12le %.12le\n", node(0), node(1), node(2));
    });
    faces.Each([&](int id, auto data) {
        auto [face] = data;
        fprintf(file, "f %d %d %d\n", face(0) + 1, face(1) + 1, face(2) + 1);
    });
    fclose(file);
}

template <class T, int dim>
VECTOR<int, 4> Read_SegMesh_Seg(const std::string& filePath,
    BASE_STORAGE<VECTOR<T, dim>>& X,
    std::vector<VECTOR<int, 2>>& segs)
{
    std::ifstream is(filePath);
    if (!is.is_open()) {
        puts((filePath + " not found!").c_str());
        exit(-1);
    }

    std::string line;
    VECTOR<T, dim> position;
    VECTOR<int, 2> seg;
    VECTOR<int, 4> counter(X.size, segs.size(), 0, 0);
    while (std::getline(is, line)) {
        std::stringstream ss(line);
        if (line[0] == 'v' && line[1] == ' ') {
            ss.ignore();
            for (size_t i = 0; i < dim; ++i)
                ss >> position(i);
            X.Append(position);
        }
        else if (line[0] == 's') {
            int cnt = 0;
            int length = line.size();
            for (int i = 0; i < 2; ++i) {
                while (cnt < length && (line[cnt] < '0' || line[cnt] > '9'))
                    cnt++;
                int index = 0;
                while (cnt < length && '0' <= line[cnt] && line[cnt] <= '9') {
                    index = index * 10 + line[cnt] - '0';
                    cnt++;
                }
                seg(i) = index - 1;
                while (cnt < length && line[cnt] != ' ')
                    cnt++;
            }
            for (int i = 0; i < 2; ++i) {
                seg[i] += counter[0];
            }
            segs.emplace_back(seg);
        }
    }

    is.close();

    counter(2) = X.size;
    counter(3) = segs.size();

    return counter;
}

template <class T, int dim>
void Write_SegMesh_Obj(MESH_NODE<T, dim>& nodes,
    std::vector<VECTOR<int, 2>>& segs, const std::string& filename)
{
    FILE *file = fopen(filename.c_str(), "w");
    if (!file) {
        puts("failed to create file");
        exit(-1);
    }

    std::vector<bool> isSegNode(nodes.size, false);
    for (const auto& segI : segs) {
        isSegNode[segI[0]] = isSegNode[segI[1]] = true;
    }

    std::vector<int> global2local(nodes.size, -1);
    int localVI = 0;
    for (int vI = 0; vI < isSegNode.size(); ++vI) {
        if (isSegNode[vI]) {
            global2local[vI] = localVI++;

            const VECTOR<T, dim>& x0 = std::get<0>(nodes.Get_Unchecked(vI));
            if constexpr (dim == 2) {
                fprintf(file, "v %le %le 0\n", x0(0), x0(1));
            }
            else if (dim == 3) {
                fprintf(file, "v %le %le %le\n", x0(0), x0(1), x0(2));
            }
        }
    }

    for (const auto segI : segs) {
        fprintf(file, "f %d %d %d\n", global2local[segI[0]] + 1, global2local[segI[1]] + 1, global2local[segI[1]] + 1);
    }
    
    fclose(file);
}

template <class T>
void Convert_Seg_File(const std::string& folderPath,
    const std::string& filePrefix)
{
    for (int frameI = 0; true; ++frameI) {
        std::string filePath = folderPath + "/" + filePrefix + std::to_string(frameI) + ".obj";
        MESH_NODE<T, 3> X;
        MESH_ELEM<2> triangles;
        VECTOR<int, 4> counter = Read_TriMesh_Obj(filePath, X, triangles);
        if (counter[0] < 0) {
            break;
        }

        filePath = folderPath + "/" + filePrefix + std::to_string(frameI) + ".poly";
        FILE *out = fopen(filePath.c_str(), "w+");
        if (!out) {
            printf("can't create %s\n", filePath.c_str());
            break;
        }

        fprintf(out, "POINTS\n");
        for (int vI = 0; vI < X.size; ++vI) {
            const VECTOR<T, 3>& x = std::get<0>(X.Get_Unchecked(vI));
            fprintf(out, "%d: %le %le %le\n", vI + 1, x[0], x[1], x[2]);
        }
        fprintf(out, "POLYS\n");
        for (int tI = 0; tI < triangles.size; ++tI) {
            const VECTOR<int, 3>& triVInd = std::get<0>(triangles.Get_Unchecked(tI));
            fprintf(out, "%d: %d %d\n", tI + 1, triVInd[0] + 1, triVInd[1] + 1);
        }
        fprintf(out, "END\n");
        fclose(out);
    }
}

void Find_Boundary_Edge_And_Node(int Xsize, 
    MESH_ELEM<2>& triangles,
    std::vector<int>& boundaryNode,
    std::vector<VECTOR<int, 2>>& boundaryEdge)
{
    std::set<std::pair<int, int>> edgeSet;
    triangles.Each([&](int id, auto data) {
        auto &[elemVInd] = data;
        edgeSet.insert(std::pair<int, int>(elemVInd[0], elemVInd[1]));
        edgeSet.insert(std::pair<int, int>(elemVInd[1], elemVInd[2]));
        edgeSet.insert(std::pair<int, int>(elemVInd[2], elemVInd[0]));
    });

    boundaryEdge.resize(0);
    for (const auto& eI : edgeSet) {
        if (edgeSet.find(std::pair<int, int>(eI.second, eI.first)) == edgeSet.end()) {
            boundaryEdge.emplace_back(eI.first, eI.second);
        }
    }

    std::vector<bool> isBoundaryNode(Xsize, false);
    for (const auto& beI : boundaryEdge) {
        isBoundaryNode[beI[0]] = isBoundaryNode[beI[1]] = true;
    }
    boundaryNode.resize(0);
    for (int nI = 0; nI < isBoundaryNode.size(); ++nI) {
        if (isBoundaryNode[nI]) {
            boundaryNode.emplace_back(nI);
        }
    }
}

void Find_Surface_Primitives(
    int Xsize, MESH_ELEM<2>& Tri,
    std::vector<int>& boundaryNode,
    std::vector<VECTOR<int, 2>>& boundaryEdge,
    std::vector<VECTOR<int, 3>>& boundaryTri)
{
    boundaryTri.reserve(Tri.size);
    std::set<VECTOR<int, 2>> boundaryEdgeSet;
    std::vector<bool> isBoundaryNode(Xsize, false);
    Tri.Each([&](int id, auto data) {
        auto &[triVInd] = data;
        
        boundaryTri.emplace_back(triVInd);
        
        auto finder = boundaryEdgeSet.find(VECTOR<int, 2>(triVInd[1], triVInd[0]));
        if (finder == boundaryEdgeSet.end()) {
            boundaryEdgeSet.insert(VECTOR<int, 2>(triVInd[0], triVInd[1]));
        }
        finder = boundaryEdgeSet.find(VECTOR<int, 2>(triVInd[2], triVInd[1]));
        if (finder == boundaryEdgeSet.end()) {
            boundaryEdgeSet.insert(VECTOR<int, 2>(triVInd[1], triVInd[2]));
        }
        finder = boundaryEdgeSet.find(VECTOR<int, 2>(triVInd[0], triVInd[2]));
        if (finder == boundaryEdgeSet.end()) {
            boundaryEdgeSet.insert(VECTOR<int, 2>(triVInd[2], triVInd[0]));
        }

        isBoundaryNode[triVInd[0]] = isBoundaryNode[triVInd[1]] = isBoundaryNode[triVInd[2]] = true;
    });

    boundaryEdge = std::move(std::vector<VECTOR<int, 2>>(boundaryEdgeSet.begin(),
        boundaryEdgeSet.end()));
    
    for (int vI = 0; vI < isBoundaryNode.size(); ++vI) {
        if (isBoundaryNode[vI]) {
            boundaryNode.emplace_back(vI);
        }
    }
}

template <class T>
void Find_Surface_Primitives_And_Compute_Area(
    MESH_NODE<T, 3>& X, MESH_ELEM<2>& Tri,
    std::vector<int>& boundaryNode,
    std::vector<VECTOR<int, 2>>& boundaryEdge,
    std::vector<VECTOR<int, 3>>& boundaryTri,
    std::vector<T>& BNArea,
    std::vector<T>& BEArea,
    std::vector<T>& BTArea)
{
    boundaryTri.reserve(Tri.size);
    BTArea.reserve(Tri.size);
    std::map<VECTOR<int, 2>, T> boundaryEdgeSet;
    std::vector<T> isBoundaryNode(X.size, 0);
    Tri.Each([&](int id, auto data) {
        auto &[triVInd] = data;

        const VECTOR<T, 3>& v0 = std::get<0>(X.Get_Unchecked(triVInd[0]));
        const VECTOR<T, 3>& v1 = std::get<0>(X.Get_Unchecked(triVInd[1]));
        const VECTOR<T, 3>& v2 = std::get<0>(X.Get_Unchecked(triVInd[2]));
        BTArea.emplace_back(0.5 * cross(v1 - v0, v2 - v0).length());
        
        boundaryTri.emplace_back(triVInd);
        
        auto finder = boundaryEdgeSet.find(VECTOR<int, 2>(triVInd[1], triVInd[0]));
        if (finder == boundaryEdgeSet.end()) {
            boundaryEdgeSet[VECTOR<int, 2>(triVInd[0], triVInd[1])] = BTArea.back() / 3;
        }
        else {
            finder->second += BTArea.back() / 3;
        }
        finder = boundaryEdgeSet.find(VECTOR<int, 2>(triVInd[2], triVInd[1]));
        if (finder == boundaryEdgeSet.end()) {
            boundaryEdgeSet[VECTOR<int, 2>(triVInd[1], triVInd[2])] = BTArea.back() / 3;
        }
        else {
            finder->second += BTArea.back() / 3;
        }
        finder = boundaryEdgeSet.find(VECTOR<int, 2>(triVInd[0], triVInd[2]));
        if (finder == boundaryEdgeSet.end()) {
            boundaryEdgeSet[VECTOR<int, 2>(triVInd[2], triVInd[0])] = BTArea.back() / 3;
        }
        else {
            finder->second += BTArea.back() / 3;
        }

        isBoundaryNode[triVInd[0]] += BTArea.back() / 3;
        isBoundaryNode[triVInd[1]] += BTArea.back() / 3;
        isBoundaryNode[triVInd[2]] += BTArea.back() / 3;

        BTArea.back() /= 2; // due to PT approx of \int_T PP
    });

    boundaryEdge.reserve(boundaryEdgeSet.size());
    BEArea.reserve(boundaryEdgeSet.size());
    for (const auto& i : boundaryEdgeSet) {
        boundaryEdge.emplace_back(i.first);
        BEArea.emplace_back(i.second / 2); // due to PE approx of \int_E PP and EE approx of \int_E PE
    }
    
    for (int vI = 0; vI < isBoundaryNode.size(); ++vI) {
        if (isBoundaryNode[vI]) {
            boundaryNode.emplace_back(vI);
            BNArea.emplace_back(isBoundaryNode[vI]);
        }
    }
}

template <class T, int dim>
void Transform_Points(
    const VECTOR<T, dim>& trans,
    const VECTOR<T, dim>& rotCenter,
    const VECTOR<T, dim>& rotAxis,
    const T rotAngDeg,
    const VECTOR<T, dim>& scale,
    const VECTOR<int, 4>& meshCounter,
    MESH_NODE<T, dim>& X)
{
    T rotAngRad = rotAngDeg / 180 * M_PI;
    for (int i = meshCounter[0]; i < meshCounter[2]; ++i) {
        VECTOR<T, dim>& xI = std::get<0>(X.Get_Unchecked(i));
        if constexpr (dim == 3) {
            const Eigen::Matrix3d rotMtr = Eigen::AngleAxis<double>(rotAngRad,
                Eigen::Vector3d(rotAxis[0], rotAxis[1], rotAxis[2]).normalized()).toRotationMatrix();
            Eigen::Vector3d x((xI - rotCenter).data);
            x = x.array() * Eigen::Array3d(scale.data);
            const Eigen::Vector3d rotx = rotMtr * x;
            for (int d = 0; d < dim; ++d) {
                xI[d] = rotx[d] + rotCenter[d] + trans[d];
            }
        }
        else {
            MATRIX<T, dim> rotMtr;
            rotMtr(0, 0) = rotMtr(1, 1) = std::cos(rotAngRad);
            rotMtr(0, 1) = -std::sin(rotAngRad);
            rotMtr(1, 0) = -rotMtr(0, 1);
            VECTOR<T, dim> x = xI - rotCenter;
            x[0] *= scale[0]; x[1] *= scale[1];
            xI = rotMtr * x + rotCenter + trans;
        }
    }
}

template <class T, int dim>
void Transform_Points_Range(
    const VECTOR<T, dim>& relBoxMin,
    const VECTOR<T, dim>& relBoxMax,
    const VECTOR<T, dim>& trans,
    const VECTOR<T, dim>& rotCenter,
    const VECTOR<T, dim>& rotAxis,
    const T rotAngDeg,
    const VECTOR<T, dim>& scale,
    MESH_NODE<T, dim>& X,
    const VECTOR<int, 4>& vIndRange = VECTOR<int, 4>(0, 0, __INT_MAX__, -1))
{
    if (!X.size) {
        puts("no nodes in the model!");
        exit(-1);
    }

    VECTOR<T, dim> bboxMin;
    VECTOR<T, dim> bboxMax;
    X.Each([&](int id, auto data) {
        if (id >= vIndRange[0] && id < vIndRange[2]) {
            auto &[x] = data;
            if (id == vIndRange[0]) {
                bboxMin = x;
                bboxMax = x;
            }
            else {
                for (int dimI = 0; dimI < dim; ++dimI) {
                    if (bboxMax(dimI) < x(dimI)) {
                        bboxMax(dimI) = x(dimI);
                    }
                    if (bboxMin(dimI) > x(dimI)) {
                        bboxMin(dimI) = x(dimI);
                    }
                }
            }
        }
    });

    VECTOR<T, dim> rangeMin = relBoxMin;
    VECTOR<T, dim> rangeMax = relBoxMax;
    for (int dimI = 0; dimI < dim; ++dimI) {
        rangeMin(dimI) *= bboxMax(dimI) - bboxMin(dimI);
        rangeMin(dimI) += bboxMin(dimI);
        rangeMax(dimI) *= bboxMax(dimI) - bboxMin(dimI);
        rangeMax(dimI) += bboxMin(dimI);
    }

    std::vector<int> nodesToTransform;
    X.Each([&](int id, auto data) {
        if (id >= vIndRange[0] && id < vIndRange[2]) {
            auto &[x] = data;
            if constexpr (dim == 3) {
                if (x(0) >= rangeMin(0) && x(0) <= rangeMax(0) &&
                    x(1) >= rangeMin(1) && x(1) <= rangeMax(1) &&
                    x(2) >= rangeMin(2) && x(2) <= rangeMax(2))
                {
                    nodesToTransform.emplace_back(id);
                }
            }
            else {
                if (x(0) >= rangeMin(0) && x(0) <= rangeMax(0) &&
                    x(1) >= rangeMin(1) && x(1) <= rangeMax(1))
                {
                    nodesToTransform.emplace_back(id);
                }
            }
        }
    });

    T rotAngRad = rotAngDeg / 180 * M_PI;
    for (const auto& i : nodesToTransform) {
        VECTOR<T, dim>& xI = std::get<0>(X.Get_Unchecked(i));
        if constexpr (dim == 3) {
            const Eigen::Matrix3d rotMtr = Eigen::AngleAxis<double>(rotAngRad,
                Eigen::Vector3d(rotAxis[0], rotAxis[1], rotAxis[2]).normalized()).toRotationMatrix();
            Eigen::Vector3d x((xI - rotCenter).data);
            x = x.array() * Eigen::Array3d(scale.data);
            const Eigen::Vector3d rotx = rotMtr * x;
            for (int d = 0; d < dim; ++d) {
                xI[d] = rotx[d] + rotCenter[d] + trans[d];
            }
        }
        else {
            MATRIX<T, dim> rotMtr;
            rotMtr(0, 0) = rotMtr(1, 1) = std::cos(rotAngRad);
            rotMtr(0, 1) = -std::sin(rotAngRad);
            rotMtr(1, 0) = -rotMtr(0, 1);
            VECTOR<T, dim> x = xI - rotCenter;
            x[0] *= scale[0]; x[1] *= scale[1];
            xI = rotMtr * x + rotCenter + trans;
        }
    }
}

template <class T, int dim = 3>
void Load_Velocity(
    const std::string& meshFolderPath,
    int lastFrame, T h,
    MESH_NODE_ATTR<T, dim>& nodeAttr)
{
    MESH_NODE<T, dim> X0, X1;
    MESH_ELEM<dim - 1> Elem0, Elem1;
    Read_TriMesh_Obj(meshFolderPath + "/" + std::to_string(lastFrame - 1) + ".obj", X0, Elem0);
    Read_TriMesh_Obj(meshFolderPath + "/" + std::to_string(lastFrame) + ".obj", X1, Elem1);
    if (X0.size != nodeAttr.size || X1.size != nodeAttr.size) {
        printf("node count does not match!\n");
        exit(-1);
    }
    X0.Join(X1).Par_Each([&](int id, auto data) {
        auto &[x0, x1] = data;
        std::get<FIELDS<MESH_NODE_ATTR<T, dim>>::v>(nodeAttr.Get_Unchecked(id)) = (x1 - x0) / h;
    });
}

template <class T, int dim = 3>
void Load_Velocity_X0(
    const std::string& meshFolderPath,
    int lastFrame, T h,
    MESH_NODE<T, dim>& X0,
    MESH_NODE_ATTR<T, dim>& nodeAttr)
{
    MESH_NODE<T, dim> X1;
    MESH_ELEM<dim - 1> Elem1;
    Read_TriMesh_Obj(meshFolderPath + "/" + std::to_string(lastFrame) + ".obj", X1, Elem1);
    if (X0.size != nodeAttr.size || X1.size != nodeAttr.size) {
        printf("node count does not match!\n");
        exit(-1);
    }
    X0.Join(X1).Par_Each([&](int id, auto data) {
        auto &[x0, x1] = data;
        std::get<FIELDS<MESH_NODE_ATTR<T, dim>>::v>(nodeAttr.Get_Unchecked(id)) = (x1 - x0) / h;
    });
}

template <class T, int dim = 3>
void Zero_Velocity(
    MESH_NODE_ATTR<T, dim>& nodeAttr)
{
    nodeAttr.Par_Each([&](int id, auto data) {
        auto &[x0, v, g, m] = data;
        v.setZero();
    });
}

template <class T, int dim>
void Append_Attribute(VECTOR_STORAGE<T, dim>& src,
    VECTOR_STORAGE<T, dim>& dst)
{
    dst.Reserve(dst.size + src.size);
    src.Each([&](int id, auto data) {
        auto &[x] = data;
        dst.Append(x);
    });
}

template <class T>
void Deep_Copy_Attribute(T& src, T& dst)
{
    src.deep_copy_to(dst);
}

template <class T, int dim = 3>
void Modify_UV(
    const std::string& filePath,
    int vStep, T dist)
{
    MESH_NODE<T, dim> X, X_tex;
    MESH_ELEM<2> triangles, triangles_tex;
    std::vector<VECTOR<int, 3>> stitchNodes;
    std::vector<T> stitchRatio;
    Read_TriMesh_Tex_Obj(filePath, X, X_tex, triangles, triangles_tex, stitchNodes, stitchRatio);
    X_tex.Each([&](int id, auto data) {
        auto &[x] = data;
        x[0] += id / vStep * dist;
    });

    std::string outputFilePath = filePath.substr(0, filePath.find_last_of('.')) + "_newUV.obj";
    FILE *out = fopen(outputFilePath.c_str(), "w+");
    if (out) {
        X.Each([&](int id, auto data) {
            auto &[x] = data;
            if constexpr (dim == 2) {
                fprintf(out, "v %le %le 0\n", x(0), x(1));
            }
            else if (dim == 3) {
                fprintf(out, "v %le %le %le\n", x(0), x(1), x(2));
            }
            else {
                puts("invalid node coordinate dimension");
                exit(-1);
            }
        });
        X_tex.Each([&](int id, auto data) {
            auto &[x] = data;
            fprintf(out, "vt %le %le\n", x(0), x(1));
        });
        triangles.Each([&](int id, auto data) {
            auto &[t] = data;
            fprintf(out, "f %d/%d %d/%d %d/%d\n", 
                t(0) + 1, t(0) + 1, 
                t(1) + 1, t(1) + 1, 
                t(2) + 1, t(2) + 1);
        });
        fclose(out);
    }
    else {
        printf("cannot create file %s", outputFilePath.c_str());
    }
}

void addIndToBiMap(
    std::map<int, int>& global2local,
    std::vector<int>& local2global,
    int globalI)
{
    if (global2local.find(globalI) == global2local.end()) {
        global2local[globalI] = local2global.size();
        local2global.emplace_back(globalI);
    }
}

template <class T, int dim = 3>
void Output_Spatial_Hash(
    const std::string& filePath)
{
    MESH_NODE<T, dim> X;
    MESH_ELEM<dim - 1> Elem;
    Read_TriMesh_Obj(filePath, X, Elem);

    std::vector<int> boundaryNode;
    std::vector<VECTOR<int, 2>> boundaryEdge;
    std::vector<VECTOR<int, 3>> boundaryTri;
    Find_Surface_Primitives(X.size, Elem, boundaryNode, boundaryEdge, boundaryTri);

    SPATIAL_HASH<T, dim> sh;
    sh.Build(X, boundaryNode, boundaryEdge, boundaryTri, 1);

    Eigen::Matrix<int, dim, 1> vIndStep(1, sh.voxelCount[0] + 1, (sh.voxelCount[0] + 1) * (sh.voxelCount[1] + 1));
    std::map<int, int> gridNodeI2meshVI;
    std::vector<int> meshVI2gridNodeI;
    for (const auto& voxelI : sh.voxel) {
        Eigen::Matrix<int, dim, 1> voxelInd;
        voxelInd[0] = voxelI.first % sh.voxelCount[0];
        voxelInd[1] = voxelI.first % sh.voxelCount0x1 / sh.voxelCount[0];
        voxelInd[2] = voxelI.first / sh.voxelCount0x1;
        int leftBottomVInd = voxelInd.dot(vIndStep);
        addIndToBiMap(gridNodeI2meshVI, meshVI2gridNodeI, leftBottomVInd);
        addIndToBiMap(gridNodeI2meshVI, meshVI2gridNodeI, leftBottomVInd + 1);
        addIndToBiMap(gridNodeI2meshVI, meshVI2gridNodeI, leftBottomVInd + vIndStep[1]);
        addIndToBiMap(gridNodeI2meshVI, meshVI2gridNodeI, leftBottomVInd + vIndStep[1] + 1);
        addIndToBiMap(gridNodeI2meshVI, meshVI2gridNodeI, leftBottomVInd + vIndStep[2]);
        addIndToBiMap(gridNodeI2meshVI, meshVI2gridNodeI, leftBottomVInd + vIndStep[2] + 1);
        addIndToBiMap(gridNodeI2meshVI, meshVI2gridNodeI, leftBottomVInd + vIndStep[2] + vIndStep[1]);
        addIndToBiMap(gridNodeI2meshVI, meshVI2gridNodeI, leftBottomVInd + vIndStep[2] + vIndStep[1] + 1);
    }

    std::string outputFilePath = filePath.substr(0, filePath.find_last_of('.')) + "_voxel.obj";
    FILE *out = fopen(outputFilePath.c_str(), "w+");
    if (out) {
        for (const auto& globalI : meshVI2gridNodeI) {
            fprintf(out, "v %le %le %le\n",
                sh.leftBottomCorner[0] + (globalI % vIndStep[1]) / sh.one_div_voxelSize,
                sh.leftBottomCorner[1] + (globalI % vIndStep[2] / vIndStep[1]) / sh.one_div_voxelSize,
                sh.leftBottomCorner[2] + (globalI / vIndStep[2]) / sh.one_div_voxelSize);
        }
        fclose(out);
    }
    else {
        printf("file creation error!\n");
        exit(-1);
    }
}

template <class T, int dim = 3>
void Subdivide(
    const std::string& filePath)
{
    MESH_NODE<T, dim> X;
    MESH_ELEM<dim - 1> Tri;
    Read_TriMesh_Obj(filePath, X, Tri);

    std::map<VECTOR<int, 2>, int> edgeInd;
    int newEInd = 0;
    Tri.Each([&](int id, auto data) {
        auto &[triVInd] = data;
        for (int d = 0; d < dim; ++d) {
            auto finder = edgeInd.find(VECTOR<int, 2>(triVInd[(d + 1) % dim], triVInd[d]));
            if (finder == edgeInd.end()) {
                edgeInd[VECTOR<int, 2>(triVInd[d], triVInd[(d + 1) % dim])] = newEInd++;
            }
            else {
                edgeInd[VECTOR<int, 2>(triVInd[d], triVInd[(d + 1) % dim])] = finder->second;
            }
        }
    });

    MESH_NODE<T, dim> newX = X;
    newX.Reserve(X.size + newEInd);
    for (int i = 0; i < newEInd; ++i) {
        newX.Append(VECTOR<T, dim>());
    }
    for (const auto& eI : edgeInd) {
        const VECTOR<T, dim>& v0 = std::get<0>(X.Get_Unchecked(eI.first[0]));
        const VECTOR<T, dim>& v1 = std::get<0>(X.Get_Unchecked(eI.first[1]));
        std::get<0>(newX.Get_Unchecked(X.size + eI.second)) = (v0 + v1) / 2;
    }
    MESH_ELEM<dim - 1> newTri(Tri.size * 4);
    Tri.Each([&](int id, auto data) {
        auto &[triVInd] = data;
        int newVInds[3] = {
            X.size + edgeInd[VECTOR<int, 2>(triVInd[0], triVInd[1])],
            X.size + edgeInd[VECTOR<int, 2>(triVInd[1], triVInd[2])],
            X.size + edgeInd[VECTOR<int, 2>(triVInd[2], triVInd[0])]
        };
        newTri.Append(VECTOR<int, dim>(triVInd[0], newVInds[0], newVInds[2]));
        newTri.Append(VECTOR<int, dim>(triVInd[1], newVInds[1], newVInds[0]));
        newTri.Append(VECTOR<int, dim>(triVInd[2], newVInds[2], newVInds[1]));
        newTri.Append(VECTOR<int, dim>(newVInds[0], newVInds[1], newVInds[2]));
    });

    std::string outputFilePath = filePath.substr(0, filePath.find_last_of('.')) + "_linSubdiv.obj";
    Write_TriMesh_Obj(newX, newTri, outputFilePath);
}

template <class T, int dim = 3>
void Subdivide_Loop(
    const std::string& filePath)
{
    MESH_NODE<T, dim> X;
    MESH_ELEM<dim - 1> Tri;
    Read_TriMesh_Obj(filePath, X, Tri);

    std::map<VECTOR<int, 2>, int> edgeInd;
    std::map<VECTOR<int, 2>, int> oppoVInd;
    std::vector<std::set<int>> vNeighbor(X.size);
    int newEInd = 0;
    Tri.Each([&](int id, auto data) {
        auto &[triVInd] = data;
        for (int d = 0; d < dim; ++d) {
            auto finder = edgeInd.find(VECTOR<int, 2>(triVInd[(d + 1) % dim], triVInd[d]));
            if (finder == edgeInd.end()) {
                edgeInd[VECTOR<int, 2>(triVInd[d], triVInd[(d + 1) % dim])] = newEInd++;
            }
            else {
                edgeInd[VECTOR<int, 2>(triVInd[d], triVInd[(d + 1) % dim])] = finder->second;
            }
            oppoVInd[VECTOR<int, 2>(triVInd[d], triVInd[(d + 1) % dim])] = triVInd[(d + 2) % dim];

            vNeighbor[triVInd[d]].insert(triVInd[(d + 1) % dim]);
            vNeighbor[triVInd[d]].insert(triVInd[(d + 2) % dim]);
        }
    });

    MESH_NODE<T, dim> newX;
    newX.Reserve(X.size + newEInd);
    for (int i = 0; i < X.size + newEInd; ++i) {
        newX.Append(VECTOR<T, dim>());
    }
    std::map<int, std::vector<int>> BVBNeighbor;
    for (const auto& eI : edgeInd) {
        auto finder = oppoVInd.find(VECTOR<int, 2>(eI.first[1], eI.first[0]));
        if (finder == oppoVInd.end()) {
            // boundary edge
            const VECTOR<T, dim>& v0 = std::get<0>(X.Get_Unchecked(eI.first[0]));
            const VECTOR<T, dim>& v1 = std::get<0>(X.Get_Unchecked(eI.first[1]));
            std::get<0>(newX.Get_Unchecked(X.size + eI.second)) = (v0 + v1) / 2;

            BVBNeighbor[eI.first[0]].emplace_back(eI.first[1]);
            BVBNeighbor[eI.first[1]].emplace_back(eI.first[0]);
        }
        else {
            // interior edge
            const VECTOR<T, dim>& v0 = std::get<0>(X.Get_Unchecked(eI.first[0]));
            const VECTOR<T, dim>& v1 = std::get<0>(X.Get_Unchecked(eI.first[1]));
            const VECTOR<T, dim>& v01next = std::get<0>(X.Get_Unchecked(oppoVInd[eI.first]));
            const VECTOR<T, dim>& v10next = std::get<0>(X.Get_Unchecked(finder->second));
            std::get<0>(newX.Get_Unchecked(X.size + eI.second)) = (3.0 * (v0 + v1) + (v01next + v10next)) / 8;
        }
    }
    for (int i = 0; i < X.size; ++i) {
        auto finder = BVBNeighbor.find(i);
        if (finder != BVBNeighbor.end()) {
            // boundary nodes
            const VECTOR<T, dim>& v0 = std::get<0>(X.Get_Unchecked(finder->second[0]));
            const VECTOR<T, dim>& v1 = std::get<0>(X.Get_Unchecked(finder->second[1]));
            const VECTOR<T, dim>& vself = std::get<0>(X.Get_Unchecked(i));
            std::get<0>(newX.Get_Unchecked(i)) = (6.0 * vself + v0 + v1) / 8;
        }
        else {
            // interior nodes
            T beta = 5.0 / 8 - std::pow(3 + 2 * std::cos(2 * M_PI / vNeighbor[i].size()), 2) / 64;
            VECTOR<T, dim> newPos = (1 - beta) * std::get<0>(X.Get_Unchecked(i));
            for (const auto& nbVI : vNeighbor[i]) {
                newPos += beta / vNeighbor[i].size() * std::get<0>(X.Get_Unchecked(nbVI));
            }
            std::get<0>(newX.Get_Unchecked(i)) = newPos;
        }
    }
    MESH_ELEM<dim - 1> newTri(Tri.size * 4);
    Tri.Each([&](int id, auto data) {
        auto &[triVInd] = data;
        int newVInds[3] = {
            X.size + edgeInd[VECTOR<int, 2>(triVInd[0], triVInd[1])],
            X.size + edgeInd[VECTOR<int, 2>(triVInd[1], triVInd[2])],
            X.size + edgeInd[VECTOR<int, 2>(triVInd[2], triVInd[0])]
        };
        newTri.Append(VECTOR<int, dim>(triVInd[0], newVInds[0], newVInds[2]));
        newTri.Append(VECTOR<int, dim>(triVInd[1], newVInds[1], newVInds[0]));
        newTri.Append(VECTOR<int, dim>(triVInd[2], newVInds[2], newVInds[1]));
        newTri.Append(VECTOR<int, dim>(newVInds[0], newVInds[1], newVInds[2]));
    });

    std::string outputFilePath = filePath.substr(0, filePath.find_last_of('.')) + "_loopSubdiv.obj";
    Write_TriMesh_Obj(newX, newTri, outputFilePath);
}

template <class T, int dim = 3>
void Extrude_Cylinder(const std::string& filePath)
{
    MESH_NODE<T, dim> X;
    MESH_ELEM<dim - 1> Tri;
    Read_TriMesh_Obj(filePath, X, Tri);

    std::vector<int> boundaryNode;
    std::vector<VECTOR<int, 2>> boundaryEdge;
    Find_Boundary_Edge_And_Node(X.size, Tri, boundaryNode, boundaryEdge);

    FILE *out = fopen("aaaaaa.obj", "w+");
    if (!out) {
        printf("cannot create file!\n");
        exit(-1);
    }
    X.Each([&](int id, auto data) {
        auto &[pos] = data;
        fprintf(out, "v %le %le %le\n", pos[0], pos[1] * 1.003, pos[2] * 1.003);
        fprintf(out, "v %le %le %le\n", pos[0], pos[1] * 0.997, pos[2] * 0.997);
    });
    Tri.Each([&](int id, auto data) {
        auto &[triVInd] = data;
        fprintf(out, "f %d %d %d\n", triVInd[0] * 2 + 1, triVInd[1] * 2 + 1, triVInd[2] * 2 + 1);
        fprintf(out, "f %d %d %d\n", triVInd[0] * 2 + 2, triVInd[2] * 2 + 2, triVInd[1] * 2 + 2);
    });
    for (const auto& eI : boundaryEdge) {
        fprintf(out, "f %d %d %d\n", eI[1] * 2 + 1, eI[0] * 2 + 1, eI[0] * 2 + 2);
        fprintf(out, "f %d %d %d\n", eI[1] * 2 + 1, eI[0] * 2 + 2, eI[1] * 2 + 2);
    }
    fclose(out);
}

//#####################################################################

void Export_MeshIO(py::module& m) {
    m.def("Read_TriMesh_Obj", &Read_TriMesh_Obj<double, 2>, "read triangle mesh from obj file");
    m.def("Read_TriMesh_Obj", &Read_TriMesh_Obj<double, 3>, "read triangle mesh from obj file");
    m.def("Read_TriMesh_Tex_Obj", &Read_TriMesh_Tex_Obj<double, 3>, "read triangle mesh with texture to obj file");
    m.def("Write_TriMesh_Obj", &Write_TriMesh_Obj<double, 2>, "write planar triangle mesh to obj file");
    m.def("Write_TriMesh_Obj", &Write_TriMesh_Obj<double, 3>, "write triangle mesh to obj file");
    m.def("Write_TriMesh_Tex_Obj", &Write_TriMesh_Tex_Obj<double, 3>, "write triangle mesh with texture to obj file");
    m.def("Read_TetMesh_Vtk", &Read_TetMesh_Vtk<double>, "read tetrahedra mesh from vtk file");
    m.def("Read_TetMesh_Mesh", &Read_TetMesh_Mesh<double>, "read tetrahedra mesh from mesh file");
    m.def("Write_TetMesh_Vtk", &Write_TetMesh_Vtk<double>, "write tetrahedra mesh from vtk file");
    m.def("Find_Surface_TriMesh", &Find_Surface_TriMesh<double>, "find surface triangle mesh of tetrahedra mesh");
    m.def("Write_Surface_TriMesh_Obj", &Write_Surface_TriMesh_Obj<double>, "write surface triangle mesh of tetrahedra mesh");
    m.def("Read_SegMesh_Seg", &Read_SegMesh_Seg<double, 2>, "read segment mesh");
    m.def("Read_SegMesh_Seg", &Read_SegMesh_Seg<double, 3>, "read segment mesh");
    m.def("Write_SegMesh_Obj", &Write_SegMesh_Obj<double, 2>, "write segment mesh");
    m.def("Write_SegMesh_Obj", &Write_SegMesh_Obj<double, 3>, "write segment mesh");

    m.def("Transform_Points", &Transform_Points<double, 2>, "transform 2d points");
    m.def("Transform_Points", &Transform_Points<double, 3>, "transform 3d points");
    m.def("Transform_Points_Range", &Transform_Points_Range<double, 3>, "transform 3d selected points");

    m.def("Append_Attribute", &Append_Attribute<double, 2>, "append vector_storage to the other");
    m.def("Append_Attribute", &Append_Attribute<double, 3>, "append vector_storage to the other");
    
    m.def("Modify_UV", &Modify_UV<double, 3>, "modify uv");
    m.def("Load_Velocity", &Load_Velocity<double, 3>, "Load_Velocity");
    m.def("Load_Velocity_X0", &Load_Velocity_X0<double, 3>, "Load_Velocity_X0");
    m.def("Zero_Velocity", &Zero_Velocity<double, 3>, "Zero_Velocity");
    m.def("Convert_Seg_File", &Convert_Seg_File<double>, "Convert_Seg_File");

    m.def("Output_Spatial_Hash", &Output_Spatial_Hash<double, 3>, "Output_Spatial_Hash");

    m.def("Subdivide", &Subdivide<double, 3>, "Subdivide");
    m.def("Subdivide_Loop", &Subdivide_Loop<double, 3>, "Subdivide_Loop");
    m.def("Extrude_Cylinder", &Extrude_Cylinder<double, 3>, "Extrude_Cylinder");
    
    m.def("Deep_Copy_Attribute", &Deep_Copy_Attribute<MESH_NODE_ATTR<double, 2>>, "append vector_storage to the other");
    m.def("Deep_Copy_Attribute", &Deep_Copy_Attribute<MESH_NODE_ATTR<double, 3>>, "append vector_storage to the other");
    m.def("Deep_Copy_Attribute", &Deep_Copy_Attribute<MESH_ELEM_ATTR<double, 2>>, "append vector_storage to the other");
    m.def("Deep_Copy_Attribute", &Deep_Copy_Attribute<MESH_ELEM_ATTR<double, 3>>, "append vector_storage to the other");
    m.def("Deep_Copy_Attribute", &Deep_Copy_Attribute<FIXED_COROTATED<double, 2>>, "append vector_storage to the other");
    m.def("Deep_Copy_Attribute", &Deep_Copy_Attribute<FIXED_COROTATED<double, 3>>, "append vector_storage to the other");
}

};