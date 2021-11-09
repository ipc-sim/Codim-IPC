#pragma once
//#####################################################################
// Class CSR_MATRIX
//#####################################################################
#include <Utils/PROFILER.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

namespace JGSL{

//#####################################################################
// Class CSR_MATRIX
//#####################################################################
template <class T>
class CSR_MATRIX
{
    typedef Eigen::SparseMatrix<T, Eigen::RowMajor> MATRIX_TYPE;
    MATRIX_TYPE A;

    // for sorting triplets per row w.r.t col index
    struct customLess {
        const std::vector<Eigen::Triplet<T>>& tripletsRef;
        customLess(const std::vector<Eigen::Triplet<T>>& triplets) :
            tripletsRef(triplets) {}
        bool operator()(int a, int b) const {
            return tripletsRef[a].col() < tripletsRef[b].col();
        }
    };
public:
    //#################################################################
    // Function Construct_From_CSR
    //#################################################################
    void Construct_From_CSR(const std::vector<int>& ptr, const std::vector<int>& col, const std::vector<T>& val)
    {
        int rowsize = ptr.size() - 1;
        A.resize(rowsize, rowsize);
        A.setZero();
        A.reserve(val.size());

        memcpy(A.valuePtr(), val.data(),
            val.size() * sizeof(val[0]));
        memcpy(A.innerIndexPtr(), col.data(),
            col.size() * sizeof(col[0]));
        memcpy(A.outerIndexPtr(), ptr.data(),
            ptr.size() * sizeof(ptr[0]));
        A.finalize();
    }

    void Construct_From_Triplet(int rows, int cols, const std::vector<Eigen::Triplet<T>>& triplets)
    {
        TIMER_FLAG("constructCSRMatrixFromTriplet");
        A.resize(rows, cols);
        A.reserve(triplets.size());
        A.setFromTriplets(triplets.begin(), triplets.end());
        A.finalize();

        // // a parallel version:
        // // reorganize triplets per row
        // std::vector<std::vector<int>> tripletsPerRow(rows);
        // for (int i = 0; i < triplets.size(); ++i) {
        //     // if(triplets[i].row() <= triplets[i].col()) { // for symmetric systems
        //         tripletsPerRow[triplets[i].row()].emplace_back(i);
        //     // }
        // }
        // //TODO: take advantage of block structures! block triplets?

        // BASE_STORAGE<int> threads(rows);
        // for (int i = 0; i < rows; ++i) {
        //     threads.Append(i);
        // }

        // struct customLess compare(triplets);
        // std::vector<std::vector<int>> colIndPerRow(rows);
        // std::vector<std::vector<T>> valuePerRow(rows);
        // A.resize(rows, cols);
        // A.outerIndexPtr()[0] = 0;
        // // sort triplets per row w.r.t. col index and then merge
        // threads.Par_Each([&](int rowI, auto data) {
        //     std::sort(tripletsPerRow[rowI].begin(), tripletsPerRow[rowI].end(), compare);

        //     int curColI = -1;
        //     for (const auto& tripletI : tripletsPerRow[rowI]) {
        //         if (triplets[tripletI].col() != curColI) {
        //             curColI = triplets[tripletI].col();
        //             colIndPerRow[rowI].emplace_back(curColI);
        //             valuePerRow[rowI].emplace_back(triplets[tripletI].value());
        //         }
        //         else {
        //             valuePerRow[rowI].back() += triplets[tripletI].value();
        //         }
        //     }
        //     A.outerIndexPtr()[rowI + 1] = colIndPerRow[rowI].size();
        // });

        // // compute outer index of CSR
        // for (int i = 1; i <= rows; ++i) {
        //     A.outerIndexPtr()[i] += A.outerIndexPtr()[i - 1];
        // }
        
        // A.reserve(A.outerIndexPtr()[rows]);
        // // copy inner index and values of CSR
        // threads.Par_Each([&](int rowI, auto data) {
        //     std::memcpy(A.valuePtr() + A.outerIndexPtr()[rowI], valuePerRow[rowI].data(), 
        //         valuePerRow[rowI].size() * sizeof(T));
        //     std::memcpy(A.innerIndexPtr() + A.outerIndexPtr()[rowI], colIndPerRow[rowI].data(),
        //         colIndPerRow[rowI].size() * sizeof(colIndPerRow[rowI][0]));
        // });
        // A.finalize();

        // for (int i = 0; i < rows + 1; ++i) {
        //     std::cout << A.outerIndexPtr()[i] << " ";
        // }
        // std::cout << std::endl;
        // for (int i = 0; i < A.outerIndexPtr()[rows]; ++i) {
        //     std::cout << A.innerIndexPtr()[i] << " ";
        // }
        // std::cout << std::endl;
        // for (int i = 0; i < A.outerIndexPtr()[rows]; ++i) {
        //     std::cout << A.valuePtr()[i] << " ";
        // }
        // std::cout << std::endl;
        // exit(0);
    }
    // sort triplet with i, j being the keys
    // record the sorting map
    // construct CSR from sorted triplet
    //TODO: void Update_From_Triplet
    // with the sorting map, the triplet should have O(1) access to entry

    void Project_DBC(const std::vector<bool>& DBCb, const int dim)
    {
        TIMER_FLAG("projectDBC");
        for (int k = 0; k < A.outerSize(); ++k) {
            typename MATRIX_TYPE::InnerIterator it(A,k);
            for (; it; ++it) {
                if(DBCb[it.row() / dim] || DBCb[it.col() / dim]) {
                    it.valueRef() = (it.row() == it.col());
                }
            }
        }
    }

    void output(const std::string& filePath)
    {
        FILE *out = fopen(filePath.c_str(), "w+");
        if (!out) {
            printf("file creation error!\n");
            exit(-1);
        }

        for (int k = 0; k < A.outerSize(); ++k) {
            typename MATRIX_TYPE::InnerIterator it(A,k);
            for (; it; ++it) {
                fprintf(out, "%d %d %le\n", it.row() + 1, it.col() + 1, it.value());
            }
        }

        fclose(out);
    }

    //#################################################################
    // Function Get_Matrix
    //#################################################################
    MATRIX_TYPE &Get_Matrix() { return A; }

    //#################################################################
    // Function Get_Item
    //#################################################################
    T& Get_Item(int i, int j) { return A.coeffRef(i,j); }

    //#################################################################
    // Function Resize
    //#################################################################
    void Resize(int i, int j) { A.resize(i,j); }

    //#################################################################
    // Function Size
    //#################################################################
    std::pair<size_t, size_t> Size() {return std::make_pair<size_t, size_t>(A.rows(), A.cols());} 
};

//#####################################################################
template<class T>
void Export_SPARSE_MATRIX_Impl(py::module& m) {
    auto suffix = std::string("_") + (std::is_same<T, float>::value ? "F" : "D");
    py::class_<CSR_MATRIX<T>>(m, ("CSR_MATRIX" + suffix).c_str())
        .def(py::init<>())
        .def("Get_Matrix", &CSR_MATRIX<T>::Get_Matrix, py::return_value_policy::reference_internal) // sparse matrix doesn't support return by reference with builtin api currently.
        .def("Construct_From_CSR", &CSR_MATRIX<T>::Construct_From_CSR)
        .def("__getitem__", [](CSR_MATRIX<T>&A, std::tuple<int, int> index){return A.Get_Item(std::get<0>(index), std::get<1>(index));})
        .def("__setitem__", [](CSR_MATRIX<T>&A, std::tuple<int, int> index, T v){A.Get_Item(std::get<0>(index), std::get<1>(index)) = v;})
        .def("Resize", &CSR_MATRIX<T>::Resize)
        .def("Size", &CSR_MATRIX<T>::Size)
        ;
}

void Export_SPARSE_MATRIX(py::module& m) {
    Export_SPARSE_MATRIX_Impl<float>(m);
    Export_SPARSE_MATRIX_Impl<double>(m);
}
}
