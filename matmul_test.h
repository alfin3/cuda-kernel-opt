// Template classes for testing matmul kernels.
//
// The allocated source vectors with the maximal dimensions are reused to create
// sub-matrices for test purposes. A derived class may set the values of
// matrices to pseudorandom or constant values, apply a different scheme
// for reusing the source vectors, and/or a different method for efficiently
// testing matrix multiplication.

#ifndef MATMUL_TEST_H
#define MATMUL_TEST_H

#include <iostream>
#include <cassert>
#include <cstdlib>
#include <vector>
#include <algorithm>

namespace matmul_test {

struct MatDim {int m, n, k;};

// Helper functions.

template <typename DT>
static void PrintHelper(const std::vector<DT>& vec, const int& dim, const int& dim_max) {
    const typename std::vector<DT>::const_iterator itr_end = vec.end();
    for (typename std::vector<DT>::const_iterator itr = vec.begin(); itr != itr_end; itr += dim_max) {
        std::for_each(itr, itr + dim, [](const DT& val) {std::cout << " " << val;});
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

// Abstract class with an interface prototype that cannot be instantiated.
template <typename DT, typename DT_ACC>
class MatmulTest {
 public:
    virtual ~MatmulTest() {};

    virtual const DT *GetA() const = 0;
    virtual const DT *GetB() const = 0;
    virtual DT_ACC *GetRes() = 0;
    virtual const MatDim GetMatDim() const = 0;
    virtual const MatDim GetMatDimMax() const = 0;
    virtual void Rerand() = 0;
};

// MatmulTestSquare.

template <typename DT, typename DT_ACC>
class MatmulTestSquare : public MatmulTest<DT, DT_ACC> {
 public:
    MatmulTestSquare(const int& dim_max, const int& tile_dim, const float& val_min, const float& val_max);

    virtual const DT *GetA() const;
    virtual const DT *GetB() const;
    virtual DT_ACC *GetRes();
    virtual const MatDim GetMatDim() const;
    virtual const MatDim GetMatDimMax() const;
    virtual void Rerand();

    bool IsCorrect(const DT_ACC *res, const MatDim& dim, const DT_ACC& atol) const;

    void PrintA(const int& dim) const;
    void PrintB(const int& dim) const;
    void PrintRes(const int& dim) const;

 private:
    // Constants for allocation, initialization, and comparison.
    const int dim_max_;
    const int tile_dim_;
    const float val_min_;
    const float val_max_;

    // Source vectors of contiguous random values, allocated once.
    std::vector<DT> vec_a_;
    std::vector<DT> vec_b_;
    std::vector<DT_ACC> vec_res_;
};

template <typename DT, typename DT_ACC>
MatmulTestSquare<DT, DT_ACC>::MatmulTestSquare(const int& dim_max, const int& tile_dim, const float& val_min,
    const float& val_max) :
    dim_max_(dim_max), tile_dim_(tile_dim), val_min_(val_min), val_max_(val_max), vec_a_(dim_max * dim_max),
    vec_b_(dim_max * dim_max), vec_res_(dim_max * dim_max) {
    std::srand(static_cast<unsigned int>(time(NULL)));
    Rerand();
}

template <typename DT, typename DT_ACC>
const DT *MatmulTestSquare<DT, DT_ACC>::GetA() const {
    return &vec_a_[0];
}

template <typename DT, typename DT_ACC>
const DT *MatmulTestSquare<DT, DT_ACC>::GetB() const {
    return &vec_b_[0];
}

template <typename DT, typename DT_ACC>
DT_ACC *MatmulTestSquare<DT, DT_ACC>::GetRes() {
    return &vec_res_[0];
}

template <typename DT, typename DT_ACC>
const MatDim MatmulTestSquare<DT, DT_ACC>::GetMatDim() const {
    int dim = rand() % (dim_max_ + 1);
    dim = (dim <= tile_dim_) ? tile_dim_ : dim - (dim % tile_dim_);
    return {dim, dim, dim};
}

template <typename DT, typename DT_ACC>
const MatDim MatmulTestSquare<DT, DT_ACC>::GetMatDimMax() const {
    return {dim_max_, dim_max_, dim_max_};
}

template <typename DT, typename DT_ACC>
bool MatmulTestSquare<DT, DT_ACC>::IsCorrect(const DT_ACC *res, const MatDim& dim, const DT_ACC& atol) const {
    assert(res == &vec_res_[0]);
    for (int i = 0; i < dim.m; ++i) {
        for (int j = 0; j < dim.n; ++j) {

            // Compute matmul value and compare.
            DT_ACC temp = static_cast<DT_ACC>(0);
            for (int k = 0; k < dim.k; ++k) {
                temp += static_cast<DT_ACC>(vec_a_[i * dim.k + k]) * static_cast<DT_ACC>(vec_b_[k * dim.n + j]);
            }
            if (std::abs(temp - res[i * dim.n + j]) > atol) {
                return false;
            }
        }
    }
    return true;
}

template <typename DT, typename DT_ACC>
void MatmulTestSquare<DT, DT_ACC>::Rerand() {
    const typename std::vector<DT>::iterator itr_a_end = vec_a_.end();
    typename std::vector<DT>::iterator itr_b = vec_b_.begin();
    for (typename std::vector<DT>::iterator itr_a = vec_a_.begin(); itr_a != itr_a_end; ++itr_a, ++itr_b) {
        *itr_a = static_cast<DT>(std::rand() / (RAND_MAX + 1.0f) * (val_max_ - val_min_) + val_min_);
        *itr_b = static_cast<DT>(std::rand() / (RAND_MAX + 1.0f) * (val_max_ - val_min_) + val_min_);
    }
}

template <typename DT, typename DT_ACC>
void MatmulTestSquare<DT, DT_ACC>::PrintA(const int& dim) const {
    PrintHelper<DT>(vec_a_, dim, dim_max_);
};

template <typename DT, typename DT_ACC>
void MatmulTestSquare<DT, DT_ACC>::PrintB(const int& dim) const {
    PrintHelper<DT>(vec_b_, dim, dim_max_);
}

template <typename DT, typename DT_ACC>
void MatmulTestSquare<DT, DT_ACC>::PrintRes(const int& dim) const {
    PrintHelper<DT_ACC>(vec_res_, dim, dim_max_);
}

// MatmulTestGemmSquare.

template <typename DT, typename DT_ACC>
class MatmulTestGemmSquare : public MatmulTest<DT, DT_ACC> {
 public:
    MatmulTestGemmSquare(const int& dim_max, const int& tile_dim, const float& val_min, const float& val_max);

    virtual const DT *GetA() const;
    virtual const DT *GetB() const;
    const DT_ACC *GetC() const;
    virtual DT_ACC *GetRes();
    virtual const MatDim GetMatDim() const;
    virtual const MatDim GetMatDimMax() const;
    virtual void Rerand();

    bool IsCorrect(const DT_ACC *res, const MatDim& dim, const DT_ACC& alpha, const DT_ACC& beta, const DT_ACC& atol) const;

    void PrintA(const int& dim) const;
    void PrintB(const int& dim) const;
    void PrintC(const int& dim) const;
    void PrintRes(const int& dim) const;

 private:
    // Constants for allocation, initialization, and comparison.
    const int dim_max_;
    const int tile_dim_;
    const float val_min_;
    const float val_max_;

    // Source vectors of contiguous random values, allocated once.
    std::vector<DT> vec_a_;
    std::vector<DT> vec_b_;
    std::vector<DT_ACC> vec_c_;
    std::vector<DT_ACC> vec_res_;
};

template <typename DT, typename DT_ACC>
MatmulTestGemmSquare<DT, DT_ACC>::MatmulTestGemmSquare(const int& dim_max, const int& tile_dim, const float& val_min,
    const float& val_max) :
    dim_max_(dim_max), tile_dim_(tile_dim), val_min_(val_min), val_max_(val_max), vec_a_(dim_max * dim_max), vec_b_(dim_max * dim_max), vec_c_(dim_max * dim_max),
    vec_res_(dim_max * dim_max) {
    std::srand(static_cast<unsigned int>(time(NULL)));
    Rerand();
}

template <typename DT, typename DT_ACC>
const DT *MatmulTestGemmSquare<DT, DT_ACC>::GetA() const {
    return &vec_a_[0];
}

template <typename DT, typename DT_ACC>
const DT *MatmulTestGemmSquare<DT, DT_ACC>::GetB() const {
    return &vec_b_[0];
}

template <typename DT, typename DT_ACC>
const DT_ACC *MatmulTestGemmSquare<DT, DT_ACC>::GetC() const {
    return &vec_c_[0];
}

template <typename DT, typename DT_ACC>
DT_ACC *MatmulTestGemmSquare<DT, DT_ACC>::GetRes() {
    return &vec_res_[0];
}

template <typename DT, typename DT_ACC>
const MatDim MatmulTestGemmSquare<DT, DT_ACC>::GetMatDim() const {
    int dim = rand() % (dim_max_ + 1);
    dim = (dim <= tile_dim_) ? tile_dim_ : dim - (dim % tile_dim_);
    return {dim, dim, dim};
}

template <typename DT, typename DT_ACC>
const MatDim MatmulTestGemmSquare<DT, DT_ACC>::GetMatDimMax() const {
    return {dim_max_, dim_max_, dim_max_};
}

template <typename DT, typename DT_ACC>
bool MatmulTestGemmSquare<DT, DT_ACC>::IsCorrect(const DT_ACC *res, const MatDim& dim, const DT_ACC& alpha, const DT_ACC& beta,
    const DT_ACC& atol) const {
    assert(res == &vec_res_[0]);
    for (int i = 0; i < dim.m; ++i) {
        for (int j = 0; j < dim.n; ++j) {

            // Compute matmul value and compare.
            DT_ACC temp = static_cast<DT_ACC>(0);
            for (int k = 0; k < dim.k; ++k) {
                temp += static_cast<DT_ACC>(vec_a_[i * dim.k + k]) * static_cast<DT_ACC>(vec_b_[k * dim.n + j]);
            }
            if (std::abs(alpha * temp + beta * vec_c_[i * dim.n + j] - res[i * dim.n + j]) > atol) {
                return false;
            }
        }
    }
    return true;
}

template <typename DT, typename DT_ACC>
void MatmulTestGemmSquare<DT, DT_ACC>::Rerand() {
    const typename std::vector<DT>::iterator itr_a_end = vec_a_.end();
    typename std::vector<DT>::iterator itr_b = vec_b_.begin();
    typename std::vector<DT_ACC>::iterator itr_c = vec_c_.begin();
    for (typename std::vector<DT>::iterator itr_a = vec_a_.begin(); itr_a != itr_a_end; ++itr_a, ++itr_b, ++itr_c) {
        *itr_a = static_cast<DT>(std::rand() / (RAND_MAX + 1.0f) * (val_max_ - val_min_) + val_min_);
        *itr_b = static_cast<DT>(std::rand() / (RAND_MAX + 1.0f) * (val_max_ - val_min_) + val_min_);
        *itr_c = static_cast<DT_ACC>(std::rand() / (RAND_MAX + 1.0f) * (val_max_ - val_min_) + val_min_);
    }
}

template <typename DT, typename DT_ACC>
void MatmulTestGemmSquare<DT, DT_ACC>::PrintA(const int& dim) const {
    PrintHelper<DT>(vec_a_, dim, dim_max_);
};

template <typename DT, typename DT_ACC>
void MatmulTestGemmSquare<DT, DT_ACC>::PrintB(const int& dim) const {
    PrintHelper<DT>(vec_b_, dim, dim_max_);
}

template <typename DT, typename DT_ACC>
void MatmulTestGemmSquare<DT, DT_ACC>::PrintC(const int& dim) const {
    PrintHelper<DT_ACC>(vec_c_, dim, dim_max_);
}

template <typename DT, typename DT_ACC>
void MatmulTestGemmSquare<DT, DT_ACC>::PrintRes(const int& dim) const {
    PrintHelper<DT_ACC>(vec_res_, dim, dim_max_);
}

} // Namespace matmul_test.

#endif  // MATMUL_TEST_H
