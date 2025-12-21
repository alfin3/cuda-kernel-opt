// Template classes for testing the correctness of matmul kernels.
//
// The allocated source vectors with the maximal dimensions are reused to create
// sub-matrices for test purposes. A derived class may set the values of
// matrices to pseudorandom or constant values, apply a different scheme
// for reusing the source vectors, and/or a different method for efficiently
// testing the correctness of matrix multiplication.

#ifndef MATMUL_TEST_H
#define MATMUL_TEST_H

#include <iostream>
#include <cassert>
#include <cstdlib>
#include <vector>
#include <algorithm>

namespace matmul_test {

struct MatDim {int m, n, k;};

// Abstract class with an interface prototype that cannot be instantiated.
template <typename T>
class MatmulTest {
 public:
    virtual ~MatmulTest() {};

    virtual const T *GetA() const = 0;
    virtual const T *GetB() const = 0;
    virtual T *GetC() = 0;
    virtual const MatDim GetMatDim() const = 0;
    virtual const MatDim GetMatDimMax() const = 0;
    virtual bool IsCorrect(const T *C, const MatDim& dim, const T& atol) const = 0;
    virtual void Rerand() = 0;
};

template <typename T>
class MatmulTestSquare : public MatmulTest<T> {
 public:
    MatmulTestSquare(const int& dim_max, const int& tile_dim, const T& val_min, const T& val_max);

    virtual const T *GetA() const;
    virtual const T *GetB() const;
    virtual T *GetC();
    virtual const MatDim GetMatDim() const;
    virtual const MatDim GetMatDimMax() const;
    virtual bool IsCorrect(const T *C, const MatDim& dim, const T& atol) const;
    virtual void Rerand();

    void PrintA() const;
    void PrintB() const;
    void PrintC() const;

 private:
    // Constants for allocation, initialization, and comparison.
    const int dim_max_;
    const int tile_dim_;
    const T zero_;
    const T val_min_;
    const T val_max_;

    // Source vectors of contiguous random values, allocated once.
    std::vector<T> vec_a_;
    std::vector<T> vec_b_;
    std::vector<T> vec_c_;
    void PrintHelper(const std::vector<T>& vec, const int& dim_max) const;
};

template <typename T>
void MatmulTestSquare<T>::PrintHelper(const std::vector<T>& vec, const int& dim_max) const {
    const typename std::vector<T>::const_iterator itr_end = vec.end();
    for (typename std::vector<T>::const_iterator itr = vec.begin(); itr != itr_end;
         itr += dim_max_) {
        std::for_each(itr, itr + dim_max_, [](const T& val) {std::cout << " " << val;});
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

template <typename T>
MatmulTestSquare<T>::MatmulTestSquare(const int& dim_max, const int& tile_dim, const T& val_min,
                                      const T& val_max) :
    dim_max_(dim_max), tile_dim_(tile_dim), zero_(static_cast<T>(0)), val_min_(val_min),
    val_max_(val_max), vec_a_(dim_max * dim_max), vec_b_(dim_max * dim_max),
    vec_c_(dim_max * dim_max) {
    std::srand(static_cast<unsigned int>(time(NULL)));
    Rerand();
}

template <typename T>
const T *MatmulTestSquare<T>::GetA() const {
    return &vec_a_[0];
}

template <typename T>
const T *MatmulTestSquare<T>::GetB() const {
    return &vec_b_[0];
}

template <typename T>
T *MatmulTestSquare<T>::GetC() {
    return &vec_c_[0];
}

template <typename T>
const MatDim MatmulTestSquare<T>::GetMatDim() const {
    int dim = rand() % (dim_max_ + 1);
    dim = (dim <= tile_dim_) ? tile_dim_ : dim - (dim % tile_dim_);
    return {dim, dim, dim};
}

template <typename T>
const MatDim MatmulTestSquare<T>::GetMatDimMax() const {
    return {dim_max_, dim_max_, dim_max_};
}

template <typename T>
bool MatmulTestSquare<T>::IsCorrect(const T *C, const MatDim& dim, const T& atol) const {
    assert(C == &vec_c_[0]);
    for (int i = 0; i < dim.m; ++i) {
        for (int j = 0; j < dim.n; ++j) {

            // Compute matmul value and compare.
            T val_c = zero_;
            for (int k = 0; k < dim.k; ++k) {
                val_c += vec_a_[i * dim.k + k] * vec_b_[k * dim.n + j];
            }
            if (std::abs(val_c - C[i * dim.n + j]) > atol) {
                return false;
            }
        }
    }
    return true;
}

template <typename T>
void MatmulTestSquare<T>::Rerand() {
    const typename std::vector<T>::iterator itr_a_end = vec_a_.end();
    typename std::vector<T>::iterator itr_b = vec_b_.begin();
    for (typename std::vector<T>::iterator itr_a = vec_a_.begin(); itr_a != itr_a_end; ++itr_a,
         ++itr_b) {
        *itr_a = std::rand() / (RAND_MAX + 1.0) * (val_max_ - val_min_) + val_min_;
        *itr_b = std::rand() / (RAND_MAX + 1.0) * (val_max_ - val_min_) + val_min_;
    }
    // TODO: transpose of B.
}

template <typename T>
void MatmulTestSquare<T>::PrintA() const {
    PrintHelper(vec_a_, dim_max_);
};

template <typename T>
void MatmulTestSquare<T>::PrintB() const {
    PrintHelper(vec_b_, dim_max_);
}

template <typename T>
void MatmulTestSquare<T>::PrintC() const {
    PrintHelper(vec_c_, dim_max_);
}

} // Namespace matmul_test.

#endif  // MATMUL_TEST_H
