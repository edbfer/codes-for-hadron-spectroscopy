#pragma once 

#include <cstdlib>
#include <cmath>
#include <iostream>
#include <complex>
#include <vector>
#include <cstring>

using namespace std;

namespace cfhs {
    class matrix {

        struct swapOut {
            complex<double> pivot;
            int swapped;
            int l1;
            int l2;
        };

        struct gaussInfo {
            int ispermutation;
            int p1;
            int p2;
            complex<double> v;
        };

    public:
        //complex ** mat;
        complex<double> *mat = nullptr;
        int *perms = nullptr;
        int n, m;
        mutable complex<double> det;
        mutable int dirty;
        mutable int sigma;

        matrix(const int n = 3, const int m = 3);
        matrix(const matrix &m1);
        ~matrix();

        static matrix id(int n);
        friend matrix permutation(int n, int l1, int l2) ;
        static matrix random(int n, int m);
        static matrix hermitian(int n);
        static matrix tridiagonal(complex<double> d0, complex<double> dp, complex<double> d1, int x);
        friend matrix slash(matrix& vector);

        complex<double> &operator()(int x, int y);
        complex<double> &operator()(int x, int y) const;
        complex<double> *operator[](const size_t i);
        complex<double> *operator[](const size_t i) const;

        matrix operator=(const matrix &m1);

        matrix operator-();
        matrix operator++(int);

        friend matrix transpose(matrix &m1);
        static matrix inverte(matrix &m1);
        friend matrix extend(matrix &m1, matrix &m2);
        friend matrix extract(matrix &m1, int x0, int x1, int y0, int y1);

        void swapLine(int l1, int l2);
        matrix swapColForVector(int col, matrix &vec);

        friend matrix operator^(matrix &m1, int e);
        friend matrix operator+(const matrix &m1, const matrix &m2);
        friend matrix operator+(complex<double> v, const matrix &m1);
        friend matrix operator+(const matrix &m1, complex<double> v);
        friend matrix operator-(const matrix &m1, const matrix &m2);
        friend matrix operator*(matrix m1, matrix m2) ;

        matrix operator~();

        friend matrix operator*(matrix &m1, complex<double> v);
        friend matrix operator*(complex<double> v, matrix &m1);
        friend matrix operator/(matrix &m1, complex<double> v);
        friend complex<double> dot(matrix &m1, matrix &m2);

        friend ostream &operator<<(ostream &out, matrix &m1);
        friend istream &operator>>(istream &in, matrix &m1);

        friend complex<double> determinant(matrix &m1);
        complex<double> trace();

        swapOut swapLineForPivot(int col, int inv);

        friend complex<double> normInf(matrix &m1);
        matrix gauss(int inv, int c1, int c2);
        friend matrix gramschmidt(matrix &m1);
        matrix LUDecomposition();
        matrix LUBacksubstitution(matrix b);
        matrix QRDecomposition();
        static matrix eigenpairs(matrix m1, int nmax);

        void fill(istream &in);
        void fill(complex<double> val);

        void printCoord(ostream &o);

        matrix execute(complex<double> (*func)(int i, int j, complex<double> vij));

        complex<double> gamma1[16] = {0, 0, 0, -1i, 0, 0, -1i, 0, 0, 1i, 0, 0, 1i, 0, 0, 0};
        complex<double> gamma2[16] = {0, 0, 0, -1., 0, 0, 1., 0, 0, 1., 0, 0, -1., 0, 0, 0};
        complex<double> gamma3[16] = {0, 0, -1i, 0, 0, 0, 0, 1i, 1i, 0, 0, 0, 0, -1i, 0, 0};
        complex<double> gamma4[16] = {1., 0, 0, 0, 0, 1., 0, 0, 0, 0, -1., 0, 0, 0, 0, -1.};
    };

    matrix permutation(int n, int l1, int l2);
    matrix transpose(matrix &m1);
    matrix slash(matrix &m);
    matrix extend(matrix &m1, matrix &m2);
    matrix extract(matrix &m1, int x0, int y0, int x1, int y1);
    matrix operator^(matrix &m1, int e);
    matrix operator+(complex<double> v, const matrix &m1);
    matrix operator+(const matrix &m1, complex<double> v);
    matrix operator+(const matrix &m1, const matrix &m2);
    matrix operator-(const matrix &m1, const matrix &m2);
    matrix operator*(matrix m1, matrix m2);
    matrix operator*(matrix &m1, complex<double> v);
    matrix operator*(complex<double> v, matrix &m1);
    matrix operator/(matrix &m1, complex<double> v);
    complex<double> dot(matrix &m1, matrix &m2);
    ostream &operator<<(ostream &out, matrix &m1);
    istream &operator>>(istream &in, matrix &m1);
    complex<double> determinant(matrix &m1);
    complex<double> normInf(matrix &m1);
    matrix gramschmidt(matrix &m1);
}