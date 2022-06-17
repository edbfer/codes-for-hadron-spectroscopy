#include "matrix.h"
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>

using namespace std;
using namespace cfhs;

matrix::matrix(const int n, const int m): n(n), m(m)
{
    mat = new complex<double>[n * m];
    perms = new int[n];
    dirty = 1;
    det = (complex<double>) 0.;
}


matrix::matrix(const matrix& m1): n(m1.n), m(m1.m)
{
    mat = new complex<double>[n * m];
    perms = new int[n];
    memcpy(mat, m1.mat, sizeof(complex<double>) * n * m);
    memcpy(perms, m1.perms, sizeof(int) * n);
    det = m1.det;
    dirty = m1.dirty;
    sigma = m1.sigma;
}


matrix matrix::id(int n)
{
    matrix res(n, n);
    for (int i = 0; i < n; i++) {
        res(i, i) = ((complex<double>) 1.);
    }
    return res;
}


matrix cfhs::permutation(int n, int l1, int l2)
{
	matrix res = matrix::id(n);
	res.swapLine(l1, l2);
	return res;
}


matrix matrix::random(int n, int m)
{
    matrix res(n, m);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            res(i, j) = (complex<double>) ((complex<double>) rand() / (complex<double>) RAND_MAX) * 10.0;
        }
    }
    return res;
}

matrix matrix::hermitian(int n)
{
    matrix res(n, n);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < i + 1; j++) {
            res(i, j) = (complex<double>) ((complex<double>) rand() / (complex<double>) RAND_MAX) * 10.0;
            res(j, i) = res(i, j);
        }
    }
    return res;
}

matrix matrix::tridiagonal(complex<double> d0, complex<double> dp, complex<double> d1, int x) {
    matrix res(x, x);
    for (int i = 0; i < x - 1; i++) {
        res(i, i) = dp;
        int j2 = i + 1;
        res(i, j2) = d1;
        res(j2, i) = d0;
    }
    int j2 = x - 1;
    res(j2, j2) = dp;
    return res;
}

matrix::~matrix()
{
	delete[] mat;
	delete[] perms;
}

complex<double> &matrix::operator()(int x, int y) {
    return mat[x * m + y];
}

complex<double> &matrix::operator()(int x, int y) const {
    return mat[x * m + y];
}

complex<double> *matrix::operator[](const size_t i) 
{
	return &mat[i * m];
}

complex<double> *matrix::operator[](const size_t i) const
{
	return &mat[i * m];
}


matrix matrix::operator= (const matrix& m1)
{
    if(mat != nullptr) delete[] mat;
    if(perms != nullptr) delete[] perms;

    n = m1.n;
    m = m1.m;

    mat = new complex<double>[n * m];
    perms = new int[n];
    memcpy(mat, m1.mat, sizeof(complex<double>) * n * m);
    memcpy(perms, m1.perms, sizeof(int) * n);

    dirty = m1.dirty;
    det = m1.det;
    sigma = m1.sigma;

    return *this;
}


matrix matrix::operator-()
{
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            (*this)(i, j) = -(*this)(i, j);
        }
    }
    return *this;
}

matrix matrix::operator++(int) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            (*this)(i, j) = (*this)(i, j) + (complex<double>) 1.;
        }
    }
    return *this;
}

matrix cfhs::operator-(const matrix& m1, const matrix& m2)
{
    matrix res(m1.n, m1.m);
    for (int i = 0; i < m1.n; i++) {
        for (int j = 0; j < m1.m; j++) {
            res(i, j) = m1(i, j) - m2(i, j);
        }
    }
    return res;
}

matrix matrix::operator~()
{
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            (*this)(i, j) = conj((*this)(i, j));
        }
    }
    return *this;
}


matrix cfhs::operator+(const matrix& m1, const matrix& m2)
{
    matrix res(m1.n, m1.m);
    for (int i = 0; i < m1.n; i++) {
        for (int j = 0; j < m1.m; j++) {
            res(i, j) = m1(i, j) + m2(i, j);
        }
    }
    return res;
}

matrix cfhs::operator+(const matrix& m1, complex<double> v)
{
    matrix res(m1.n, m1.m);
    for(int i = 0; i < m1.n; i++)
    {
        //for(int j = 0; j < m1.m; j++)
            res(i, i) = m1(i, i) + v;
    }

    return res;
}

matrix cfhs::operator+(complex<double> v, const matrix& m1)
{
    return operator+(m1, v);
}

matrix cfhs::operator*(matrix m1, matrix m2)
{
    matrix newm(m1.n, m2.m);
    for (int i = 0; i < newm.n; i++) {
        for (int j = 0; j < newm.m; j++) {
            complex<double> v = 0.;
            for (int k = 0; k < m1.m; k++) {
                complex<double> v1 = m1(i, k);
                complex<double> v2 = m2(k, j);
                v += v1 * v2;
            }
            newm(i, j) = v;
        }
    }

    return newm;
}


matrix cfhs::operator*(matrix& m1, complex<double> v)
{
    matrix res = m1;
    int i = 0, j = 0;
    #pragma omp parallel for
    for (i = 0; i < m1.n; i++) {
        for (j = 0; j < m1.m; j++) {
            res(i, j) = m1(i, j) * v;
        }
    }

    return res;
}

matrix cfhs::operator/(matrix& m1, complex<double> v)
{
    return operator*(m1, 1./v);
}

matrix cfhs::operator*(complex<double> v, matrix& m1)
{
    return cfhs::operator*(m1, v);
}


matrix cfhs::operator^(matrix& m1, int e)
{
    matrix r = m1;
    for (int i = 0; i < e; i++) {
        r = r * m1;
    }
    return r;
}


ostream& cfhs::operator<<(ostream& out, matrix& m1)
{
    for (int i = 0; i < m1.n; i++) {
        for (int j = 0; j < m1.m; j++) {
            out << m1(i, j) << " ";
        }

        out << endl;
    }
    return out;
}


istream& cfhs::operator>>(istream& in, matrix& m1)
{
    for (int i = 0; i < m1.n; i++) {
        for (int j = 0; j < m1.m; j++) {
            in >> m1(i, j);
        }
    }
    return in;
}


complex<double> cfhs::determinant(matrix& m1)
{
    if (!m1.dirty)
        return m1.det;
    else {
        m1.dirty = 0;
        m1.gauss(0, 0, m1.m );
        return m1.det;
    }
}

complex<double> cfhs::normInf(matrix& m1)
{
    vector<complex<double>> v;
    complex<double> s = 0.;
    for (int i = 0; i < m1.n; i++) {
        s = 0.;
        for (int j = 0; j < m1.m; j++) {
            s = s + m1(i, j);
        }
        v.push_back(s);
    }

    //sort(v.begin(), v.end());

    return v[v.size() - 1];
}


cfhs::matrix::swapOut matrix::swapLineForPivot(int col, int inv)
{
    int firstZero = -1;
    int notZero = -1;
    int inc = (inv) ? -1 : +1;
    int i = 0;
    for (i = col; (inv) ? (i >= col) : (i < m); i += inc) {
        if ((*this)(i, col) != 0.)
            break;

    }

    swapOut out = {(*this)(i, col), (i == col) ? 0 : 1, i, col};

    swapLine(i, col);

    return out;
}


matrix matrix::inverte(matrix& m1)
{
    if(m1.n != m1.m)
    {
        cout << "Singular!" << endl;
        exit(-1);
    }

    matrix temp = m1;
    temp.LUDecomposition();

    matrix res(m1.n, m1.n);

    matrix c(m1.n, 1);
    for(int j = 0; j < m1.n; j++) {
        for (int i = 0; i < m1.n; i++) {
            c(i, 0) = 0.0;
        }
        c(j, 0) = 1.0;

        matrix tt = temp.LUBacksubstitution(c);
        for(int i = 0; i < m1.n; i++)
        {
            res(i, j) = tt(i, 0);
        }
    }

    return res;
/*
    matrix i = matrix::id(m1.m);
    matrix t = extend(m1, i);
    //cout << t << endl;
    matrix res = t.gauss(0, 0, m1.n);
    //cout << res << endl;

    res = res.gauss(1, 0, m1.n - 1);
    //cout << res << endl;


    for (int i = 0; i < m1.n; i++) {
        complex<double> v = res(i, i);
        for (int j = 0; j < res.m; j++) {
            res(i, j) = res(i, j) / v;
        }
    }

    res = extract(res, 0, m1.n - 1, m1.m, t.m - 1);

    return res;*/
}


matrix matrix::LUDecomposition() {
    //Flannery et Al, Crout algo w/ partial pivoting
    if(n != m)
    {
        cout << "This matrix is not square!" << endl;
        exit(-1);
    }

    auto *scalings = new complex<double>[n]();
    complex<double> large = 0, temp = 0;
    int imax = 0;

    //vamos procurar os scalings das linhas
    for (int i = 0; i < n; i++)
    {
        large = 0;
        int colindex = 0;
        for(int j = 0; j < n; j++) {
            temp = abs((*this)(i, j));
            if (abs(temp) > abs(large)) {
                large = temp;
                colindex = j;
            }
        }
        /*if (large == 0)
        {
            cout << "This matrix is singular!" << endl;
            exit(1);
        }*/

        //save the scaling -> this is the largest pivot in the row
        scalings[i] = (1.0 / (*this)(i, colindex));
    }

    //agora vamos ao método de crout propriamente dito
    for(int j = 0; j < n; j++) //for each col
    {
        for(int i = 0; i < j; i++) //for each row
        {
            complex<double> sum = (*this)(i, j);

            for(int k = 0; k < i; k++)
                sum -= (*this)(i, k) * (*this)(k, j);

            (*this)(i, j) = sum;
        }

        //procura again pelo pivot maior
        large = 0;
        for(int i = j; i < n; i++)
        {
            complex<double> sum = (*this)(i, j);
            for(int k = 0; k < j; k++) sum -= (*this)(i, k) * (*this)(k, j);
            (*this)(i, j) = sum;

            auto dum = abs(scalings[i] * sum);
            if (dum >= abs(large))
            {
                large = dum;
                imax = i;
            }
        }

        //change rows?
        if(j != imax)
        {
            for(int k = 0; k < n; k++)
            {
                complex<double> t = (*this)(imax, k);
                (*this)(imax, k) = (*this)(j, k);
                (*this)(j, k) = t;
            }

            sigma *= -1.;
            //auto teee = scalings[imax];
            scalings[imax] = scalings[j];
            //scalings[j] = teee;
        }
        perms[j] = imax;

        if((*this)(j, j) == 0.0) (*this)(j, j) = 1e-50; //para nao dar barraca
        if((j+1) != n)
        {
            auto dummy1 = 1.0/((*this)(j, j));
            for(int i = j+1; i < n; i++)
            {
                (*this)(i, j) *= dummy1;
            }
        }
    }
    delete[] scalings;
    return *this;
}

matrix matrix::LUBacksubstitution(matrix b) {

    int ii = 0;

    for(int i = 0; i < n; i++) {
        int ip = perms[i];
        complex<double> sum = b(ip, 0);
        b(ip, 0) = b(i, 0);

        if (ii) {
            for (int j = ii-1; j < i; j++) {
                sum -= (*this)(i, j) * b(j, 0);
            }
        } else if (abs(sum)) ii = i+1;

        b(i, 0) = sum;
    }
    for(int i = n - 1; i >= 0; i--)
    {
        complex<double> sum = b(i, 0);
        for(int j = i+1; j < n; j++)
        {
            sum -= (*this)(i, j) * b(j, 0);
        }
        b(i, 0) = sum / (*this)(i, i);
    }

    return b;
}

matrix matrix::QRDecomposition()
{
    matrix A = *this;
    *this = gramschmidt(*this);
    matrix qinv = transpose(*this);
    matrix R = qinv * A;
    return R;
}


matrix cfhs::transpose(matrix& m1)
{
    matrix r(m1.m, m1.n);
    for (int i = 0; i < m1.n; i++) {
        for (int j = 0; j < m1.m; j++) {
            r(j, i) = m1(i, j);
        }
    }
    return r;
}


matrix matrix::gauss(int inv, int c1, int c2)
{
    matrix m1 = *this;
    int inc = (inv) ? -1 : 1;
    sigma = 1.;
    for (int col = (inv) ? c2 : c1; (inv) ? col >= c1 : col < c2; col += inc) {
        swapOut inf = m1.swapLineForPivot(col, inv);
        sigma = (inf.swapped) ? sigma * -1 : sigma;

        for (int row = (inv) ? col - 1 : col + 1; (inv) ? row >= c1 : row < c2; row += inc) {
            complex<double> v1;
            if ((v1 = m1(row, col)) == 0.0)
                continue;

            complex<double> val = v1 / inf.pivot;
            for (int k = 0; k < m; k++) {
                m1(row, k) = m1(row, k) - (val) * m1(col, k);
            }

        }
    }

    this->det = 1.0;
    for (int i = 0; i < n; i++) {
        this->det *= m1(i, i);
    }

    this->det = det * complex<double>(sigma);
    m1.det = det;

    return m1;
}


matrix cfhs::extend(matrix& m1, matrix& m2)
{
    matrix t(m1.n, m1.m + m2.m);
    for (int i = 0; i < t.n; i++) {
        for (int j = 0; j < m1.m; j++) {
            t(i, j) = m1(i, j);
        }

        for (int j = 0; j < m2.m; j++) {
            t(i, j + m1.m) = m2(i, j);
        }
    }

    return t;
}


matrix cfhs::extract(matrix& m1, int x0, int x1, int y0, int y1)
{
    matrix r(x1 - x0 + 1, y1 - y0 + 1);
    for (int i = x0; i <= x1; i++) {
        for (int j = y0; j <= y1; j++) {
            r(i - x0, j - y0) = m1(i, j);
        }
    }
    return r;
}


complex<double> cfhs::dot(matrix& m1, matrix& m2)
{
    matrix temp1 = transpose(m1);
    matrix temp =  (~temp1) * m2;
    return temp(0, 0);
}


void matrix::swapLine(int l1, int l2)
{
    complex<double> *r = new complex<double>[m];
    //memcpy(r, (*this)[l1], sizeof(complex<double>) * m);
    for(int i = 0; i < m; i++)
    {
        r[i] = (*this)(l1, i);
    }
    //memcpy(r, mat+(l1*m), sizeof(complex<double>)*m);
    for(int i = 0; i < m; i++)
    {
        (*this)(l1, i) = (*this)(l2, i);
    }
    for(int i = 0; i < m; i++)
    {
        (*this)(l2, i) = r[i];
    }
    //memcpy((*this)[l1], (*this)[l2], sizeof(complex<double>) * m);
    //memcpy(mat + (l1*m), mat + (l2*m), sizeof(complex<double>) * m);
    //memcpy((*this)[l2], r, sizeof(complex<double>) * m);
    //memcpy(mat + (l2*m), r, sizeof(complex<double>) * m);
    delete[] r;
}

matrix matrix::swapColForVector(int col, matrix& vec)
{
    for (int i = 0; i < n; i++) {
        (*this)(i, col) = vec(i, 0);
    }
    return *this;
}


matrix cfhs::gramschmidt(matrix& m1)
{
    matrix res = m1;

    for (int i = 0; i < m1.m; i++) {
        matrix v = extract(m1, 0, m1.n - 1, i, i);
        for (int j = 0; j < i; j++) {
            matrix u = extract(res, 0, res.n - 1, j, j);
            /*complex vu = dot(u, v);
            complex mod = dot(u, u);*/
            matrix temp = u * ((complex<double>) (dot(u, v) / dot(u, u)));
            v = v - temp;
        }
        complex<double> mod = (complex<double>) sqrt(dot(v, v));
        v = v * (complex<double>) (1.0 / mod);
        res.swapColForVector(i, v);
    }
    return res;
}


matrix matrix::eigenpairs(matrix m1, int nmax)
{
    matrix temp = m1;
    matrix v0 = random(m1.n, m1.m);
    v0 = gramschmidt(v0);
    matrix v1 = matrix::id(m1.n);

    for (int i = 0; i < nmax; i++) {
        v1 = temp * v0;
        v0 = gramschmidt(v1);
        cout << "Iteração: " << i << " de " << nmax << endl;
    }

    matrix eigenvalues(m1.n, 1);
    //rayleigh quotien
    for (int i = 0; i < m1.m; i++) {
        matrix x = extract(v0, 0, v0.n - 1, i, i);
        matrix t1 = transpose(x);
        matrix t2 = t1 * temp;
        complex<double> t = (complex<double>) (t2 * x)(0, 0);
        complex<double> d = (complex<double>) dot(x, x);
        eigenvalues(i, 0) = t / d;
    }

    v0 = extend(v0, eigenvalues);
    return v0;

    /*//raleigh coefficient
    for(int i = 0; i < m1.m; i++)
    {
        //start with initial guess for eigenval eigenvec
        matrix eigenvec = random(m1.n, 1);
        complex<double> eigenval = random(1, 1)(0, 0);

        matrix matE = id(m1.m);

        double eps = 10.0;
        while(eps > 1e-6)
        {
            matrix new_eigenvec = random(m1.n, 1);

            matrix temp1 = m1 * eigenvec;
            matrix temp2 = ~eigenvec;
            matrix temp3 = temp2 * temp1;

            complex<double> nr = dot(eigenvec, eigenvec);

            complex<double> mu = temp3(0, 0)/nr;

            matrix temp4 = 
        }
    }*/
}

void matrix::fill(istream &in) {
    while (!in.eof()) {
        int x, y;
        in >> x >> y;
        in >> (*this)(x, y);
    }
}

void matrix::fill(complex<double> val) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            (*this)(i, j) = val;
        }
    }
}

void matrix::printCoord(ostream &o) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            o << i << "\t" << j << "\t" << (*this)(i, j) << endl;
        }
    }
}

matrix matrix::execute(complex<double> (*func)(int i, int j, complex<double> vij)) {
    matrix res(n, m);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            res(i, j) = func(i, j, (*this)(i, j));
        }
    }
    return res;
}

matrix cfhs::slash(matrix& vec)
{
    matrix temp(4, 4);

    for(int i = 0; i < 4; i++)    
    {
        for(int j = 0; j < 4; j++)
        {
            int offset = i * 4 + j;
            temp(i, j) = vec(0, 0) * vec.gamma1[offset] + vec(1, 0) * vec.gamma2[offset] + vec(2, 0) * vec.gamma3[offset] + vec(3, 0) * vec.gamma4[offset];
        }
    }

    return temp;
}

complex<double> matrix::trace()
{
    complex<double> tr = 0;
    for(int i = 0; i < n; i++)
    {
        tr += (*this)(i, i);
    }
    return tr;
}