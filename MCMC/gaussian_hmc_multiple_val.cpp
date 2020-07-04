#include <iostream>
#include <cmath>
#include <fstream>
#include <iomanip>

const int niter = 10000;
const int ntau = 40; // リープフロッグのステップ数
const double dtau = 1e0; // リープルフロッグのステップ幅
const int ndim = 3; // 変数の個数


// ガウス乱数生成
int BoxMuller(double &p, double &q) {
    double pi = 2e0 * asin(1e0);
    double r = (double) rand() / RAND_MAX;
    double s = (double) rand() / RAND_MAX;

    p = sqrt(-2e0 * log(r)) * sin(2e0 * pi * s);
    q = sqrt(-2e0 * log(r)) * cos(2e0 * pi * s);

    return 0;
}

// s[x] の計算
double calc_action(const double x[ndim], const double A[ndim][ndim]) {
    double action = 0e0;
    for (int idim = 0; idim < ndim; idim++) {
        for (int jdim = 0; jdim < idim; jdim++) {
            action += x[idim] * A[idim][jdim] * x[jdim];
        }
        action += 0.5e0 * x[idim] * A[idim][idim] * x[idim];
    }
    return action;
}

// ハミルトニアンの計算
double calc_hamiltonian(const double x[ndim], const double p[ndim], const double A[ndim][ndim]) {
    double ham = calc_action(x, A);
    for (int idim = 0; idim < ndim; idim++) {
        ham += 0.5e0 * p[idim] * p[idim];
    }
    return ham;
}

// dh/dxの計算
double calc_delh(const double x[ndim], const double A[ndim][ndim], double (&delh)[ndim]) {
    for (int idim = 0; idim < ndim; idim++) {
        delh[idim] = 0e0;
    }
    for (int idim = 0; idim < ndim; idim++) {
        for (int jdim = 0; jdim < ndim; jdim++) {
            delh[idim] += A[idim][jdim] * x[jdim];
        }
    }
    return 0;
}

// 分子軌道法による時間発展
// リープフロッグで時間発展させる
int Molecular_Dynamics(double (&x)[ndim], const double A[ndim][ndim], double &ham_init,
                       double &ham_fin) {
    double p[ndim];
    double delh[ndim];
    double r1, r2;
    for (int idim = 0; idim < ndim; idim++) {
        BoxMuller(r1, r2);
        p[idim] = r1;
    }
    ham_init = calc_hamiltonian(x, p, A);

    for (int idim = 0; idim < ndim; idim++) {
        x[idim] += p[idim] * 0.5e0 * dtau;
    }

    for (int step = 1; step < ntau; step++) {
        calc_delh(x, A, delh);
        for (int idim = 0; idim < ndim; idim++) {
            p[idim] -= delh[idim] * dtau;
        }
        for (int idim = 0; idim < ndim; idim++) {
            x[idim] += p[idim] * dtau;
        }
    }
    calc_delh(x, A, delh);
    for (int idim = 0; idim < ndim; idim++) {
        p[idim] -= delh[idim] * dtau;
    }
    for (int idim = 0; idim < ndim; idim++) {
        x[idim] += p[idim] * 0.5e0 * dtau;
    }

    ham_fin = calc_hamiltonian(x, p, A);
    return 0;
}


int main() {
    double x[ndim];
    double A[ndim][ndim];

    A[0][0] = 1e0;
    A[1][1] = 2e0;
    A[2][2] = 2e0;
    A[0][1] = 1e0;
    A[0][2] = 1e0;
    A[1][2] = 1e0;
    for (int idim = 1; idim < ndim; idim++) {
        for (int jdim = 0; jdim < idim; jdim++) {
            A[idim][jdim] = A[jdim][idim];
        }
    }
    srand((unsigned) time(NULL));

    for (int idim = 0; idim < ndim; idim++) {
        x[idim] = 0e0;
    }

    std::ofstream outputfile("output.txt");
    int naccept = 0;

    for (int iter = 0; iter < niter; iter++) {
        double backup_x[ndim];
        for (int idim = 0; idim < ndim; idim++) {
            backup_x[idim] = x[idim];
        }
        double ham_init, ham_fin;
        Molecular_Dynamics(x, A, ham_init, ham_fin);
        double metropolis = (double) rand() / RAND_MAX;
        if (exp(ham_init - ham_fin) > metropolis) {
            naccept++;
        } else {
            for (int idim = 0; idim < ndim; idim++) {
                x[idim] = backup_x[idim];
            }
        }
        if ((iter + 1) % 10 == 0) {
            std::cout << std::fixed << std::setprecision(6)
                      << x[0] << "   "
                      << x[1] << "   "
                      << x[2] << "   "
                      << ((double) naccept) / ((double) iter + 1)
                      << std::endl;

            outputfile << std::fixed << std::setprecision(6)
                       << x[0] << "   "
                       << x[1] << "   "
                       << x[2] << "   "
                       << ((double) naccept) / ((double) iter + 1)
                       << std::endl;
        }
    }
    outputfile.close();
    return 0;
}
