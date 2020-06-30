/**
 * Created by hiramekun at 01:27 on 2020-07-01.
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

int main(void) {
    int niter = 100; // サンプルの数
    double step_size = 0.5e0;
    srand((unsigned) time(nullptr));

    double x = 0e0;
    int naccept = 0; // 受理回数カウンター

    for (int iter = 1; iter < niter + 1; iter++) {
        double backup_x = x;
        double action_init = 0.5e0 * x * x;

        double dx = (double) rand() / RAND_MAX;
        dx = (dx - 0.5e0) * step_size * 2e0;
        x += dx;

        double action_fin = 0.5e0 * x * x;

        // メトロポリステスト
        double metropolis = (double) rand() / RAND_MAX;
        if (exp(action_init - action_fin) > metropolis) {
            // 受理
            naccept++;
        } else {
            // 棄却して元に戻す
            x = backup_x;
        }
        printf("%.10f\t%f\n", x, (double) naccept / iter);
    }
}
