/**
 * Created by hiramekun at 01:27 on 2020-07-01.
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

int main(void) {
    int niter = 10000; // サンプルの数
    double step_size_x = 0.5e0;
    double step_size_y = 0.5e0;

    srand((unsigned) time(nullptr));

    double x = 0e0;
    double y = 0e0;
    int naccept = 0; // 受理回数カウンター

    for (int iter = 1; iter < niter + 1; iter++) {
        double backup_x = x;
        double backup_y = y;
        double action_init = 0.5e0 * (x * x + y * y + x * y);

        double dx = (double) rand() / RAND_MAX;
        double dy = (double) rand() / RAND_MAX;
        dx = (dx - 0.5e0) * step_size_x * 2e0;
        dy = (dy - 0.5e0) * step_size_y * 2e0;
        x += dx;
        y += dy;

        double action_fin = 0.5e0 * (x * x + y * y + x * y);

        // メトロポリステスト
        double metropolis = (double) rand() / RAND_MAX;
        if (exp(action_init - action_fin) > metropolis) {
            // 受理
            naccept++;
        } else {
            // 棄却して元に戻す
            x = backup_x;
            y = backup_y;
        }
        if (iter % 10 == 0) {
            printf("%.10f\t%.10f\t%f\n", x, y, (double) naccept / iter);
        }
    }
}