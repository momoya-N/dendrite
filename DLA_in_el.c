#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <time.h>

#define N 512             // 系の大きさ
#define CEN (int)(N / 2)  // 中心座標

void Initialize_int(int **data, int a) {  // data , initial value
  int i, j;

  for (i = 0; i < N; i++) {
    for (j = 0; j < N; j++) {
      data[i][j] = a;
    }
  }
}

void Initialize_each_step_duble(double **data1, double a, int **data2) {  // data1(Phi) , initial value , data2(Shape)
  int i, j;

  for (i = 0; i < N; i++) {
    for (j = 0; j < N; j++) {
      if (data2[i][j] == 1) {
        data1[i][j] = a;
      }
    }
  }
}

void Initialize_double(double **data, double a) {  // data , initial value
  int i, j;

  for (i = 0; i < N; i++) {
    for (j = 0; j < N; j++) {
      data[i][j] = a;
    }
  }
}

int step(double x) {  // Heviside's step function
  int r = 0;          // return

  if (x <= 0) {
    r = 0;
  } else if (x > 0) {
    r = 1;
  }

  return r;
}

double rr(int i, int j) {  // 中心からの距離の2乗
  double rr;

  rr = (CEN - i) * (CEN - i) + (CEN - j) * (CEN - j);

  return rr;
}

void circle(int **data, double r) {  // 円形領域の形成,用意した配列のポインタと半径を渡す
  int i, j;

  for (i = 0; i < N; i++) {
    for (j = 0; j < N; j++) {
      if (rr(i, j) < r * r) {
        data[i][j] = 1;
      }
    }
  }
}

double p(void) {  // 0〜1の乱数発生

  double rn;
  rn = rand() / (double)RAND_MAX;

  return rn;
}

void DLA(int **data1, int Particle, double ***data2, double alpha, double C, int *n_p1, int *n_p2, int *n_p3, int *n_p4,
         int *n_q) {  // DLA,形状の配列ポインタ(data1),(Particle)粒子分成長後終了,各点での移動確立の異方性(data2),異方性の影響の大きさ(alpha),分散の大きさ(C),各粒子が0-1を超えた回数
  int const D = 30;   // 粒子発生位置のフロントラインからの距離
  int const R_C = CEN;   // DLAの棄却領域
  int const RM = CEN;    // DLAの最大成長半径
  double const P = 1.0;  // 粒子の固着確率
  // double const C = 3.0 / 16;                                          // variance (constant),2*C

  int x0, y0;  // 粒子の移動後保存用
  int x, y;    // 粒子の位置
  int n = 0;   // 粒子No.
  int dr = 0;  // 粒子発生半径調整用、粒子のフロントライン
  int r = 0;   // 粒子発生半径
  int t = 0;   // ステップ数
  int i, j;    // index for "for"

  double th;                 // 角度θ
  double tmp;                // 確率保存用
  double p1, p2, p3, p4, q;  // step probability

  x = 0;
  y = 0;
  x0 = 0;
  y0 = 0;

  for (i = 0; i < N; i++) {  // 粒子のフロントラインの探索(初期値決定)
    for (j = 0; j < N; j++) {
      if (data1[i][j] == 1 && rr(i, j) > (dr * dr)) {
        dr = sqrt(rr(i, j));
      }
    }
  }

  srand((unsigned int)time(NULL));  // 現在時刻の情報でrandの初期化
  // printf("Start DLA\n");
  while (n < Particle) {  // 粒子No.n
    r = D + dr;
    x0 = 0;
    y0 = 0;

    if (r > RM) {
      r = RM;
    }

    /*粒子発生*/
    th = 2.0 * M_PI * p();         // 角度θ、0から2πの範囲
    x = (int)(CEN + r * cos(th));  // 粒子の発生位置、半径rの円周上
    y = (int)(CEN + r * sin(th));

    while (1) {
      t++;

      // //printf("{%d,%d}\n", x, y);
      tmp = p();
      // x,y方向の推移確率&その場に留まる確率
      p1 = 2 * alpha * alpha * data2[x][y][0] * data2[x][y][0] + alpha * data2[x][y][0] + C;
      p3 = 2 * alpha * alpha * data2[x][y][0] * data2[x][y][0] - alpha * data2[x][y][0] + C;
      p2 = 2 * alpha * alpha * data2[x][y][1] * data2[x][y][1] - alpha * data2[x][y][1] + C;
      p4 = 2 * alpha * alpha * data2[x][y][1] * data2[x][y][1] + alpha * data2[x][y][1] + C;
      q = 1 - 4 * C - 4 * alpha * alpha * (data2[x][y][0] * data2[x][y][0] + data2[x][y][1] * data2[x][y][1]);

      // //printf("q=%f\n", q);
      // 確率が0,1に入っているかの判定→入ってなかった場合：終了、回数の計測、計算のし直し、アルファとともに分散も変えることで、電場が強くなっても一定範囲内かつ同じ物理状態にできそう。

      if (p1 < 0.0 || 1.0 < p1) {  // 範囲外の確率になった回数を記録
        (*n_p1)++;
      }
      if (p2 < 0.0 || 1.0 < p2) {
        (*n_p2)++;
      }
      if (p3 < 0.0 || 1.0 < p3) {
        (*n_p3)++;
      }
      if (p4 < 0.0 || 1.0 < p4) {
        (*n_p4)++;
      }
      if (q < 0.0 || 1.0 < q) {
        (*n_q)++;
      }

      if (p1 > 1.0) {  // alphaが大きく、確率が1を超えている場合(特にp1)、この処理を入れないと、n_p1がどんどん大きくなる→どこにもくっつかない？なぜp1が大きいのか？アルファによっては収束しない？
        x++;
      } else if (p2 > 1.0) {
        y--;
      } else if (p3 > 1.0) {
        x--;
      } else if (p4 > 1.0) {
        y++;
      } else {
        if (tmp < q) {              // その場に留まる
        } else if (tmp < q + p1) {  // xに+1
          x++;
        } else if (tmp < q + p1 + p2) {  // yに-1
          y--;
        } else if (tmp < q + p1 + p2 + p3) {  // xに-1
          x--;
        } else if (tmp < q + p1 + p2 + p3 + p4) {  // yに+1
          y++;
        }
      }

      //  判定//
      if (rr(x, y) >= R_C * R_C) {  // 動いた後の粒子の位置が棄却領域なら...
        break;
      } else if (data1[x][y] == 1) {  // 動いた後の位置に粒子がいれば...
        if (p() <= P) {               // 粒子が固着する場合
          data1[x0][y0] = 1;          // 補足
          n++;
          if (rr(x0, y0) > (dr * dr)) {  // 粒子発生位置のフロントライン調整
            dr = sqrt(rr(x0, y0));
          }
          break;
        } else {   // 粒子が固着しない場合
          x = x0;  // 固着直前の座標に戻る
          y = y0;
        }
      } else {
        x0 = x;  // 移動後の粒子の座標を保存、yも同様
        y0 = y;
      }
    }
  }
  // printf("Current iteration: %d\n", n);
  // printf("Current r: %d\n", r);
}

int main(int argc, char *argv[]) {
  if (argc != 2) {
    printf("./DLA alpha\n");
    return 1;
  }

  /*処理開始時の経過時間とCPU時間を取得*/
  time_t start_time, end_time;
  clock_t start_clock, end_clock;
  start_time = time(NULL);
  start_clock = clock();

  /*DLA関係*/
  const int dla_n = 15000;       // DLAの総粒子数
  const int dla_step = 150;      // DLA形状取得のステップ数,(dla_step)粒子ごとにDLA取得、電位計算
  const double C = 1.0 / 16;     // 分散の大きさ
  double alpha = atof(argv[1]);  // RWの電場による異方性の大きさ

  int i, j, k;

  int n;                                                // DLA粒子数計測用
  int n_p1 = 0, n_p2 = 0, n_p3 = 0, n_p4 = 0, n_q = 0;  // countes for range over probability

  /*電場計算関係*/
  const double dif = 1.0e-5;  // 収束判定,前回ループとの差

  double MaxPhi = 1.0;            // 最大電位
  double MaxErr;                  // 最大誤差
  double CurErr;                  // 現在の誤差
  double Phi_tmp;                 // 電位一時保管用
  double E_max = 0;               // 電場の最大値
  double r = 2 / (1 + M_PI / N);  // relaxation parameter
  double Ex, Ey;                  // 電場

  int loop;  // 計算ループ数

  // El_fieldを３次元配列として確保
  double ***El_field = (double ***)malloc(sizeof(double **) * N);  // 電場配列
  for (i = 0; i < N; i++) {
    El_field[i] = (double **)malloc(sizeof(double *) * N);
    for (j = 0; j < N; j++) {
      El_field[i][j] = (double *)malloc(sizeof(double) * 2);
    }
  }

  // phi, rho, Prev_phi, sh_in, sh_out を malloc で 2次元配列として確保
  double **phi = (double **)malloc(sizeof(double *) * N);       // 電位配列
  double **rho = (double **)malloc(sizeof(double *) * N);       // 電荷密度配列
  double **Prev_phi = (double **)malloc(sizeof(double *) * N);  // 現在の電位配列
  int **sh_in = (int **)malloc(sizeof(int *) * N);              // 内側形状
  int **sh_out = (int **)malloc(sizeof(int *) * N);             // 外側形状
  for (i = 0; i < N; i++) {
    phi[i] = (double *)malloc(sizeof(double) * N);
    rho[i] = (double *)malloc(sizeof(double) * N);
    Prev_phi[i] = (double *)malloc(sizeof(double) * N);
    sh_in[i] = (int *)malloc(sizeof(int) * N);
    sh_out[i] = (int *)malloc(sizeof(int) * N);
  }

  // phi, rho, Prev_phi, sh_in, sh_out El_fieldを初期化
  Initialize_double(phi, MaxPhi);
  Initialize_double(rho, 0);
  Initialize_double(Prev_phi, 0);
  Initialize_int(sh_in, 0);
  Initialize_int(sh_out, 0);

  // El_fieldの初期化
  for (i = 0; i < N; i++) {
    for (j = 0; j < N; j++) {
      for (k = 0; k < 2; k++) {
        El_field[i][j][k] = 0;
      }
    }
  }

  FILE *f;  // ファイルハンドラ

  ////printf("Initializing...\n");

  /*極板の形状決定*/

  sh_in[CEN][CEN] = 1;      // 中央に核を置く
  circle(sh_out, CEN - 1);  // 陰極(外側)形状,半径はCEN-1にしないとi+1などの計算でセグフォになる。

  /*繰り返し計算*/
  loop = 0;
  n = 0;
  // MaxErr = 0.0;

  // MaxPhi = 5.0;  // 系内の最大電位、0除算の防止用のため有限値を入れる

  for (k = 0; k < (int)(dla_n / dla_step); k++) {
    DLA(sh_in, dla_step, El_field, alpha, C, &n_p1, &n_p2, &n_p3, &n_p4, &n_q);  // DLAの計算
    // printf("%d\t%d\t%d\t%d\t%d\n", n_p1, n_p2, n_p3, n_p4, n_q);
    n = (k + 1) * dla_step;  // DLAの現在の総粒子数

    Initialize_each_step_duble(phi, 0.0, sh_in);  // 極板形状をDLAにして、極板の電位を0にする

    // printf("current particle:%d\n", n);
    do {
      MaxErr = 0.0;
      CurErr = 0.0;

      if (loop % 2 == 0) {  // 偶数回目のループならば
        for (i = 0; i < N; i++) {
          for (j = 0; j < N; j++) {
            if (sh_out[i][j] == 1 && sh_in[i][j] == 0) {  // 電極でない場所(計算したい領域)ならば

              Phi_tmp = phi[i][j];                                                                                             // 前回ループのPhi
              phi[i][j] = (1 - r) * phi[i][j] + r * (0.25 * (phi[i + 1][j] + phi[i - 1][j] + phi[i][j + 1] + phi[i][j - 1]));  // 電極なので電荷密度rho=0なので省略した。
              // if (MaxPhi < fabs(phi[i][j])) {
              //   MaxPhi = phi[i][j];  // 最大電位の更新
              // }

              CurErr = (fabs(phi[i][j] - Phi_tmp)) / MaxPhi;  // 前回ループとの差を電位の最大値で規格化

              if (MaxErr < CurErr) {
                MaxErr = CurErr;  // 系内の誤差の最大値の更新
              }
            }
          }
        }
      } else {  // 奇数回目のループならば
        for (i = N - 1; i > 0; i--) {
          for (j = N - 1; j > 0; j--) {
            if (sh_out[i][j] == 1 && sh_in[i][j] == 0) {  // 電極でない場所(計算したい領域)ならば

              Phi_tmp = phi[i][j];                                                                                             // 前回ループのPhi
              phi[i][j] = (1 - r) * phi[i][j] + r * (0.25 * (phi[i + 1][j] + phi[i - 1][j] + phi[i][j + 1] + phi[i][j - 1]));  // 電極なので電荷密度rho=0なので省略した。

              // if (MaxPhi < fabs(phi[i][j])) {
              //   MaxPhi = phi[i][j];  // 最大電位の更新
              // }

              CurErr = (fabs(phi[i][j] - Phi_tmp)) / MaxPhi;  // 前回ループとの差を電位の最大値で規格化

              if (MaxErr < CurErr) {
                MaxErr = CurErr;  // 系内の誤差の最大値の更新
              }
            }
          }
        }
      }

      loop++;
      if (loop % 1000 == 0) {
        // printf("loop: %d,  MaxErr: %f\n", loop, MaxErr);
      }
    } while (MaxErr > dif);  // 系内全ての前回ループとの誤差が規定値以下になったら終了

    /*電場計算*/
    for (i = 1; i < N - 1; i++) {
      for (j = 1; j < N - 1; j++) {
        Ex = -(phi[i + 1][j] - phi[i - 1][j]) / 2;
        Ey = -(phi[i][j + 1] - phi[i][j - 1]) / 2;

        El_field[i][j][0] = Ex;  // 電場の保存
        El_field[i][j][1] = Ey;

        if (E_max < sqrt(Ex * Ex + Ey * Ey)) {
          E_max = sqrt(Ex * Ex + Ey * Ey);
        }
      }
    }

    // printf("Particle: %d\n", n);
  }

  // printf("finish calculation\tEmax:%f\n", E_max);

  /*電位の出力*/
  char dirname[100];
  char fname[100];

  sprintf(dirname, "./data/C=%f", C);
  mkdir(dirname, 0777);
  sprintf(dirname, "./data/C=%f/Phi_data", C);
  mkdir(dirname, 0777);
  sprintf(fname, "./data/C=%f/Phi_data/Phi_alpha=%f.dat", C, alpha);  // ディレクトリ、ファイル作成

  f = fopen(fname, "w");
  for (i = 0; i < N; i++) {
    for (j = 0; j < N; j++) {
      fprintf(f, "%d %d %e\n", i, j, phi[i][j]);
    }
  }
  fclose(f);

  /*電場出力、最終のEl_field*/
  sprintf(dirname, "./data/C=%f/El_data", C);
  mkdir(dirname, 0777);
  sprintf(fname, "./data/C=%f/El_data/El_alpha=%f.dat", C, alpha);  // ディレクトリ作成

  f = fopen(fname, "w");
  for (i = 1; i < N - 1; i++) {
    for (j = 1; j < N - 1; j++) {
      Ex = El_field[i][j][0];
      Ey = El_field[i][j][1];
      fprintf(f, "%d %d %e %e %e\n", i, j, sqrt(Ex * Ex + Ey * Ey), Ex, Ey);
    }
  }
  fclose(f);

  /*形状出力*/
  sprintf(dirname, "./data/C=%f/DLA_data", C);
  mkdir(dirname, 0777);
  sprintf(fname, "./data/C=%f/DLA_data/DLA_alpha=%f.dat", C, alpha);

  f = fopen(fname, "w");
  for (i = 0; i < N; i++) {
    for (j = 0; j < N; j++) {
      fprintf(f, "%d ", sh_in[i][j]);
    }
    fprintf(f, "\n");
  }
  fclose(f);

  /*その他データ出力,読み込みはmatplotlib用(コメントアウト文字が#)*/
  sprintf(dirname, "./data/C=%f/other_data", C);
  mkdir(dirname, 0777);
  sprintf(fname, "./data/C=%f/other_data/other_alpha=%f.dat", C, alpha);

  f = fopen(fname, "w");
  fprintf(f, "#alpha\tn_p1\tn_p2\tn_p3\tn_p4\tn_q\n");
  fprintf(f, "%f\t%d\t%d\t%d\t%d\t%d\n", alpha, n_p1, n_p2, n_p3, n_p4, n_q);
  fclose(f);

  // printf("final loop:%d\n", loop);

  // メモリの解放
  for (i = 0; i < N; i++) {
    free(phi[i]);
    free(rho[i]);
    free(Prev_phi[i]);
    free(sh_in[i]);
    free(sh_out[i]);
  }
  free(phi);
  free(rho);
  free(Prev_phi);
  free(sh_in);
  free(sh_out);

  for (i = 0; i < N; i++) {
    for (j = 0; j < N; j++) {
      free(El_field[i][j]);
    }
    free(El_field[i]);
  }

  free(El_field);

  /*処理終了後の経過時間とCPU時間の取得*/
  end_time = time(NULL);
  end_clock = clock();

  /*経過時間の表示*/
  printf("real_time:%ld\tCPU_time:%lu\n", end_time - start_time, (end_clock - start_clock) / CLOCKS_PER_SEC);

  return 0;
}
