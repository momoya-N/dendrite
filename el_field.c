#define _USE_MATH_DEFINES  // windowsのVCだとmath.hを使うのに必要
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 512             // 系の大きさ
#define CEN (int)(N / 2)  // 中心座標
#define DEBUG 3

void Initialize_int(int **data, double a) {  // data , initial value
  int i, j;

  for (i = 0; i < N; i++) {
    for (j = 0; j < N; j++) {
      data[i][j] = a;
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

void DLA(int **data) {        // DLA形状の形成,用意した配列のポインタを渡す
  int const Particl = 15000;  // 粒子数
  int const D = 30;           // 粒子発生位置のフロントラインからの距離
  int const R_C = CEN;        // DLAの棄却領域
  int const RM = CEN;         // DLAの最大成長半径
  double const P = 0.9;       // 粒子の固着確率

  int x0,
      y0;  // 粒子の移動後保存用
  int x,
      y;      // 粒子の位置
  int n = 0;  // 粒子No.
  int dr;     // 粒子発生半径調整用、粒子のフロントライン
  int r;      // 粒子発生半径
  int t;      // ステップ数
  int i,
      j;  // index for "for"

  double th;   // 角度θ
  double tmp;  // 確率保存用

  x = 0;
  y = 0;
  x0 = 0;
  y0 = 0;
  dr = 0;
  r = 0;
  t = 0;

  data[CEN][CEN] = 1;  // 中央に核を置く

  srand((unsigned int)time(NULL));  // 現在時刻の情報でrandの初期化
  printf("Start DLA\n");
  while (n < Particl) {  // 粒子No.n
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
      tmp = p();
      if (tmp < 0.25) {  // 確率1/4で右へ
        x++;
      } else if (0.25 <= tmp && tmp < 0.5) {  // 確率1/4で左へ
        x--;
      } else if (0.5 <= tmp && tmp < 0.75) {  // 確率1/4で上へ
        y++;
      } else if (0.75 <= tmp) {  // 確率1/4で下へ
        y--;
      }

      // 判定//
      if (rr(x, y) >= R_C * R_C) {  // 動いた後の粒子の位置が棄却領域なら...
        break;
      } else if (data[x][y] == 1) {  // 動いた後の位置に粒子がいれば...
        if (p() <= P) {              // 粒子が固着する場合
          data[x0][y0] = 1;          // 補足
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
  printf("Current iteration: %d\n", n);
}

int main(void) {
  const double dif = 1.0e-1;  // 収束判定,前回ループとの差

  double MaxPhi;  // 最大電位
  double MaxErr;  // 最大誤差
  double CurErr;  // 現在の誤差
  double Ex, Ey;  // 電場

  int i, j;
  int loop;  // 計算ループ数

  // phi, rho, Prev_phi, sh_in, sh_out を malloc で 2次元配列として確保
  double **phi = (double **)malloc(sizeof(double *) * N);       // 電位配列
  double **rho = (double **)malloc(sizeof(double *) * N);       // 電荷密度配列
  double **Prev_phi = (double **)malloc(sizeof(double *) * N);  // 現在の電位配列
  // double **errtest = (double **)malloc(sizeof(double *) * N);  // 非対称性計測用
  int **sh_in = (int **)malloc(sizeof(int *) * N);   // 内側形状
  int **sh_out = (int **)malloc(sizeof(int *) * N);  // 外側形状

  for (i = 0; i < N; i++) {
    phi[i] = (double *)malloc(sizeof(double) * N);
    rho[i] = (double *)malloc(sizeof(double) * N);
    Prev_phi[i] = (double *)malloc(sizeof(double) * N);
    // errtest[i] = (double *)malloc(sizeof(double) * N);
    sh_in[i] = (int *)malloc(sizeof(int) * N);
    sh_out[i] = (int *)malloc(sizeof(int) * N);
  }

  // phi, rho, Prev_phi, sh_in, sh_out を初期化
  Initialize_double(phi, 5.0);
  Initialize_double(rho, 0);
  Initialize_double(Prev_phi, 0);
  // Initialize_double(errtest, 0);
  Initialize_int(sh_in, 0);
  Initialize_int(sh_out, 0);

  /*
    for (i = 0; i < N; i++) {
      for (j = 0; j < N; j++) {
        phi[i][j] = 5.0;
        rho[i][j] = 0;
        Prev_phi[i][j] = 0;
        sh_in[i][j] = 0;
        sh_out[i][j] = 0;
      }
    }
  */

  FILE *f;  // ファイルハンドラ

  printf("Initializing...\n");

  /*極板の形状決定*/
  // DLA(sh_in);  // 陽極(内側)形状,1が電極の存在する場所,DLA形状
  circle(sh_in, CEN * 0.05);  // 陽極(内側)形状,1が電極の存在する場所,円形
  circle(sh_out, CEN - 1);    // 陰極(外側)形状,半径はCEN-1にしないとi+1などの計算でセグフォになる。
  printf("Start Caliculation\n");

  /*電位配置*/
  for (i = 0; i < N; i++) {
    for (j = 0; j < N; j++) {
      if (sh_in[i][j] == 1) {
        phi[i][j] = 0.0;
      }
    }
  }

  time_t start_time, end_time;
  clock_t start_clock, end_clock;

  /*処理開始時の経過時間とCPU時間を取得*/
  start_time = time(NULL);
  start_clock = clock();

  /*繰り返し計算*/
  loop = 0;

  MaxPhi = 5.0;  // 系内の最大電位、0除算の防止用のため有限値を入れる

#if DEBUG == 0  // 一方向に値を計算、更新前と更新後の混在データによる電位計算
  double Phi_tmp = 0;

  do {
    MaxErr = CurErr = 0.0;

    for (i = 0; i < N; i++) {
      for (j = 0; j < N; j++) {
        if (sh_out[i][j] == 1 && sh_in[i][j] == 0) {  // 電極でない場所(計算したい領域)ならば

          Phi_tmp = phi[i][j];                                                                 // 前回ループのPhi
          phi[i][j] = 0.25 * (phi[i + 1][j] + phi[i - 1][j] + phi[i][j + 1] + phi[i][j - 1]);  // 電極なので電荷密度rho=0なので省略した。

          if (MaxPhi < fabs(phi[i][j])) {
            MaxPhi = phi[i][j];  // 最大電位の更新
          }

          CurErr = (fabs(phi[i][j] - Phi_tmp)) / MaxPhi;  // 前回ループとの差を電位の最大値で規格化

          if (MaxErr < CurErr) {
            MaxErr = CurErr;  // 誤差の最大値の更新
          }
        }
      }
    }
    loop++;

    if (loop % 1000 == 0) {
      printf("loop: %d, MaxErr: %f\n", loop, MaxErr);
    }
  } while (MaxErr > dif);  // 系内全ての前回ループとの誤差が規定値以下になったら終了

#elif DEBUG == 1  // 一方向に計算、loopごとに計算方向を逆方向にする
  double Phi_tmp = 0;

  do {
    MaxErr = CurErr = 0.0;

    if (loop % 2 == 0) {  // 偶数回目のループならば
      for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
          if (sh_out[i][j] == 1 && sh_in[i][j] == 0) {  // 電極でない場所(計算したい領域)ならば

            Phi_tmp = phi[i][j];                                                                 // 前回ループのPhi
            phi[i][j] = 0.25 * (phi[i + 1][j] + phi[i - 1][j] + phi[i][j + 1] + phi[i][j - 1]);  // 電極なので電荷密度rho=0なので省略した。

            if (MaxPhi < fabs(phi[i][j])) {
              MaxPhi = phi[i][j];  // 最大電位の更新
            }

            CurErr = (fabs(phi[i][j] - Phi_tmp)) / MaxPhi;  // 前回ループとの差を電位の最大値で規格化

            if (MaxErr < CurErr) {
              MaxErr = CurErr;  // 誤差の最大値の更新
            }
          }
        }
      }
    } else {  // 奇数回目のループならば
      for (i = N - 1; i > 0; i--) {
        for (j = N - 1; j > 0; j--) {
          if (sh_out[i][j] == 1 && sh_in[i][j] == 0) {  // 電極でない場所(計算したい領域)ならば

            Phi_tmp = phi[i][j];                                                                 // 前回ループのPhi
            phi[i][j] = 0.25 * (phi[i + 1][j] + phi[i - 1][j] + phi[i][j + 1] + phi[i][j - 1]);  // 電極なので電荷密度rho=0なので省略した。

            if (MaxPhi < fabs(phi[i][j])) {
              MaxPhi = phi[i][j];  // 最大電位の更新
            }

            CurErr = (fabs(phi[i][j] - Phi_tmp)) / MaxPhi;  // 前回ループとの差を電位の最大値で規格化

            if (MaxErr < CurErr) {
              MaxErr = CurErr;  // 誤差の最大値の更新
            }
          }
        }
      }
    }

    loop++;

    if (loop % 1000 == 0) {
      printf("loop: %d, MaxErr: %f\n", loop, MaxErr);
    }
  } while (MaxErr > dif);  // 系内全ての前回ループとの誤差が規定値以下になったら終了

#elif DEBUG == 2  // 一方向に計算、前回ループの値のみで計算
  do {
    MaxErr = CurErr = 0.0;

    for (i = 0; i < N; i++) {  // 前回ループの電位を保存
      for (j = 0; j < N; j++) {
        Prev_phi[i][j] = phi[i][j];
      }
    }

    for (i = 0; i < N; i++) {
      for (j = 0; j < N; j++) {
        if (sh_out[i][j] == 1 && sh_in[i][j] == 0) {  // 電極でない場所(計算したい領域)ならば

          phi[i][j] = 0.25 * (Prev_phi[i + 1][j] + Prev_phi[i - 1][j] + Prev_phi[i][j + 1] + Prev_phi[i][j - 1]);  // 電極なので電荷密度rho=0なので省略した。
          if (MaxPhi < fabs(phi[i][j])) {
            MaxPhi = phi[i][j];  // 最大電位の更新
          }
          CurErr = (fabs(phi[i][j] - Prev_phi[i][j])) / MaxPhi;  // 前回ループとの差を電位の最大値で規格化
          if (MaxErr < CurErr) {
            MaxErr = CurErr;  // 誤差の最大値の更新
          }
        }
      }
    }
    loop++;
    if (loop % 1000 == 0) {
      printf("loop: %d, MaxErr: %f\n", loop, MaxErr);
    }
  } while (MaxErr > dif);  // 系内全ての前回ループとの誤差が規定値以下になったら終了

#elif DEBUG == 3  // SOR(Successive Over Relaxation)かつ計算順の入れ替え
  double Phi_tmp = 0;
  double r = 2 / (1 + M_PI / N);  // relaxation parameter

  do {
    MaxErr = CurErr = 0.0;

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
              MaxErr = CurErr;  // 誤差の最大値の更新
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
              MaxErr = CurErr;  // 誤差の最大値の更新
            }
          }
        }
      }
    }

    loop++;

    if (loop % 1000 == 0) {
      printf("loop: %d, MaxErr: %f\n", loop, MaxErr);
    }
  } while (MaxErr > dif);  // 系内全ての前回ループとの誤差が規定値以下になったら終了

#endif

  /*処理終了後の経過時間とCPU時間の取得*/
  end_time = time(NULL);
  end_clock = clock();

  /*経過時間の表示*/
  printf("real_time:%ld\tCPU_time:%lu\n", end_time - start_time, (end_clock - start_clock) / CLOCKS_PER_SEC);

  /*電位の出力*/
  f = fopen("Phi.dat", "w");
  for (i = 0; i < N; i++) {
    for (j = 0; j < N; j++) {
      fprintf(f, "%d %d %e\n", i, j, phi[i][j]);
    }
  }
  fclose(f);

  /*電場出力*/
  f = fopen("El.dat", "w");
  for (i = 1; i < N - 1; i++) {
    for (j = 1; j < N - 1; j++) {
      Ex = -(phi[i + 1][j] - phi[i - 1][j]) / 2;
      Ey = -(phi[i][j + 1] - phi[i][j - 1]) / 2;

      fprintf(f, "%d %d %e %e %e\n", i, j, sqrt(Ex * Ex + Ey * Ey), Ex, Ey);
    }
  }
  fclose(f);

  /*極の形状出力*/
  f = fopen("Shape.dat", "w");
  for (i = 0; i < N; i++) {
    for (j = 0; j < N; j++) {
      fprintf(f, "%d ", sh_in[i][j]);
    }
    fprintf(f, "\n");
  }
  fclose(f);

  // /*対称性とのずれの出力*/
  // f = fopen("errtest.dat", "w");
  // double err = 0;

  // for (i = 0; i < N; i++) {
  //   for (j = 0; j < N; j++) {
  //     err = fabs(phi[i][j] - phi[N - 1 - i][N - 1 - j]);
  //     fprintf(f, "%e ", err);
  //   }
  //   fprintf(f, "\n");
  // }
  // fclose(f);

  printf("loop:%d\n", loop);

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

  return 0;
}
