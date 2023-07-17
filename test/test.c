#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <time.h>

// int main(void) {
//   char dirname[50];
//   const double C = 1.0 / 8;
//   const double alpha = 1.0;
//   char fname[50];
//   FILE *f;

#define R 128                // 枠の大きさ
#define D 30                 // 粒子の発生位置のフロントラインからの距離
#define N 1500               // 粒子数
#define P 1.00               // 粒子の固着確率
#define CEN (int)(R / 2)     // 中心座標
#define R_C (int)(R / 2)     // 粒子の棄却半径
#define RM (int)(R / 2) - 3  // フラクタルの成長限界半径

int rr(int x, int y) {  // 中心からの距離の2乗
  int rr;

  rr = (x - CEN) * (x - CEN) + (y - CEN) * (y - CEN);

  return rr;
}

double p(void) {  // 0〜1の乱数発生

  double rn;
  rn = rand() / (double)RAND_MAX;

  return rn;
}

int flag(int x, int y) {  // ある座標が配列内にあるかどうか調べる
  int flag = 0;
  if (0 <= x && x < R && 0 <= y && y < R) {
    flag = 1;
  }

  return flag;
}

// メイン//
// int main(void) {
//   int x0, y0;        // 粒子の移動後保存用
//   int x, y;          // 粒子の位置
//   int s[R][R] = {};  // 状態1なら固着,配列の大きさは中心から上下左右にR,{}は全ての要素を0で初期化
//   int n = 0;         // 粒子No.
//   double th;         // 角度θ
//   double tmp;        // 確率保存用
//   int dr;            // 粒子発生半径調整用、粒子のフロントライン
//   int r;             // 粒子発生半径
//   int t;             // ステップ数
//   int i, j;

//   x = 0;
//   y = 0;
//   x0 = 0;
//   y0 = 0;
//   dr = 0;
//   r = 0;
//   t = 0;

//   s[CEN][CEN] = 1;  // 中央に核を置く

//   srand((unsigned int)time(NULL));  // 現在時刻の情報でrandの初期化

//   while (n < N) {  // 粒子No.n
//     r = D + dr;
//     x0 = 0;
//     y0 = 0;

//     if (r > RM) {
//       r = RM;
//     }

//     /*粒子発生*/
//     th = 2.0 * M_PI * p();         // 角度θ、0から2πの範囲
//     x = (int)(CEN + r * cos(th));  // 粒子の発生位置、半径rの円周上
//     y = (int)(CEN + r * sin(th));

//     while (1) {
//       t++;
//       tmp = p();
//       if (tmp < 0.25) {  // 確率1/4で右へ
//         x++;
//       } else if (0.25 <= tmp && tmp < 0.5) {  // 確率1/4で左へ
//         x--;
//       } else if (0.5 <= tmp && tmp < 0.75) {  // 確率1/4で上へ
//         y++;
//       } else if (0.75 <= tmp) {  // 確率1/4で下へ
//         y--;
//       }
//       // printf("[x,y]=[%d,%d]\n", x, y);
//       //  判定//
//       if (rr(x, y) >= R_C * R_C) {  // 動いた後の粒子の位置が棄却領域なら...
//         break;
//       } else if (s[x][y] == 1) {  // 動いた後の位置に粒子がいれば...
//         if (p() <= P) {           // 粒子が固着する場合
//           s[x0][y0] = 1;          // 補足
//           n++;
//           if (rr(x0, y0) > (dr * dr)) {  // 粒子発生位置のフロントライン調整
//             dr = sqrt(rr(x0, y0));
//           }
//           break;
//         } else {   // 粒子が固着しない場合
//           x = x0;  // 固着直前の座標に戻る
//           y = y0;
//         }
//       } else {
//         x0 = x;  // 移動後の粒子の座標を保存、yも同様
//         y0 = y;
//       }
//     }
//     // if (n % 150 == 0) {
//     //   printf("Particle:%d\n", n);
//     // }
//   }

//   FILE *f;
//   char fname[100];

//   sprintf(fname, "./test_DLA_1d.dat");
//   sprintf(fname, "./test_DLA.dat");

//   f = fopen(fname, "w");
//   for (i = 0; i < R; i++) {
//     for (j = 0; j < R; j++) {
//       fprintf(f, "%d\n", s[i][j]);
//     }
//   }
//   fclose(f);

//   f = fopen(fname, "w");
//   for (i = 0; i < R; i++) {
//     for (j = 0; j < R; j++) {
//       fprintf(f, "%d\t", s[i][j]);
//     }
//     fprintf(f, "\n");
//   }
//   fclose(f);

//   return 0;
// }

/*FFTテスト用*/
int main(void) {
  const int n = 512;
  const double delta = 5.0 / n;
  double x;

  FILE *f;
  char fname[100];

  sprintf(fname, "./FFT_test_1d.dat");

  f = fopen(fname, "w");
  for (int i = 0; i < n; i++) {
    x = (i - n / 2) * delta;
    fprintf(f, "%f\n", exp(-x * x));
  }
  fclose(f);
}