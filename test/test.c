#include "SFMT.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <time.h>

// /*FFTテスト用*/
// int main(void) {
//   const int n = 512;
//   const double delta = 5.0 / n;
//   double x;

//   FILE *f;
//   char fname[100];

//   sprintf(fname, "./FFT_test_1d.dat");

//   f = fopen(fname, "w");
//   for (int i = 0; i < n; i++) {
//     x = (i - n / 2) * delta;
//     fprintf(f, "%f\n", exp(-x * x));
//   }
//   fclose(f);
// }

/*メルセンヌツイスタテスト用*/
int main(int argc, char *argv[]) {
  /* 状態を保持する構造体 */
  sfmt_t sfmt;

  /* シードを指定して初期化 */
  int seed = 0;
  sfmt_init_gen_rand(&sfmt, seed);

  /* 32bit整数を生成する場合： */
  uint32_t r_int_32 = sfmt_genrand_uint32(&sfmt);

  /* 64bit整数を生成する場合（ただし64bit環境のみ）： */
  uint64_t r_int_64 = sfmt_genrand_uint64(&sfmt);

  /* 0以上1未満の実数を生成する場合： */
  double r_real = sfmt_genrand_real2(&sfmt);

  /* 0以上1未満の52bit精度実数を生成する場合： */

  /* 1. 64bit環境ではこちら */
  uint64_t v = sfmt_genrand_uint64(&sfmt);
  double r_real_52 = sfmt_to_res53(v);

  //   /* 2. 32bit環境ではこちら */
  //   uint32_t v1 = sfmt_genrand_uint32(&sfmt);
  //   uint32_t v2 = sfmt_genrand_uint32(&sfmt);
  //   double r_real_52 = sfmt_to_res53_mix(v1, v2);

  printf("%ld", r_int_64);

  return 0;
}