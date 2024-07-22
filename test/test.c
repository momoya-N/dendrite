#include <stdio.h>

void sum(int a,int b, int *c);

int main(void){
    int a=1,b=2;
    int d=0;
    int *c=&d;
    
    // c=&d;
    // *c=10;
    printf("c = %p\n",c);
    printf("c = %d\n",*c);
    sum(a,b,c);
    printf("c = %d\n",*c);
    printf("%d + %d = %d\n",a,b,*c);

    return 0;
}

void sum(int a,int b, int *c){
    printf("c = %d\n",*c);
    *c = a+b;
    printf("c = %d\n",*c);
}