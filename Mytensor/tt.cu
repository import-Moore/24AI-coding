#include<iostream>
int main(){
    int res = 0;
    for (int i = 1; i <= 1 << 6;++i)
        for (int j = 1; j <= i;j*=2)
            res++;
    printf("%d", res);
}