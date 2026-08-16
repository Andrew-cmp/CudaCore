

void 

int main(){

    int num[1000000];
    for(int i = 0;i < 1000000;i++){
        num[i] = i;
    }
    int output[1000000];
    memset(output,0,sizeof(int)*1000000);
    prefix_sum(num,output,1000000);

}