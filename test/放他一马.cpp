#include <bits/stdc++.h>
#define INVALID -1e18
using namespace std;
typedef long long ll;
int main() {
    int n;
    cin>>n;
    vector<ll> nums(n);
    for(int i=0;i<n;i++) cin>>nums[i];
    
    vector<vector<ll>> dp(n+5,vector<ll>(10,INVALID));
    dp[0][0]=1;
    dp[0][1]=nums[0]*2;
    //dp[i][j]到位置i （0-base）时击杀j%10只怪物所获经验值
    for(int i=1;i<n;i++){
    	for(int j=0;j<=min(i+1,10);j++){
        	dp[i][j]=max(dp[i][j],dp[i-1][j]+1+i);
            //之前的状态，之前击杀的怪物数量应当为(j - 1 + 10) % 10，比如当前击杀了
            dp[i][j]=max(dp[i][j],dp[i-1][(j - 1 + 10) % 10] + nums[i] + j*nums[i]);
        }
    }
    cout<<*max_element(dp[n-1].begin(),dp[n-1].end());
    return 0;
}
// 64 位输出请用 printf("%lld")