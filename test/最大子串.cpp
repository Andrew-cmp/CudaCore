#include <bits/stdc++.h>
using namespace std;

int main(){
  int T;  cin >> T;
  while(T --){   
    vector<long long> nums(26);
    long long mx = 0 , sum = 0;
    for (int i = 0; i < 26; ++i) {
       cin >> nums[i];
       mx = max(mx , nums[i]);
       sum += nums[i];
    }
    cout << (long long)min(sum , 2 * (sum - mx) + 1) << endl;
  }       
  return 0;
}