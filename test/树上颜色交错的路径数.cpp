#include <iostream>
#include <bits/stdc++.h>
using namespace std;
long long ans = 0;
//以当前节点为起点，向下延伸，并且颜色交替的路径数量。
int dfs(int node , const vector<vector<int>> &tree, const string &s, int n, vector<bool> &visit) {
    int node_paths = 1;
    if(visit[node])return 0;
    visit[node]=1;
    if(tree[node].empty()){
        return node_paths;
    }
    vector<int> child_cnt;
    for(int i = 0;i < tree[node].size();i++){
        if(visit[tree[node][i]])continue;
        int child_path_count = dfs(tree[node][i], tree, s, n,visit);
        if(s[tree[node][i]] != s[node]){
            child_cnt.push_back(child_path_count);
            node_paths += child_path_count;
        }
    }
    for(int i = 0;i < child_cnt.size();i++){
        for(int j = i+1;j < child_cnt.size();j++){
            ans += child_cnt[i]*child_cnt[j];
        }
    }
    ans += node_paths;
    return node_paths;

}
int main() {
    int n;
    cin >> n;
    vector<vector<int>> tree(n+1);
    ///必须得是两条边都放进去，因为给的测试用例不一定是父节点在前面
    for(int i = 1;i <= n-1;i++){
        int u ,v;
        cin >> u >> v;
        tree[u].push_back(v);
        tree[v].push_back(u);
    }
    string s;
    cin >> s;
    s= ' '+ s;
    vector<bool>visit(n+1,false);
    dfs(1, tree, s, n,visit);
    cout << ans;
}
// 64 位输出请用 printf("%lld")