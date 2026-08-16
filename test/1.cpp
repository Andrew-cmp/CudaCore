#include <bits/stdc++.h>
using namespace std;

int countGood(const vector<int>& arr) {
    int n = arr.size();
    vector<int> suf(n);
    suf[n - 1] = INT_MAX;
    for (int i = n - 2; i >= 0; --i) {
        suf[i] = min(arr[i + 1], suf[i + 1]);
    }
    int cnt = 0;
    for (int i = 0; i < n - 1; ++i) {
        if (arr[i] > suf[i]) cnt++;
    }
    return cnt;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    const int MAX_TRIES = 30;
    mt19937 rng(chrono::steady_clock::now().time_since_epoch().count());

    int T;
    cin >> T;
    while (T--) {
        int n;
        cin >> n;
        vector<int> p(n), q(n);
        for (int i = 0; i < n; ++i) {
            cin >> p[i];
        }

        int base = countGood(p);
        if (base == 0 || base == n - 1) {
            cout << "NO\n";
            continue;
        }

        bool found = false;
        uniform_int_distribution<int> dist(0, n - 1);
        for (int t = 0; t < MAX_TRIES; ++t) {
            q = p;
            int i = dist(rng), j = dist(rng);
            if (i == j) j = (i + 1) % n;
            swap(q[i], q[j]);
            if (countGood(q) == base) {
                cout << "YES\n";
                for (int x : q) cout << x << ' ';
                cout << "\n";
                found = true;
                break;
            }
        }
        if (!found) {
            cout << "NO\n";
        }
    }

    return 0;
}
