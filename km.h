// !!! Change: this is a new file

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <vector>
#define MAXN 5000
#define INF 0x3fffffff

using namespace std;

void swap(int& a, int& b) {
    int temp = a;
    a = b;
    b = temp;
}

template <class T>
struct KM {
    vector<vector<T> > mp;
    // T mp[MAXN][MAXN];
    int link_x[MAXN], link_y[MAXN], N, cntx, cnty;
    bool visx[MAXN], visy[MAXN];
    int que[MAXN << 1], top, fail, pre[MAXN];
    T hx[MAXN], hy[MAXN], slack[MAXN];
    KM(int cntx, int cnty) : cntx(cntx), cnty(cnty) { N = max(cntx, cnty) + 1; }
    KM(int cntx, int cnty, vector<vector<T> > m)
        : cntx(cntx), cnty(cnty), mp(m) {
        N = max(cntx, cnty) + 1;
    }
    void resize(int x, int y){
        cntx = x;
        cnty = y;
        N = max(cntx, cnty) + 1;
    }
    inline int check(int i) {
        visx[i] = true;
        if (link_x[i]) {
            que[fail++] = link_x[i];
            return visy[link_x[i]] = true;
        }
        while (i) {
            link_x[i] = pre[i];
            swap(i, link_y[pre[i]]);
        }
        return 0;
    }
    void bfs(int S) {
        for (int i = 1; i <= N; i++) {
            slack[i] = INF;
            visx[i] = visy[i] = false;
        }
        top = 0;
        fail = 1;
        que[0] = S;
        visy[S] = true;
        while (true) {
            T d;
            while (top < fail) {
                for (int i = 1, j = que[top++]; i <= N; i++) {
                    if (!visx[i] &&
                        slack[i] >= (d = hx[i] + hy[j] - mp[i - 1][j - 1])) {
                        pre[i] = j;
                        if (d > 0)
                            slack[i] = d;
                        else if (!check(i))
                            return;
                    }
                }
            }
            d = INF;
            for (int i = 1; i <= N; i++) {
                if (!visx[i] && d > slack[i]) d = slack[i];
            }
            for (int i = 1; i <= N; i++) {
                if (visx[i])
                    hx[i] += d;
                else
                    slack[i] -= d;
                if (visy[i]) hy[i] -= d;
            }
            for (int i = 1; i <= N; i++) {
                if (!visx[i] && !slack[i] && !check(i)) return;
            }
        }
    }
    void init() {
        for (int i = 1; i <= N; i++) {
            link_x[i] = link_y[i] = 0;
            visy[i] = false;
        }
        for (int i = 1; i <= N; i++) {
            hx[i] = 0;
            for (int j = 1; j <= N; j++) {
                if (hx[i] < mp[i - 1][j - 1]) hx[i] = mp[i - 1][j - 1];
            }
        }
    }
    T compute() {
        T ans = 0;
        init();
        for (int i = 1; i <= N; i++) bfs(i);
        for (int i = 1; i <= cntx; i++) ans += mp[i - 1][link_x[i] - 1];
        return ans;
    }
    T compute(vector<vector<T> > m) {
        mp = m;
        int x = m.size() - 1;
        int y = m[0].size() - 1;
        resize(x, y);
        T ans = 0;
        init();
        for (int i = 1; i <= N; i++) bfs(i);
        for (int i = 1; i <= cntx; i++) ans += mp[i - 1][link_x[i] - 1];
        return ans;
    }
};