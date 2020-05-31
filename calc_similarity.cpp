// !!! Change: this is a new file

#include <cstdio>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "getopt.h"
#include "km.h"

using namespace std;

typedef vector<float> Vector;
typedef vector<Vector> Vector2D;
typedef vector<Vector2D> Vector3D;
// typedef vector<Vector3D> Vector4D;
// 可以写成一个类
#define MAXN 5000

Vector3D* read(ifstream& ifile) {
    // ifstream ifile;
    // ifile.open(filename, ios::in);
    // if (!infile) {
    //     cerr << "Failed to open file" << endl;
    //     exit(1);
    // }
    int d1, d2, d3, d4;
    ifile >> d1 >> d2 >> d3 >> d4;
    int tsize = d1 * d2 * d3 * d4;
    Vector3D* pv3d = new Vector3D;
    pv3d->resize(d1);
    for (int i = 0; i < d1; ++i) {
        (*pv3d)[i].resize(d2);
        for (int j = 0; j < d2; ++j) {
            (*pv3d)[i][j].resize(d3 * d4);
        }
    }
    for (int i = 0; i < d1; ++i) {
        for (int j = 0; j < d2; ++j) {
            for (int k = 0; k < d3 * d4; ++k) {
                ifile >> (*pv3d)[i][j][k];
            }
        }
    }
    // ifile.close();
    return pv3d;
}

float kernel_similarity(Vector& a, Vector& b) {
    int l = a.size();
    float ans = 0;
    for (int i = 0; i < l; ++i) {
        if (a[i] == b[i]) ans += 1;
    }
    return ans / float(l);
}

float kernel_similarity_iou(Vector& a, Vector& b) {
    int l = a.size();
    int I = 0;
    int U = 0;
    for (int i = 0; i < l; ++i) {
        if (a[i] == 1 || b[i] == 1) {
            U++;
        }
        if (a[i] == 1 && b[i] == 1) {
            I++;
        }
    }
    if (I == 0) return 0;
    return float(I) / float(U);
}

inline float sim_conbine(float a, float b) { return (a + b) / 2; }

float max_matching_approx(Vector2D& psimm) {
    int l = psimm.size();

    bool visX[MAXN] = {};
    bool visY[MAXN] = {};
    int curX = 0;
    int curY = 0;
    float m1 = 0;
    float m2 = 0;
    while (1) {
        float tmpmax = -1;
        bool finished = true;
        for (int j = 0; j < l; ++j) {
            if (visY[j]) continue;
            if (psimm[curX][j] > tmpmax) {
                tmpmax = psimm[curX][j];
                finished = false;
                curY = j;
            }
        }
        if (finished) break;
        visY[curY] = true;
        m1 += tmpmax;
        tmpmax = -1;
        finished = true;
        for (int j = 0; j < l; ++j) {
            if (visX[j]) continue;
            if (psimm[j][curY] > tmpmax) {
                tmpmax = psimm[j][curY];
                finished = false;
                curX = j;
            }
        }
        if (finished) break;
        visX[curX] = true;
        m2 += tmpmax;
    }
    return ((m1 > m2) ? m1 : m2) / float(l);
}

KM<float> km(MAXN, MAXN);
float max_matching(Vector2D sim_matrix) {
    int cntx = sim_matrix.size() - 1;
    // for (auto i : sim_matrix) {
    //     for (auto j : i) {
    //         cout << j << ' ';
    //     }
    //     cout << endl;
    // }
    return km.compute(sim_matrix) / float(cntx);
}

Vector2D net_similarity(vector<Vector3D>& ks1, vector<Vector3D>& ks2,
                        Vector2D& sim_init) {
    /*
     *  *sim_init维度和*pks第一层输入维度(实际的第二维)相同
     */

    int num_kernels = ks1.size();
    vector<Vector2D> simvec;
    simvec.push_back(sim_init);
    for (int i = 0; i < num_kernels; ++i) {
        Vector2D vtemp;
        vtemp.resize(ks1[i].size() + 1);
        for (int o = 0; o < ks1[i].size() + 1; ++o) {
            vtemp[o].resize(ks1[i].size() + 1, 0);
        }
        simvec.push_back(vtemp);
    }
    // cout << simvec.size() << endl;
    clock_t start_time = clock();
    for (int i = 0; i < num_kernels; ++i) {
        int num_nodes = ks1[i][0].size();
        int num_nodes_nxt = ks1[i].size();
        bool cout_flag = false;
        for (int j = 0; j < num_nodes_nxt; ++j) {
            for (int k = 0; k < num_nodes_nxt; ++k) {
                // cout << i << ' ' << j << ' ' << k << ' ';
                Vector2D sim_matrix;
                sim_matrix.resize(num_nodes + 1);
                for (int o = 0; o < num_nodes + 1; ++o) {
                    sim_matrix[o].resize(num_nodes + 1, 0);
                }
                for (int l = 0; l < num_nodes; ++l) {
                    for (int m = 0; m < num_nodes; ++m) {
                        float line_sim =
                            kernel_similarity(ks1[i][j][l], ks2[i][k][m]);
                        // cout<<simvec[0].size()<<endl;
                        // cout<<simvec[0][0].size()<<endl;

                        float node_sim = simvec[i][l][m];
                        sim_matrix[l][m] = sim_conbine(line_sim, node_sim);
                    }
                }
                if (num_nodes < 1000) {
                    simvec[i + 1][j][k] = max_matching(sim_matrix);
                } else {
                    if (!cout_flag) {
                        cout << "running approximate algorithm for max matching"
                             << endl;
                        cout_flag = true;
                    }
                    simvec[i + 1][j][k] = max_matching_approx(sim_matrix);
                }
                // cout << fixed << setprecision(20) << simvec[i + 1][j][k]
                //     << endl;
            }
        }
        // cout << "layer " << i << ", time:" << fixed << setprecision(3)
        //      << (clock() - start_time) / float(CLOCKS_PER_SEC)
        //      << ", similarity: " << max_matching(simvec[i + 1]) << endl;
    }
    // cout << "total time: " << (clock() - start_time) / float(CLOCKS_PER_SEC)
    //      << endl;
    return simvec[num_kernels];
}

inline void bool_output(bool b) {
    if (b)
        cout << "true" << endl;
    else
        cout << "false" << endl;
}

int main(int argc, char** argv) {
    int opt;
    int option_index = 0;
    const char* optstring = "";
    string X_seed = "2";
    string Y_seed = "2";
    bool X_trained = false;
    bool Y_trained = false;
    bool init_state = false;
    string config = "conv6_ukn_unsigned";
    static struct option long_options[] = {
        {"X_seed", required_argument, NULL, 0},
        {"Y_seed", required_argument, NULL, 1},
        {"X_trained", no_argument, NULL, 2},
        {"Y_trained", no_argument, NULL, 3},
        {"config", required_argument, NULL, 4},
        {"init-state", no_argument, NULL, 5},  // 未处理，默认为false
        {0, 0, 0, 0}  // 添加 {0, 0, 0, 0} 是为了防止输入空值
    };
    while ((opt = getopt_long(argc, argv, optstring, long_options,
                              &option_index)) != -1) {
        switch (opt) {
            case 0:
                X_seed = optarg;
                break;
            case 1:
                Y_seed = optarg;
                break;
            case 2:
                X_trained = true;
                break;
            case 3:
                Y_trained = true;
                break;
            case 4:
                config = optarg;
                break;
            case 5:
                init_state = true;
        }
    }
    string X_path = "seed_" + X_seed;
    if (X_trained) X_path = X_path + "_weight_trained";

    string Y_path = "seed_" + Y_seed;
    if (Y_trained) Y_path = Y_path + "_weight_trained";

    if (config == "conv6_usc_unsigned") {
        X_path = "./runs/conv6_usc_unsigned/" + X_path +
                 "/prune_rate=0.7/checkpoints/";
        Y_path = "./runs/conv6_usc_unsigned/" + Y_path +
                 "/prune_rate=0.7/checkpoints/";
    } else if (config == "conv6_ukn_unsigned") {
        X_path = "./runs_ukn/" + X_path + "/prune_rate=0.7/checkpoints/";
        Y_path = "./runs_ukn/" + Y_path + "/prune_rate=0.7/checkpoints/";
    }

    string filename = "m.txt";
    if(init_state) filename = "i.txt";
    
    ifstream ifile, ifile2;
    ifile.open(X_path + filename, ios::in);
    if (!ifile) {
        cerr << "file path: " << X_path << filename << endl;
        cerr << "若m.txt不存在，请先运行data_processing.py" << endl;
        exit(1);
    }
    Vector3D* conv0_1 = read(ifile);
    Vector3D* conv2_1 = read(ifile);
    Vector3D* conv5_1 = read(ifile);
    Vector3D* conv7_1 = read(ifile);
    Vector3D* conv10_1 = read(ifile);
    Vector3D* conv12_1 = read(ifile);
    Vector3D* linear0_1 = read(ifile);
    Vector3D* linear2_1 = read(ifile);
    Vector3D* linear4_1 = read(ifile);
    // cout << conv0_1->size() << ' ' << (*conv0_1)[0].size() << ' '
    //      << (*conv0_1)[0][0].size() << endl;
    // cout << conv2_1->size() << ' ' << (*conv2_1)[0].size() << ' '
    //      << (*conv2_1)[0][0].size() << endl;
    // cout << conv5_1->size() << ' ' << (*conv5_1)[0].size() << ' '
    //      << (*conv5_1)[0][0].size() << endl;
    // cout << conv7_1->size() << ' ' << (*conv7_1)[0].size() << ' '
    //      << (*conv7_1)[0][0].size() << endl;
    // cout << conv10_1->size() << ' ' << (*conv10_1)[0].size() << ' '
    //      << (*conv10_1)[0][0].size() << endl;
    // cout << conv12_1->size() << ' ' << (*conv12_1)[0].size() << ' '
    //      << (*conv12_1)[0][0].size() << endl;
    // cout << linear0_1->size() << ' ' << (*linear0_1)[0].size() << ' '
    //      << (*linear0_1)[0][0].size() << endl;
    // cout << linear2_1->size() << ' ' << (*linear2_1)[0].size() << ' '
    //      << (*linear2_1)[0][0].size() << endl;
    // cout << linear4_1->size() << ' ' << (*linear4_1)[0].size() << ' '
    //      << (*linear4_1)[0][0].size() << endl;
    ifile.close();

    ifile2.open(Y_path + filename, ios::in);
    if (!ifile2) {
        cerr << "file path: " << Y_path << filename << endl;
        cerr << "若m.txt不存在，请先运行data_processing.py" << endl;
        exit(1);
    }
    Vector3D* conv0_2 = read(ifile2);
    Vector3D* conv2_2 = read(ifile2);
    Vector3D* conv5_2 = read(ifile2);
    Vector3D* conv7_2 = read(ifile2);
    Vector3D* conv10_2 = read(ifile2);
    Vector3D* conv12_2 = read(ifile2);
    Vector3D* linear0_2 = read(ifile2);
    Vector3D* linear2_2 = read(ifile2);
    Vector3D* linear4_2 = read(ifile2);
    // cout << conv0_2->size() << ' ' << (*conv0_2)[0].size() << ' '
    //      << (*conv0_2)[0][0].size() << endl;
    // cout << conv2_2->size() << ' ' << (*conv2_2)[0].size() << ' '
    //      << (*conv2_2)[0][0].size() << endl;
    // cout << conv5_2->size() << ' ' << (*conv5_2)[0].size() << ' '
    //      << (*conv5_2)[0][0].size() << endl;
    // cout << conv7_2->size() << ' ' << (*conv7_2)[0].size() << ' '
    //      << (*conv7_2)[0][0].size() << endl;
    // cout << conv10_2->size() << ' ' << (*conv10_2)[0].size() << ' '
    //      << (*conv10_2)[0][0].size() << endl;
    // cout << conv12_2->size() << ' ' << (*conv12_2)[0].size() << ' '
    //      << (*conv12_2)[0][0].size() << endl;
    // cout << linear0_2->size() << ' ' << (*linear0_2)[0].size() << ' '
    //      << (*linear0_2)[0][0].size() << endl;
    // cout << linear2_2->size() << ' ' << (*linear2_2)[0].size() << ' '
    //      << (*linear2_2)[0][0].size() << endl;
    // cout << linear4_2->size() << ' ' << (*linear4_2)[0].size() << ' '
    //      << (*linear4_2)[0][0].size() << endl;
    ifile2.close();

    vector<Vector3D> ks1, ks2;
    ks1.push_back(*conv0_1);
    ks1.push_back(*conv2_1);
    ks1.push_back(*conv5_1);
    ks1.push_back(*conv7_1);
    ks1.push_back(*conv10_1);
    ks1.push_back(*conv12_1);

    ks2.push_back(*conv0_2);
    ks2.push_back(*conv2_2);
    ks2.push_back(*conv5_2);
    ks2.push_back(*conv7_2);
    ks2.push_back(*conv10_2);
    ks2.push_back(*conv12_2);

    Vector2D sim_init;
    sim_init.resize((*conv0_1)[0].size() + 1);
    for (int i = 0; i < (*conv0_1)[0].size() + 1; ++i) {
        sim_init[i].resize((*conv0_1)[0].size() + 1, 1);
        sim_init[i][(*conv0_1)[0].size()] = 0;
    }
    // cout << sim_init.size() << endl;
    // cout << sim_init[0].size() << endl;
    Vector2D a = net_similarity(ks1, ks2, sim_init);
    cout << max_matching(a) << endl;

    Vector2D sim_init_2;
    // int l = a.size() - 1;
    // sim_init_2.resize(16 * l + 1);
    // for (int i = 0; i < 16 * l + 1; ++i) {
    //     sim_init_2[i].resize(16 * l + 1, 0);
    // }
    // for (int i = 0; i < l; ++i) {
    //     for (int j = 0; j < l; ++j) {
    //         for (int k = 0; k < 16; ++k) {
    //             for (int l = 0; l < 16; ++l) {
    //                 sim_init_2[16 * i + k][16 * j + l] = a[i][j];
    //             }
    //         }
    //     }
    // }
    int l = (*linear2_1)[0].size() + 1;
    sim_init_2.resize(l);
    for (int o = 0; o < l; ++o) {
        sim_init_2[o].resize(l, 1);
    }

    vector<Vector3D> ls1, ls2;
    // ls1.push_back(*linear0_1);
    ls1.push_back(*linear2_1);
    ls1.push_back(*linear4_1);
    // ls2.push_back(*linear0_2);
    ls2.push_back(*linear2_2);
    ls2.push_back(*linear4_2);

    Vector2D res = net_similarity(ls1, ls2, sim_init_2);
    cout << max_matching(res) << endl;
    bool_output(init_state);
    bool_output(X_trained);
    bool_output(Y_trained);
    cout << X_seed << endl;
    cout << Y_seed << endl;
    cout << config << endl;
}