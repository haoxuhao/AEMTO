#ifndef __H_UTIL_H__
#define __H_UTIL_H__

#include "config.h"
#include <time.h>
#include <sys/time.h>

using namespace std;

void mkdirs(const char *muldir);
typedef vector<vector<Real> > Mat;

vector<string> split(const string &str, char delim, bool skip_empty=true);
vector<string> &split(const string &str, char delim, vector<string> &elems, bool skip_empty = true);
vector<int> argsort_population(const Population &pop);
vector<int> argsort(const vector<Real> &v);
Real median(vector<Real>);
Mat matrix_multiply(const Mat &a, const Mat &b);
Mat matrix_transpose(const Mat &a);
void print_mat(const Mat & m);

string time_now();
double get_wall_time();
double get_cpu_time();

Real L2_dist(vector<Real> &x, vector<Real> &y);

template<typename T> int vector_index(vector<T> &arr,T v)
{
    auto it = std::find(arr.begin(), arr.end(), v); 
    if (it != arr.end()) 
    { 
        return it - arr.begin();
    }
    else
    {
        return -1;
    }   
}

#endif