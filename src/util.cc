#include <algorithm>
#include <string.h>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <sstream>
#include <ctime>
#include "util.h"

using namespace std;

vector<string> &split(const string &str, char delim, vector<string> &elems, bool skip_empty)
{
    istringstream iss(str);
    for (string item; getline(iss, item, delim);)
        if (skip_empty && item.empty()) continue;
        else elems.push_back(item);
        return elems;
}

vector<string> split(const string &str, char delim, bool skip_empty)
{
    istringstream iss(str);
    vector<string> elems;
    for (string item; getline(iss, item, delim);)
        if (skip_empty && item.empty()) continue;
        else elems.push_back(item);
        return elems;
}

void mkdirs(const char *muldir) 
{
    int i,len;
    char str[512];    
    strncpy(str, muldir, 512);
    len=strlen(str);
    for( i=0; i<len; i++ )
    {
        if( str[i]=='/' )
        {
            str[i] = '\0';
            if( access(str,0)!=0 )
            {
                mkdir( str, 0700 );
            }
            str[i]='/';
        }
    }
    if( len>0 && access(str,0)!=0 )
    {
        mkdir( str, 0700 );
    }
    return;
}

string time_now()
{
    std::string stime;
    std::stringstream strtime;
    std::time_t currenttime = std::time(0);
    char tAll[255];
    std::strftime(tAll, sizeof(tAll), "%Y-%m-%d-%H:%M:%S", std::localtime(&currenttime));
    strtime << tAll;
    stime = strtime.str();
    return stime;
}

vector<int> argsort_population(const Population &v)
{
    // initialize original index locations
    vector<int> idx(v.size());
    for (int i = 0; i != idx.size(); ++i)
        idx[i] = i;
    // sort indexes based on comparing values in v
    sort(idx.begin(), idx.end(),
        [&v](int i1, int i2) { return v[i1].fitness_value < v[i2].fitness_value; });
    return idx;
}

vector<int> argsort(const vector<real> &v)
{
    // initialize original index locations
    vector<int> idx(v.size());
    for (int i = 0; i != idx.size(); ++i)
        idx[i] = i;
    // sort indexes based on comparing values in v
    sort(idx.begin(), idx.end(),
        [&v](int i1, int i2) { return v[i1] < v[i2]; });
    return idx; 
}

real median(vector<real> len)
{
    if (len.size() < 1)
        return std::numeric_limits<double>::signaling_NaN();

    const auto alpha = len.begin();
    const auto omega = len.end();

    // Find the two middle positions (they will be the same if size is odd)
    const auto i1 = alpha + (len.size()-1) / 2;
    const auto i2 = alpha + len.size() / 2;

    // Partial sort to place the correct elements at those indexes (it's okay to modify the vector,
    // as we've been given a copy; otherwise, we could use std::partial_sort_copy to populate a
    // temporary vector).
    std::nth_element(alpha, i1, omega);
    std::nth_element(i1, i2, omega);

    return 0.5 * (*i1 + *i2);
}

real L2_dist(vector<real> &x, vector<real> &y)
{
    real sum = 0;
    for(int i = 0; i < x.size(); i++)
    {
        sum += (x[i] - y[i]) * (x[i] - y[i]); 
    }
    return sqrt(sum + 1e-12);
}

double get_wall_time()
{
    struct timeval time;
    if (gettimeofday(&time,NULL)){
        fprintf(stderr, "Fatal : get wall clock time error.\n");
        return 0;
    }
    return (double)time.tv_sec + (double)time.tv_usec * .000001;
}
double get_cpu_time()
{
    return (double)clock() / CLOCKS_PER_SEC;
}

Mat matrix_multiply(const Mat &a, const Mat &b)
{
    assert(a.size() > 0 && b.size() > 0 && a[0].size() > 0 && b[0].size() > 0);
    assert(a[0].size() == b.size() && "a*b, a's columns == b's rows required.");
    int m = a.size(), n = b[0].size();
    Mat res(m, vector<real>(n, 0.0));
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            for (int k = 0; k < b.size(); k++)
            {
                res[i][j] += (a[i][k]*b[k][j]);
            }
        }
    }
    return res;
}

Mat matrix_transpose(const Mat &a)
{
    assert(a.size() > 0 && a[0].size() > 0);
    int m = a.size(), n = a[0].size();
    Mat res(n, vector<real>(m, 0.0));
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < m; j++)
        {
            res[i][j] = a[j][i];
        }
    }
    return res;
}

void print_mat(const Mat & m)
{
    for (const auto &row : m)
    {
        for (const auto &e : row)
        {
            cout << e << ", ";
        }
        cout << endl;
    }
}