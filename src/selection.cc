#include "selection.h"
#include "util.h"
#include <sstream>
#include <queue>
#include <algorithm>

int bin_search(vector<pair<real, real>> &wheel, real p, int left, int right)
{
    if (left > right)
        return -1;

    int mid = (left + right) / 2;
    real low = wheel[mid].first;
    real high = wheel[mid].second;

    if (low <= p && p <= high) //make sure that 0 posibility will not be selected ever.
    {
        return mid;
    }
    else if (p < low)
    {
        return bin_search(wheel, p, left, mid);
    }
    else
    {
        return bin_search(wheel, p, mid + 1, right);
    }
}

int Selection::Initialize(IslandInfo &island_info)
{
    island_info_ = island_info;
    LP = 10;
    epsilon = 0.0001;
    base_reuse_prob = 0.1;
    export_rewards_transfer_interval = 1;
    last_export_rewards_received_time = 1;

    for (int i = 0; i < LP; i++)
    {
        vector<real> row1(island_info_.island_num, 0);
        vector<real> row2(island_info_.island_num, 0);
        import_rewards_table.emplace_back(row1);
        import_selections_table.emplace_back(row1);
        export_rewards_table.emplace_back(row2);
    }

    export_pdf.resize(island_info_.island_num, 1.0 / (island_info_.island_num - 1));
    import_pdf.resize(island_info_.island_num, 1.0 / (island_info_.island_num - 1));
    init_pdf.resize(island_info_.island_num, 1.0 / (island_info_.island_num - 1));
    Q_value_import_selection.first.resize(island_info_.island_num, 0.0);
    Q_value_import_selection.second = 0;

    export_pdf[island_info_.island_ID] = 0;
    import_pdf[island_info_.island_ID] = 0;
    init_pdf[island_info_.island_ID] = 0;

    export_rewards_table_update_count = 0;
    import_rewards_table_update_count = 0;

    return 0;
}

int Selection::UnInitialize()
{
    export_pdf.clear();
    import_pdf.clear();
    import_rewards_table.clear();
    import_selections_table.clear();
    export_rewards_table.clear();
    return 0;
}

real Selection::get_self_rewards()
{
    return 0;
}

int Selection::Q_func_update(pair<vector<real>, int> &Q,
                             unordered_map<int, int> &ns, int gen)
{
    real alpha = island_info_.ada_param.alpha;
    real delta_T = island_info_.ada_param.Delta_T;
    int pre_gen = Q_value_import_selection.second;

    real pre_decay = delta_T / std::max<real>(delta_T, gen - pre_gen);

    for (int i = 0; i < island_info_.island_num; i++)
    {
        if (i != island_info_.island_ID)
        {
            if (selection_times_.find(i) != selection_times_.end())
            {
                auto it = ns.find(i);
                if (it != ns.end())
                {
                    real r = (real)ns[i] / (real)selection_times_[i];
                    Q_value_import_selection.first.at(i) = \
                        Q_value_import_selection.first.at(i) * alpha * pre_decay + (1 - alpha) * r;
                }
                else{
                    Q_value_import_selection.first.at(i) = Q_value_import_selection.first.at(i) * alpha * pre_decay;
                }
            }
        }
    }
    Q_value_import_selection.second = gen;
    return 0; 
}

int Selection::update_pdf_matching(vector<real> &Q)
{
    int K = island_info_.island_num - 1;
    // real pmin = 1 / (2.0 * K);
    real pmin = island_info_.ada_param.pbase * (1 / (real)K);
    real Q_sum = 0;
    real epslion = 1e-4;
    Q_sum = accumulate(std::begin(Q), std::end(Q), Q_sum);
    for (int i = 0; i < island_info_.island_num; i++)
    {
        if (i != island_info_.island_ID)
        {
            import_pdf[i] = \
                pmin + (1 - K*pmin) * (Q.at(i) + epslion) / (Q_sum + K*epslion);
        }
    }
    return 0;
}

int Selection::update_pdf_pursuit(vector<real> &Q)
{
    int K = island_info_.island_num - 1;
    // real pmin = 1 / (2.0 * K);
    real pmin = island_info_.ada_param.pbase * (1 / (real)K);
    real pmax = 1 - pmin * (K-1);
    real beta = 0.2;
    real max_q = -1.0;
    int max_sel = -1;
    for (int i = 0; i < island_info_.island_num; i++)
    {
        if (i != island_info_.island_ID){
            if (Q.at(i) > max_q)
            {
                max_q = Q.at(i);
                max_sel = i;
            }
        }
    }
    for (int i = 0; i < island_info_.island_num; i++)
    {
        if (i != island_info_.island_ID)
        {
            if (i == max_sel)
            {
                import_pdf[i] += beta * (pmax - import_pdf[i]); // increase to pmax
            }else{
                import_pdf[i] += beta * (pmin - import_pdf[i]); // decrease to pmin
            }
        }
    }
    return 0;
}

int Selection::update_pdf(vector<real> &averaged_rewards, vector<real> &pdf)
{
    real sum = 0;
    // division normalization
    for (int i = 0; i < averaged_rewards.size(); i++)
    {
        // if(i == island_info_.island_ID) continue;
        sum += averaged_rewards[i];
    }

    for (int i = 0; i < pdf.size(); i++)
    {
        // if(i == island_info_.island_ID) continue;
        pdf[i] = (averaged_rewards[i]) / sum;
    }

    return 0;
}
int Selection::update_export_pdf(unordered_map<int, real> &rewards, int current_export_rewards_received_time)
{
    int ret = 0;
    int row_to_insert;

    if ((current_export_rewards_received_time -
         last_export_rewards_received_time) > export_rewards_transfer_interval)
    {
        export_rewards_table_update_count++;

        // update the pdf
        if (export_rewards_table_update_count > LP)
        {
            vector<real> tmp(island_info_.island_num, 0);
            for (int j = 0; j < island_info_.island_num; j++)
            {
                if (j == island_info_.island_ID)
                    continue;
                for (int i = 0; i < LP; i++)
                {
                    tmp[j] += export_rewards_table[i][j];
                }
                tmp[j] /= LP;
                tmp[j] += epsilon; //add a constant and each the final prob could be larger than 1/10*tasks.
            }
            update_pdf(tmp, export_pdf);
            ret = 1;
        }

        row_to_insert = export_rewards_table_update_count % LP;

        // reset the next row
        for (int i = 0; i < export_rewards_table[0].size(); i++)
        {
            export_rewards_table[row_to_insert][i] = 0.0;
        }
    }
    else
    {
        row_to_insert = export_rewards_table_update_count % LP;
    }

    for (int i = 0; i < island_info_.island_num; i++)
    {
        auto it = rewards.find(i);
        if (it != rewards.end())
        {
            export_rewards_table[row_to_insert][i] = it->second;
        }
    }

    last_export_rewards_received_time = current_export_rewards_received_time;
    return ret;
}
int Selection::update_import_pdf(unordered_map<int, int> &ns, int gen)
{
    if (island_info_.ada_import_strategy == "Pursuit")
    {
        Q_func_update(Q_value_import_selection, ns, gen);
        update_pdf_pursuit(Q_value_import_selection.first);
        return 0;
    } else if (island_info_.ada_import_strategy == "Matching")
    {
        Q_func_update(Q_value_import_selection, ns, gen);
        update_pdf_matching(Q_value_import_selection.first);
        return 0;
    } else {
        // Table of SaDE can be considered as Q function estimate
        throw invalid_argument("invalid probability update strategy");
    }
}

unordered_map<int, int> Selection::sus_sampling(vector<real> &pdf, int sample_size)
{
    unordered_map<int, int> ret;
    if (sample_size <= 0)
    {
        return ret;
    }
    //make wheel
    vector<pair<real, real>> wheel;
    real top = 0;
    for (int i = 0; i < pdf.size(); i++)
    {
        wheel.push_back(std::make_pair(top, top + pdf[i]));
        top += pdf[i];
    }
    //sus smaple
    real step_size = top / sample_size;
    real r = random_.RandRealUnif(0, step_size);
    int s = bin_search(wheel, r, 0, pdf.size());
    ret.insert(std::make_pair(s, 1));
    int count = 1;

    while ((++count) <= sample_size)
    {
        r += step_size;
        if (r > 1)
            r -= (int)r;
        s = bin_search(wheel, r, 0, pdf.size());
        if (ret.find(s) != ret.end())
        {
            ret[s]++;
        }
        else
        {
            ret.insert(std::make_pair(s, 1));
        }
    }
    return ret;
}

unordered_map<int, int> Selection::get_export_destinations(int select_num, bool rand_)
{
    unordered_map<int, int> ret;
    if (rand_)
    {
        ret = sus_sampling(init_pdf, select_num);
    }
    else
    {
        ret = sus_sampling(export_pdf, select_num);
    }
    return ret;
}

// return the top N indices
struct cmpPairSecondFloatGreat{
    bool operator() (const std::pair<int, real>&a,
        const std::pair<int, real>& b) {
        return a.second > b.second;
    }
};


vector<pair<int, real>> Selection::top_N(const vector<real> &pdf, int N)
{
    // random indices
    vector<int> rand_indices(pdf.size(), 0);
    for (int i = 0; i < rand_indices.size(); i++) rand_indices[i] = i;
    std::random_shuffle(rand_indices.begin(), rand_indices.end());
    // build top N
    std::priority_queue<std::pair<int, real>, 
            std::vector<std::pair<int, real>>, 
            cmpPairSecondFloatGreat> pq;
    for (const auto &idx : rand_indices)
    {
        if (pq.size() < N)
        {
            pq.push(make_pair(idx, pdf[idx]));
        } else if (pdf[idx] > pq.top().second)
        {
            pq.pop();
            pq.push(make_pair(idx, pdf[idx]));
        } else {
            continue;
        }
    }
    vector<pair<int, real>> res;
    while (!pq.empty()) {
        res.push_back(pq.top());
        pq.pop();
    }
    return res;
}

unordered_map<int, int> Selection::get_import_selections(bool rand_)
{
#ifdef DEEPINSIGHT
    stringstream ss;
    vector<real> &pdf_print = import_pdf;
    if (rand_)
    {
        pdf_print = init_pdf;   
    }
    for (int i = 0; i < island_info_.island_num; i++)
    {
        if (i != island_info_.island_num - 1)
            ss << pdf_print[i] << ", ";
        else
            ss << pdf_print[i];
    }
    fprintf(stdout, "island %d selection probability: [%s]\n",
        island_info_.island_ID, ss.str().c_str());
#endif
    // tackle large task number, e.g. 1000+
    if (island_info_.island_num > island_info_.ada_param.sample_batch)
    {
        vector<pair<int, real>> topk = top_N(import_pdf, island_info_.ada_param.sample_batch);
        vector<real> sample_pdf(topk.size(), 0);
        real sum = 0;
        for (const auto &e : topk)
        {
            sum += e.second;
        }
        for (int i = 0; i < topk.size(); i++)
        {
            sample_pdf[i] = topk[i].second / (sum + 1e-12);
        }
        unordered_map<int, int> tmp;
        if (!rand_)
        {
            tmp = sus_sampling(sample_pdf, island_info_.island_size);
            selection_times_.clear();
            for (auto &e : tmp)
            {
                selection_times_.insert(make_pair(topk[e.first].first, e.second));
            }
        }
        else
        {
            selection_times_ = sus_sampling(init_pdf, island_info_.island_size);
        }
        
    }
    else {
        if (!rand_)
        {
            selection_times_ = sus_sampling(import_pdf, island_info_.island_size);
        }
        else
        {
            selection_times_ = sus_sampling(init_pdf, island_info_.island_size);
        }
    }
    
    return selection_times_;
}

unordered_map<int, int> Selection::get_import_selections(int select_num, bool rand_)
{
    if (rand_)
    {
        selection_times_ = sus_sampling(init_pdf, select_num);
    }
    else
    {
        selection_times_ = sus_sampling(import_pdf, select_num);
    }
    return selection_times_;
}
