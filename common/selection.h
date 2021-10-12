#ifndef __H_ADAPTIVE_SELECT_H__
#define __H_ADAPTIVE_SELECT_H__

#include "config.h"
#include <unordered_map>
#include "random.h"

class Selection
{
    private:
        IslandInfo island_info_;
        int LP;
        real epsilon;
        real base_reuse_prob;
        int export_rewards_transfer_interval;
        vector<real> import_pdf;
        vector<real> init_pdf;
        vector<real> export_pdf;
        vector<vector < real > > export_rewards_table;
        vector<vector < real > > import_rewards_table;
        vector<vector < real > > import_selections_table;
        /**
         * first: Q values
         * second: previous generation
        */
        pair<vector<real>, int> Q_value_import_selection;
        int export_rewards_table_update_count;
        int import_rewards_table_update_count;
        int last_export_rewards_received_time;

        Random random_;
        unordered_map<int, int> selection_times_;

        int Q_func_update(pair<vector<real>, int> &, unordered_map<int, int> &ns, int gen=0);
        int update_pdf(vector<real> &averaged_rewards, vector<real> &pdf);
        real get_self_rewards();
        unordered_map<int, int> sus_sampling(vector<real> &pdf, int sample_size);
        int update_pdf_pursuit(vector<real> &Q);
        int update_pdf_matching(vector<real> &Q);
        vector<pair<int, real>> top_N(const vector<real> &pdf, int N);

    public:
        Selection();
        ~Selection();
        int Initialize(IslandInfo &island_info);
        int UnInitialize();
        int update_export_pdf(unordered_map<int, real> &rewards, int current_export_rewards_received_time);
        int update_import_pdf(unordered_map<int, int> &rewards, int gen=0);
        unordered_map<int, int> get_export_destinations(int select_num=1, bool rand_=false);
        unordered_map<int, int> get_import_selections(int select_num, bool rand_=false);
        unordered_map<int, int> get_import_selections(bool rand_=false);
        const vector<real> &get_import_pdf() { return import_pdf; }
};

inline Selection::Selection()
{

}
inline Selection::~Selection()
{
    
}
#endif