#ifndef __MIGRATE_HH__
#define __MIGRATE_HH__
#include "random.h"
#include "config.h"
#include "buffer_manage.h"
#include "comm.h"
#include "EA.h"


class Migrate
{
private:
    NodeInfo                node_info_;
    IslandInfo              island_info_;
    ProblemInfo             problem_info_;
    EAInfo                  EA_info_;

    Random                  random_;
    Comm                    comm_;
    list<Message>           message_queue_;
    vector<real>            transfer_table_;
    vector<real>            rewards_table_;
    int                     export_count;
    int                     ask_import_count;
    int                     rewards_count;
    int                     comm_choked_count;
    int                     base_tag_;
    int                     clean_mpi_inner_buffer();

    // adaptive reuse params
    real                    lower_import_prob;
    real                    upper_import_prob;
    int                     reuse_ineffect_count;
    real                    reuse_ineffect_thresh;
    vector<vector<real>>    reuse_record_table;
    int                     row_to_insert;
    int                     loop_count;

    /**
     * Private definition of adaptive import prob
     */
    pair<real, int>         Q_value_update_self;
    pair<real, int>         Q_value_update_reuse;
    int                     update_Q_func(pair<real, int> &Q, real update_rate, int gen=0);
    int                     update_import_prob_matching();
    int                     update_import_prob_pursuit();

public:

                            Migrate(const NodeInfo node_info);
                            ~Migrate();

    int                     send_rewards_table(unordered_map<int, real> &success_insert_table);
    vector<real>            get_success_insert_table();

    int                     ExportCriteria(uint generation);
    int                     ImportCriteria(uint generation);

    int                     AskForImport();
    int                     Initialize(IslandInfo island_info, ProblemInfo problem_info, EAInfo EA_info);
    int                     Uninitialize();
    int                     Finish();
    real                    GetImportProb(); 
    real                    FlushMessageQueue(uint generation);

    int                     ExportEmigrants(Population &population);

    Population              PrepareEmigrations(Population &population);
    int                     RecvImmigrations(Population &recv_pop, uint generation);

    int 					migration_counter;
    int                     times_record_other;
    int                     times_record_self;
    int                     best_times_record_other;
    int                     best_times_record_self;
    void                    update_import_prob_EBS();
    /**
     *  adaptive import prob public functions
     */
    void                    add_update_rate_self(real update_rate_self, int gen);
    real                    get_Q_update_rate_self() { return Q_value_update_self.first; };
    void                    add_update_rate_reuse(real update_rate_reuse, int gen);
    real                    get_Q_update_rate_reuse() { return Q_value_update_reuse.first; };
    int                     UpdateImportProb();
    int RandomImportProb() 
    {
        return random_.RandRealUnif(island_info_.lower_import_prob, island_info_.upper_import_prob);
    };

};
#endif
