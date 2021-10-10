#ifndef __COMMUNICATOR_HH__
#define __COMMUNICATOR_HH__
#include "random.h"
#include "config.h"
#include "buffer_manage.h"
#include "comm.h"
#include "selection.h"
#include <sstream>
#include <memory>

class Communicator
{
private:
    vector<int>             success_recv_table_count;
    vector<int>             success_send_table_count;
    vector<int>             success_send_rewards_count;
    vector<int>             success_recv_rewards_count;

    int                     base_tag;
    
    
protected:
    Random                  random_;
    IslandInfo              island_info_;
    ProblemInfo             problem_info_;
    NodeInfo                node_info_;
    vector<real>            selected_import_count;
    vector<int>             selected_export_count;
    
#ifndef EMTO
    BufferManage *          buffer_manage_;
#endif

    //each island has a task pecified buffer
    unordered_map<int, BufferManage * > multi_buffer_manage_;


    list<Message>           message_queue_;
    
    int                     SendFlagFinish();
    int                     init_tables();
    int                     de_init_tables();
    int                     show_tables();
    int                     update_buffer(Message &msg, BufferManage *buffer_manage);
    int                     priority_check_receive(Message &msg);
    
public:
    Population              select_to_import(int ntasks_other=0); 
    vector<int>             select_to_export();                

    int                     Initialize(IslandInfo island_info, ProblemInfo problem_info);
    int                     update_buffer(Population &immigrants, int origin_sender);
    int                     Uninitialize();
    int                     Execute();
    Selection               selection;
    int                     record_import_times_count = 0;
    int                     record_export_times_count = 0;

    Comm                    comm_;
    shared_ptr<BufferManage> shared_buffer;
    void                    update_shared_buffer(Population &emigrants);
    unordered_map<int, int> get_selections(int num_other);
                            Communicator(const NodeInfo node_info);
                            ~Communicator();
};
#endif
