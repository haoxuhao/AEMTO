#include "communicator.h"
#include <math.h>
#include "util.h"


Communicator::Communicator(NodeInfo node_info) : comm_(node_info)
{
    node_info_ = node_info;
    
}

Communicator::~Communicator()
{
    
}

int Communicator::Initialize(IslandInfo island_info, ProblemInfo problem_info)
{
    island_info_ = island_info;
    problem_info_ = problem_info;
    BufferManage *tmp_buffer_manage = generate_buffer(island_info_);
    multi_buffer_manage_.insert(std::make_pair(island_info_.island_ID, tmp_buffer_manage));

    comm_.Initialize(problem_info_.dim);
    base_tag = 1000 + 10 * problem_info_.run_ID;
    // init_tables();
    
    selection.Initialize(island_info);
    shared_buffer.reset(std::move(generate_buffer(island_info)));
    shared_buffer->Initialize(island_info);
    return 0;
}
int Communicator::Uninitialize()
{
    for (auto buffer_item : multi_buffer_manage_)
    {
        buffer_item.second->Uninitialize();
        delete buffer_item.second;
    }
    multi_buffer_manage_.clear();
    message_queue_.clear();
    comm_.Uninitialize();
    // de_init_tables();
    selection.UnInitialize();
    shared_buffer->Uninitialize();
    return 0;
}

void Communicator::update_shared_buffer(Population &emigrants)
{
    for_each(emigrants.begin(), emigrants.end(), 
        [&](Individual &ind) {
            shared_buffer->UpdateBufferLock(ind);
    });
}

unordered_map<int, int> Communicator::get_selections(int ntasks_other)
{
    unordered_map<int, int> selections;
    if (random_.RandRealUnif(0, 1) <= island_info_.ada_import_epsilon)
    {
        selections = selection.get_import_selections(true);
    }
    else
    {
        selections = selection.get_import_selections();
    }
    return selections;
}

Population Communicator::select_to_import(int ntasks_other)
{
    Population immigrants;
    if (island_info_.island_num == 1)
    {
        return immigrants;
    }
    unordered_map<int, int> selections;
    if (random_.RandRealUnif(0, 1) <= island_info_.ada_import_epsilon)
    {
        selections = selection.get_import_selections(true);
    }
    else
    {
        selections = selection.get_import_selections();
    }
    for (auto sel : selections)
    {
        auto it = multi_buffer_manage_.find(sel.first);
        if (it == multi_buffer_manage_.end())
        {
            continue;
        }
        // int sele_num = (int)island_info_.import_rate * sel.second;
        int sele_num = sel.second;
        Population sel_pop = it->second->SelectFromBuffer(sele_num);
        immigrants.insert(immigrants.end(), sel_pop.begin(), sel_pop.end());
        selected_import_count[sel.first] += sele_num / (real)island_info_.island_size;
    }
    ++record_import_times_count;
    return immigrants;
}

vector<int> Communicator::select_to_export()
{
    vector<int> destinations;
    if (island_info_.island_num == 1)
    {
        return destinations;
    }
    int outbound_num = island_info_.island_num - 1;
    unordered_map<int, int> sus_select;

    if (island_info_.export_strategy == "ADA")
    {
        sus_select = selection.get_export_destinations(outbound_num);
    }
    else
    {
        sus_select = selection.get_export_destinations(outbound_num, true);
    }

    for (auto e : sus_select)
    {
        if (e.first == island_info_.island_ID)
        {
            continue;
        }
        destinations.push_back(e.first);
        selected_export_count[e.first] += 1;
    }
    stringstream ss;
    for (auto e : destinations)
    {
        ss << e << ", ";
    }
    ++record_export_times_count;
    return destinations;
}
int Communicator::update_buffer(Message &msg, BufferManage *buffer_manage)
{
    for (int i = 0; i < msg.data.size(); i++)
    {
        buffer_manage->UpdateBuffer(msg.data[i]);
        Message tmp_msg_from_COMM;
        tmp_msg_from_COMM.flag = -1;
        if (comm_.CheckRecv(tmp_msg_from_COMM, -1, -1) == 1)
            break;
    }
    return 0;
}

int Communicator::update_buffer(Population &immigrants, int origin_sender)
{
    auto it = multi_buffer_manage_.find(origin_sender);
    if (it != multi_buffer_manage_.end())
    {
        for (int i = 0; i < immigrants.size(); i++)
        {
            immigrants[i].skill_factor = origin_sender;
            it->second->UpdateBuffer(immigrants[i]);
        }
    }
    else
    {
        BufferManage *tmp_buffer_manage = generate_buffer(island_info_);

        for (int i = 0; i < immigrants.size(); i++)
        {
            immigrants[i].skill_factor = origin_sender;
            tmp_buffer_manage->UpdateBuffer(immigrants[i]);
        }

        multi_buffer_manage_.insert(std::make_pair(origin_sender, tmp_buffer_manage));
    }
    return 0;
}
int Communicator::init_tables()
{
    selected_import_count.resize(island_info_.island_num, 0);
    selected_export_count.resize(island_info_.island_num, 0);
    success_send_table_count.resize(island_info_.island_num, 0);
    success_recv_table_count.resize(island_info_.island_num, 0);
    success_send_rewards_count.resize(island_info_.island_num, 0);
    success_recv_rewards_count.resize(island_info_.island_num, 0);
    record_import_times_count = 0;
    record_export_times_count = 0;
    return 0;
}
int Communicator::show_tables()
{
    stringstream ss[6];
    for (int i = 0; i < island_info_.island_num; i++)
    {
        ss[0] << selected_import_count[i] << ",";
        ss[1] << selected_export_count[i] << ",";
        ss[2] << success_send_table_count[i] << ",";
        ss[3] << success_recv_table_count[i] << ",";
        ss[4] << success_send_rewards_count[i] << ",";
        ss[5] << success_recv_rewards_count[i] << ",";
    }

    return 0;
}
int Communicator::de_init_tables()
{
    success_send_table_count.clear();
    success_recv_table_count.clear();
    selected_import_count.clear();
    selected_export_count.clear();
    success_send_rewards_count.clear();
    success_recv_rewards_count.clear();
    return 0;
}

int Communicator::Execute()
{
    // int finish_count = 0;
    // int flag_send_finish = 0;
    // unsigned int loop_count = 0;

    // while (finish_count != island_info_.island_num || message_queue_.size() > 0)
    // {
    //     loop_count++;

    //     /*check the immigrants from other islands*/
    //     Message msg_from_COMM;
    //     msg_from_COMM.flag = FLAG_INDIVIDUAL;
    //     if (comm_.CheckRecv(msg_from_COMM, -1, base_tag + TAG_COMM_TO_COMM) == 1)
    //     {
    //         comm_.RecvData(msg_from_COMM);
    //         auto it1 = island_info_.comm_core_island_id_table.find(msg_from_COMM.sender);
    //         assert(it1 != island_info_.comm_core_island_id_table.end() && "the message sender is not in the known world");
    //         spdlog::info("task {}; run_id {}; loop count {}; com receive one message from task {}. for {} times.",
    //                      problem_info_.task_id,
    //                      problem_info_.run_ID,
    //                      loop_count,
    //                      island_info_.task_ids[island_info_.comm_core_island_id_table[msg_from_COMM.sender]],
    //                      success_recv_table_count[it1->second] + 1);

    //         if (flag_send_finish == 0)
    //         {
    //             success_recv_table_count[it1->second]++;
    //             update_buffer(msg_from_COMM.data, it1->second);
    //         }
    //     }
    //     //check the import singal from exec core
    //     Message import_singal;
    //     import_singal.flag = -1;
    //     if (comm_.CheckRecv(import_singal, island_info_.exec_rank, base_tag + TAG_EXEC_ASK_IMPORT) == 1)
    //     {
    //         comm_.RecvData(import_singal);
    //         if (import_singal.flag != FLAG_ASK_IMPORT)
    //         {
    //             spdlog::error("received wrong import signal message, message flag is {}", import_singal.flag);
    //         }
    //         else
    //         {

    //             if (flag_send_finish == 0)
    //             {
    //                 //import immigrants
    //                 Population pop_import;
    //                 int original_sender = select_to_import(pop_import);
    //                 if (pop_import.size() > 0)
    //                 {
    //                     vector<int> receiver(1, island_info_.exec_rank);
    //                     comm_.GenerateMsg(message_queue_, pop_import,
    //                                       receiver, base_tag + TAG_COMM_TO_EXEC, 0);
    //                 }
    //                 //generate the current success receive table
    //                 stringstream ss;

    //                 for (int i = 0; i < success_recv_table_count.size(); i++)
    //                 {
    //                     ss << success_recv_table_count[i] << ",";
    //                 }

    //                 spdlog::info("task {}: run id {}; loop count {}; com generate  {} import individuals for {} times, the former recv table: {}",
    //                              problem_info_.task_id, problem_info_.run_ID, loop_count, pop_import.size(), record_export_times_count, ss.str());
    //             }
    //         }
    //     }

    //     /*check the rewards signal*/
    //     Message msg_reward;
    //     msg_reward.flag = FLAG_REWARDS;
    //     if (comm_.CheckRecv(msg_reward, -1, base_tag + TAG_REWARD) == 1)
    //     {
    //         comm_.RecvData(msg_reward);
    //         if (flag_send_finish == 0)
    //         {
    //             if (msg_reward.sender == island_info_.exec_rank)
    //             {
    //                 success_recv_rewards_count[island_info_.island_ID]++;
    //                 if (island_info_.import_strategy == "Epsilon_ADA")
    //                 {
    //                     //update the import pdf
    //                     // selection.update_import_pdf(msg_reward.rewards_table);
    //                 }

    //                 if (island_info_.export_strategy == "ADA")
    //                 {
    //                     //send the rewards to other islands
    //                     for (auto e : msg_reward.rewards_table)
    //                     {
    //                         unordered_map<int, real> reward;
    //                         reward.insert(std::make_pair(island_info_.island_ID, e.second));
    //                         vector<int> receivers(1, island_info_.global_islands_comm_ranks[e.first]);
    //                         comm_.GenerateRewardsMsg(message_queue_, receivers, base_tag + TAG_REWARD, 1, reward);

    //                         spdlog::info("task {} send rewards message to task {}, rewards: {:2}", problem_info_.task_id, island_info_.task_ids[e.first], e.second);
    //                     }
    //                 }
    //             }
    //             else
    //             {
    //                 success_recv_rewards_count[island_info_.comm_core_island_id_table[msg_reward.sender]]++;
    //                 spdlog::info("task {}: com receive one reward message from task {}. rewards: {:2}.",
    //                              problem_info_.task_id, island_info_.comm_core_island_id_table[msg_reward.sender], msg_reward.rewards_table.begin()->second);

    //                 if (island_info_.export_strategy == "ADA")
    //                 {
    //                     //update the export pdf
    //                     selection.update_export_pdf(msg_reward.rewards_table, record_export_times_count);
    //                 }
    //             }
    //         }
    //     }
    //     //check the export individuals from exec core
    //     Message msg;
    //     msg.flag = FLAG_INDIVIDUAL;
    //     if (comm_.CheckRecv(msg, island_info_.exec_rank, base_tag + TAG_EXEC_TO_COMM) == 1)
    //     {
    //         comm_.RecvData(msg);
    //         if (flag_send_finish == 0)
    //         {
    //             //export immigrants
    //             update_buffer(msg.data, island_info_.island_ID);
    //             success_recv_table_count[island_info_.island_ID]++;
    //             spdlog::debug("run id {}, task {}, comm core received {} times emigrants from exec core. loop count: {}.", problem_info_.run_ID, problem_info_.task_id,
    //                           success_recv_table_count[island_info_.island_ID], loop_count);
    //             vector<int> receiver = select_to_export();

    //             for (int i = 0; i < receiver.size(); i++)
    //             {
    //                 receiver[i] = island_info_.global_islands_comm_ranks[receiver[i]];
    //             }

    //             comm_.GenerateMsg(message_queue_, msg.data, receiver, base_tag + TAG_COMM_TO_COMM, 0); //garentee that the newly come export individuals is in the head of the send queue.

    //             //generate the current success send table
    //             stringstream ss;

    //             for (int i = 0; i < success_send_table_count.size(); i++)
    //             {
    //                 ss << success_send_table_count[i] << ",";
    //             }
    //             spdlog::info("task {}: run id {}; loop count {}; com receive {} export individuals for {} times, the former send table: {}",
    //                          problem_info_.task_id, problem_info_.run_ID, loop_count, msg.data.size(), record_import_times_count, ss.str());
    //         }
    //     }

    //     /*check the fininsh flag*/
    //     Message msg_flag;
    //     msg_flag.flag = -1;
    //     if (comm_.CheckRecv(msg_flag, -1, base_tag + TAG_FINISH) == 1)
    //     {
    //         fprintf(stderr, "task %d, recv finish flag from: %d, flag is: %d\n", problem_info_.task_id, msg_flag.sender, msg_flag.flag);
    //         comm_.RecvData(msg_flag);
    //         if (msg_flag.flag == FLAG_FINISH)
    //         {
    //             finish_count++;
    //             if (msg_flag.sender == island_info_.exec_rank)
    //             {
    //                 flag_send_finish = SendFlagFinish();
    //                 spdlog::info("island {}: comm recv finish flag from exec core", island_info_.island_ID);
    //             }
    //             else
    //             {
    //                 spdlog::info("island {}: comm recv finish flag from other island {}", island_info_.island_ID, island_info_.comm_core_island_id_table[msg_flag.sender]);
    //             }
    //         }
    //         else
    //         {
    //             spdlog::error("received wrong finish flag message, received flag is {}", msg_flag.flag);
    //         }
    //     }

    //     if (flag_send_finish == -1)
    //     {
    //         if (finish_count != island_info_.island_num)
    //         {
    //             // spdlog::error("task {}, still wait for other tasks send finish flag. loop count: {}.", problem_info_.task_id, loop_count);
    //         }
    //     }

    //     if (flag_send_finish == -1 && message_queue_.size())
    //     {
    //         // spdlog::info("task {} send finish flag to exec core and other islands, message len: {}, loop count {}, message receiver is {}, exec rank {}", problem_info_.task_id, message_queue_.size(), loop_count, message_queue_.begin()->receiver, island_info_.exec_rank);
    //         // fprintf(stderr, "task %d send finish flag to exec core and other islands, message len: %d, loop count %d, message receiver is %d, exec rank %d\n",
    //                 // problem_info_.task_id, (int)message_queue_.size(), loop_count, message_queue_.begin()->receiver, island_info_.exec_rank);
    //         if (message_queue_.size() > 1)
    //         {
    //             auto it = message_queue_.begin();
    //             ++it;
    //             spdlog::info("task {} send finish flag to exec core and other islands, message len: {}, loop count {}, next message receiver is {}, exec rank {}", problem_info_.task_id, message_queue_.size(), loop_count, it->receiver, island_info_.exec_rank);
    //         }
    //     }

    //     /*flush the data in the message queue*/
    //     for (auto it = message_queue_.begin(); it != message_queue_.end(); it++)
    //     {
    //         if (comm_.SendData(*it))
    //         {
    //             if ((it->flag == FLAG_INDIVIDUAL))
    //             {
    //                 if (it->receiver == island_info_.exec_rank)
    //                 {
    //                     success_send_table_count[island_info_.island_ID]++;
    //                     spdlog::info("task {}; run id {}; send import individuals to exec core for {} times; loop count {}",
    //                                  problem_info_.task_id, problem_info_.run_ID, success_send_table_count[island_info_.island_ID], loop_count);
    //                 }
    //                 else
    //                 {
    //                     success_send_table_count[island_info_.comm_core_island_id_table[it->receiver]]++;
    //                     spdlog::info("success send: task {} run id {} send export individuals to task {} for {} times; loop count {}",
    //                                  problem_info_.task_id, problem_info_.run_ID, island_info_.task_ids[island_info_.comm_core_island_id_table[it->receiver]],
    //                                  success_send_table_count[island_info_.comm_core_island_id_table[it->receiver]], loop_count);
    //                 }
    //             }
    //             if (it->flag == FLAG_REWARDS)
    //             {
    //                 success_send_rewards_count[island_info_.comm_core_island_id_table[it->receiver]]++;
    //                 spdlog::debug("success send: task {} send rewards to task {} for {} times",
    //                               problem_info_.task_id, island_info_.task_ids[island_info_.comm_core_island_id_table[it->receiver]],
    //                               success_send_rewards_count[island_info_.comm_core_island_id_table[it->receiver]]++);
    //             }
    //             it = message_queue_.erase(it);
    //             // break;
    //         }
    //         else if (flag_send_finish == -1 &&
    //                  it->receiver == island_info_.exec_rank &&
    //                  it->flag == FLAG_INDIVIDUAL)
    //         {
    //             it = message_queue_.erase(it);
    //         }
    //     }
    //     if (message_queue_.size() > 0 && flag_send_finish == -1)
    //         spdlog::info("task {}; run id {}; loop count {}; on comm core send queque size; {}", problem_info_.task_id, problem_info_.run_ID, loop_count, message_queue_.size());
    // }

    // //show all tables
    // show_tables();
    // return 0;
}

int Communicator::SendFlagFinish()
{
    Message message;
    message.flag = FLAG_FINISH;
    message.tag = base_tag + TAG_FINISH;
    for (int i = 0; i < island_info_.island_num; i++)
    {
        if (island_info_.island_ID != i)
        {
            message.receiver = island_info_.global_islands_comm_ranks[i];
            message_queue_.push_back(message);
        }
        else
        {
            message.receiver = island_info_.exec_rank;
            message_queue_.push_front(message);
        }
    }
    return -1;
}
