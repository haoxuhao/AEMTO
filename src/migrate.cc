#include <math.h>
#include <sstream>

#include "migrate.h"
#include "util.h"


Migrate::Migrate(NodeInfo node_info) : comm_(node_info)
{
    node_info_ = node_info;
}

Migrate::~Migrate()
{
}

int Migrate::Initialize(IslandInfo island_info, ProblemInfo problem_info, EAInfo EA_info)
{
    island_info_ = island_info;
    problem_info_ = problem_info;
    EA_info_ = EA_info;

    comm_.Initialize(problem_info_.dim);
    migration_counter = 0;

    export_count = 0;
    ask_import_count = 0;
    rewards_count = 0;
    comm_choked_count = 0;
    base_tag_ = 1000 + 10 * problem_info_.run_ID; //base tag

    rewards_table_.resize(island_info_.island_num, 1);
    transfer_table_.resize(island_info_.island_num, 0);

    reuse_ineffect_count = 0;
    reuse_ineffect_thresh = 0.01;
    if(island_info_.ada_import_prob)
        island_info_.import_prob = \
            (island_info_.lower_import_prob + island_info_.upper_import_prob) * 0.5;
    
    // adaptive import prob Q init
    Q_value_update_self.first = 0.0;
    Q_value_update_self.second = 0;
    Q_value_update_reuse.first = 0.0;
    Q_value_update_reuse.second = 0;

    times_record_self=0;
    times_record_other=0;
    best_times_record_self = 0;
    best_times_record_other = 0;

    return 0;
}
int Migrate::Uninitialize()
{
    message_queue_.clear();
    comm_.Uninitialize();
    rewards_table_.clear();
    transfer_table_.clear();

    return 0;
}
int Migrate::clean_mpi_inner_buffer()
{
    Message test_msg;
    test_msg.flag = FLAG_INDIVIDUAL;
    test_msg.tag = base_tag_ + TAG_COMM_TO_EXEC;
    
    if (comm_.CheckRecv(test_msg, island_info_.comm_rank, test_msg.tag) == 1)
    {
        fprintf(stderr, "task %d receive any message, tag: %d, size: %d, sender: %d, comm rank: %d\n", problem_info_.task_id, test_msg.tag, test_msg.msg_length, test_msg.sender, island_info_.comm_rank);
        if (test_msg.msg_length > 0 && (test_msg.tag % 10) == TAG_COMM_TO_EXEC)
        {
            comm_.RecvData(test_msg);
            fprintf(stderr, "task %d: receive current generation invidividuals from mpi inner buffer.\n", problem_info_.task_id);
        }
        return 1;
    }
    else if (comm_.CheckRecv(test_msg, island_info_.comm_rank, -1) == 1)
    {
        fprintf(stderr, "task %d receive any message, tag: %d, size: %d, sender: %d, comm rank: %d\n", problem_info_.task_id, test_msg.tag, test_msg.msg_length, test_msg.sender, island_info_.comm_rank);
        if (test_msg.msg_length > 0 && (test_msg.tag % 10) == TAG_COMM_TO_EXEC)
        {
            comm_.RecvData(test_msg);
            fprintf(stderr, "task %d: receive previous generation invidividuals from mpi inner buffer.\n", problem_info_.task_id);
        }
        return 1;
    }
    else
    {
        return 0;
    }
}
int Migrate::Finish()
{
    Message message;
    message.flag = FLAG_FINISH;
    message.tag = base_tag_ + TAG_FINISH;
    message.receiver = island_info_.comm_rank;
    int flag_send = 0;
    int flag_recv = 0;

    while (flag_send * flag_recv == 0)
    {
        if (flag_send == 0 && comm_.SendData(message) == 1)
        {
            flag_send = 1;
        }

        //receive callback of comm core
        Message msg_flag;
        msg_flag.flag = FLAG_FINISH;
        if (comm_.CheckRecv(msg_flag, island_info_.comm_rank, base_tag_ + TAG_FINISH) == 1)
        {
            comm_.RecvData(msg_flag);
            if (msg_flag.flag == FLAG_FINISH)
            {
                fprintf(stderr, "task %d: exec core receive finish flag from comm core, out EA.", problem_info_.task_id);
                flag_recv = 1;
                while (clean_mpi_inner_buffer())
                {
                    //still need to clean the inner buffer of mpi
                    fprintf(stderr, "task: %d, receive after finish.\n", problem_info_.task_id);
                }
                // fprintf(stderr, "task %d; out clean buffer.\n", problem_info_.task_id);
            }
        }

        if (flag_send && flag_recv != 1)
        {
            //fprintf(stderr, "task %d send finish flag to comm and wait for comm to recall.\n", problem_info_.task_id);
        }
        clean_mpi_inner_buffer();
    }

    stringstream ss;
    for (int i = 0; i < transfer_table_.size(); i++)
    {
        ss << "(" << i << "|" << island_info_.task_ids[i] << ", " << transfer_table_[i] << ")"
           << "; ";
    }

    ss.str("");
    for (int i = 0; i < rewards_table_.size(); i++)
    {
        ss << "(" << island_info_.task_ids[i] << ", " << rewards_table_[i] << ")"
           << "; ";
    }

    fprintf(stderr, "task: %d out EA finish.\n", problem_info_.task_id);
    return 0;
}

vector<real> Migrate::get_success_insert_table()
{
    return transfer_table_;
}

int Migrate::ExportCriteria(uint generation)
{
    return random_.RandRealUnif(0, 1) <= island_info_.export_prob;
    //return generation % island_info_.export_interval == 0;
}

int Migrate::ImportCriteria(uint generation)
{
    // return generation % island_info_.import_interval == 0;
    return random_.RandRealUnif(0, 1) <= island_info_.import_prob;
}

real Migrate::GetImportProb()
{
    return island_info_.import_prob;
}

int Migrate::update_import_prob_matching()
{
    real epsilon = 1e-6;
    real ratio = (Q_value_update_reuse.first + epsilon) /
        (Q_value_update_reuse.first + Q_value_update_self.first + 2*epsilon);
    real range = island_info_.upper_import_prob - island_info_.lower_import_prob;
    island_info_.import_prob = island_info_.lower_import_prob + ratio * range;
    return 0;
}

int Migrate::update_import_prob_pursuit()
{
    real beta = island_info_.ada_param.beta;
    real over_thresh = 0.00;
    real diff = Q_value_update_reuse.first - Q_value_update_self.first;
    real over_ratio = diff / (Q_value_update_self.first + 1e-12);
    if (over_ratio < -over_thresh)
    {
        island_info_.import_prob += \
            beta * (island_info_.lower_import_prob - island_info_.import_prob);
    }
    else if (over_ratio > over_thresh)
    {
        island_info_.import_prob += \
            beta * (island_info_.upper_import_prob - island_info_.import_prob);
    }
    return 0;
}

int Migrate::update_Q_func(pair<real, int> &Q, real update_rate, int gen)
{
    real alpha = island_info_.ada_param.alpha;
    real delta_T = island_info_.ada_param.Delta_T;
    int pre_gen = Q.second;
    real pre_decay = delta_T / std::max<real>(delta_T, gen - pre_gen);
    Q.first = Q.first * alpha * pre_decay + (1 - alpha) * update_rate;
    Q.second = gen;
    return 0;
}

void Migrate::add_update_rate_reuse(real rate, int gen)
{
    update_Q_func(Q_value_update_reuse, rate, gen);
}

void Migrate::add_update_rate_self(real rate, int gen)
{
    update_Q_func(Q_value_update_self, rate, gen);
}

void Migrate::update_import_prob_EBS()
{
    real Ro = (real) best_times_record_other / (times_record_other + 1e-12);
    real Rs = (real) best_times_record_self / (times_record_self + 1e-12);
    island_info_.import_prob = (Ro + 1e-12) / (Ro + Rs + 2*1e-12);
    if (island_info_.import_prob < island_info_.lower_import_prob) island_info_.import_prob = island_info_.lower_import_prob;
    // printf("best times other %d, best times self %d, times self %d times other %d \n", best_times_record_other, best_times_record_self, times_record_self, times_record_other);
}

int Migrate::UpdateImportProb()
{
    if (island_info_.ada_import_prob != 1)
    {
        return 0;
    }
    if(island_info_.ada_import_strategy == "Matching")
    {
        update_import_prob_matching();
    }else if (island_info_.ada_import_strategy == "Pursuit")
    {
        update_import_prob_pursuit();
    } else if (island_info_.ada_import_strategy == "EBS")
    {
        update_import_prob_EBS();
    }
    
    return 0;
}

int Migrate::AskForImport()
{
    Message message;
    message.receiver = island_info_.comm_rank;
    message.tag = base_tag_ + TAG_EXEC_ASK_IMPORT;
    message.flag = FLAG_ASK_IMPORT;
    message.sender = node_info_.node_ID;
    message_queue_.push_front(message); //insert to the head
    return 0;
}

real Migrate::FlushMessageQueue(uint generation)
{
    real communication_time = comm_.Time();

    for (auto it = message_queue_.begin(); it != message_queue_.end(); it++)
    {
        if (comm_.SendData(*it))
        {
            if (it->flag == FLAG_INDIVIDUAL)
            {
            }
            else if (it->flag == FLAG_REWARDS)
            {
            }
            else if (it->flag == FLAG_ASK_IMPORT)
            {
            }
            it = message_queue_.erase(it);
        }
    }
    if (message_queue_.size() > 1)
    {
        comm_choked_count++;
    }

    return comm_.Time() - communication_time;
}

Population Migrate::PrepareEmigrations(Population &population)
{
    Population emigrants;
    if (island_info_.emmigration_strategy == "tournament")
    {
        for (int i = 0; i < (int)ceil(island_info_.export_rate * island_info_.island_size); i++)
        {
            vector<int> rand_index = random_.Permutate(population.size(), 2);
            if (population[rand_index[0]].fitness_value < population[rand_index[1]].fitness_value)
                emigrants.emplace_back(population[rand_index[0]]);
            else
                emigrants.emplace_back(population[rand_index[1]]);
        }
    }
    else if (island_info_.emmigration_strategy == "best")
    {
        vector<int> sorted_indexes = argsort_population(population);
        for (int i = 0; i < (int)ceil(island_info_.export_rate * island_info_.island_size) && i < island_info_.island_size; i++)
        {
            emigrants.emplace_back(population[sorted_indexes[i]]);
        }
    }
    else
    {
        vector<int> rand_index = random_.Permutate(population.size(), (int)ceil(island_info_.export_rate * island_info_.island_size));
        for (int i = 0; i < rand_index.size(); i++)
        {
            emigrants.emplace_back(population[rand_index[i]]);
        }
    }

    return emigrants;
}

int Migrate::ExportEmigrants(Population &population)
{
    Population emigrants = PrepareEmigrations(population);
    // message_queue_.clear(); //if former sending failed, the last messages should be cleaned.
    comm_.GenerateMsg(message_queue_, emigrants, vector<int>(1, island_info_.comm_rank), base_tag_ + TAG_EXEC_TO_COMM, 0);
}

int Migrate::RecvImmigrations(Population &recv_pop, uint generation)
{
    Message recv_msg;
    recv_msg.flag = FLAG_INDIVIDUAL;
    int sender = island_info_.comm_rank;
    int tag = base_tag_ + TAG_COMM_TO_EXEC;

    if (comm_.CheckRecv(recv_msg, sender, tag) == 1)
    {
        comm_.RecvData(recv_msg);
        if (recv_msg.data.size())
        {
            recv_pop.insert(recv_pop.end(), recv_msg.data.begin(), recv_msg.data.end());
            migration_counter++;
            return 1;
        }
    }
    return 0;
}

