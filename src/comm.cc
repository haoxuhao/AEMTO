#include "comm.h"


Comm::Comm(NodeInfo node_info):node_info_(node_info)
{
}

Comm::~Comm()
{
}

int Comm::Initialize(int dim)
{
    for (int i = 0; i < node_info_.node_num; i++)
    {
        MPI_Request req;
        send_request_statuses.insert(make_pair(i, req));
        is_first_send.insert(make_pair(i, true));
    }
    return 0;
}

int Comm::Uninitialize()
{
    send_request_statuses.clear();
    is_first_send.clear();
    return 0;
}

vector<TsfDataUnitType> Comm::SerializePopData(Population &pop)
{
    /*
    * headsize | popsize | dim | objectives | data ... |
    */
    vector<TsfDataUnitType> data;
    if(pop.size() == 0)
    {
        fprintf(stderr, "warning: rank %d no data for sending\n", get_rank());
        return data;
    }
    int dim = pop[0].elements.size();
    int data_size = pop.size() * (dim + 1);
    data.resize(data_size + 4);

    data[0] = 4; data[1] = (TsfDataUnitType)pop.size();
    data[2] = dim; data[3] = 1; //only one objectives
    int begin = 4;
    for (const auto &ind : pop)
    {
        for (int i = 0; i < dim; i++)
        {
            data[begin++] = ind.elements[i];
        }
        data[begin++] = ind.fitness_value;
    }
    return data;
}

Population Comm::DeserializePopData(vector<TsfDataUnitType> &data, int sender)
{
    Population pop;
    // check the receive data; length check
    assert(data.size() >= 4 && "Error: receive data size >= 4 required.");
    int headsize = (int)data[0];
    int popsize = (int)data[1];
    int dim = (int)data[2];
    int objs = (int)data[3];
    
    if (popsize * (dim + objs) + 4 != data.size())
    {
        printf("warning: rank %d recv data len %d, head size %d, pop size %d, dim %d, objs %d. head data %.2f, %.2f, %.2f, %.2f\n",
            get_rank(), data.size(), headsize, popsize, dim, objs,
            data[0], data[1], data[2], data[3]);
        return pop;
    }
    // assert(popsize * (dim + objs) + 4 == data.size() && 
    //     "Error: message length doesn't match");
    // parse the data
    int begin = 4;
    for (int i = 0; i < popsize; i++)
    {
        Individual ind;
        ind.elements.resize(dim);
        for (int j = 0; j < dim; j++)
        {
            ind.elements[j] = data[begin++];
        }
        ind.fitness_value = data[begin++];
        ind.skill_factor = sender;
        pop.emplace_back(ind);
    }
    return pop;
}

RecvInfo Comm::CheckRecv(int sender, int tag)
{
    int flag_incoming = 0;
    MPI_Status mpi_status;
    if (sender == -1 && tag == -1)
        MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag_incoming, &mpi_status);
    else if (sender == -1 && tag != -1)
        MPI_Iprobe(MPI_ANY_SOURCE, tag, MPI_COMM_WORLD, &flag_incoming, &mpi_status);
    else if (sender != -1 && tag == -1)
        MPI_Iprobe(sender, MPI_ANY_TAG, MPI_COMM_WORLD, &flag_incoming, &mpi_status);
    else if (sender != -1 && tag != -1)
        MPI_Iprobe(sender, tag, MPI_COMM_WORLD, &flag_incoming, &mpi_status);
    
    RecvInfo info;
    if (flag_incoming == 1)
    {
#ifdef DOUBLE_PRECISION
        MPI_Get_count(&mpi_status, MPI_DOUBLE, &info.len);
#else
        MPI_Get_count(&mpi_status, MPI_FLOAT, &info.len);
#endif 
        info.status = ok;
        info.sender = mpi_status.MPI_SOURCE;
        info.tag = mpi_status.MPI_TAG; 
    }else{
        info.status = none;
    }
    return info;
}

vector<TsfDataUnitType> Comm::RecvData(RecvInfo &info)
{
    std::lock_guard<std::mutex> guard(recv_mtx);
    vector<TsfDataUnitType> recv_data(info.len); 
    MPI_Status mpi_status; 
#ifdef DOUBLE_PRECISION
    MPI_Recv(recv_data.data(), info.len, MPI_DOUBLE, info.sender, info.tag, MPI_COMM_WORLD, &mpi_status);
#else
    MPI_Recv(recv_data.data(), info.len, MPI_FLOAT, info.sender, info.tag, MPI_COMM_WORLD, &mpi_status);
#endif
    if (mpi_status.MPI_ERROR != MPI_SUCCESS)
    {
        fprintf(stderr, "Error: receive data error, code: %d\n", mpi_status.MPI_ERROR);
        recv_data.clear();
        return recv_data;
    }
    return recv_data;
}

Status Comm::CheckSend(int receiver)
{
    std::lock_guard<std::mutex> guard(send_check_mtx);
    if(is_first_send[receiver])
    {
        is_first_send[receiver] = false; 
        return ok;
    }
    MPI_Status mpi_status;
    int test_flag = 0;
    if(send_request_statuses.find(receiver) == send_request_statuses.end())
    {
        printf("rank %d send request status map size %d for receiver %d \n", get_rank(), send_request_statuses.size(), receiver);
    }

    assert(send_request_statuses.find(receiver) != send_request_statuses.end());
    MPI_Test(&send_request_statuses[receiver], &test_flag, &mpi_status);
    if (test_flag != 0)
    {
        return ok;
    } else {
        return failed;
    }
}
void Comm::SendData(const vector<TsfDataUnitType> &data, RecvInfo &info)
{
    std::lock_guard<std::mutex> guard(send_check_mtx);
#ifdef DOUBLE_PRECISION
    MPI_Isend(data.data(), data.size(), MPI_DOUBLE, info.receiver, info.tag, MPI_COMM_WORLD, &send_request_statuses[info.receiver]);
#else
    MPI_Isend(data.data(), data.size(), MPI_FLOAT, info.receiver, info.tag, MPI_COMM_WORLD, &send_request_statuses[info.receiver]);
#endif
}

void Comm::flush_send_queques(unordered_map<int, queue<pair<vector<TsfDataUnitType>, bool>>> &q, int tag)
{
    for (auto &sendque_pair : q)
    {
        queue<pair<vector<TsfDataUnitType>, bool>> &sendque = sendque_pair.second;
        int receiver = sendque_pair.first;
        if (sendque.size() > 0)
        {
            RecvInfo info(receiver, tag);
            if (CheckSend(info.receiver) == ok)
            {
                if (sendque.front().second == true)
                {
                    sendque.pop();
                }
                if (sendque.size())
                {
                    sendque.front().second = true;
                    SendData(sendque.front().first, info);
                }
            }
        }
    }
}

void Comm::flush_send_queques(unordered_map<int, queue<pair<pair<vector<TsfDataUnitType>, int>, bool>>> &q)
{
    for (auto &sendque_pair : q)
    {
        queue<pair<pair<vector<TsfDataUnitType>, int>, bool>> &sendque = sendque_pair.second;
        int receiver = sendque_pair.first;
        if (sendque.size() > 0)
        {
            RecvInfo info(receiver);
            if (CheckSend(info.receiver) == ok)
            {
                if (sendque.front().second == true)
                {
                    sendque.pop();
                }
                if (sendque.size())
                {
                    sendque.front().second = true;
                    info.tag = sendque.front().first.second; 
                    SendData(sendque.front().first.first, info);
                }
            }
        }
    }
}

int Comm::CheckRecv(Message & message, int sender, int tag)
{
    int flag_incoming_msg = 0;
    MPI_Status mpi_status;
    if(sender == -1 && tag == -1)
        MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag_incoming_msg, &mpi_status);
    if(sender == -1 && tag != -1)
        MPI_Iprobe(MPI_ANY_SOURCE, tag, MPI_COMM_WORLD, &flag_incoming_msg, &mpi_status);
    if(sender != -1 && tag == -1)
        MPI_Iprobe(sender, MPI_ANY_TAG, MPI_COMM_WORLD, &flag_incoming_msg, &mpi_status);
    if(sender != -1 && tag != -1)
        MPI_Iprobe(sender, tag, MPI_COMM_WORLD, &flag_incoming_msg, &mpi_status);
    if(flag_incoming_msg == 1)
    {
    	message.sender = mpi_status.MPI_SOURCE;
    	message.receiver = node_info_.node_ID;
    	message.tag = mpi_status.MPI_TAG;
        if(message.flag >= 0) //check data length for data message
        {
#ifdef DOUBLE_PRECISION
            MPI_Get_count(&mpi_status, MPI_DOUBLE, &message.msg_length);
#else
            MPI_Get_count(&mpi_status, MPI_FLOAT, &message.msg_length);
#endif   
        }
    }
    return flag_incoming_msg;
}

int Comm::RecvData(Message &message)
{
    MPI_Status mpi_status;

    if(message.flag >= 0)
    {
        real * msg_recv = new real[message.msg_length];
#ifdef DOUBLE_PRECISION
	    MPI_Recv(msg_recv, message.msg_length, MPI_DOUBLE, message.sender, message.tag, MPI_COMM_WORLD, &mpi_status);
#else
	    MPI_Recv(msg_recv, message.msg_length, MPI_FLOAT, message.sender, message.tag, MPI_COMM_WORLD, &mpi_status);
#endif 
        if(message.flag == FLAG_REWARDS)
        {
            for(int i=0; i < message.msg_length;)
            {
                message.rewards_table.insert(std::make_pair((int)msg_recv[i], msg_recv[i+1]));
                i+=2;
            }
        }
        else if(message.flag == FLAG_INDIVIDUAL)
        {
            DeserialMsgToIndividual(message.data, msg_recv, message.msg_length/(individual_len_));
        }else{
            return -1;
        }

	    delete [] msg_recv; 
    }
    else
    {
	    MPI_Recv(&message.flag, 1, MPI_INT, message.sender, message.tag, MPI_COMM_WORLD, &mpi_status);
    }

    return 0;
}

int Comm::CheckSend(int receiver, int msg_flag)
{
    //create an independent request for a receiver
    MPI_Status mpi_status;
    auto it = send_status_table.find(receiver);
    if(it != send_status_table.end())
    {
        int test_flag = 0;
        MPI_Test(&it->second.first, &test_flag, &mpi_status);
        if(test_flag == 0)
        {
            // if(try_to_send_count>1000)
            // {
            //     MPI_Cancel(&it->second.first);
            //     try_to_send_count = 0;
            // }
        }
        return test_flag;
    }
    else
    {
        return -1;
    }
}


int Comm::CheckSend()
{
    MPI_Status mpi_status;
    
    if(flag_ready_to_send_ == 0)
    {
        MPI_Test(&mpi_request_, &flag_ready_to_send_, &mpi_status);
        if(flag_ready_to_send_ == 0)
        {
            // MPI_Cancel(&mpi_request_);
            // spdlog::error("Node {}, canceled a send request; total canceled count: {}", node_info_.node_ID, ++total_cancel_count);
            // if(try_to_send_count >= max_wait_count)
            // {
            //     Cancel();
            //     try_to_send_count = 0;
            //     spdlog::error("Node {}, canceled a send request; total canceled count: {}", node_info_.node_ID, ++total_cancel_count);
            // }
        }
    }
    
    return flag_ready_to_send_;
}

real Comm::Time()
{
    return MPI_Wtime();
}


int Comm::SendData(Message &message)
{
    int check_status;
    if(message.flag >= 0)
    {
        check_status = CheckSend(message.receiver, message.flag);
        if(check_status != 0)
        {
            // serialize the message to array
            //individuals data length
            int msg_length = message.data.size() * (individual_len_); 
            // append the rewards message
            msg_length += message.rewards_table.size()*2; //key & value

            real *msg_to_send = new real[msg_length];
            int offset = 0;
            if(message.flag == FLAG_INDIVIDUAL)
            {
                offset = SerialIndividualToMsg(msg_to_send, message.data);
            }
            else if(message.flag == FLAG_REWARDS)
            {
                for(auto e : message.rewards_table)
                {
                    msg_to_send[offset] = (real)e.first;
                    offset++;
                    msg_to_send[offset] = (real)e.second;
                    offset++;
                }
            }
            else
            {
            }
            
            //need to new a request for the receiver
            if(check_status == -1) 
            {
                MPI_Request local_request;
                send_status_table.insert(std::make_pair(message.receiver, std::make_pair(local_request, msg_to_send)));
            }
            else
            {
                auto it = send_status_table.find(message.receiver);
                delete[] it->second.second;
                it->second.second = msg_to_send;
            }
            
            auto it = send_status_table.find(message.receiver);
            // non-blocking send data out
#ifdef DOUBLE_PRECISION
            MPI_Isend(msg_to_send, msg_length, MPI_DOUBLE, message.receiver, message.tag, MPI_COMM_WORLD, &(it->second.first));
#else
            MPI_Isend(msg_to_send, msg_length, MPI_FLOAT, message.receiver, message.tag, MPI_COMM_WORLD, &(it->second.first));
#endif

            // flag_ready_to_send_ = 0;
            // delete []msg_to_send; // ? release data here may cause some problem, maybe fixed later.
            return 1;
        }
        return  0;
    }
    else
    {
        check_status = CheckSend(message.receiver, message.flag);
        if(check_status != 0)
        {
            //need to new a request for the receiver
            if(check_status == -1) 
            {
                MPI_Request local_request;
                send_status_table.insert(std::make_pair(message.receiver, std::make_pair(local_request, (real*)NULL)));
            }
            auto it = send_status_table.find(message.receiver);

            MPI_Isend(&message.flag, 1, MPI_INT, message.receiver, message.tag, MPI_COMM_WORLD,  &(it->second.first));
            // flag_ready_to_send_ = 0;
            return  1;
        }
        return  0;
    }
}
int Comm::Cancel()
{
    // if(CheckSend() == 0)
    // {
    //     int flag_cancel = 0;
    //     MPI_Status mpi_stutus;
    //     while(flag_cancel == 0)
    //     {
    //         MPI_Cancel(&mpi_request_);
    //         MPI_Test_cancelled(&mpi_stutus, &flag_cancel);
    //     }
    //     flag_ready_to_send_ = 1;
    // }
    return 0;
}

int Comm::DeserialMsgToIndividual(Population &individual, real *msg, int length)
{
    int count = 0;
    individual.clear();
    for (int i = 0; i < length; i++)
    {
        Individual local_individual;
        for(int j = 0; j < dim_; j++)
        {
            local_individual.elements.push_back(msg[count]);
            count++;
        }
        local_individual.fitness_value = msg[count];
        count++;
        local_individual.skill_factor = (int)msg[count];
        count++;

        individual.push_back(local_individual);
    }
    return count;
}


int Comm::SerialIndividualToMsg(real *msg, Population &individual)
{
    int count = 0;
    for (int i = 0; i < individual.size(); i++)
    {
        for (int j = 0; j < dim_; j++)
        {
            msg[count] = individual[i].elements[j];
            count++;
        }
        msg[count] = individual[i].fitness_value;
        count++;
        msg[count] = (real)individual[i].skill_factor;
        count++;
    }
    return count;
}

int Comm::GenerateMsg(list<Message> & message_queue, \
                        Population msg_data, \
                        vector<int> destinations, \
                        int tag, int pos, int original_sender, real rewards)
{
    for(int i = 0; i < destinations.size(); i++)
    {
        Message message;
        message.data = msg_data;
        message.sender = node_info_.node_ID;
        message.receiver = destinations[i];
        message.original_sender = original_sender;
        message.rewards = rewards;
        message.tag = tag;
        message.flag = FLAG_INDIVIDUAL;
        if(pos == 0)
        {
            auto it = message_queue.begin();
            if(it!=message_queue.end() && it->flag == FLAG_ASK_IMPORT)
            {
                message_queue.insert(++it, message);
            }else{
                message_queue.push_front(message);
            }
        }
        else
            message_queue.push_back(message);
    }
    return 0;
}

int Comm::GenerateRewardsMsg(list<Message> & message_queue, \
                                        vector<int> destinations, \
                                        int tag, int pos, unordered_map<int, real>& rewards_table)
{
    for(auto dst : destinations)
    {
        Message message;
        message.sender = node_info_.node_ID;
        message.receiver = dst;
        message.flag = FLAG_REWARDS;
        message.tag = tag;
        message.rewards_table = rewards_table;
        if(pos == 0)
        {
            message_queue.push_front(message);
        }else{
            message_queue.push_back(message);
        }
    }
    return 0;
}

int Comm::Finish(int base_tag)
{
    return 0;
}