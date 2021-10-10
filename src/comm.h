#ifndef __COMM_H__
#define __COMM_H__

#include <mpi.h>
#include "config.h"
#include <mutex>
#include <queue>

#define TAG_EXEC_TO_COMM   		    1
#define TAG_COMM_TO_EXEC    	    2
#define TAG_COMM_TO_COMM     		3
#define TAG_RECORD					4
#define TAG_REWARD					5
#define TAG_EXEC_ASK_IMPORT			6
#define TAG_FINISH     			    0


enum TagMsg {
    TagAny = -1,
    DataReq,
    DataIndvidual,
    ReqDataBegin = 1000000, // must no more than 100M tasks
};


//data message
//0 for check data message
#define FLAG_INDIVIDUAL             1
#define FLAG_REWARDS                2

//flag message
// -1 is reserved flag test
#define FLAG_FINISH                 -2
#define FLAG_ASK_IMPORT             -3

enum Status {ok, none, failed};

typedef real TsfDataUnitType;


struct Message
{
	Population data;
	int flag;
	int sender;
	int receiver;
	int original_sender;
	real rewards;
	unordered_map<int, real> rewards_table;
	int msg_length;
	int tag;
};

struct RecvInfo
{
    int sender;
    int receiver;
    int len;
    int tag;
    Status status;
    RecvInfo(int recv, int tag=-1) : receiver(recv), tag(tag) {};
    RecvInfo() {};
};

class Comm
{
private:
    MPI_Request             mpi_request_;
    NodeInfo                node_info_;
    int                     dim_;
    int                     flag_ready_to_send_;

    // real *                  send_msg_to_other_EA_;
    unsigned int            try_to_send_count;
    unsigned int            max_wait_count;
    unsigned int            total_cancel_count;
    int                     individual_len_;
    unordered_map<int, pair<MPI_Request, real*>>  send_status_table;
    int                     append_msg_len_;

    unordered_map<int, MPI_Request> send_request_statuses;
    unordered_map<int, bool> is_first_send;
    mutex send_check_mtx;
    mutex recv_mtx;
   
    Population              Best(Population &population, int select_num);
    int                     SerialIndividualToMsg(real *msg_of_node_EA, Population &individual);
    int                     DeserialMsgToIndividual(Population &individual, real *msg_of_node_EA, int length);
    int                     FindNearestIndividual(Individual &individual, Population &population);

public:
                            Comm(const NodeInfo node_info);
                            ~Comm();
    int                     Initialize(int dim);
    int                     Uninitialize();
    int                     GenerateMsg(list<Message> & message_queue, \
                                        Population msg_data, \
                                        vector<int> destinations, \
                                        int tag, int pos, int original_sender=-1,\
                                        real rewards=0);

    int                     GenerateRewardsMsg(list<Message> & message_queue, \
                                        vector<int> destinations, \
                                        int tag, int pos, unordered_map<int, real> &rewards_table);

    int                     CheckRecv(Message & message, int sender, int tag);
    int                     SendData(Message &message);
    int                     RecvData(Message &message);
    int                     Cancel();
    int                     CheckSend();
    int                     CheckSend(int receiver, int msg_flag);
    int                     Finish(int base_tag);
    real                    Time();

    /**
     * check specified data received?
     * return RecvInfo
     * */
    RecvInfo                CheckRecv(int sender = -1, int tag = TagAny);
    Status CheckSend(int receiver);
    /**
     * Receive raw data to local buffer
     * */
    vector<TsfDataUnitType> RecvData(RecvInfo &info);

    /**
     * Send raw data
     * */
    void SendData(const vector<TsfDataUnitType> &data, RecvInfo &info);
    void flush_send_queques(unordered_map<int, queue<pair<vector<TsfDataUnitType>, bool>>> &q, int tag);
    /// with tag in each message data
    void flush_send_queques(unordered_map<int, queue<pair<pair<vector<TsfDataUnitType>, int>, bool>>> &q);

    vector<TsfDataUnitType>  SerializePopData(Population &pop);
    Population DeserializePopData(vector<TsfDataUnitType> &data, int sender=0);
    int get_rank() const {return node_info_.node_ID;}

};
#endif