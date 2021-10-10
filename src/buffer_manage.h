#ifndef __BUFFER_MANAGE__
#define __BUFFER_MANAGE__

#include <mutex>
#include <algorithm>
#include "comm.h"
#include "random.h"
#include "config.h"


class BufferManage
{
protected:
    Population          recv_buffer_;
    Population          tmp_recv_buffer_;

    Random              random_;
    IslandInfo          island_info_;
    int                 max_buffer_size_;
    mutex               mtx;

    Population          sample_best(int num, bool resample=true);
    Population          sample_random(int num, bool resample=true);
    Population          sample_roulette(int num, bool resample=true);

public:
                        BufferManage();
    int                 Initialize(IslandInfo island_info);
    int                 Uninitialize();
    virtual             ~BufferManage();
    virtual int         UpdateBuffer(Individual &individual) = 0;
    virtual int         UpdateBufferLock(Individual &individual);
    virtual Population  SelectFromBuffer(int emigration_num);
    virtual Population  SelectFromBufferLock(int emigration_num);
    int                 size() const {return recv_buffer_.size();};
    real                CalDiversity();
};

class DiversityPreserving: public BufferManage
{
protected:
    real                CalDistance(Individual &individual1, Individual &individual2);
    int                 FindNearestIndividual(Individual &individual, Population &recv_buffer);
public:
                        DiversityPreserving();
    virtual             ~DiversityPreserving();
    virtual int         UpdateBuffer(Individual &individual);
    virtual Population  SelectFromBuffer(int emigration_num);

};

class TournamentPreserving: public BufferManage
{
private:
public:
                        TournamentPreserving();
    virtual             ~TournamentPreserving();
    virtual int         UpdateBuffer(Individual &individual);
    virtual Population  SelectFromBuffer(int import_num);
};

class BestPreserving : public BufferManage
{
private:
public:
                        BestPreserving();
    virtual             ~BestPreserving();
    virtual int         UpdateBuffer(Individual &individual);
};

class RandomReplaced : public BufferManage
{
private:
public:
                        RandomReplaced();
    virtual             ~RandomReplaced();
    virtual int         UpdateBuffer(Individual &individual);
};

class FirstReplaced : public BufferManage
{
private:
    int                 head;
    int                 tail;
    uint                count;
public:
                        FirstReplaced();
    virtual             ~FirstReplaced();
    virtual int         UpdateBuffer(Individual &individual);
    virtual Population  SelectFromBuffer(int import_num);
};

BufferManage* generate_buffer(IslandInfo &island_info_);

#endif
