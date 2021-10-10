#include "buffer_manage.h"
#include "util.h"
#include <array>

BufferManage::BufferManage() { }
BufferManage::~BufferManage() { }
int BufferManage::Initialize(IslandInfo island_info)
{
    island_info_ = island_info;
    max_buffer_size_ = island_info_.buffer_capacity; //*(island_info_.island_num-1);
}

int BufferManage::Uninitialize()
{
    recv_buffer_.clear();
}

int BufferManage::UpdateBufferLock(Individual &individual)
{
    std::lock_guard<std::mutex> guard(mtx);
    int ret = UpdateBuffer(individual);
    return ret;
}

Population BufferManage::SelectFromBufferLock(int emigration_num)
{
    std::lock_guard<std::mutex> guard(mtx);
    Population pop = SelectFromBuffer(emigration_num);
    return pop;
}

Population BufferManage::sample_random(int num, bool resample)
{
    assert(num >= 0 && "given num must be greater than 0.");
    Population ret;
    int buff_size = recv_buffer_.size();
    if(buff_size == 0 || num == 0){
        return ret;
    }
    else
    {
        vector<int> rand_indices = random_.Permutate(buff_size, buff_size);
        if (!resample) num = min(num, buff_size);
        while (ret.size() != num) {
            for (const auto &i : rand_indices)
            {
                ret.emplace_back(recv_buffer_.at(i));
                if (ret.size() == num) break;
            }
        }
    }
    return ret;
}

Population BufferManage::sample_best(int num, bool resample)
{
    assert(num >= 0 && "given num must be greater than 0.");
    Population ret;
    int buff_size = recv_buffer_.size();
    if(buff_size == 0 || num == 0){
        return ret;
    }
    else
    {
        vector<int> sorted_indices = argsort_population(recv_buffer_);
        if (!resample) num = min(num, buff_size);
        while(ret.size() != num) {
            for (const auto &i : sorted_indices)
            {
                ret.emplace_back(recv_buffer_.at(i));
                if (ret.size() == num) break;
            }
        }
        return ret;
    }
}

Population BufferManage::sample_roulette(int num, bool resample)
{
    assert(num >= 0 && "given num must be greater than 0.");
    Population ret;
    int buff_size = recv_buffer_.size();
    if(buff_size == 0 || num == 0){
        return ret;
    }
    vector<real> fitnesses;
    for(const auto &e : recv_buffer_)
    {
        fitnesses.push_back(e.fitness_value);
    }
    if(!resample) num = min(num, buff_size);
    vector<int> roulette_indices = random_.roulette_sample(fitnesses, num);
    for (const auto &i : roulette_indices)
    {
        ret.emplace_back(recv_buffer_.at(i));
    }
    return ret;
}

Population BufferManage::SelectFromBuffer(int emigration_num)
{
    return sample_random(emigration_num);
}

real BufferManage::CalDiversity()
{
    real sum = 0;
    if(recv_buffer_.size() > 0)
    {
        for(int i = 0; i < recv_buffer_.size(); i++)
            sum += recv_buffer_[i].fitness_value;
        real mean = sum / (recv_buffer_.size() + 0.0);
        sum = 0;
        for(int i = 0; i < recv_buffer_.size(); i++)
            sum += (recv_buffer_[i].fitness_value - mean) * (recv_buffer_[i].fitness_value - mean);
        return sqrt(sum / (recv_buffer_.size() + 0.0));
    }
    else
    {
        return 0;
    }
}

TournamentPreserving::TournamentPreserving(){

}

TournamentPreserving::~TournamentPreserving(){

}

int TournamentPreserving::UpdateBuffer(Individual &individual)
{
    if(recv_buffer_.size() < max_buffer_size_)
    {
        recv_buffer_.push_back(individual);
    }
    else
    {
        vector<int> rand_indx = random_.Permutate(recv_buffer_.size(), 1);
        if(recv_buffer_[rand_indx[0]].fitness_value > individual.fitness_value)
            recv_buffer_[rand_indx[0]] = individual;
    }
    
    return 0;
}

Population TournamentPreserving::SelectFromBuffer(int import_num)
{
    Population pop_import;

    vector<int> rand_selected = random_.Permutate(recv_buffer_.size(), import_num);
    for(auto e : rand_selected)
    {
        if(e >= recv_buffer_.size())
            cerr << "error in permutate" << endl;
        pop_import.push_back(recv_buffer_[e]);
    }

    return pop_import;
}


DiversityPreserving::DiversityPreserving()
{

}

DiversityPreserving::~DiversityPreserving()
{

}
Population DiversityPreserving::SelectFromBuffer(int emigration_num)
{
    Population emigration_export;
    for(int i = 0; i < emigration_num && recv_buffer_.size() > 0; i++)
    {
        
        int best_ID = 0;
        /*
        real best_fitness_value = recv_buffer_[0].fitness_value;
        for(int j = 1; j < recv_buffer_.size(); j++)
        {
            if(best_fitness_value > recv_buffer_[i].fitness_value)
            {
                best_fitness_value = recv_buffer_[i].fitness_value;
                best_ID = j;
            }
        }
        */
        //emigration_export.push_back(recv_buffer_[best_ID]);
        //recv_buffer_.erase(recv_buffer_.begin() + best_ID);
        //recv_buffer_.push_back(emigration_export[i]);
    }
    emigration_export = recv_buffer_;
    //recv_buffer_.clear();
    return emigration_export;
}

int DiversityPreserving::FindNearestIndividual(Individual &individual, Population &recv_buffer)
{

    int min_distance = CalDistance(individual, recv_buffer[0]);
    int nearest_index = 0;
    for(int i = 1; i < recv_buffer.size(); i++)
    {
        real distances = CalDistance(individual, recv_buffer[i]);
        if(min_distance > distances)
        {
            min_distance = distances;
            nearest_index = i;
        }
    }

    return nearest_index;
}
real DiversityPreserving::CalDistance(Individual &individual1, Individual &individual2)
{
    real distance_sum = 0;
    int dim = individual1.elements.size();
    for(int i = 0; i < dim; i++)
        distance_sum += (individual1.elements[i] - individual2.elements[i]) * (individual1.elements[i] - individual2.elements[i]);
    //distance_sum = abs(individual1.fitness_value - individual2.fitness_value);

    return distance_sum;
}

int DiversityPreserving::UpdateBuffer(Individual &individual)
{

        if(recv_buffer_.size() < max_buffer_size_)
        {
            recv_buffer_.push_back(individual);
        }
        else
        {
            int nearest_individual_ID = FindNearestIndividual(individual, recv_buffer_);
            if(recv_buffer_[nearest_individual_ID].fitness_value > individual.fitness_value)
                recv_buffer_[nearest_individual_ID] = individual;
        }
    
    return 0;
}


BestPreserving::BestPreserving()
{

}

BestPreserving::~BestPreserving()
{

}
int BestPreserving::UpdateBuffer(Individual &individual)
{
    recv_buffer_.push_back(individual);
    while(recv_buffer_.size() > island_info_.buffer_capacity * island_info_.island_size)
    {
        
        int worst_ID = 0;
        real worst_fitness_value = 1e12;
        for(int i = 1; i < recv_buffer_.size(); i++)
        {
            if(worst_fitness_value < recv_buffer_[i].fitness_value)
            {
                worst_fitness_value = recv_buffer_[i].fitness_value;
                worst_ID = i;
            }
        }
        // if(worst_ID != recv_buffer_.size()-1) cout << "update_buffer" << endl;
        recv_buffer_.erase(recv_buffer_.begin() + worst_ID);
    }
    return 0;
}


RandomReplaced::RandomReplaced()
{

}

RandomReplaced::~RandomReplaced()
{

}
int RandomReplaced::UpdateBuffer(Individual &individual)
{
    recv_buffer_.push_back(individual);
    while(recv_buffer_.size() > island_info_.buffer_capacity * island_info_.island_size)
        recv_buffer_.erase(recv_buffer_.begin() + rand() % recv_buffer_.size());
    return 0;
}

FirstReplaced::FirstReplaced()
{
    head = 0, tail = 0, count = 0;
}

FirstReplaced::~FirstReplaced()
{

}
int FirstReplaced::UpdateBuffer(Individual &individual)
{
    if(recv_buffer_.size() < max_buffer_size_)
    {
        recv_buffer_.emplace_back(individual);
    }else{
        recv_buffer_[tail] = individual;
    }

    tail = (++tail) % max_buffer_size_;
    
    return 0;
}

Population FirstReplaced::SelectFromBuffer(int emigration_num)
{
    string sample_strategy = island_info_.buffer_sampling;
    if (sample_strategy == "random")
    {
        return sample_random(emigration_num);
    }
    else if (sample_strategy == "best")
    {
        return sample_best(emigration_num);
    }
    else if (sample_strategy == "roulette")
    {
        return sample_roulette(emigration_num);
    }else
    {
        throw "Error: unsupported sampling strategy\n";
    }
}

BufferManage* generate_buffer(IslandInfo &island_info_)
{
    BufferManage *tmp_buffer_manage;
    if(island_info_.buffer_manage == "diversity")
        tmp_buffer_manage = new DiversityPreserving();
    else if(island_info_.buffer_manage == "first")
    {
        tmp_buffer_manage = new FirstReplaced();
    }
    else if(island_info_.buffer_manage == "best")
        tmp_buffer_manage = new BestPreserving();
    else if(island_info_.buffer_manage == "random")
        tmp_buffer_manage = new RandomReplaced();
    else if(island_info_.buffer_manage == "tournament")
        tmp_buffer_manage = new TournamentPreserving();
    else
        tmp_buffer_manage = new RandomReplaced();
    tmp_buffer_manage->Initialize(island_info_);
    return tmp_buffer_manage;
}