#include "mem_region.hpp"

static std::map<std::string, unsigned long long int> unique_ckpt_names;

unsigned long long int get_region_uid(std::string ckpt_name, int version, int region_id) {
    unsigned long long int u_ckpt_id = 0;
    if (unique_ckpt_names.find(ckpt_name) == unique_ckpt_names.end()) {
        unique_ckpt_names[ckpt_name] = unique_ckpt_names.size()+1;
    }
    u_ckpt_id = unique_ckpt_names[ckpt_name];
    return u_ckpt_id*MAX_VERSIONS_PER_SHOT+version*MAX_REGIONS_PER_CKPT+region_id;
}
