#pragma once

#ifdef _WIN32

#include <windows.h>
#include <cstdint>
#include <mutex>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace cpu_affinity {

static std::vector<DWORD_PTR> g_physical_core_masks;
static std::once_flag g_init_flag;

static inline void init_physical_cores() {
    DWORD len = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &len);
    if (len == 0) {
        return;
    }

    std::vector<std::uint8_t> buffer(len);
    auto* base = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buffer.data());
    if (!GetLogicalProcessorInformationEx(RelationProcessorCore, base, &len)) {
        return;
    }

    DWORD offset = 0;
    while (offset < len) {
        auto* info = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buffer.data() + offset);
        if (info->Relationship == RelationProcessorCore && info->Processor.GroupCount > 0) {
            const DWORD_PTR mask = info->Processor.GroupMask[0].Mask;
            if (mask != 0) {
                const DWORD_PTR first_lp = mask & (~mask + 1);
                g_physical_core_masks.push_back(first_lp);
            }
        }
        offset += info->Size;
    }
}

static inline void pin_current_worker() {
    std::call_once(g_init_flag, init_physical_cores);
    if (g_physical_core_masks.empty()) {
        return;
    }

#ifdef _OPENMP
    const int tid = omp_get_thread_num();
#else
    const int tid = 0;
#endif

    const DWORD_PTR mask = g_physical_core_masks[static_cast<std::size_t>(tid) % g_physical_core_masks.size()];
    SetThreadAffinityMask(GetCurrentThread(), mask);
}

static inline void pin_once() {
    static thread_local bool pinned = false;
    if (!pinned) {
        pin_current_worker();
        pinned = true;
    }
}

} // namespace cpu_affinity

#else

namespace cpu_affinity {
static inline void pin_once() {}
} // namespace cpu_affinity

#endif

