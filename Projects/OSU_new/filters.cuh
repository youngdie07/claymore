#ifndef __FILTERS_CUH_
#define __FILTERS_CUH_

#include "settings.h"
#include "mgmpm_kernels.cuh"

#include <MnBase/Math/Vec.h>

namespace mn {

// __forceinline__ __device__ int 
// arrayIndex(int x, int y, int z, int height, int width) {
// 	return x + y*height + z*height*width;
// }

template <typename shT = float(*) [3][8][8][8], typename T = float>
__forceinline__ __device__ void
wenchia_filter(shT buffer, vec<T,9> &Out) {

    //https://www.researchgate.net/publication/314097527_An_Enhanced_Smoothing_Algorithm_for_MPM_to_Stabilize_Hydrodynamic_Impact_Problems_with_Embedded_Solids

    // Incremental values of two adjacent cells from localized smoothing
    // dq+ = y (m- / (m- + m+))(q- - q+)
    // dq- = y (m+ / (m- + m+))(q+ - q-)

    // Oscillation limiter
    // (qr - qc)(qc - ql) < 0;
    // If > 0 then y = 0
    // Else 

    // Linear Limiter
    // Sc = (qr - ql) / (xor - xol) if l and r exist
    // Sc = (qc - ql) / (xoc - xol) if l exist
    // Sc = (qr - qc) / (xor - xoc) if r exist

    // eac = (xm - xoc)Sc
    // xm = (m- xo+ + m+ xp-)/(m- + m+)

    // y = min{max{((q- - q+) + (ea- - ea+))/(q- - q+), 0}, 1}
    
    // Grid-to-particle buffer size set-up
    static constexpr uint64_t numViPerBlock = mn::config::g_blockvolume * 3;  //< Velocities per block
    static constexpr uint64_t numViInArena  = numViPerBlock << 3; //< Velocities per arena
    static constexpr unsigned blocksize = mn::config::g_blocksize;
    static constexpr unsigned blockbits = mn::config::g_blockbits;
    static constexpr unsigned blockmask = mn::config::g_blockmask;
    
    T* splits[8][8][8];
    for (int base = threadIdx.x; base < numViInArena; base += blockDim.x) {
        char local_block_id = base / numViPerBlock;
        int channelid = base % numViPerBlock;
        //char c = channelid & 0x3f;
        char cz = channelid & blockmask;
        char cy = (channelid >>= blockbits) & blockmask;
        char cx = (channelid >>= blockbits) & blockmask;
        char i = cx + (local_block_id & 4 ? blocksize : 0);
        char j = cy + (local_block_id & 2 ? blocksize : 0);
        char k = cz + (local_block_id & 1 ? blocksize : 0);

        channelid >>= blockbits;

        // Pull from grid buffer (device)

        // T *empt = buffer[0][0][0][0];
        // Out[0] = *empt;
        if (channelid == 1)
            //splits[i][j][k] = buffer[channelid][i][j][k]; //< Grid-node vx (m/s)
            //*splits[i][j][k] = 0.f; 
            *buffer[channelid][i][j][k] = 0.f;
            // T qc = *splits[i][j][k] / 2.f;
            // T* qp = splits[i][j][k];
            // T q = 0.f;
            //Out[0] = *buffer[channelid][i][j][k];
            //Out[0] = *splits[i][j][k];

            //*buffer[channelid][i][j][k] = qc;
    }

// #pragma unroll 6
//     for (char i = 1; i < 7; i++)
// #pragma unroll 6
//       for (char j = 1; j < 7; j++)
// #pragma unroll 6
//         for (char k = 1; k < 7; k++) 
// #pragma unroll 6
//             for (char s = 0; s < 6; s++) {
//         }

}



}
#endif