//
//  kernel.metal
//  FDM-Metal (Serial+Parallel)
//
//  Created by 조일현 on 2023/03/08.
//


#include <metal_stdlib>
using namespace metal;



kernel void fdmKernel(
    device float *a [[ buffer(0) ]],
    device float *b [[ buffer(1) ]],
    device float *temp [[ buffer(2) ]],
                      
   const uint2 id [[ thread_position_in_grid ]],
 //   uint id2 [[threadgroup_position_in_grid]],
//    uint id [[thread_index_in_threadgroup]],
 // const  uint2 id [[thread_position_in_threadgroup]],
 // const  uint2 tg       [[ threads_per_threadgroup ]],
  //  const device int  &stepT [[buffer(3)]],
    const device uint  &nx [[buffer(4)]])

{

 //   threadgroup_barrier(mem_flags::mem_threadgroup   );
    
    if ((id.x <= (nx-3)) && (id.y <= (nx-3))) {
        temp[((id.x+1)*nx+id.y+1)]  = (a[(id.x+1+1)*nx+id.y+1]+a[(id.x+1-1)*nx+id.y+1]+a[(id.x+1)*nx+id.y+1+1]+a[(id.x+1)*nx+id.y-1+1]-4*a[(id.x+1)*nx+id.y+1]) ;
        b[(id.x+1)*nx+id.y+1] = a[(id.x+1)*nx+id.y+1] + float(0.1 /(pow(1.0, 2)))*temp[(id.x+1)*nx+id.y+1];
    }


    
  //  threadgroup_barrier(mem_flags::mem_device);
  //      b[0]++;
    
}



  

