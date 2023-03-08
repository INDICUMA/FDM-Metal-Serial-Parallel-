//
//  main.swift
//  FDM-Metal (Serial+Parallel)
//
//  Created by 조일현 on 2023/03/08.
//



import MetalKit



//FDM parameters
let start =  CFAbsoluteTimeGetCurrent()
var nx : Int = 32
var ny : Int = 32
let t : Float = 1000000.0
let diviN : Int = 4
print(diviN)
let divi : Int = nx/diviN
let dx : Float = 1.0
let dy : Float = 1.0

let dt : Float = 0.1
let bndL : Float = 0.0
let bndR : Float = 100.0
let initV : Float = 50.0
var stepT = Int(t/dt)

// matrix
var u = [Float](repeating: initV, count: nx*ny)
var uR = [Float](repeating: initV, count: nx*ny)
var temp = [Float](repeating: 0.0, count: nx*ny)
// Boundary condition
for dy in 0..<nx{
    u[dy*ny] = bndL
    u[dy*ny+(nx-1)] = bndR
}
// uR = u
uR = u

//GPU work setup
guard
let device  = MTLCreateSystemDefaultDevice(),
let commandQueue = device.makeCommandQueue(),
let defaultLibrary = device.makeDefaultLibrary(),
let kernelFunction = defaultLibrary.makeFunction(name: "fdmKernel")//  let sumFunction = defaultLibrary.makeFunction(name: "sumKernel")
else {fatalError()}

let computePipelineState = try device.makeComputePipelineState(function: kernelFunction)


//let computePipelineState3 = try device.makeComputePipelineState(function: sumFunction)

//create Buffers
var fdmBuffer : MTLBuffer = device.makeBuffer(bytes: &u, length: MemoryLayout<Float>.stride * u.count,  options : .storageModeShared)!
var fdmBuffer2 : MTLBuffer = device.makeBuffer(bytes: &uR,length: MemoryLayout<Float>.stride * uR.count, options : .storageModeShared)!
var tempBuffer = device.makeBuffer(bytes: &temp, length: MemoryLayout<Float>.stride * temp.count,options : .storageModeShared)
var errorBuffer = device.makeBuffer(bytes: &temp, length: MemoryLayout<Float>.stride * temp.count,options : .storageModeShared)!
var eSumBuffer = device.makeBuffer(bytes: &nx, length: MemoryLayout<Int>.stride ,options : .storageModeShared)!

//set ThreadExecution ( threadgroup_position_in_grid)
//let w = computePipelineState.threadExecutionWidth
//let h = computePipelineState.maxTotalThreadsPerThreadgroup/(w)
let threadsPerThreadgroup = MTLSize(width:nx/divi, height:ny/divi , depth: 1)
var threadGroupSizeIsMultipleOfThreadExecutionWidth: Bool { true }
// computePipelineState.threadExecutionWidth (thread_position_in_threadgroup)
let threadGroupCount = MTLSize(width:divi, height: divi, depth: 1)

let threadsPerThreadgroup2 = MTLSize(width:nx, height:1 , depth: 1)
//var threadGroupSizeIsMultipleOfThreadExecutionWidth: Bool { true }
// computePipelineState.threadExecutionWidth (thread_position_in_threadgroup)
let threadGroupCount2 = MTLSize(width:nx, height: 1, depth: 1)


let startTime = CFAbsoluteTimeGetCurrent()

var count : Int = 0
var conv = "not converged"
var sum : Float = 0.0
var errorsum : Int = 0


//iterate
for _ in 0..<stepT{
    guard
        let commandBuffer : MTLCommandBuffer = commandQueue.makeCommandBuffer(),
        let commandBuffer2 : MTLCommandBuffer = commandQueue.makeCommandBuffer(),
  //      let commandBuffer4 : MTLCommandBuffer = commandQueue.makeCommandBuffer(),
        let computeEncoder : MTLComputeCommandEncoder = commandBuffer.makeComputeCommandEncoder(),
   //     let computeEncoder3 : MTLComputeCommandEncoder = commandBuffer4.makeComputeCommandEncoder(),
        let blitCommandEncoder = commandBuffer2.makeBlitCommandEncoder()
            
    else {fatalError()}
    
    
    computeEncoder.setComputePipelineState(computePipelineState)
    
    //set Buffers to computeEncoder
    computeEncoder.setBuffer(fdmBuffer, offset: 0, index: 0)
    computeEncoder.setBuffer(fdmBuffer2, offset: 0, index: 1)
    computeEncoder.setBuffer(errorBuffer, offset: 0, index: 2)
   // computeEncoder.setBytes(&stepT, length: MemoryLayout<Int>.stride, index: 3)
    computeEncoder.setBytes(&nx, length: MemoryLayout<Int>.stride, index: 4)
    //computeEncoder.setBytes(&ny, length: MemoryLayout<Int>.stride, index: 5)
    computeEncoder.dispatchThreadgroups(threadGroupCount, threadsPerThreadgroup: threadsPerThreadgroup)
    //start GPU time
    computeEncoder.endEncoding()

    commandBuffer.commit()
   commandBuffer.waitUntilCompleted()

let rawpointer2 = fdmBuffer2.contents()
let datau2 = rawpointer2.bindMemory(to: Float.self, capacity: u.count)
//  let rawpointer1 = fdmBuffer.contents()
//   let datau = rawpointer1.bindMemory(to: Float.self, capacity: u.count)
for i in 1..<nx{
    datau2[i] = datau2[nx+i]
    datau2[nx*(nx-1)+i] = datau2[nx*(nx-2)+i]
    //      for j in 0..<nx{
    //  sum += (abs(datau2[i*nx+j] - datau[i*nx+j]))
    //     }
}
   
    /*  let rawpointer2 = fdmBuffer2?.contents()
     var datau2 = rawpointer2?.bindMemory(to: Float.self, capacity: u.count)
     let rawpointer1 = fdmBuffer?.contents()
     var datau = rawpointer1?.bindMemory(to: Float.self, capacity: u.count)
     for i in 0..<nx*nx {
     sum += (datau![i]-datau2![i])
     }
     
     */
    //copy

    
    //   let rawpointer3 = tempBuffer?.contents()
    //  let datau3 = rawpointer3?.bindMemory(to: Float.self, capacity: u.count)
    
    
  
    
    blitCommandEncoder.copy(from: fdmBuffer2,
                            sourceOffset: 0,
                            to: fdmBuffer,
                            destinationOffset: 0,
                            size:
                                MemoryLayout<Int>.stride*u.count/2)
    
    blitCommandEncoder.endEncoding()
    
    commandBuffer2.commit()
  //  commandBuffer2.waitUntilCompleted()
    
}
//capture GPU time
print("GPU conversion time:", CFAbsoluteTimeGetCurrent() - startTime)

let rawpointer2 = fdmBuffer2.contents()
var datau2 = rawpointer2.bindMemory(to: Float.self, capacity: u.count)


for i in 0..<nx{
    for j in 0..<ny{
        print(datau2[i*nx+j], terminator: " ")
    }
    print("\n")
}

print(conv, count,"diff sum:",sum)

