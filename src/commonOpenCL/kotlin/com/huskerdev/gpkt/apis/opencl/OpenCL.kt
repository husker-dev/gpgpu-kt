package com.huskerdev.gpkt.apis.opencl

import com.huskerdev.gpkt.MemoryUsage
import kotlin.math.ceil

private const val CL_DEVICE_TYPE_GPU = 1L shl 2
private const val CL_CONTEXT_PLATFORM = 0x1084L
private const val CL_DEVICE_NAME = 0x102B
private const val CL_DEVICE_GLOBAL_MEM_SIZE = 0x101F

private const val CL_MEM_READ_WRITE = 1L shl 0
private const val CL_MEM_WRITE_ONLY = 1L shl 1
private const val CL_MEM_READ_ONLY = 1L shl 2
private const val CL_MEM_COPY_HOST_PTR = 1L shl 5
internal const val CL_KERNEL_WORK_GROUP_SIZE = 0x11B0

internal expect fun isCLSupported(): Boolean

expect class CLPlatformId
expect class CLDeviceId
expect class CLContext
expect class CLCommandQueue
expect class CLMem
expect class CLProgram
expect class CLKernel

expect fun clGetPlatformIDs(): Array<CLPlatformId>
internal expect fun clGetDeviceIDs(platform: CLPlatformId, type: Long): Array<CLDeviceId>
internal expect fun clGetDeviceInfo(device: CLDeviceId, param: Int): ByteArray
internal expect fun clCreateContext(properties: Array<Any>, device: CLDeviceId): CLContext
internal expect fun clReleaseContext(context: CLContext): Int
expect fun clCreateCommandQueue(context: CLContext, device: CLDeviceId): CLCommandQueue
expect fun clReleaseCommandQueue(queue: CLCommandQueue): Int

internal expect fun clReleaseMemObject(mem: CLMem): Int
internal expect fun clReleaseProgram(program: CLProgram): Int
internal expect fun clReleaseKernel(kernel: CLKernel): Int

internal expect fun clCreateBuffer(context: CLContext, usage: Long, size: Long): CLMem
internal expect fun clCreateBuffer(context: CLContext, usage: Long, size: Long, array: FloatArray): CLMem
internal expect fun clCreateBuffer(context: CLContext, usage: Long, size: Long, array: IntArray): CLMem
internal expect fun clCreateBuffer(context: CLContext, usage: Long, size: Long, array: ByteArray): CLMem

internal expect fun clEnqueueReadBuffer(commandQueue: CLCommandQueue, mem: CLMem, blockingRead: Boolean, offset: Long, size: Long, dst: FloatArray): Int
internal expect fun clEnqueueReadBuffer(commandQueue: CLCommandQueue, mem: CLMem, blockingRead: Boolean, offset: Long, size: Long, dst: IntArray): Int
internal expect fun clEnqueueReadBuffer(commandQueue: CLCommandQueue, mem: CLMem, blockingRead: Boolean, offset: Long, size: Long, dst: ByteArray): Int

internal expect fun clEnqueueWriteBuffer(commandQueue: CLCommandQueue, mem: CLMem, blockingRead: Boolean, offset: Long, size: Long, src: FloatArray, srcOffset: Int): Int
internal expect fun clEnqueueWriteBuffer(commandQueue: CLCommandQueue, mem: CLMem, blockingRead: Boolean, offset: Long, size: Long, src: IntArray, srcOffset: Int): Int
internal expect fun clEnqueueWriteBuffer(commandQueue: CLCommandQueue, mem: CLMem, blockingRead: Boolean, offset: Long, size: Long, src: ByteArray, srcOffset: Int): Int

internal expect fun clCreateProgramWithSource(context: CLContext, source: String, error: IntArray): CLProgram
internal expect fun clBuildProgram(program: CLProgram, options: String): Int
internal expect fun clGetProgramBuildInfo(program: CLProgram, device: CLDeviceId): String
internal expect fun clCreateKernel(program: CLProgram, main: String): CLKernel
internal expect fun clGetKernelWorkGroupInfo(kernel: CLKernel, device: CLDeviceId, param: Int): LongArray
internal expect fun clEnqueueNDRangeKernel(commandQueue: CLCommandQueue, kernel: CLKernel, workDim: Int, globalWorkSize: LongArray, localWorkSize: LongArray?): Int

internal expect fun clSetKernelArg(kernel: CLKernel, index: Int, mem: CLMem): Int
internal expect fun clSetKernelArg1f(kernel: CLKernel, index: Int, value: Float): Int
internal expect fun clSetKernelArg1i(kernel: CLKernel, index: Int, value: Int): Int
internal expect fun clSetKernelArg1b(kernel: CLKernel, index: Int, value: Byte): Int

class OpenCL {
    companion object {
        val supported = isCLSupported()
    }

    fun getDevices() =
        clGetDeviceIDs(clGetPlatformIDs()[0], CL_DEVICE_TYPE_GPU)

    fun getDeviceName(device: CLDeviceId) = clGetDeviceInfo(device, CL_DEVICE_NAME).run {
        String(this, 0, this.size-1)
    }

    fun getDeviceMemory(device: CLDeviceId) = clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE).toLong()

    fun createContext(platform: CLPlatformId, device: CLDeviceId) =
        clCreateContext(arrayOf(CL_CONTEXT_PLATFORM, platform), device)

    fun disposeContext(context: CLContext){
        clReleaseContext(context).checkError()
    }

    private fun MemoryUsage.toCL(with: Long = 0) = when(this){
        MemoryUsage.READ_ONLY -> CL_MEM_READ_ONLY
        MemoryUsage.WRITE_ONLY -> CL_MEM_WRITE_ONLY
        MemoryUsage.READ_WRITE -> CL_MEM_READ_WRITE
    } or with

    fun deallocMemory(mem: CLMem) {
        clReleaseMemObject(mem).checkError()
    }

    fun deallocProgram(program: CLProgram) {
        clReleaseProgram(program).checkError()
    }

    fun deallocKernel(kernel: CLKernel) {
        clReleaseKernel(kernel).checkError()
    }

    fun allocate(context: CLContext, size: Int, usage: MemoryUsage) =
        clCreateBuffer(context, usage.toCL(), size.toLong())

    fun wrapFloats(context: CLContext, array: FloatArray, usage: MemoryUsage) = clCreateBuffer(
        context, usage.toCL(CL_MEM_COPY_HOST_PTR),
        array.size.toLong() * Float.SIZE_BYTES, array
    )

    fun wrapInts(context: CLContext, array: IntArray, usage: MemoryUsage) = clCreateBuffer(
        context, usage.toCL(CL_MEM_COPY_HOST_PTR),
        array.size.toLong() * Int.SIZE_BYTES, array
    )

    fun wrapBytes(context: CLContext, array: ByteArray, usage: MemoryUsage) = clCreateBuffer(
        context, usage.toCL(CL_MEM_COPY_HOST_PTR),
        array.size.toLong(), array
    )

    fun readFloats(commandQueue: CLCommandQueue, src: CLMem, length: Int, offset: Int) = FloatArray(length).apply {
        clEnqueueReadBuffer(
            commandQueue, src, true, offset.toLong() * Float.SIZE_BYTES,
            length.toLong() * Float.SIZE_BYTES, this
        ).checkError()
    }

    fun readInts(commandQueue: CLCommandQueue, src: CLMem, length: Int, offset: Int) = IntArray(length).apply {
        clEnqueueReadBuffer(
            commandQueue, src, true, offset.toLong() * Int.SIZE_BYTES,
            length.toLong() * Int.SIZE_BYTES, this
        ).checkError()
    }

    fun readBytes(commandQueue: CLCommandQueue, src: CLMem, length: Int, offset: Int) = ByteArray(length).apply {
        clEnqueueReadBuffer(
            commandQueue, src, true, offset.toLong(),
            length.toLong(), this
        ).checkError()
    }

    fun writeFloats(commandQueue: CLCommandQueue, dst: CLMem, src: FloatArray, length: Int, srcOffset: Int, dstOffset: Int){
        clEnqueueWriteBuffer(
            commandQueue, dst, true, dstOffset.toLong() * Float.SIZE_BYTES,
            length.toLong() * Float.SIZE_BYTES, src, srcOffset
        ).checkError()
    }

    fun writeInts(commandQueue: CLCommandQueue, dst: CLMem, src: IntArray, length: Int, srcOffset: Int, dstOffset: Int){
        clEnqueueWriteBuffer(
            commandQueue, dst, true, dstOffset.toLong() * Int.SIZE_BYTES,
            length.toLong() * Int.SIZE_BYTES, src, srcOffset
        ).checkError()
    }

    fun writeBytes(commandQueue: CLCommandQueue, dst: CLMem, src: ByteArray, length: Int, srcOffset: Int, dstOffset: Int){
        clEnqueueWriteBuffer(
            commandQueue, dst, true, dstOffset.toLong() * Byte.SIZE_BYTES,
            length.toLong() * Byte.SIZE_BYTES, src, srcOffset
        ).checkError()
    }

    fun compileProgram(device: CLDeviceId, context: CLContext, code: String): CLProgram {
        val error = IntArray(1)
        val program = clCreateProgramWithSource(context, code, error)
        if(error[0] != 0)
            throw Exception("Failed to compile OpenCL program: error code: ${error[0]}")
        if(clBuildProgram(program, "-cl-fp32-correctly-rounded-divide-sqrt") != 0)
            throw Exception("Failed to build OpenCL program:\n${clGetProgramBuildInfo(program, device)}")
        return program
    }

    fun createKernel(program: CLProgram, main: String): CLKernel =
        clCreateKernel(program, main)

    fun executeKernel(commandQueue: CLCommandQueue, kernel: CLKernel, maxGroupSize: Long, instances: Long) {
        val count: Long
        val groups: Long
        if(instances < maxGroupSize){
            count = instances
            groups = 1
        }else {
            count = ceil(instances.toDouble() / maxGroupSize).toLong() * maxGroupSize
            groups = maxGroupSize
        }

        clEnqueueNDRangeKernel(commandQueue, kernel, 1,
            longArrayOf(count),
            null
        ).checkError()
    }

    fun setArgument(kernel: CLKernel, index: Int, memory: CLMem) =
        clSetKernelArg(kernel, index, memory).checkError()

    fun setArgument1f(kernel: CLKernel, index: Int, value: Float) =
        clSetKernelArg1f(kernel, index, value).checkError()

    fun setArgument1i(kernel: CLKernel, index: Int, value: Int) =
        clSetKernelArg1i(kernel, index, value).checkError()

    fun setArgument1b(kernel: CLKernel, index: Int, value: Byte) =
        clSetKernelArg1b(kernel, index, value).checkError()

}

private fun Int.checkError(){
    if(this != 0)
        throw Exception(errorToString(this))
}

private fun ByteArray.toLong(): Long {
    var result = 0L
    var shift = 0
    for (byte in this) {
        result = result or (byte.toLong() shl shift)
        shift += 8
    }
    return result
}

private fun errorToString(code: Int) = when(code){
    // run-time and JIT compiler errors
    0 -> "CL_SUCCESS"
    -1 -> "CL_DEVICE_NOT_FOUND"
    -2 -> "CL_DEVICE_NOT_AVAILABLE"
    -3 -> "CL_COMPILER_NOT_AVAILABLE"
    -4 -> "CL_MEM_OBJECT_ALLOCATION_FAILURE"
    -5 -> "CL_OUT_OF_RESOURCES"
    -6 -> "CL_OUT_OF_HOST_MEMORY"
    -7 -> "CL_PROFILING_INFO_NOT_AVAILABLE"
    -8 -> "CL_MEM_COPY_OVERLAP"
    -9 -> "CL_IMAGE_FORMAT_MISMATCH"
    -10 -> "CL_IMAGE_FORMAT_NOT_SUPPORTED"
    -11 -> "CL_BUILD_PROGRAM_FAILURE"
    -12 -> "CL_MAP_FAILURE"
    -13 -> "CL_MISALIGNED_SUB_BUFFER_OFFSET"
    -14 -> "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST"
    -15 -> "CL_COMPILE_PROGRAM_FAILURE"
    -16 -> "CL_LINKER_NOT_AVAILABLE"
    -17 -> "CL_LINK_PROGRAM_FAILURE"
    -18 -> "CL_DEVICE_PARTITION_FAILED"
    -19 -> "CL_KERNEL_ARG_INFO_NOT_AVAILABLE"

    // compile-time errors
    -30 -> "CL_INVALID_VALUE"
    -31 -> "CL_INVALID_DEVICE_TYPE"
    -32 -> "CL_INVALID_PLATFORM"
    -33 -> "CL_INVALID_DEVICE"
    -34 -> "CL_INVALID_CONTEXT"
    -35 -> "CL_INVALID_QUEUE_PROPERTIES"
    -36 -> "CL_INVALID_COMMAND_QUEUE"
    -37 -> "CL_INVALID_HOST_PTR"
    -38 -> "CL_INVALID_MEM_OBJECT"
    -39 -> "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR"
    -40 -> "CL_INVALID_IMAGE_SIZE"
    -41 -> "CL_INVALID_SAMPLER"
    -42 -> "CL_INVALID_BINARY"
    -43 -> "CL_INVALID_BUILD_OPTIONS"
    -44 -> "CL_INVALID_PROGRAM"
    -45 -> "CL_INVALID_PROGRAM_EXECUTABLE"
    -46 -> "CL_INVALID_KERNEL_NAME"
    -47 -> "CL_INVALID_KERNEL_DEFINITION"
    -48 -> "CL_INVALID_KERNEL"
    -49 -> "CL_INVALID_ARG_INDEX"
    -50 -> "CL_INVALID_ARG_VALUE"
    -51 -> "CL_INVALID_ARG_SIZE"
    -52 -> "CL_INVALID_KERNEL_ARGS"
    -53 -> "CL_INVALID_WORK_DIMENSION"
    -54 -> "CL_INVALID_WORK_GROUP_SIZE"
    -55 -> "CL_INVALID_WORK_ITEM_SIZE"
    -56 -> "CL_INVALID_GLOBAL_OFFSET"
    -57 -> "CL_INVALID_EVENT_WAIT_LIST"
    -58 -> "CL_INVALID_EVENT"
    -59 -> "CL_INVALID_OPERATION"
    -60 -> "CL_INVALID_GL_OBJECT"
    -61 -> "CL_INVALID_BUFFER_SIZE"
    -62 -> "CL_INVALID_MIP_LEVEL"
    -63 -> "CL_INVALID_GLOBAL_WORK_SIZE"
    -64 -> "CL_INVALID_PROPERTY"
    -65 -> "CL_INVALID_IMAGE_DESCRIPTOR"
    -66 -> "CL_INVALID_COMPILER_OPTIONS"
    -67 -> "CL_INVALID_LINKER_OPTIONS"
    -68 -> "CL_INVALID_DEVICE_PARTITION_COUNT"

    // extension errors
    -1000 -> "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR"
    -1001 -> "CL_PLATFORM_NOT_FOUND_KHR"
    -1002 -> "CL_INVALID_D3D10_DEVICE_KHR"
    -1003 -> "CL_INVALID_D3D10_RESOURCE_KHR"
    -1004 -> "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR"
    -1005 -> "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR"
    else -> "Unknown"
}
