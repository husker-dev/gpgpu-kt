package com.huskerdev.gpkt.apis.opencl

import com.huskerdev.gpkt.MemoryUsage
import kotlin.math.ceil

private const val CL_DEVICE_TYPE_ALL = 0xFFFFFFFFL
private const val CL_CONTEXT_PLATFORM = 0x1084L
private const val CL_DEVICE_NAME = 0x102B

private const val CL_MEM_READ_WRITE = (1L shl 0)
private const val CL_MEM_WRITE_ONLY = (1L shl 1)
private const val CL_MEM_READ_ONLY = (1L shl 2)
private const val CL_MEM_COPY_HOST_PTR = (1L shl 5)
private const val CL_KERNEL_WORK_GROUP_SIZE = 0x11B0

internal expect fun isCLSupported(): Boolean
internal expect fun createCL(): OpenCL

abstract class CLPlatformId
abstract class CLDeviceId
abstract class CLContext
abstract class CLCommandQueue
abstract class CLMem
abstract class CLProgram
abstract class CLKernel

abstract class OpenCL {
    companion object {
        val supported = isCLSupported()
    }

    abstract fun clGetPlatformIDs(): Array<CLPlatformId>
    abstract fun clGetDeviceIDs(platform: CLPlatformId, type: Long): Array<CLDeviceId>
    abstract fun clGetDeviceInfo(device: CLDeviceId, param: Int): ByteArray
    abstract fun clCreateContext(properties: Array<Any>, device: CLDeviceId): CLContext
    abstract fun clReleaseContext(context: CLContext)
    abstract fun clCreateCommandQueue(context: CLContext, device: CLDeviceId): CLCommandQueue

    abstract fun clReleaseMemObject(mem: CLMem)
    abstract fun clReleaseProgram(program: CLProgram)
    abstract fun clReleaseKernel(kernel: CLKernel)

    abstract fun clCreateBuffer(context: CLContext, usage: Long, size: Long): CLMem
    abstract fun clCreateBuffer(context: CLContext, usage: Long, size: Long, array: FloatArray): CLMem
    abstract fun clCreateBuffer(context: CLContext, usage: Long, size: Long, array: IntArray): CLMem
    abstract fun clCreateBuffer(context: CLContext, usage: Long, size: Long, array: ByteArray): CLMem

    abstract fun clEnqueueReadBuffer(commandQueue: CLCommandQueue, mem: CLMem, blockingRead: Boolean, offset: Long, size: Long, dst: FloatArray)
    abstract fun clEnqueueReadBuffer(commandQueue: CLCommandQueue, mem: CLMem, blockingRead: Boolean, offset: Long, size: Long, dst: IntArray)
    abstract fun clEnqueueReadBuffer(commandQueue: CLCommandQueue, mem: CLMem, blockingRead: Boolean, offset: Long, size: Long, dst: ByteArray)

    abstract fun clEnqueueWriteBuffer(commandQueue: CLCommandQueue, mem: CLMem, blockingRead: Boolean, offset: Long, size: Long, src: FloatArray, srcOffset: Int)
    abstract fun clEnqueueWriteBuffer(commandQueue: CLCommandQueue, mem: CLMem, blockingRead: Boolean, offset: Long, size: Long, src: IntArray, srcOffset: Int)
    abstract fun clEnqueueWriteBuffer(commandQueue: CLCommandQueue, mem: CLMem, blockingRead: Boolean, offset: Long, size: Long, src: ByteArray, srcOffset: Int)

    abstract fun clCreateProgramWithSource(context: CLContext, source: String, error: IntArray): CLProgram
    abstract fun clBuildProgram(program: CLProgram): Int
    abstract fun clGetProgramBuildInfo(program: CLProgram, device: CLDeviceId): String
    abstract fun clCreateKernel(program: CLProgram, main: String): CLKernel
    abstract fun clGetKernelWorkGroupInfo(kernel: CLKernel, device: CLDeviceId, param: Int): LongArray
    abstract fun clEnqueueNDRangeKernel(commandQueue: CLCommandQueue, kernel: CLKernel, workDim: Int, globalWorkSize: LongArray, localWorkSize: LongArray)

    abstract fun clSetKernelArg(kernel: CLKernel, index: Int, mem: CLMem)
    abstract fun clSetKernelArg1f(kernel: CLKernel, index: Int, value: Float)
    abstract fun clSetKernelArg1d(kernel: CLKernel, index: Int, value: Double)
    abstract fun clSetKernelArg1i(kernel: CLKernel, index: Int, value: Int)
    abstract fun clSetKernelArg1b(kernel: CLKernel, index: Int, value: Byte)

    fun getDevices() =
        clGetDeviceIDs(clGetPlatformIDs()[0], CL_DEVICE_TYPE_ALL)

    fun getDeviceName(device: CLDeviceId) = clGetDeviceInfo(device, CL_DEVICE_NAME).run {
        String(this, 0, this.size-1)
    }

    fun createContext(platform: CLPlatformId, device: CLDeviceId) =
        clCreateContext(arrayOf(CL_CONTEXT_PLATFORM, platform), device)

    fun disposeContext(context: CLContext){
        clReleaseContext(context)
    }

    private fun MemoryUsage.toCL(with: Long = 0) = when(this){
        MemoryUsage.READ_ONLY -> CL_MEM_READ_ONLY
        MemoryUsage.WRITE_ONLY -> CL_MEM_WRITE_ONLY
        MemoryUsage.READ_WRITE -> CL_MEM_READ_WRITE
    } or with

    fun deallocMemory(mem: CLMem) {
        clReleaseMemObject(mem)
    }

    fun deallocProgram(program: CLProgram) {
        clReleaseProgram(program)
    }

    fun deallocKernel(kernel: CLKernel) {
        clReleaseKernel(kernel)
    }

    fun allocate(context: CLContext, size: Int, usage: MemoryUsage): CLMem = clCreateBuffer(
        context, usage.toCL(), size.toLong()
    )

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
            commandQueue, src, true, offset.toLong(),
            length.toLong() * Float.SIZE_BYTES, this
        )
    }

    fun readInts(commandQueue: CLCommandQueue, src: CLMem, length: Int, offset: Int) = IntArray(length).apply {
        clEnqueueReadBuffer(
            commandQueue, src, true, offset.toLong(),
            length.toLong() * Int.SIZE_BYTES, this
        )
    }

    fun readBytes(commandQueue: CLCommandQueue, src: CLMem, length: Int, offset: Int) = ByteArray(length).apply {
        clEnqueueReadBuffer(
            commandQueue, src, true, offset.toLong(),
            length.toLong(), this
        )
    }


    fun writeFloats(commandQueue: CLCommandQueue, dst: CLMem, src: FloatArray, length: Int, srcOffset: Int, dstOffset: Int){
        clEnqueueWriteBuffer(
            commandQueue, dst, true, dstOffset.toLong(),
            length.toLong() * Float.SIZE_BYTES, src, srcOffset
        )
    }

    fun writeInts(commandQueue: CLCommandQueue, dst: CLMem, src: IntArray, length: Int, srcOffset: Int, dstOffset: Int){
        clEnqueueWriteBuffer(
            commandQueue, dst, true, dstOffset.toLong(),
            length.toLong() * Int.SIZE_BYTES, src, srcOffset
        )
    }

    fun writeBytes(commandQueue: CLCommandQueue, dst: CLMem, src: ByteArray, length: Int, srcOffset: Int, dstOffset: Int){
        clEnqueueWriteBuffer(
            commandQueue, dst, true, dstOffset.toLong(),
            length.toLong() * Byte.SIZE_BYTES, src, srcOffset
        )
    }

    fun compileProgram(device: CLDeviceId, context: CLContext, code: String): CLProgram {
        val error = IntArray(1)
        val program = clCreateProgramWithSource(context, code, error)
        if(error[0] != 0)
            throw Exception("Failed to compile OpenCL program: error code: ${error[0]}")
        if(clBuildProgram(program) != 0)
            throw Exception("Failed to build OpenCL program:\n${clGetProgramBuildInfo(program, device)}")
        return program
    }

    fun createKernel(program: CLProgram, main: String): CLKernel =
        clCreateKernel(program, main)

    fun executeKernel(commandQueue: CLCommandQueue, kernel: CLKernel, device: CLDeviceId, workGroupSize: Long) {
        val maxGroupSize = clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_WORK_GROUP_SIZE)[0]

        val count: Long
        val groups: Long
        if(workGroupSize < maxGroupSize){
            count = workGroupSize
            groups = 1
        }else {
            count = ceil(workGroupSize.toDouble() / maxGroupSize).toLong() * maxGroupSize
            groups = maxGroupSize
        }

        clEnqueueNDRangeKernel(commandQueue, kernel, 1,
            longArrayOf(count),
            longArrayOf(groups))
    }

    fun setArgument(kernel: CLKernel, index: Int, memory: CLMem) =
        clSetKernelArg(kernel, index, memory)

    fun setArgument1f(kernel: CLKernel, index: Int, value: Float) =
        clSetKernelArg1f(kernel, index, value)

    fun setArgument1d(kernel: CLKernel, index: Int, value: Double) =
        clSetKernelArg1d(kernel, index, value)

    fun setArgument1i(kernel: CLKernel, index: Int, value: Int) =
        clSetKernelArg1i(kernel, index, value)

    fun setArgument1b(kernel: CLKernel, index: Int, value: Byte) =
        clSetKernelArg1b(kernel, index, value)

}