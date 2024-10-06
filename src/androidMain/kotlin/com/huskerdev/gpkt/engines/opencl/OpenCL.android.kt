package com.huskerdev.gpkt.engines.opencl

import com.huskerdev.gpkt.engines.opencl.OpenCLBindings.Companion.nBuildProgram
import com.huskerdev.gpkt.engines.opencl.OpenCLBindings.Companion.nCreateBuffer
import com.huskerdev.gpkt.engines.opencl.OpenCLBindings.Companion.nCreateByteBuffer
import com.huskerdev.gpkt.engines.opencl.OpenCLBindings.Companion.nCreateCommandQueue
import com.huskerdev.gpkt.engines.opencl.OpenCLBindings.Companion.nCreateContext
import com.huskerdev.gpkt.engines.opencl.OpenCLBindings.Companion.nCreateDoubleBuffer
import com.huskerdev.gpkt.engines.opencl.OpenCLBindings.Companion.nCreateFloatBuffer
import com.huskerdev.gpkt.engines.opencl.OpenCLBindings.Companion.nCreateIntBuffer
import com.huskerdev.gpkt.engines.opencl.OpenCLBindings.Companion.nCreateKernel
import com.huskerdev.gpkt.engines.opencl.OpenCLBindings.Companion.nCreateProgram
import com.huskerdev.gpkt.engines.opencl.OpenCLBindings.Companion.nEnqueueNDRangeKernel
import com.huskerdev.gpkt.engines.opencl.OpenCLBindings.Companion.nGetBuildInfo
import com.huskerdev.gpkt.engines.opencl.OpenCLBindings.Companion.nGetDeviceInfo
import com.huskerdev.gpkt.engines.opencl.OpenCLBindings.Companion.nGetDevices
import com.huskerdev.gpkt.engines.opencl.OpenCLBindings.Companion.nGetKernelWorkGroupInfo
import com.huskerdev.gpkt.engines.opencl.OpenCLBindings.Companion.nGetPlatforms
import com.huskerdev.gpkt.engines.opencl.OpenCLBindings.Companion.nReadByteBuffer
import com.huskerdev.gpkt.engines.opencl.OpenCLBindings.Companion.nReadDoubleBuffer
import com.huskerdev.gpkt.engines.opencl.OpenCLBindings.Companion.nReadFloatBuffer
import com.huskerdev.gpkt.engines.opencl.OpenCLBindings.Companion.nReadIntBuffer
import com.huskerdev.gpkt.engines.opencl.OpenCLBindings.Companion.nReleaseKernel
import com.huskerdev.gpkt.engines.opencl.OpenCLBindings.Companion.nReleaseMem
import com.huskerdev.gpkt.engines.opencl.OpenCLBindings.Companion.nReleaseProgram
import com.huskerdev.gpkt.engines.opencl.OpenCLBindings.Companion.nSetKernelArg
import com.huskerdev.gpkt.engines.opencl.OpenCLBindings.Companion.nSetKernelArg1b
import com.huskerdev.gpkt.engines.opencl.OpenCLBindings.Companion.nSetKernelArg1d
import com.huskerdev.gpkt.engines.opencl.OpenCLBindings.Companion.nSetKernelArg1f
import com.huskerdev.gpkt.engines.opencl.OpenCLBindings.Companion.nSetKernelArg1i
import com.huskerdev.gpkt.engines.opencl.OpenCLBindings.Companion.nTryInitCL
import com.huskerdev.gpkt.engines.opencl.OpenCLBindings.Companion.nWriteByteBuffer
import com.huskerdev.gpkt.engines.opencl.OpenCLBindings.Companion.nWriteDoubleBuffer
import com.huskerdev.gpkt.engines.opencl.OpenCLBindings.Companion.nWriteFloatBuffer
import com.huskerdev.gpkt.engines.opencl.OpenCLBindings.Companion.nWriteIntBuffer


class OpenCLBindings {
    companion object {
        @JvmStatic external fun nTryInitCL(): Boolean
        @JvmStatic external fun nGetPlatforms(): LongArray
        @JvmStatic external fun nGetDevices(platform: Long, type: Long): LongArray
        @JvmStatic external fun nGetDeviceInfo(device: Long, param: Int): ByteArray
        @JvmStatic external fun nCreateContext(properties: LongArray, device: Long): Long
        @JvmStatic external fun nCreateCommandQueue(context: Long, device: Long): Long

        @JvmStatic external fun nReleaseMem(buffer: Long)
        @JvmStatic external fun nReleaseProgram(program: Long)
        @JvmStatic external fun nReleaseKernel(kernel: Long)

        @JvmStatic external fun nCreateBuffer(context: Long, flags: Long, size: Long): Long
        @JvmStatic external fun nCreateFloatBuffer(context: Long, flags: Long, array: FloatArray): Long
        @JvmStatic external fun nCreateDoubleBuffer(context: Long, flags: Long, array: DoubleArray): Long
        @JvmStatic external fun nCreateIntBuffer(context: Long, flags: Long, array: IntArray): Long
        @JvmStatic external fun nCreateByteBuffer(context: Long, flags: Long, array: ByteArray): Long

        @JvmStatic external fun nReadFloatBuffer(commandQueue: Long, mem: Long, blockingRad: Boolean, offset: Long, size: Long, dst: FloatArray)
        @JvmStatic external fun nReadDoubleBuffer(commandQueue: Long, mem: Long, blockingRad: Boolean, offset: Long, size: Long, dst: DoubleArray)
        @JvmStatic external fun nReadIntBuffer(commandQueue: Long, mem: Long, blockingRad: Boolean, offset: Long, size: Long, dst: IntArray)
        @JvmStatic external fun nReadByteBuffer(commandQueue: Long, mem: Long, blockingRad: Boolean, offset: Long, size: Long, dst: ByteArray)

        @JvmStatic external fun nWriteFloatBuffer(commandQueue: Long, mem: Long, blockingRad: Boolean, offset: Long, size: Long, src: FloatArray, srcOffset: Long)
        @JvmStatic external fun nWriteDoubleBuffer(commandQueue: Long, mem: Long, blockingRad: Boolean, offset: Long, size: Long, src: DoubleArray, srcOffset: Long)
        @JvmStatic external fun nWriteIntBuffer(commandQueue: Long, mem: Long, blockingRad: Boolean, offset: Long, size: Long, src: IntArray, srcOffset: Long)
        @JvmStatic external fun nWriteByteBuffer(commandQueue: Long, mem: Long, blockingRad: Boolean, offset: Long, size: Long, src: ByteArray, srcOffset: Long)

        @JvmStatic external fun nCreateProgram(context: Long, program: String, error: IntArray): Long
        @JvmStatic external fun nBuildProgram(program: Long): Int
        @JvmStatic external fun nGetBuildInfo(program: Long, device: Long): String
        @JvmStatic external fun nCreateKernel(program: Long, main: String): Long
        @JvmStatic external fun nGetKernelWorkGroupInfo(program: Long, device: Long, param: Int): LongArray
        @JvmStatic external fun nEnqueueNDRangeKernel(commandQueue: Long, kernel: Long, workDim: Int, globalWorkSize: LongArray, localWorkSize: LongArray)

        @JvmStatic external fun nSetKernelArg(kernel: Long, index: Int, mem: Long)
        @JvmStatic external fun nSetKernelArg1f(kernel: Long, index: Int, value: Float)
        @JvmStatic external fun nSetKernelArg1d(kernel: Long, index: Int, value: Double)
        @JvmStatic external fun nSetKernelArg1i(kernel: Long, index: Int, value: Int)
        @JvmStatic external fun nSetKernelArg1b(kernel: Long, index: Int, value: Byte)

        init {
            System.loadLibrary("gpgpu-kt")
        }
    }
}

internal actual fun isCLSupported(): Boolean = try {
    nTryInitCL()
}catch (e: Exception){
    false
}

class NCLPlatformId(val ptr: Long): CLPlatformId()
class NCLDeviceId(val ptr: Long): CLDeviceId()
class NCLContext(val ptr: Long): CLContext()
class NCLCommandQueue(val ptr: Long): CLCommandQueue()
class NCLMem(val ptr: Long): CLMem()
class NCLProgram(val ptr: Long): CLProgram()
class NCLKernel(val ptr: Long): CLKernel()

private val CLPlatformId.ptr: Long
    get() = (this as NCLPlatformId).ptr
private val CLDeviceId.ptr: Long
    get() = (this as NCLDeviceId).ptr
private val CLContext.ptr: Long
    get() = (this as NCLContext).ptr
private val CLCommandQueue.ptr: Long
    get() = (this as NCLCommandQueue).ptr
private val CLMem.ptr: Long
    get() = (this as NCLMem).ptr
private val CLProgram.ptr: Long
    get() = (this as NCLProgram).ptr
private val CLKernel.ptr: Long
    get() = (this as NCLKernel).ptr


internal actual fun createCL(requestedDeviceId: Int): OpenCL = object: OpenCL(requestedDeviceId) {
    override fun clGetPlatformIDs(): Array<CLPlatformId> =
        nGetPlatforms().map { NCLPlatformId(it) }.toTypedArray()

    override fun clGetDeviceIDs(platform: CLPlatformId, type: Long): Array<CLDeviceId> =
        nGetDevices(platform.ptr, type).map { NCLDeviceId(it) }.toTypedArray()

    override fun clGetDeviceInfo(device: CLDeviceId, param: Int) =
        nGetDeviceInfo(device.ptr, param)

    override fun clCreateContext(properties: Array<Any>, device: CLDeviceId) =
        NCLContext(nCreateContext(properties.map {
            when (it) {
                is Number -> it.toLong()
                is CLPlatformId -> it.ptr
                else -> throw UnsupportedOperationException()
            }
        }.toLongArray(), device.ptr))

    override fun clCreateCommandQueue(context: CLContext, device: CLDeviceId) =
        NCLCommandQueue(nCreateCommandQueue(context.ptr, device.ptr))

    override fun clReleaseMemObject(mem: CLMem) =
        nReleaseMem(mem.ptr)

    override fun clReleaseProgram(program: CLProgram) =
        nReleaseProgram(program.ptr)

    override fun clReleaseKernel(kernel: CLKernel) =
        nReleaseKernel(kernel.ptr)

    override fun clCreateBuffer(context: CLContext, usage: Long, size: Long) =
        NCLMem(nCreateBuffer(context.ptr, usage, size))

    override fun clCreateBuffer(context: CLContext, usage: Long, size: Long, array: FloatArray) =
        NCLMem(nCreateFloatBuffer(context.ptr, usage, array))

    override fun clCreateBuffer(context: CLContext, usage: Long, size: Long, array: DoubleArray) =
        NCLMem(nCreateDoubleBuffer(context.ptr, usage, array))

    override fun clCreateBuffer(context: CLContext, usage: Long, size: Long, array: IntArray) =
        NCLMem(nCreateIntBuffer(context.ptr, usage, array))

    override fun clCreateBuffer(context: CLContext, usage: Long, size: Long, array: ByteArray) =
        NCLMem(nCreateByteBuffer(context.ptr, usage, array))

    override fun clEnqueueReadBuffer(
        commandQueue: CLCommandQueue,
        mem: CLMem,
        blockingRead: Boolean,
        offset: Long,
        size: Long,
        dst: FloatArray
    ) = nReadFloatBuffer(commandQueue.ptr, mem.ptr, blockingRead, offset, size, dst)

    override fun clEnqueueReadBuffer(
        commandQueue: CLCommandQueue,
        mem: CLMem,
        blockingRead: Boolean,
        offset: Long,
        size: Long,
        dst: DoubleArray
    )  = nReadDoubleBuffer(commandQueue.ptr, mem.ptr, blockingRead, offset, size, dst)

    override fun clEnqueueReadBuffer(
        commandQueue: CLCommandQueue,
        mem: CLMem,
        blockingRead: Boolean,
        offset: Long,
        size: Long,
        dst: IntArray
    ) = nReadIntBuffer(commandQueue.ptr, mem.ptr, blockingRead, offset, size, dst)

    override fun clEnqueueReadBuffer(
        commandQueue: CLCommandQueue,
        mem: CLMem,
        blockingRead: Boolean,
        offset: Long,
        size: Long,
        dst: ByteArray
    ) = nReadByteBuffer(commandQueue.ptr, mem.ptr, blockingRead, offset, size, dst)

    override fun clEnqueueWriteBuffer(
        commandQueue: CLCommandQueue,
        mem: CLMem,
        blockingRead: Boolean,
        offset: Long,
        size: Long,
        src: FloatArray,
        srcOffset: Int
    ) = nWriteFloatBuffer(commandQueue.ptr, mem.ptr, blockingRead, offset, size, src, srcOffset.toLong())

    override fun clEnqueueWriteBuffer(
        commandQueue: CLCommandQueue,
        mem: CLMem,
        blockingRead: Boolean,
        offset: Long,
        size: Long,
        src: DoubleArray,
        srcOffset: Int
    ) = nWriteDoubleBuffer(commandQueue.ptr, mem.ptr, blockingRead, offset, size, src, srcOffset.toLong())

    override fun clEnqueueWriteBuffer(
        commandQueue: CLCommandQueue,
        mem: CLMem,
        blockingRead: Boolean,
        offset: Long,
        size: Long,
        src: IntArray,
        srcOffset: Int
    ) = nWriteIntBuffer(commandQueue.ptr, mem.ptr, blockingRead, offset, size, src, srcOffset.toLong())

    override fun clEnqueueWriteBuffer(
        commandQueue: CLCommandQueue,
        mem: CLMem,
        blockingRead: Boolean,
        offset: Long,
        size: Long,
        src: ByteArray,
        srcOffset: Int
    ) = nWriteByteBuffer(commandQueue.ptr, mem.ptr, blockingRead, offset, size, src, srcOffset.toLong())

    override fun clCreateProgramWithSource(context: CLContext, source: String, error: IntArray) =
        NCLProgram(nCreateProgram(context.ptr, source, error))

    override fun clBuildProgram(program: CLProgram) =
        nBuildProgram(program.ptr)

    override fun clGetProgramBuildInfo(program: CLProgram, device: CLDeviceId) =
        nGetBuildInfo(program.ptr, device.ptr)

    override fun clCreateKernel(program: CLProgram, main: String) =
        NCLKernel(nCreateKernel(program.ptr, main))

    override fun clGetKernelWorkGroupInfo(kernel: CLKernel, device: CLDeviceId, param: Int) =
        nGetKernelWorkGroupInfo(kernel.ptr, device.ptr, param)

    override fun clEnqueueNDRangeKernel(
        commandQueue: CLCommandQueue,
        kernel: CLKernel,
        workDim: Int,
        globalWorkSize: LongArray,
        localWorkSize: LongArray
    ) = nEnqueueNDRangeKernel(commandQueue.ptr, kernel.ptr, workDim, globalWorkSize, localWorkSize)

    override fun clSetKernelArg(kernel: CLKernel, index: Int, mem: CLMem) =
        nSetKernelArg(kernel.ptr, index, mem.ptr)

    override fun clSetKernelArg1f(kernel: CLKernel, index: Int, value: Float) =
        nSetKernelArg1f(kernel.ptr, index, value)

    override fun clSetKernelArg1d(kernel: CLKernel, index: Int, value: Double) =
        nSetKernelArg1d(kernel.ptr, index, value)

    override fun clSetKernelArg1i(kernel: CLKernel, index: Int, value: Int) =
        nSetKernelArg1i(kernel.ptr, index, value)

    override fun clSetKernelArg1b(kernel: CLKernel, index: Int, value: Byte) =
        nSetKernelArg1b(kernel.ptr, index, value)

}