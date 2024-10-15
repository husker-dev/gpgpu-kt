package com.huskerdev.gpkt.apis.opencl

import com.huskerdev.gpkt.apis.opencl.OpenCLBindings.Companion.nBuildProgram
import com.huskerdev.gpkt.apis.opencl.OpenCLBindings.Companion.nCreateBuffer
import com.huskerdev.gpkt.apis.opencl.OpenCLBindings.Companion.nCreateByteBuffer
import com.huskerdev.gpkt.apis.opencl.OpenCLBindings.Companion.nCreateCommandQueue
import com.huskerdev.gpkt.apis.opencl.OpenCLBindings.Companion.nCreateContext
import com.huskerdev.gpkt.apis.opencl.OpenCLBindings.Companion.nCreateFloatBuffer
import com.huskerdev.gpkt.apis.opencl.OpenCLBindings.Companion.nCreateIntBuffer
import com.huskerdev.gpkt.apis.opencl.OpenCLBindings.Companion.nCreateKernel
import com.huskerdev.gpkt.apis.opencl.OpenCLBindings.Companion.nCreateProgram
import com.huskerdev.gpkt.apis.opencl.OpenCLBindings.Companion.nEnqueueNDRangeKernel
import com.huskerdev.gpkt.apis.opencl.OpenCLBindings.Companion.nGetBuildInfo
import com.huskerdev.gpkt.apis.opencl.OpenCLBindings.Companion.nGetDeviceInfo
import com.huskerdev.gpkt.apis.opencl.OpenCLBindings.Companion.nGetDevices
import com.huskerdev.gpkt.apis.opencl.OpenCLBindings.Companion.nGetKernelWorkGroupInfo
import com.huskerdev.gpkt.apis.opencl.OpenCLBindings.Companion.nGetPlatforms
import com.huskerdev.gpkt.apis.opencl.OpenCLBindings.Companion.nReadByteBuffer
import com.huskerdev.gpkt.apis.opencl.OpenCLBindings.Companion.nReadFloatBuffer
import com.huskerdev.gpkt.apis.opencl.OpenCLBindings.Companion.nReadIntBuffer
import com.huskerdev.gpkt.apis.opencl.OpenCLBindings.Companion.nReleaseContext
import com.huskerdev.gpkt.apis.opencl.OpenCLBindings.Companion.nReleaseKernel
import com.huskerdev.gpkt.apis.opencl.OpenCLBindings.Companion.nReleaseMem
import com.huskerdev.gpkt.apis.opencl.OpenCLBindings.Companion.nReleaseProgram
import com.huskerdev.gpkt.apis.opencl.OpenCLBindings.Companion.nSetKernelArg
import com.huskerdev.gpkt.apis.opencl.OpenCLBindings.Companion.nSetKernelArg1b
import com.huskerdev.gpkt.apis.opencl.OpenCLBindings.Companion.nSetKernelArg1f
import com.huskerdev.gpkt.apis.opencl.OpenCLBindings.Companion.nSetKernelArg1i
import com.huskerdev.gpkt.apis.opencl.OpenCLBindings.Companion.nTryInitCL
import com.huskerdev.gpkt.apis.opencl.OpenCLBindings.Companion.nWriteByteBuffer
import com.huskerdev.gpkt.apis.opencl.OpenCLBindings.Companion.nWriteFloatBuffer
import com.huskerdev.gpkt.apis.opencl.OpenCLBindings.Companion.nWriteIntBuffer


class OpenCLBindings {
    companion object {
        @JvmStatic external fun nTryInitCL(): Boolean
        @JvmStatic external fun nGetPlatforms(): LongArray
        @JvmStatic external fun nGetDevices(platform: Long, type: Long): LongArray
        @JvmStatic external fun nGetDeviceInfo(device: Long, param: Int): ByteArray
        @JvmStatic external fun nCreateContext(properties: LongArray, device: Long): Long
        @JvmStatic external fun nReleaseContext(context: Long)
        @JvmStatic external fun nCreateCommandQueue(context: Long, device: Long): Long

        @JvmStatic external fun nReleaseMem(buffer: Long)
        @JvmStatic external fun nReleaseProgram(program: Long)
        @JvmStatic external fun nReleaseKernel(kernel: Long)

        @JvmStatic external fun nCreateBuffer(context: Long, flags: Long, size: Long): Long
        @JvmStatic external fun nCreateFloatBuffer(context: Long, flags: Long, array: FloatArray): Long
        @JvmStatic external fun nCreateIntBuffer(context: Long, flags: Long, array: IntArray): Long
        @JvmStatic external fun nCreateByteBuffer(context: Long, flags: Long, array: ByteArray): Long

        @JvmStatic external fun nReadFloatBuffer(commandQueue: Long, mem: Long, blockingRad: Boolean, offset: Long, size: Long, dst: FloatArray)
        @JvmStatic external fun nReadIntBuffer(commandQueue: Long, mem: Long, blockingRad: Boolean, offset: Long, size: Long, dst: IntArray)
        @JvmStatic external fun nReadByteBuffer(commandQueue: Long, mem: Long, blockingRad: Boolean, offset: Long, size: Long, dst: ByteArray)

        @JvmStatic external fun nWriteFloatBuffer(commandQueue: Long, mem: Long, blockingRad: Boolean, offset: Long, size: Long, src: FloatArray, srcOffset: Long)
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

actual class CLPlatformId(val ptr: Long)
actual class CLDeviceId(val ptr: Long)
actual class CLContext(val ptr: Long)
actual class CLCommandQueue(val ptr: Long)
actual class CLMem(val ptr: Long)
actual class CLProgram(val ptr: Long)
actual class CLKernel(val ptr: Long)


actual fun clGetPlatformIDs(): Array<CLPlatformId> =
    nGetPlatforms().map { CLPlatformId(it) }.toTypedArray()

internal actual fun clGetDeviceIDs(platform: CLPlatformId, type: Long): Array<CLDeviceId> =
    nGetDevices(platform.ptr, type).map { CLDeviceId(it) }.toTypedArray()

internal actual fun clGetDeviceInfo(device: CLDeviceId, param: Int) =
    nGetDeviceInfo(device.ptr, param)

internal actual fun clCreateContext(properties: Array<Any>, device: CLDeviceId) =
    CLContext(nCreateContext(properties.map {
        when (it) {
            is Number -> it.toLong()
            is CLPlatformId -> it.ptr
            else -> throw UnsupportedOperationException()
        }
    }.toLongArray(), device.ptr))

internal actual fun clReleaseContext(context: CLContext) =
    nReleaseContext(context.ptr)

actual fun clCreateCommandQueue(context: CLContext, device: CLDeviceId) =
    CLCommandQueue(nCreateCommandQueue(context.ptr, device.ptr))

internal actual fun clReleaseMemObject(mem: CLMem) =
    nReleaseMem(mem.ptr)

internal actual fun clReleaseProgram(program: CLProgram) =
    nReleaseProgram(program.ptr)

internal actual fun clReleaseKernel(kernel: CLKernel) =
    nReleaseKernel(kernel.ptr)

internal actual fun clCreateBuffer(context: CLContext, usage: Long, size: Long) =
    CLMem(nCreateBuffer(context.ptr, usage, size))

internal actual fun clCreateBuffer(context: CLContext, usage: Long, size: Long, array: FloatArray) =
    CLMem(nCreateFloatBuffer(context.ptr, usage, array))

internal actual fun clCreateBuffer(context: CLContext, usage: Long, size: Long, array: IntArray) =
    CLMem(nCreateIntBuffer(context.ptr, usage, array))

internal actual fun clCreateBuffer(context: CLContext, usage: Long, size: Long, array: ByteArray) =
    CLMem(nCreateByteBuffer(context.ptr, usage, array))

internal actual fun clEnqueueReadBuffer(
    commandQueue: CLCommandQueue,
    mem: CLMem,
    blockingRead: Boolean,
    offset: Long,
    size: Long,
    dst: FloatArray
) = nReadFloatBuffer(commandQueue.ptr, mem.ptr, blockingRead, offset, size, dst)

internal actual fun clEnqueueReadBuffer(
    commandQueue: CLCommandQueue,
    mem: CLMem,
    blockingRead: Boolean,
    offset: Long,
    size: Long,
    dst: IntArray
) = nReadIntBuffer(commandQueue.ptr, mem.ptr, blockingRead, offset, size, dst)

internal actual fun clEnqueueReadBuffer(
    commandQueue: CLCommandQueue,
    mem: CLMem,
    blockingRead: Boolean,
    offset: Long,
    size: Long,
    dst: ByteArray
) = nReadByteBuffer(commandQueue.ptr, mem.ptr, blockingRead, offset, size, dst)

internal actual fun clEnqueueWriteBuffer(
    commandQueue: CLCommandQueue,
    mem: CLMem,
    blockingRead: Boolean,
    offset: Long,
    size: Long,
    src: FloatArray,
    srcOffset: Int
) = nWriteFloatBuffer(commandQueue.ptr, mem.ptr, blockingRead, offset, size, src, srcOffset.toLong())

internal actual fun clEnqueueWriteBuffer(
    commandQueue: CLCommandQueue,
    mem: CLMem,
    blockingRead: Boolean,
    offset: Long,
    size: Long,
    src: IntArray,
    srcOffset: Int
) = nWriteIntBuffer(commandQueue.ptr, mem.ptr, blockingRead, offset, size, src, srcOffset.toLong())

internal actual fun clEnqueueWriteBuffer(
    commandQueue: CLCommandQueue,
    mem: CLMem,
    blockingRead: Boolean,
    offset: Long,
    size: Long,
    src: ByteArray,
    srcOffset: Int
) = nWriteByteBuffer(commandQueue.ptr, mem.ptr, blockingRead, offset, size, src, srcOffset.toLong())

internal actual fun clCreateProgramWithSource(context: CLContext, source: String, error: IntArray) =
    CLProgram(nCreateProgram(context.ptr, source, error))

internal actual fun clBuildProgram(program: CLProgram) =
    nBuildProgram(program.ptr)

internal actual fun clGetProgramBuildInfo(program: CLProgram, device: CLDeviceId) =
    nGetBuildInfo(program.ptr, device.ptr)

internal actual fun clCreateKernel(program: CLProgram, main: String) =
    CLKernel(nCreateKernel(program.ptr, main))

internal actual fun clGetKernelWorkGroupInfo(kernel: CLKernel, device: CLDeviceId, param: Int) =
    nGetKernelWorkGroupInfo(kernel.ptr, device.ptr, param)

internal actual fun clEnqueueNDRangeKernel(
    commandQueue: CLCommandQueue,
    kernel: CLKernel,
    workDim: Int,
    globalWorkSize: LongArray,
    localWorkSize: LongArray
) = nEnqueueNDRangeKernel(commandQueue.ptr, kernel.ptr, workDim, globalWorkSize, localWorkSize)

internal actual fun clSetKernelArg(kernel: CLKernel, index: Int, mem: CLMem) =
    nSetKernelArg(kernel.ptr, index, mem.ptr)

internal actual fun clSetKernelArg1f(kernel: CLKernel, index: Int, value: Float) =
    nSetKernelArg1f(kernel.ptr, index, value)

internal actual fun clSetKernelArg1i(kernel: CLKernel, index: Int, value: Int) =
    nSetKernelArg1i(kernel.ptr, index, value)

internal actual fun clSetKernelArg1b(kernel: CLKernel, index: Int, value: Byte) =
    nSetKernelArg1b(kernel.ptr, index, value)

