package com.huskerdev.gpkt.engines.opencl

import org.jocl.*
import org.jocl.CL.*

actual fun isCLSupported(): Boolean = try{
    val numPlatformsArray = IntArray(1)
    clGetPlatformIDs(0, null, numPlatformsArray)
    val numPlatforms = numPlatformsArray[0]

    val platforms = arrayOfNulls<cl_platform_id>(numPlatforms)
    clGetPlatformIDs(platforms.size, platforms, null)
    val platform = platforms[0]

    val contextProperties = cl_context_properties()
    contextProperties.addProperty(CL_CONTEXT_PLATFORM.toLong(), platform)

    val numDevicesArray = IntArray(1)
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, null, numDevicesArray)
    val numDevices = numDevicesArray[0]

    numDevices > 0
}catch (e: Exception){
    false
}

class JOCLPlatformId(val ptr: cl_platform_id): CLPlatformId()
class JOCLDeviceId(val ptr: cl_device_id): CLDeviceId()
class JOCLContext(val ptr: cl_context): CLContext()
class JOCLCommandQueue(val ptr: cl_command_queue): CLCommandQueue()
class JOCLMem(val ptr: cl_mem): CLMem()
class JOCLProgram(val ptr: cl_program): CLProgram()
class JOCLKernel(val ptr: cl_kernel): CLKernel()


private val CLPlatformId.ptr: cl_platform_id
    get() = (this as JOCLPlatformId).ptr
private val CLDeviceId.ptr: cl_device_id
    get() = (this as JOCLDeviceId).ptr
private val CLContext.ptr: cl_context
    get() = (this as JOCLContext).ptr
private val CLCommandQueue.ptr: cl_command_queue
    get() = (this as JOCLCommandQueue).ptr
private val CLMem.ptr: cl_mem
    get() = (this as JOCLMem).ptr
private val CLProgram.ptr: cl_program
    get() = (this as JOCLProgram).ptr
private val CLKernel.ptr: cl_kernel
    get() = (this as JOCLKernel).ptr

internal actual fun createCL(requestedDeviceId: Int): OpenCL = object: OpenCL(requestedDeviceId){

    override fun clGetPlatformIDs(): Array<CLPlatformId> {
        val numPlatformsArray = IntArray(1)
        clGetPlatformIDs(0, null, numPlatformsArray)
        val numPlatforms = numPlatformsArray[0]

        val platforms = arrayOfNulls<cl_platform_id>(numPlatforms)
        clGetPlatformIDs(platforms.size, platforms, null)

        return platforms.map { JOCLPlatformId(it!!) }.toTypedArray()
    }

    override fun clGetDeviceIDs(platform: CLPlatformId, type: Long): Array<CLDeviceId> {
        val numDevicesArray = IntArray(1)
        clGetDeviceIDs(platform.ptr, type, 0, null, numDevicesArray)
        val numDevices = numDevicesArray[0]

        val devices = arrayOfNulls<cl_device_id>(numDevices)
        clGetDeviceIDs(platform.ptr, type, numDevices, devices, null)

        return devices.map { JOCLDeviceId(it!!) }.toTypedArray()
    }

    override fun clGetDeviceInfo(device: CLDeviceId, param: Int): ByteArray {
        val buffer = LongArray(1)
        clGetDeviceInfo(device.ptr, param, 0, null, buffer)
        val nameBuffer = ByteArray(buffer[0].toInt())
        clGetDeviceInfo(device.ptr, param, buffer[0], Pointer.to(nameBuffer), null)
        return nameBuffer
    }

    override fun clCreateContext(properties: Array<Any>, device: CLDeviceId): CLContext {
        val contextProperties = cl_context_properties()
        contextProperties.addProperty(
            properties[0] as Long,
            (properties[1] as CLPlatformId).ptr
        )
        return JOCLContext(clCreateContext(
            contextProperties, 1, arrayOf(device.ptr),
            null, null, null
        ))
    }

    override fun clCreateCommandQueue(context: CLContext, device: CLDeviceId): CLCommandQueue {
        return JOCLCommandQueue(clCreateCommandQueue(
            context.ptr, device.ptr, 0, null
        ))
    }

    override fun clReleaseMemObject(mem: CLMem) {
        clReleaseMemObject(mem.ptr)
    }

    override fun clReleaseProgram(program: CLProgram) {
        clReleaseProgram(program.ptr)
    }

    override fun clReleaseKernel(kernel: CLKernel) {
        clReleaseKernel(kernel.ptr)
    }

    override fun clCreateBuffer(context: CLContext, usage: Long, size: Long) =
        JOCLMem(clCreateBuffer(context.ptr, usage, size, null, null))

    override fun clCreateBuffer(context: CLContext, usage: Long, size: Long, array: FloatArray) =
        JOCLMem(clCreateBuffer(context.ptr, usage, size, Pointer.to(array), null))

    override fun clCreateBuffer(context: CLContext, usage: Long, size: Long, array: IntArray) =
        JOCLMem(clCreateBuffer(context.ptr, usage, size, Pointer.to(array), null))

    override fun clCreateBuffer(context: CLContext, usage: Long, size: Long, array: ByteArray) =
        JOCLMem(clCreateBuffer(context.ptr, usage, size, Pointer.to(array), null))

    override fun clEnqueueReadBuffer(
        commandQueue: CLCommandQueue,
        mem: CLMem,
        blockingRead: Boolean,
        offset: Long,
        size: Long,
        dst: FloatArray
    ) {
        clEnqueueReadBuffer(
            commandQueue.ptr, mem.ptr, blockingRead, offset,
            size, Pointer.to(dst),
            0, null, null
        )
    }

    override fun clEnqueueReadBuffer(
        commandQueue: CLCommandQueue,
        mem: CLMem,
        blockingRead: Boolean,
        offset: Long,
        size: Long,
        dst: IntArray
    ) {
        clEnqueueReadBuffer(
            commandQueue.ptr, mem.ptr, blockingRead, offset,
            size, Pointer.to(dst),
            0, null, null
        )
    }

    override fun clEnqueueReadBuffer(
        commandQueue: CLCommandQueue,
        mem: CLMem,
        blockingRead: Boolean,
        offset: Long,
        size: Long,
        dst: ByteArray
    ) {
        clEnqueueReadBuffer(
            commandQueue.ptr, mem.ptr, blockingRead, offset,
            size, Pointer.to(dst),
            0, null, null
        )
    }

    override fun clEnqueueWriteBuffer(
        commandQueue: CLCommandQueue,
        mem: CLMem,
        blockingRead: Boolean,
        offset: Long,
        size: Long,
        src: FloatArray,
        srcOffset: Int
    ) {
        clEnqueueWriteBuffer(
            commandQueue.ptr, mem.ptr, blockingRead, offset,
            size, Pointer.to(src).withByteOffset(srcOffset.toLong()),
            0, null, null
        )
    }

    override fun clEnqueueWriteBuffer(
        commandQueue: CLCommandQueue,
        mem: CLMem,
        blockingRead: Boolean,
        offset: Long,
        size: Long,
        src: IntArray,
        srcOffset: Int
    ) {
        clEnqueueWriteBuffer(
            commandQueue.ptr, mem.ptr, blockingRead, offset,
            size, Pointer.to(src).withByteOffset(srcOffset.toLong()),
            0, null, null
        )
    }

    override fun clEnqueueWriteBuffer(
        commandQueue: CLCommandQueue,
        mem: CLMem,
        blockingRead: Boolean,
        offset: Long,
        size: Long,
        src: ByteArray,
        srcOffset: Int
    ) {
        clEnqueueWriteBuffer(
            commandQueue.ptr, mem.ptr, blockingRead, offset,
            size, Pointer.to(src).withByteOffset(srcOffset.toLong()),
            0, null, null
        )
    }

    override fun clCreateProgramWithSource(context: CLContext, source: String, error: IntArray) =
        JOCLProgram(clCreateProgramWithSource(context.ptr, 1, arrayOf(source), null, error))

    override fun clBuildProgram(program: CLProgram) =
        clBuildProgram(program.ptr, 0, null, null, null, null)

    override fun clGetProgramBuildInfo(program: CLProgram, device: CLDeviceId): String {
        val size = LongArray(1)
        clGetProgramBuildInfo(program.ptr, device.ptr, CL_PROGRAM_BUILD_LOG, 0, null, size)

        val buffer = ByteArray(size[0].toInt())
        clGetProgramBuildInfo(program.ptr, device.ptr, CL_PROGRAM_BUILD_LOG, size[0], Pointer.to(buffer), null)
        return String(buffer, 0, buffer.lastIndex)
    }

    override fun clCreateKernel(program: CLProgram, main: String) =
        JOCLKernel(clCreateKernel(program.ptr, main, null))

    override fun clGetKernelWorkGroupInfo(kernel: CLKernel, device: CLDeviceId, param: Int): LongArray {
        val buffer = LongArray(1)
        clGetKernelWorkGroupInfo(kernel.ptr, device.ptr, param, 0, null, buffer)
        val info = LongArray(buffer[0].toInt())
        clGetKernelWorkGroupInfo(kernel.ptr, device.ptr, param, buffer[0], Pointer.to(info), null)
        return info
    }

    override fun clEnqueueNDRangeKernel(
        commandQueue: CLCommandQueue,
        kernel: CLKernel,
        workDim: Int,
        globalWorkSize: LongArray,
        localWorkSize: LongArray
    ) {
        clEnqueueNDRangeKernel(
            commandQueue.ptr, kernel.ptr, workDim,
            null,
            globalWorkSize,
            localWorkSize,
            0, null, null)
    }

    override fun clSetKernelArg(kernel: CLKernel, index: Int, mem: CLMem) {
        clSetKernelArg(kernel.ptr, index, Sizeof.cl_mem.toLong(), Pointer.to(mem.ptr))
    }

    override fun clSetKernelArg1f(kernel: CLKernel, index: Int, value: Float) {
        clSetKernelArg(kernel.ptr, index, Sizeof.cl_float.toLong(), Pointer.to(floatArrayOf(value)))
    }

    override fun clSetKernelArg1d(kernel: CLKernel, index: Int, value: Double) {
        clSetKernelArg(kernel.ptr, index, Sizeof.cl_double.toLong(), Pointer.to(doubleArrayOf(value)))
    }

    override fun clSetKernelArg1i(kernel: CLKernel, index: Int, value: Int) {
        clSetKernelArg(kernel.ptr, index, Sizeof.cl_int.toLong(), Pointer.to(intArrayOf(value)))
    }

    override fun clSetKernelArg1b(kernel: CLKernel, index: Int, value: Byte) {
        clSetKernelArg(kernel.ptr, index, Sizeof.cl_char.toLong(), Pointer.to(byteArrayOf(value)))
    }


}