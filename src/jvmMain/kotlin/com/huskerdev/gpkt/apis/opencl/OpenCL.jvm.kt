package com.huskerdev.gpkt.apis.opencl

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

actual class CLPlatformId(val ptr: cl_platform_id)
actual class CLDeviceId(val ptr: cl_device_id)
actual class CLContext(val ptr: cl_context)
actual class CLCommandQueue(val ptr: cl_command_queue)
actual class CLMem(val ptr: cl_mem)
actual class CLProgram(val ptr: cl_program)
actual class CLKernel(val ptr: cl_kernel)


internal actual fun clGetPlatformIDs(): Array<CLPlatformId> {
    val numPlatformsArray = IntArray(1)
    clGetPlatformIDs(0, null, numPlatformsArray)
    val numPlatforms = numPlatformsArray[0]

    val platforms = arrayOfNulls<cl_platform_id>(numPlatforms)
    clGetPlatformIDs(platforms.size, platforms, null)

    return platforms.map { CLPlatformId(it!!) }.toTypedArray()
}

internal actual fun clGetDeviceIDs(platform: CLPlatformId, type: Long): Array<CLDeviceId> {
    val numDevicesArray = IntArray(1)
    clGetDeviceIDs(platform.ptr, type, 0, null, numDevicesArray)
    val numDevices = numDevicesArray[0]

    val devices = arrayOfNulls<cl_device_id>(numDevices)
    clGetDeviceIDs(platform.ptr, type, numDevices, devices, null)

    return devices.map { CLDeviceId(it!!) }.toTypedArray()
}

internal actual fun clGetDeviceInfo(device: CLDeviceId, param: Int): ByteArray {
    val buffer = LongArray(1)
    clGetDeviceInfo(device.ptr, param, 0, null, buffer)
    val nameBuffer = ByteArray(buffer[0].toInt())
    clGetDeviceInfo(device.ptr, param, buffer[0], Pointer.to(nameBuffer), null)
    return nameBuffer
}

internal actual fun clCreateContext(properties: Array<Any>, device: CLDeviceId): CLContext {
    val contextProperties = cl_context_properties()
    contextProperties.addProperty(
        properties[0] as Long,
        (properties[1] as CLPlatformId).ptr
    )
    return CLContext(clCreateContext(
        contextProperties, 1, arrayOf(device.ptr),
        null, null, null
    ))
}

internal actual fun clReleaseContext(context: CLContext) {
    clReleaseContext(context.ptr)
}

internal actual fun clCreateCommandQueue(context: CLContext, device: CLDeviceId) =
    CLCommandQueue(clCreateCommandQueue(
        context.ptr, device.ptr, 0, null
    ))

internal actual fun clReleaseMemObject(mem: CLMem) {
    clReleaseMemObject(mem.ptr)
}

internal actual fun clReleaseProgram(program: CLProgram) {
    clReleaseProgram(program.ptr)
}

internal actual fun clReleaseKernel(kernel: CLKernel) {
    clReleaseKernel(kernel.ptr)
}

internal actual fun clCreateBuffer(context: CLContext, usage: Long, size: Long) =
    CLMem(clCreateBuffer(context.ptr, usage, size, null, null))

internal actual fun clCreateBuffer(context: CLContext, usage: Long, size: Long, array: FloatArray) =
    CLMem(clCreateBuffer(context.ptr, usage, size, Pointer.to(array), null))

internal actual fun clCreateBuffer(context: CLContext, usage: Long, size: Long, array: IntArray) =
    CLMem(clCreateBuffer(context.ptr, usage, size, Pointer.to(array), null))

internal actual fun clCreateBuffer(context: CLContext, usage: Long, size: Long, array: ByteArray) =
    CLMem(clCreateBuffer(context.ptr, usage, size, Pointer.to(array), null))

internal actual fun clEnqueueReadBuffer(
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

internal actual fun clEnqueueReadBuffer(
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

internal actual fun clEnqueueReadBuffer(
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

internal actual fun clEnqueueWriteBuffer(
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

internal actual fun clEnqueueWriteBuffer(
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

internal actual fun clEnqueueWriteBuffer(
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

internal actual fun clCreateProgramWithSource(context: CLContext, source: String, error: IntArray) =
    CLProgram(clCreateProgramWithSource(context.ptr, 1, arrayOf(source), null, error))

internal actual fun clBuildProgram(program: CLProgram) =
    clBuildProgram(program.ptr, 0, null, null, null, null)

internal actual fun clGetProgramBuildInfo(program: CLProgram, device: CLDeviceId): String {
    val size = LongArray(1)
    clGetProgramBuildInfo(program.ptr, device.ptr, CL_PROGRAM_BUILD_LOG, 0, null, size)

    val buffer = ByteArray(size[0].toInt())
    clGetProgramBuildInfo(program.ptr, device.ptr, CL_PROGRAM_BUILD_LOG, size[0], Pointer.to(buffer), null)
    return String(buffer, 0, buffer.lastIndex)
}

internal actual fun clCreateKernel(program: CLProgram, main: String) =
    CLKernel(clCreateKernel(program.ptr, main, null))

internal actual fun clGetKernelWorkGroupInfo(kernel: CLKernel, device: CLDeviceId, param: Int): LongArray {
    val buffer = LongArray(1)
    clGetKernelWorkGroupInfo(kernel.ptr, device.ptr, param, 0, null, buffer)
    val info = LongArray(buffer[0].toInt())
    clGetKernelWorkGroupInfo(kernel.ptr, device.ptr, param, buffer[0], Pointer.to(info), null)
    return info
}

internal actual fun clEnqueueNDRangeKernel(
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

internal actual fun clSetKernelArg(kernel: CLKernel, index: Int, mem: CLMem) {
    clSetKernelArg(kernel.ptr, index, Sizeof.cl_mem.toLong(), Pointer.to(mem.ptr))
}

internal actual fun clSetKernelArg1f(kernel: CLKernel, index: Int, value: Float) {
    clSetKernelArg(kernel.ptr, index, Sizeof.cl_float.toLong(), Pointer.to(floatArrayOf(value)))
}

internal actual fun clSetKernelArg1i(kernel: CLKernel, index: Int, value: Int) {
    clSetKernelArg(kernel.ptr, index, Sizeof.cl_int.toLong(), Pointer.to(intArrayOf(value)))
}

internal actual fun clSetKernelArg1b(kernel: CLKernel, index: Int, value: Byte) {
    clSetKernelArg(kernel.ptr, index, Sizeof.cl_char.toLong(), Pointer.to(byteArrayOf(value)))
}
