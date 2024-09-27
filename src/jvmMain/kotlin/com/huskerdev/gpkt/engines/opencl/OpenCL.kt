package com.huskerdev.gpkt.engines.opencl

import org.jocl.*
import org.jocl.CL.*
import kotlin.math.ceil
import kotlin.math.max
import kotlin.math.min

class OpenCL(
    requestedDeviceId: Int
) {
    companion object {
        const val DEVICE_TYPE = CL_DEVICE_TYPE_ALL

        val supported = try{
            // Obtain the number of platforms
            val numPlatformsArray = IntArray(1)
            clGetPlatformIDs(0, null, numPlatformsArray)
            val numPlatforms = numPlatformsArray[0]

            // Obtain a platform ID
            val platforms = arrayOfNulls<cl_platform_id>(numPlatforms)
            clGetPlatformIDs(platforms.size, platforms, null)
            val platform = platforms[0]

            // Initialize the context properties
            val contextProperties = cl_context_properties()
            contextProperties.addProperty(CL_CONTEXT_PLATFORM.toLong(), platform)

            // Obtain the number of devices for the platform
            val numDevicesArray = IntArray(1)
            clGetDeviceIDs(platform, DEVICE_TYPE, 0, null, numDevicesArray)
            val numDevices = numDevicesArray[0]

            numDevices > 0
        }catch (e: Exception){
            false
        }
    }

    val deviceId: Int
    private val device: cl_device_id
    private val context: cl_context
    private val commandQueue: cl_command_queue
    val deviceName: String

    init {
        // Enable exceptions and subsequently omit error checks in this sample
        setExceptionsEnabled(true)

        // Obtain the number of platforms
        val numPlatformsArray = IntArray(1)
        clGetPlatformIDs(0, null, numPlatformsArray)
        val numPlatforms = numPlatformsArray[0]

        // Obtain a platform ID
        val platforms = arrayOfNulls<cl_platform_id>(numPlatforms)
        clGetPlatformIDs(platforms.size, platforms, null)
        val platform = platforms[0]

        // Initialize the context properties
        val contextProperties = cl_context_properties()
        contextProperties.addProperty(CL_CONTEXT_PLATFORM.toLong(), platform)

        // Obtain the number of devices for the platform
        val numDevicesArray = IntArray(1)
        clGetDeviceIDs(platform, DEVICE_TYPE, 0, null, numDevicesArray)
        val numDevices = numDevicesArray[0]

        deviceId = max(0, min(requestedDeviceId, numDevices))

        // Obtain a device ID
        val devices = arrayOfNulls<cl_device_id>(numDevices)
        clGetDeviceIDs(platform, DEVICE_TYPE, numDevices, devices, null)
        device = devices[deviceId]!!

        val buffer = LongArray(1)
        clGetDeviceInfo(device, CL_DEVICE_NAME, 0, null, buffer)
        val nameBuffer = ByteArray(buffer[0].toInt())
        clGetDeviceInfo(device, CL_DEVICE_NAME, buffer[0], Pointer.to(nameBuffer), null)
        deviceName = String(nameBuffer, 0, nameBuffer.size-1)

        // Create a context for the selected device
        context = clCreateContext(
            contextProperties, 1, arrayOf(device),
            null, null, null
        )
        commandQueue = clCreateCommandQueueWithProperties(context, device, null, null)
    }

    fun allocate(pointer: Pointer, size: Long): cl_mem = clCreateBuffer(
        context, CL_MEM_READ_WRITE or CL_MEM_COPY_HOST_PTR,
        size, pointer, null
    )

    fun allocate(size: Int): cl_mem = clCreateBuffer(
        context, CL_MEM_READ_WRITE,
        size.toLong(), null, null
    )

    fun dealloc(mem: cl_mem) {
        clReleaseMemObject(mem)
    }

    fun dealloc(program: cl_program) {
        clReleaseProgram(program)
    }

    fun dealloc(kernel: cl_kernel) {
        clReleaseKernel(kernel)
    }

    fun read(ptr: cl_mem, size: Long, dst: Pointer) {
        clEnqueueReadBuffer(
            commandQueue, ptr, CL_TRUE, 0,
            size, dst,
            0, null, null
        )
    }

    fun compileProgram(code: String): cl_program{
        val error = IntArray(1)
        val program = clCreateProgramWithSource(context, 1, arrayOf(code), null, error)
        if(error[0] != 0)
            println("[ERROR] Failed to compile OpenCL program (error code: ${error[0]})")
        clBuildProgram(program, 0, null, null, null, null)
        return program
    }

    fun createKernel(clProgram: cl_program, main: String): cl_kernel =
        clCreateKernel(clProgram, main, null)

    fun executeKernel(kernel: cl_kernel, workGroupSize: Long) {
        val buffer = LongArray(1)
        clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_WORK_GROUP_SIZE, 0, null, buffer)
        val maxWorkGroupBuffer = LongArray(buffer[0].toInt())
        clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_WORK_GROUP_SIZE, buffer[0], Pointer.to(maxWorkGroupBuffer), null)
        val maxGroupSize = maxWorkGroupBuffer[0]

        val count: Long
        val groups: Long
        if(workGroupSize < maxGroupSize){
            count = workGroupSize
            groups = 1
        }else {
            count = ceil(workGroupSize.toDouble() / maxGroupSize).toLong() * maxGroupSize
            groups = maxGroupSize
        }

        clEnqueueNDRangeKernel(commandQueue, kernel, 1, null,
            longArrayOf(count),
            longArrayOf(groups),
            0, null, null)
    }

    fun setArgument(kernel: cl_kernel, index: Int, memory: cl_mem){
        clSetKernelArg(kernel, index, Sizeof.cl_mem.toLong(), Pointer.to(memory))
    }
}