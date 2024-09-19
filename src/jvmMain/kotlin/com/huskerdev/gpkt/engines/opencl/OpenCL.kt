package com.huskerdev.gpkt.engines.opencl

import org.jocl.*
import org.jocl.CL.*

class OpenCL {

    private val context: cl_context
    private val commandQueue: cl_command_queue

    init {
        val platformIndex = 0
        val deviceType = CL_DEVICE_TYPE_ALL
        val deviceIndex = 0

        // Enable exceptions and subsequently omit error checks in this sample
        setExceptionsEnabled(true)

        // Obtain the number of platforms
        val numPlatformsArray = IntArray(1)
        clGetPlatformIDs(0, null, numPlatformsArray)
        val numPlatforms = numPlatformsArray[0]

        // Obtain a platform ID
        val platforms = arrayOfNulls<cl_platform_id>(numPlatforms)
        clGetPlatformIDs(platforms.size, platforms, null)
        val platform = platforms[platformIndex]

        // Initialize the context properties
        val contextProperties = cl_context_properties()
        contextProperties.addProperty(CL_CONTEXT_PLATFORM.toLong(), platform)

        // Obtain the number of devices for the platform
        val numDevicesArray = IntArray(1)
        clGetDeviceIDs(platform, deviceType, 0, null, numDevicesArray)
        val numDevices = numDevicesArray[0]

        // Obtain a device ID
        val devices = arrayOfNulls<cl_device_id>(numDevices)
        clGetDeviceIDs(platform, deviceType, numDevices, devices, null)
        val device = devices[deviceIndex]

        // Create a context for the selected device
        context = clCreateContext(
            contextProperties, 1, arrayOf(device),
            null, null, null
        )
        commandQueue = clCreateCommandQueueWithProperties(context, device, null, null)
    }

    fun allocate(array: FloatArray): cl_mem = clCreateBuffer(
        context,
        CL_MEM_READ_WRITE or CL_MEM_COPY_HOST_PTR,
        (Sizeof.cl_float * array.size).toLong(), Pointer.to(array),
        null
    )

    fun allocate(length: Int): cl_mem = clCreateBuffer(
        context,
        CL_MEM_READ_WRITE,
        (Sizeof.cl_float * length).toLong(), null,
        null
    )

    fun dealloc(mem: cl_mem) =
        clReleaseMemObject(mem)

    fun read(mem: cl_mem, length: Int) = FloatArray(length).apply {
        clEnqueueReadBuffer(
            commandQueue, mem, CL_TRUE, 0,
            (length * Sizeof.cl_float).toLong(), Pointer.to(this),
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

    fun createKernel(clProgram: cl_program, main: String) =
        clCreateKernel(clProgram, main, null)

    fun executeKernel(kernel: cl_kernel, workGroupSize: Long) {
        clEnqueueNDRangeKernel(commandQueue, kernel, 1, null,
            longArrayOf(workGroupSize), longArrayOf(1), 0, null, null)
    }

    fun setArgument(kernel: cl_kernel, index: Int, source: OCLSource){
        clSetKernelArg(kernel, index, Sizeof.cl_mem.toLong(), Pointer.to(source.data))
    }
}