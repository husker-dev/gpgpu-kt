package org.example.com.huskerdev.ta.engine.gpu

import org.jocl.*

class OpenCL {

    val context: cl_context
    val commandQueue: cl_command_queue

    init {
        // The platform, device type and device number
        // that will be used
        val platformIndex = 0
        val deviceType = CL.CL_DEVICE_TYPE_ALL
        val deviceIndex = 0

        // Enable exceptions and subsequently omit error checks in this sample
        CL.setExceptionsEnabled(true)

        // Obtain the number of platforms
        val numPlatformsArray = IntArray(1)
        CL.clGetPlatformIDs(0, null, numPlatformsArray)
        val numPlatforms = numPlatformsArray[0]

        // Obtain a platform ID
        val platforms = arrayOfNulls<cl_platform_id>(numPlatforms)
        CL.clGetPlatformIDs(platforms.size, platforms, null)
        val platform = platforms[platformIndex]

        // Initialize the context properties
        val contextProperties = cl_context_properties()
        contextProperties.addProperty(CL.CL_CONTEXT_PLATFORM.toLong(), platform)


        // Obtain the number of devices for the platform
        val numDevicesArray = IntArray(1)
        CL.clGetDeviceIDs(platform, deviceType, 0, null, numDevicesArray)
        val numDevices = numDevicesArray[0]


        // Obtain a device ID
        val devices = arrayOfNulls<cl_device_id>(numDevices)
        CL.clGetDeviceIDs(platform, deviceType, numDevices, devices, null)
        val device = devices[deviceIndex]

        // Create a context for the selected device
        context = CL.clCreateContext(
            contextProperties, 1, arrayOf(device),
            null, null, null
        )

        commandQueue = CL.clCreateCommandQueue(context, device, 0, null)
    }

    fun run(src: String, arrays: Array<FloatArray>): FloatArray{
        val length = arrays.minOf { it.size }.toLong()
        val globalWorkSize = longArrayOf(length)
        val localWorkSize = longArrayOf(1)

        // Compile
        val program = CL.clCreateProgramWithSource(
            context,
            1, arrayOf(src), null, null
        )
        CL.clBuildProgram(program, 0, null, null, null, null)

        val kernel = CL.clCreateKernel(program, "kernelMain", null)

        // Cast arrays
        val inputBuffers = arrayOfNulls<cl_mem>(arrays.size)
        arrays.forEachIndexed { i, array ->
            inputBuffers[i] = CL.clCreateBuffer(
                context,
                CL.CL_MEM_READ_ONLY or CL.CL_MEM_COPY_HOST_PTR,
                Sizeof.cl_float * length, Pointer.to(array), null
            )

            CL.clSetKernelArg(
                kernel, i,
                Sizeof.cl_mem.toLong(), Pointer.to(inputBuffers[i])
            )
        }
        val outputBuffer = CL.clCreateBuffer(
            context,
            CL.CL_MEM_READ_WRITE,
            Sizeof.cl_float * length, null, null
        )
        CL.clSetKernelArg(
            kernel, inputBuffers.size,
            Sizeof.cl_mem.toLong(), Pointer.to(outputBuffer)
        )

        // Execute
        CL.clEnqueueNDRangeKernel(
            commandQueue, kernel, 1, null,
            globalWorkSize, localWorkSize, 0, null, null
        )

        // Read
        val result = FloatArray(length.toInt())
        CL.clEnqueueReadBuffer(
            commandQueue, outputBuffer, CL.CL_TRUE, 0,
            length * Sizeof.cl_float, Pointer.to(result), 0, null, null
        )

        // Release
        inputBuffers.forEach { CL.clReleaseMemObject(it) }
        CL.clReleaseMemObject(outputBuffer)
        CL.clReleaseKernel(kernel)
        CL.clReleaseProgram(program)

        return result
    }

}