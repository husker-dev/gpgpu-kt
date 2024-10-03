package com.huskerdev.gpkt.engines.opencl

import com.huskerdev.gpkt.MemoryUsage
import com.huskerdev.gpkt.utils.*
import org.lwjgl.opencl.CL10.*
import org.lwjgl.opencl.CL20.clCreateCommandQueueWithProperties
import java.nio.Buffer
import java.nio.ByteBuffer
import java.nio.DoubleBuffer
import java.nio.FloatBuffer
import java.nio.IntBuffer
import kotlin.math.ceil
import kotlin.math.max
import kotlin.math.min

class OpenCL(
    requestedDeviceId: Int
) {
    companion object {
        const val DEVICE_TYPE = CL_DEVICE_TYPE_ALL.toLong()

        val supported = try{
            useStack {
                val platform = mallocPointer(1)
                clGetPlatformIDs(platform, null as IntBuffer?)

                val numDevices = mallocInt(1)
                clGetDeviceIDs(platform[0], DEVICE_TYPE, null, numDevices)

                numDevices[0] > 0
            }
        }catch (e: Exception){
            e.printStackTrace()
            false
        }
    }

    var deviceId: Int = 0
    private var device: Long = 0
    private var context: Long = 0
    private var commandQueue: Long = 0
    lateinit var deviceName: String

    init {
        useStack {
            val platform = mallocPointer(1)
            clGetPlatformIDs(platform, null as IntBuffer?)

            val numDevices = mallocInt(1)
            clGetDeviceIDs(platform[0], DEVICE_TYPE, null, numDevices)

            val devices = mallocPointer(numDevices[0])
            clGetDeviceIDs(platform[0], DEVICE_TYPE, devices, null as IntBuffer?)

            deviceId = max(0, min(requestedDeviceId, numDevices[0]))
            device = devices[deviceId]

            val nameLength = mallocPointer(1)
            clGetDeviceInfo(device, CL_DEVICE_NAME, null as ByteBuffer?, nameLength)

            val nameBytes = malloc(nameLength[0].toInt())
            clGetDeviceInfo(device, CL_DEVICE_NAME, nameBytes, null)
            deviceName = nameBytes.readArray().decodeToString(endIndex = nameLength[0].toInt() - 1)

            val contextProperties = mallocPointer(3)
            contextProperties.put(0, CL_CONTEXT_PLATFORM.toLong())
            contextProperties.put(1, platform[0])
            contextProperties.put(2, 0)

            context = clCreateContext(
                contextProperties, device,
                null, 0, null
            )
            commandQueue = clCreateCommandQueueWithProperties(
                context, device,
                null, null as IntBuffer?)
        }
    }

    private fun MemoryUsage.toCL(with: Int = 0) = when(this){
        MemoryUsage.READ_ONLY -> CL_MEM_READ_ONLY
        MemoryUsage.WRITE_ONLY -> CL_MEM_WRITE_ONLY
        MemoryUsage.READ_WRITE -> CL_MEM_READ_WRITE
    }.toLong() or with.toLong()

    fun allocate(pointer: Buffer, usage: MemoryUsage) = when(pointer){
        is IntBuffer -> clCreateBuffer(context, usage.toCL(CL_MEM_COPY_HOST_PTR), pointer, null as IntBuffer?)
        is DoubleBuffer -> clCreateBuffer(context, usage.toCL(CL_MEM_COPY_HOST_PTR), pointer, null as IntBuffer?)
        is FloatBuffer -> clCreateBuffer(context, usage.toCL(CL_MEM_COPY_HOST_PTR), pointer, null as IntBuffer?)
        is ByteBuffer -> clCreateBuffer(context, usage.toCL(CL_MEM_COPY_HOST_PTR), pointer, null as IntBuffer?)
        else -> throw UnsupportedOperationException()
    }

    fun allocate(size: Long, usage: MemoryUsage) =
        nclCreateBuffer(
            context, usage.toCL(),
            size, 0, 0
        )

    fun deallocMemory(mem: Long) {
        clReleaseMemObject(mem)
    }

    fun deallocProgram(program: Long) {
        clReleaseProgram(program)
    }

    fun deallocKernel(kernel: Long) {
        clReleaseKernel(kernel)
    }

    fun read(src: Long, dst: Buffer, srcOffset: Long) = when(dst) {
        is IntBuffer ->
            clEnqueueReadBuffer(commandQueue, src, true, srcOffset, dst, null, null)
        is FloatBuffer ->
            clEnqueueReadBuffer(commandQueue, src, true, srcOffset, dst, null, null)
        is DoubleBuffer ->
            clEnqueueReadBuffer(commandQueue, src, true, srcOffset, dst, null, null)
        is ByteBuffer ->
            clEnqueueReadBuffer(commandQueue, src, true, srcOffset, dst, null, null)
        else -> throw UnsupportedOperationException()
    }

    fun write(dst: Long, src: Buffer, dstOffset: Long) = when(src){
        is IntBuffer ->
            clEnqueueWriteBuffer(commandQueue, dst, true, dstOffset, src, null, null)
        is FloatBuffer ->
            clEnqueueWriteBuffer(commandQueue, dst, true, dstOffset, src, null, null)
        is DoubleBuffer ->
            clEnqueueWriteBuffer(commandQueue, dst, true, dstOffset, src, null, null)
        is ByteBuffer ->
            clEnqueueWriteBuffer(commandQueue, dst, true, dstOffset, src, null, null)
        else -> throw UnsupportedOperationException()
    }

    fun compileProgram(code: String): Long = useStack {
        val error = mallocInt(1)
        println(code)
        val program = /*clCreateProgramWithSource(context, code, error)*/
        clCreateProgramWithSource(context, pointers(bytes(*code.toByteArray(), 0)), pointers(code.length.toLong() + 1), error)

        if(error[0] != 0)
            println("[ERROR] Failed to compile OpenCL program (error code: ${error[0]})")
        clBuildProgram(program, 0, "", null, 0)
        return program
    }

    fun createKernel(clProgram: Long, main: String): Long =
        clCreateKernel(clProgram, main, null as IntBuffer?)

    fun executeKernel(kernel: Long, workGroupSize: Long) = useStack {
        val maxGroupSizeBuffer = mallocInt(1)
        clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_WORK_GROUP_SIZE, maxGroupSizeBuffer, null)
        val maxGroupSize = maxGroupSizeBuffer[0]

        val count: Long
        val groups: Long
        if(workGroupSize < maxGroupSize){
            count = workGroupSize
            groups = 1
        }else {
            count = ceil(workGroupSize.toDouble() / maxGroupSize).toLong() * maxGroupSize
            groups = maxGroupSize.toLong()
        }

        clEnqueueNDRangeKernel(commandQueue, kernel, 1,
            null,
            mallocPointer(1).put(count),
            mallocPointer(1).put(groups),
            null, null)
    }

    fun setArgument(kernel: Long, index: Int, size: Long, memory: Long) = useStack {
        nclSetKernelArg(kernel, index, size, memory)
    }

    fun setArgument(kernel: Long, index: Int, buffer: Buffer) = when(buffer) {
        is IntBuffer -> clSetKernelArg(kernel, index, buffer)
        is FloatBuffer -> clSetKernelArg(kernel, index, buffer)
        is DoubleBuffer -> clSetKernelArg(kernel, index, buffer)
        is ByteBuffer -> clSetKernelArg(kernel, index, buffer)
        else -> throw UnsupportedOperationException()
    }
}