package com.huskerdev.gpkt.engines.cuda


import com.huskerdev.gpkt.utils.*
import org.lwjgl.PointerBuffer
import org.lwjgl.cuda.CU.*
import org.lwjgl.cuda.NVRTC.*
import java.nio.Buffer
import java.nio.ByteBuffer
import java.nio.DoubleBuffer
import java.nio.FloatBuffer
import java.nio.IntBuffer
import kotlin.math.max
import kotlin.math.min


class Cuda(
    requestedDeviceId: Int
) {
    companion object {
        val supported = try {
            useStack {
                cuInit(0)
                val deviceCount = mallocInt(1)
                cuDeviceGetCount(deviceCount)
                if (deviceCount[0] == 0) {
                    println("[INFO] CUDA is supported, but could not find supported devices.")
                    false
                } else true
            }
        }catch (e: Exception){
            println("[INFO] Failed to load CUDA. Check toolkit installation.")
            false
        }
    }

    var deviceId: Int = 0
    private var device: Int = 0
    private var context: Long = 0

    lateinit var deviceName: String
    private var maxBlockDimX: Int = 0

    init {
        useStack {
            val deviceCount = mallocInt(1)
            cuDeviceGetCount(deviceCount)

            deviceId = max(0, min(requestedDeviceId, deviceCount[0] - 1))

            val deviceBuffer = mallocInt(1)
            cuDeviceGet(deviceBuffer, deviceId)
            device = deviceBuffer[0]

            val contextBuffer = mallocPointer(1)
            cuCtxCreate(contextBuffer, 0, device)
            context = contextBuffer[0]

            val nameBuffer = malloc(1024)
            cuDeviceGetName(nameBuffer, device)
            val name = nameBuffer.readArray()
            deviceName = name.decodeToString(endIndex = name.indexOfFirst { it.toInt() == 0 })

            val maxBlockDimBuffer = mallocInt(1)
            cuDeviceGetAttribute(maxBlockDimBuffer, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X, device)
            maxBlockDimX = maxBlockDimBuffer[0]
        }
    }

    fun alloc(buffer: Buffer) = useStack {
        val mem = mallocPointer(1)
        cuMemAlloc(mem, buffer.capacity().toLong())
        write(mem[0], buffer, 0)
        mem[0]
    }

    fun alloc(size: Long) = useStack {
        val mem = mallocPointer(1)
        cuMemAlloc(mem, size)
        mem[0]
    }

    fun dealloc(ptr: Long) {
        cuMemFree(ptr)
    }

    fun read(src: Long, dst: Buffer, srcOffset: Int) = when(dst) {
        is IntBuffer -> cuMemcpyDtoH(dst, src + srcOffset)
        is DoubleBuffer -> cuMemcpyDtoH(dst, src + srcOffset)
        is FloatBuffer -> cuMemcpyDtoH(dst, src + srcOffset)
        is ByteBuffer -> cuMemcpyDtoH(dst, src + srcOffset)
        else -> throw UnsupportedOperationException()
    }

    fun write(dst: Long, src: Buffer, dstOffset: Int) = when(src) {
        is IntBuffer -> cuMemcpyHtoD(dst + dstOffset, src)
        is DoubleBuffer -> cuMemcpyHtoD(dst + dstOffset, src)
        is FloatBuffer -> cuMemcpyHtoD(dst + dstOffset, src)
        is ByteBuffer -> cuMemcpyHtoD(dst + dstOffset, src)
        else -> throw UnsupportedOperationException()
    }

    fun compileToModule(src: String): Long = useStack {
        val program = mallocPointer(1)
        nvrtcCreateProgram(program, src, null, null, null)
        val status = nvrtcCompileProgram(program[0], null)

        if(status != 0){
            val logSize = mallocPointer(1)
            nvrtcGetProgramLogSize(program[0], logSize)

            val logBuffer = malloc(logSize[0].toInt())
            nvrtcGetProgramLog(program[0], logBuffer)
            throw Exception("Failed to compile CUDA program: \n${logBuffer.readArray().decodeToString()}")
        }

        val ptxSize = mallocPointer(1)
        nvrtcGetPTXSize(program[0], ptxSize)

        val ptx = malloc(ptxSize[0].toInt())
        nvrtcGetPTX(program[0], ptx)

        nvrtcDestroyProgram(program)

        val module = mallocPointer(1)
        cuModuleLoadData(module, ptx)

        return module[0]
    }

    fun getFunctionPointer(module: Long, name: String) = useStack {
        val function = mallocPointer(1)
        cuModuleGetFunction(function, module, name)
        function[0]
    }

    fun launch(function: Long, count: Int, values: PointerBuffer) = useStack {
        val blockSizeX = min(maxBlockDimX, count)
        val gridSizeX = (count + blockSizeX - 1) / blockSizeX

        cuLaunchKernel(function,
            gridSizeX, 1, 1,
            blockSizeX, 1, 1,
            0, 0,
            values, null
        )
    }
}