package com.huskerdev.gpkt.cuda

import jcuda.CudaException
import jcuda.Pointer
import jcuda.Sizeof
import jcuda.driver.*
import jcuda.driver.JCudaDriver.*
import jcuda.nvrtc.JNvrtc
import jcuda.nvrtc.JNvrtc.*
import jcuda.nvrtc.nvrtcProgram


class Cuda {
    companion object {
        val supported = try {
            JNvrtc.setExceptionsEnabled(true)
            true
        }catch (e: UnsatisfiedLinkError){
            println("[INFO] Failed to load CUDA. Check toolkit installation.")
            false
        }
    }

    private val device = CUdevice()
    private val context = CUcontext()

    init {
        cuInit(0)
        cuDeviceGet(device, 0)
        cuCtxCreate(context, 0, device)
    }

    fun alloc(array: FloatArray): CUdeviceptr {
        val length = (array.size * Sizeof.FLOAT).toLong()
        val ptr = CUdeviceptr()
        cuMemAlloc(ptr, length)
        cuMemcpyHtoD(ptr, Pointer.to(array), length)
        return ptr
    }

    fun alloc(length: Int): CUdeviceptr {
        val ptr = CUdeviceptr()
        cuMemAlloc(ptr, (length * Sizeof.FLOAT).toLong())
        return ptr
    }

    fun dealloc(ptr: CUdeviceptr) =
        cuMemFree(ptr)

    fun read(ptr: CUdeviceptr, length: Int): FloatArray{
        val result = FloatArray(length)
        cuMemcpyDtoH(Pointer.to(result), ptr, (length * Sizeof.FLOAT).toLong())
        return result
    }

    fun compileToModule(src: String): CUmodule{
        val program = nvrtcProgram()
        try {
            nvrtcCreateProgram(program, src, null, 0, null, null)
            nvrtcCompileProgram(program, 0, null)
        }catch (e: CudaException){
            val logBuffer = arrayOfNulls<String>(1)
            nvrtcGetProgramLog(program, logBuffer)
            throw Exception("Failed to compile CUDA program: \n${logBuffer[0]}")
        }

        val ptx = arrayOfNulls<String>(1)
        nvrtcGetPTX(program, ptx)
        nvrtcDestroyProgram(program)

        val module = CUmodule()
        cuModuleLoadData(module, ptx[0])
        return module
    }

    fun getFunctionPointer(module: CUmodule, name: String): CUfunction{
        val function = CUfunction()
        cuModuleGetFunction(function, module, name)
        return function
    }

    fun launch(function: CUfunction, count: Int, sources: List<CudaSource>){
        val blockSizeX = 256
        val gridSizeX: Int = (count + blockSizeX - 1) / blockSizeX

        cuLaunchKernel(function,
            gridSizeX, 1, 1,
            blockSizeX, 1, 1,
            0, null,
            Pointer.to(*sources.map {
                Pointer.to(it.ptr)
            }.toTypedArray()), null
        )
        cuCtxSynchronize()
    }
}