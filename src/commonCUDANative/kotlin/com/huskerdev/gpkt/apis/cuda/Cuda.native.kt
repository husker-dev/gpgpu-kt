@file:OptIn(ExperimentalForeignApi::class)
package com.huskerdev.gpkt.apis.cuda

import kotlinx.cinterop.*


actual class CUdevice(val ptr: cuda.CUdevice)
actual class CUcontext(val ptr: cuda.CUcontext)
actual class CUdeviceptr(val ptr: cuda.CUdeviceptr)
actual class CUmodule(val ptr: cuda.CUmodule)
actual class nvrtcProgram(val ptr: nvrtc.nvrtcProgram)
actual class CUfunction(val ptr: cuda.CUfunction)

@OptIn(ExperimentalForeignApi::class)
internal actual fun cuInit(flags: Int) =
    cuda.cuInit(flags.toUInt()).toInt()

@OptIn(ExperimentalForeignApi::class)
internal actual fun cuDeviceGetCount() = memScoped {
    val buffer = alloc<Int>(0)
    cuda.cuDeviceGetCount(buffer.ptr)
    buffer.value
}

@OptIn(ExperimentalForeignApi::class)
internal actual fun cuDeviceGet(index: Int) = memScoped {
    val buffer = alloc<cuda.CUdevice>(0)
    cuda.cuDeviceGet(buffer.ptr, index)
    CUdevice(buffer.value)
}

@OptIn(ExperimentalForeignApi::class)
internal actual fun cuDeviceGetName(device: CUdevice) = memScoped {
    val nameBuffer = allocArray<ByteVar>(1024)
    cuda.cuDeviceGetName(nameBuffer, 1024, device.ptr)
    createString(nameBuffer.readBytes(1024))
}

@OptIn(ExperimentalForeignApi::class)
internal actual fun cuCtxCreate(flags: Int, device: CUdevice) = memScoped {
    val buffer = alloc<cuda.CUcontextVar>()
    cuda.cuCtxCreate!!(buffer.ptr, flags.toUInt(), device.ptr)
    CUcontext(buffer.value!!)
}

@OptIn(ExperimentalForeignApi::class)
internal actual fun cuCtxSetCurrent(context: CUcontext?) =
    cuda.cuCtxSetCurrent(context?.ptr).toInt()

@OptIn(ExperimentalForeignApi::class)
internal actual fun cuCtxDestroy(context: CUcontext) =
    cuda.cuCtxDestroy!!(context.ptr).toInt()

@OptIn(ExperimentalForeignApi::class)
internal actual fun cuMemFree(ptr: CUdeviceptr) =
    cuda.cuMemFree!!(ptr.ptr).toInt()

@OptIn(ExperimentalForeignApi::class)
internal actual fun cuMemAlloc(size: Long) = memScoped {
    val buffer = alloc<cuda.CUdeviceptrVar>()
    cuda.cuMemAlloc!!(buffer.ptr, size.toULong())
    CUdeviceptr(buffer.value)
}

@OptIn(ExperimentalForeignApi::class)
internal actual fun cuMemcpyDtoHFloats(
    dst: FloatArray,
    src: CUdeviceptr,
    byteCount: Long,
    srcOffset: Long
) = dst.usePinned {
    cuda.cuMemcpyDtoH!!(it.addressOf(0), src.ptr + srcOffset.toULong(), byteCount.toULong()).toInt()
}

@OptIn(ExperimentalForeignApi::class)
internal actual fun cuMemcpyDtoHInts(
    dst: IntArray,
    src: CUdeviceptr,
    byteCount: Long,
    srcOffset: Long
) = dst.usePinned {
    cuda.cuMemcpyDtoH!!(it.addressOf(0), src.ptr + srcOffset.toULong(), byteCount.toULong()).toInt()
}

@OptIn(ExperimentalForeignApi::class)
internal actual fun cuMemcpyDtoHBytes(
    dst: ByteArray,
    src: CUdeviceptr,
    byteCount: Long,
    srcOffset: Long
) = dst.usePinned {
    cuda.cuMemcpyDtoH!!(it.addressOf(0), src.ptr + srcOffset.toULong(), byteCount.toULong()).toInt()
}

@OptIn(ExperimentalForeignApi::class)
internal actual fun cuMemcpyHtoDFloats(
    dst: CUdeviceptr,
    src: FloatArray,
    byteCount: Long,
    srcOffset: Long,
    dstOffset: Long
) = src.usePinned {
    cuda.cuMemcpyHtoD!!(dst.ptr + dstOffset.toULong(), it.addressOf(srcOffset.toInt()), byteCount.toULong()).toInt()
}

@OptIn(ExperimentalForeignApi::class)
internal actual fun cuMemcpyHtoDInts(
    dst: CUdeviceptr,
    src: IntArray,
    byteCount: Long,
    srcOffset: Long,
    dstOffset: Long
) = src.usePinned {
    cuda.cuMemcpyHtoD!!(dst.ptr + dstOffset.toULong(), it.addressOf(srcOffset.toInt()), byteCount.toULong()).toInt()
}

@OptIn(ExperimentalForeignApi::class)
internal actual fun cuMemcpyHtoDBytes(
    dst: CUdeviceptr,
    src: ByteArray,
    byteCount: Long,
    srcOffset: Long,
    dstOffset: Long
) = src.usePinned {
    cuda.cuMemcpyHtoD!!(dst.ptr + dstOffset.toULong(), it.addressOf(srcOffset.toInt()), byteCount.toULong()).toInt()
}

@OptIn(ExperimentalForeignApi::class)
internal actual fun nvrtcCreateProgram(
    src: String
) = memScoped {
    val program = alloc<nvrtc.nvrtcProgramVar>()
    nvrtc.nvrtcCreateProgram(program.ptr, src, null, 0, null, null)
    nvrtcProgram(program.value!!)
}

@OptIn(ExperimentalForeignApi::class)
internal actual fun nvrtcCompileProgram(
    program: nvrtcProgram
) = nvrtc.nvrtcCompileProgram(program.ptr, 0, null).toInt()

@OptIn(ExperimentalForeignApi::class)
internal actual fun nvrtcGetProgramLog(program: nvrtcProgram) = memScoped {
    val length = alloc<ULongVar>()
    nvrtc.nvrtcGetProgramLogSize(program.ptr, length.ptr)

    val buffer = allocArray<ByteVar>(length.value.toInt())
    nvrtc.nvrtcGetProgramLog(program.ptr, buffer)
    createString(buffer.readBytes(length.value.toInt()))
}

@OptIn(ExperimentalForeignApi::class)
internal actual fun nvrtcGetPTX(program: nvrtcProgram) = memScoped {
    val length = alloc<ULongVar>()
    nvrtc.nvrtcGetPTXSize(program.ptr, length.ptr)

    val buffer = allocArray<ByteVar>(length.value.toInt())
    nvrtc.nvrtcGetPTX(program.ptr, buffer)
    createString(buffer.readBytes(length.value.toInt()))
}

@OptIn(ExperimentalForeignApi::class)
internal actual fun nvrtcDestroyProgram(program: nvrtcProgram) = memScoped {
    val a = alloc<nvrtc.nvrtcProgramVar>()
    a.value = program.ptr
    nvrtc.nvrtcDestroyProgram(a.ptr).toInt()
}

@OptIn(ExperimentalForeignApi::class)
internal actual fun cuModuleLoadData(ptx: String) = memScoped {
    val buffer = alloc<cuda.CUmoduleVar>()
    cuda.cuModuleLoadData(buffer.ptr, ptx.cstr)
    CUmodule(buffer.value!!)
}

@OptIn(ExperimentalForeignApi::class)
internal actual fun cuModuleUnload(module: CUmodule) =
    cuda.cuModuleUnload(module.ptr).toInt()

@OptIn(ExperimentalForeignApi::class)
internal actual fun cuModuleGetFunction(module: CUmodule, name: String) = memScoped {
    val buffer = alloc<cuda.CUfunctionVar>()
    cuda.cuModuleGetFunction(buffer.ptr, module.ptr, name)
    CUfunction(buffer.value!!)
}

@OptIn(ExperimentalForeignApi::class)
internal actual fun cuDeviceGetAttribute(attrib: Int, device: CUdevice) = memScoped {
    val buffer = alloc<IntVar>()
    cuda.cuDeviceGetAttribute(buffer.ptr, attrib.toUInt(), device.ptr)
    buffer.value
}

@OptIn(ExperimentalForeignApi::class)
internal actual fun cuLaunchKernel(
    function: CUfunction,
    gridDimX: Int,
    gridDimY: Int,
    gridDimZ: Int,
    blockDimX: Int,
    blockDimY: Int,
    blockDimZ: Int,
    sharedMemBytes: Int,
    vararg kernelParams: Any
) = memScoped {
    val args = allocArray<COpaquePointerVar>(kernelParams.size)
    kernelParams.forEachIndexed { i, it ->
        args[i] = when(it){
            is Float -> alloc(it).ptr
            is Int -> alloc(it).ptr
            is Byte -> alloc(it).ptr
            is Boolean -> alloc<Byte>(if(it == true) 1 else 0).ptr
            is CUdeviceptr -> alloc(it.ptr).ptr
            else -> throw UnsupportedOperationException(it.toString())
        }
    }

    cuda.cuLaunchKernel(
        function.ptr,
        gridDimX.toUInt(), gridDimY.toUInt(), gridDimZ.toUInt(),
        blockDimX.toUInt(), blockDimY.toUInt(), blockDimZ.toUInt(),
        sharedMemBytes.toUInt(), null,
        args, null
    ).toInt()
}

@OptIn(ExperimentalForeignApi::class)
internal actual fun cuGetErrorName(code: Int): String = memScoped {
    val ptr = this.allocPointerTo<ByteVar>()
    cuda.cuGetErrorName(code.toUInt(), ptr.ptr)
    return ptr.value!!.toKString()
}

@OptIn(ExperimentalForeignApi::class)
internal actual fun cuGetErrorString(code: Int): String = memScoped {
    val ptr = this.allocPointerTo<ByteVar>()
    cuda.cuGetErrorString(code.toUInt(), ptr.ptr)
    return ptr.value!!.toKString()
}