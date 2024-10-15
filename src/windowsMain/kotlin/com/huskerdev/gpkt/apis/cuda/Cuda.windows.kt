package com.huskerdev.gpkt.apis.cuda

import kotlinx.cinterop.*
import platform.windows.GetProcAddress
import platform.windows.LoadLibraryA

@OptIn(ExperimentalForeignApi::class)
internal actual fun isCUDASupported(): Boolean {
    val lib = LoadLibraryA("nvcuda.dll")
        ?: return false

    GetProcAddress(lib, "cuInit")
        ?: return false
    return true
}