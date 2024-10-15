package com.huskerdev.gpkt.apis.cuda

import kotlinx.cinterop.ExperimentalForeignApi
import platform.posix.RTLD_NOW
import platform.posix.dlopen
import platform.posix.dlsym

@OptIn(ExperimentalForeignApi::class)
internal actual fun isCUDASupported(): Boolean {
    val lib = dlopen("libcuda.so", RTLD_NOW)
        ?: return false

    dlsym(lib, "cuInit")
        ?: return false
    return true
}