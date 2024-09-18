package com.huskerdev.gpkt

import com.huskerdev.gpkt.opencl.OCLEngine


internal actual fun createSupportedInstance(vararg expectedEngine: String): GPGPUEngine? {
    expectedEngine.forEach {
        when(it){
            "opencl" -> return OCLEngine()
        }
    }
    return null
}