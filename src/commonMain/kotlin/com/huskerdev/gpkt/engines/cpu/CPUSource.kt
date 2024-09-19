package com.huskerdev.gpkt.engines.cpu

import com.huskerdev.gpkt.Source

class CPUSource(
    var array: FloatArray?
): Source {

    override val length = array!!.size

    override fun read() = array!!

    override fun dealloc() {
        array = null
    }
}