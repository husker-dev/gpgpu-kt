package com.huskerdev.gpkt.engines.jdk

import com.huskerdev.gpkt.Source

class JavacSource(
     val array: FloatArray
): Source {
    override val length = array.size

    override fun read() = array
    override fun dealloc() = Unit
}