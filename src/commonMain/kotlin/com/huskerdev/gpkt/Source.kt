package com.huskerdev.gpkt

interface Source {
    val length: Int
    fun read(): FloatArray
    fun dealloc()
}