package com.huskerdev.gpkt.utils

actual val threads = 1

actual fun runThread(f: () -> Unit) = object: AbstractThread {
    init {
        f()
    }
    override fun waitEnd() = Unit
}