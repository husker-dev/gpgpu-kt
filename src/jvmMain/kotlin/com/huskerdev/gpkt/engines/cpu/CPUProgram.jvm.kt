package com.huskerdev.gpkt.engines.cpu

import kotlin.concurrent.thread

actual val threads = Runtime.getRuntime().availableProcessors()

actual fun runThread(f: () -> Unit) = object: AbstractThread {
    val thread = thread(block = f, isDaemon = true)

    override fun waitEnd() = thread.join()
}