package com.huskerdev.gpkt.engines.cpu

actual val threads = 1

actual fun runThread(f: () -> Unit): AbstractThread =
    throw UnsupportedOperationException("Can't run threads in Web")