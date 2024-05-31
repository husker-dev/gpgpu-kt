package org.example

import org.example.com.huskerdev.ta.engine.cpu.EngineCPUImpl
import org.example.com.huskerdev.ta.engine.gpu.EngineGPUImpl
import org.example.com.huskerdev.ta.sequence.*


fun main() {
    //defaultEngine = EngineCPUImpl
    EngineGPUImpl.opencl    // Pre-initialize opencl

    // Arrays [0, 1, ..., n]
    val s1 = FloatArray(100_000_000) { it.toFloat() }.toSequence()
    val s2 = FloatArray(100_000_000) { it.toFloat() }.toSequence()
    val s3 = FloatArray(100_000_000) { it.toFloat() }.toSequence()

    val sum = s1 + s2 + s3
    val mult = s1 * s2 * s3
    val smth = sum + mult * 1000 / sqrt(s1) + pow(s2, 2)


    val time = System.nanoTime()
    val values = smth.get()
    println("==========================")
    println("completed in: ${(System.nanoTime() - time) / 1000000 / 1000.0} sec")

    println("size: ${values.size} ${values.toShortString()}")
}

fun FloatArray.toShortString(elements: Int = 6) =
    FloatArray(elements){ i -> this[i] }.joinToString(", ", prefix = "[", postfix = ", ...]")
