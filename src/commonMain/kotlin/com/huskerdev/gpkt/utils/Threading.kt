package com.huskerdev.gpkt.utils

import kotlin.math.ceil
import kotlin.math.max
import kotlin.math.min

private const val MIN_OPERATIONS_PER_THREAD = 20

expect val threads: Int
expect fun runThread(f: () -> Unit): AbstractThread


interface AbstractThread {
    fun waitEnd()
}

fun splitThreadInvocation(instances: Int, block: (from: Int, to: Int) -> Unit){
    if(threads == 1 || instances < MIN_OPERATIONS_PER_THREAD){
        block(0, instances)
    } else {
        val instancesPerThread = max(MIN_OPERATIONS_PER_THREAD, ceil(instances.toFloat() / threads).toInt())
        val threadList = arrayListOf<AbstractThread>()

        for(i in 0 until instances step instancesPerThread){
            val to = min(instances, i + instancesPerThread)
            threadList += runThread {
                block(i, to)
            }
        }
        threadList.forEach { it.waitEnd() }
    }
}