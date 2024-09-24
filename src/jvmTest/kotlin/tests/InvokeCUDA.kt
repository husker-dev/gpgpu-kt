package tests

import com.huskerdev.gpkt.GPType
import com.huskerdev.gpkt.utils.threads
import kotlin.test.Test

class InvokeCUDA {

    @Test
    fun invoke1() = testInvocation(GPType.CUDA, 1)

    @Test
    fun invoke5() = testInvocation(GPType.CUDA, 2)

    @Test
    fun invokeAllThreads() = testInvocation(GPType.CUDA, threads)

    @Test
    fun invoke50() = testInvocation(GPType.CUDA, 50)

    @Test
    fun invoke500() = testInvocation(GPType.CUDA, 500)

    @Test
    fun invoke100_000() = testInvocation(GPType.CUDA, 100_000)
}