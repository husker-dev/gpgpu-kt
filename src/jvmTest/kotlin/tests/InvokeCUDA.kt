package tests

import com.huskerdev.gpkt.GPApiType
import com.huskerdev.gpkt.utils.threads
import kotlin.test.Test

class InvokeCUDA {

    @Test
    fun invoke1() = testInvocation(GPApiType.CUDA, 1)

    @Test
    fun invoke5() = testInvocation(GPApiType.CUDA, 2)

    @Test
    fun invokeAllThreads() = testInvocation(GPApiType.CUDA, threads)

    @Test
    fun invoke50() = testInvocation(GPApiType.CUDA, 50)

    @Test
    fun invoke500() = testInvocation(GPApiType.CUDA, 500)

    @Test
    fun invoke100_000() = testInvocation(GPApiType.CUDA, 100_000)
}