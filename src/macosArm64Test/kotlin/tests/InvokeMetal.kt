package tests

import com.huskerdev.gpkt.GPApiType
import com.huskerdev.gpkt.utils.threads
import kotlin.test.Test

class InvokeMetal {

    @Test
    fun invoke1() = testInvocation(GPApiType.Metal, 1)

    @Test
    fun invoke5() = testInvocation(GPApiType.Metal, 2)

    @Test
    fun invokeAllThreads() = testInvocation(GPApiType.Metal, threads)

    @Test
    fun invoke50() = testInvocation(GPApiType.Metal, 50)

    @Test
    fun invoke500() = testInvocation(GPApiType.Metal, 500)

    @Test
    fun invoke100_000() = testInvocation(GPApiType.Metal, 100_000)
}