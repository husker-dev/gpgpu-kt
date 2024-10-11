package tests

import com.huskerdev.gpkt.GPApiType
import com.huskerdev.gpkt.utils.threads
import kotlin.test.Test

class InvokeJAVAC {

    @Test
    fun invoke1() = testInvocation(GPApiType.Javac, 1)

    @Test
    fun invoke5() = testInvocation(GPApiType.Javac, 2)

    @Test
    fun invokeAllThreads() = testInvocation(GPApiType.Javac, threads)

    @Test
    fun invoke50() = testInvocation(GPApiType.Javac, 50)

    @Test
    fun invoke500() = testInvocation(GPApiType.Javac, 500)

    @Test
    fun invoke100_000() = testInvocation(GPApiType.Javac, 100_000)
}