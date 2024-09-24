package tests

import com.huskerdev.gpkt.GPType
import com.huskerdev.gpkt.utils.threads
import kotlin.test.Test

class InvokeJAVAC {

    @Test
    fun invoke1() = testInvocation(GPType.Javac, 1)

    @Test
    fun invoke5() = testInvocation(GPType.Javac, 2)

    @Test
    fun invokeAllThreads() = testInvocation(GPType.Javac, threads)

    @Test
    fun invoke50() = testInvocation(GPType.Javac, 50)

    @Test
    fun invoke500() = testInvocation(GPType.Javac, 500)

    @Test
    fun invoke100_000() = testInvocation(GPType.Javac, 100_000)
}