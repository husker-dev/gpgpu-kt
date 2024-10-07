package tests

import com.huskerdev.gpkt.GPType
import com.huskerdev.gpkt.utils.threads
import kotlin.test.Test

class InvokeMetal {

    @Test
    fun invoke1() = testInvocation(GPType.Metal, 1)

    @Test
    fun invoke5() = testInvocation(GPType.Metal, 2)

    @Test
    fun invokeAllThreads() = testInvocation(GPType.Metal, threads)

    @Test
    fun invoke50() = testInvocation(GPType.Metal, 50)

    @Test
    fun invoke500() = testInvocation(GPType.Metal, 500)

    @Test
    fun invoke100_000() = testInvocation(GPType.Metal, 100_000)
}