package tests

import com.huskerdev.gpkt.GPType
import com.huskerdev.gpkt.utils.threads
import kotlin.test.Test

class InvokeCL {

    @Test
    fun invoke1() = testInvocation(GPType.OpenCL, 1)

    @Test
    fun invoke5() = testInvocation(GPType.OpenCL, 2)

    @Test
    fun invokeAllThreads() = testInvocation(GPType.OpenCL, threads)

    @Test
    fun invoke50() = testInvocation(GPType.OpenCL, 50)

    @Test
    fun invoke500() = testInvocation(GPType.OpenCL, 500)

    @Test
    fun invoke100_000() = testInvocation(GPType.OpenCL, 100_000)
}