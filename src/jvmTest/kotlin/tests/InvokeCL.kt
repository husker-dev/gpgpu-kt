package tests

import com.huskerdev.gpkt.GPApiType
import com.huskerdev.gpkt.utils.threads
import kotlin.test.Test

class InvokeCL {

    @Test
    fun invoke1() = testInvocation(GPApiType.OpenCL, 1)

    @Test
    fun invoke5() = testInvocation(GPApiType.OpenCL, 2)

    @Test
    fun invokeAllThreads() = testInvocation(GPApiType.OpenCL, threads)

    @Test
    fun invoke50() = testInvocation(GPApiType.OpenCL, 50)

    @Test
    fun invoke500() = testInvocation(GPApiType.OpenCL, 500)

    @Test
    fun invoke100_000() = testInvocation(GPApiType.OpenCL, 100_000)
}