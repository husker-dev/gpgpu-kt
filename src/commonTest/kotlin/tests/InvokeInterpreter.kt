package tests

import com.huskerdev.gpkt.GPApiType
import com.huskerdev.gpkt.utils.threads
import kotlin.test.Test

class InvokeInterpreter {

    @Test
    fun invoke1() = testInvocation(GPApiType.Interpreter, 1)

    @Test
    fun invoke5() = testInvocation(GPApiType.Interpreter, 2)

    @Test
    fun invokeAllThreads() = testInvocation(GPApiType.Interpreter, threads)

    @Test
    fun invoke50() = testInvocation(GPApiType.Interpreter, 50)

    @Test
    fun invoke500() = testInvocation(GPApiType.Interpreter, 500)

    @Test
    fun invoke100_000() = testInvocation(GPApiType.Interpreter, 100_000)
}