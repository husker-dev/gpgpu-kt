package tests

import com.huskerdev.gpkt.GPType
import com.huskerdev.gpkt.utils.threads
import kotlin.test.Test

class InvokeInterpreter {

    @Test
    fun invoke1() = testInvocation(GPType.Interpreter, 1)

    @Test
    fun invoke5() = testInvocation(GPType.Interpreter, 2)

    @Test
    fun invokeAllThreads() = testInvocation(GPType.Interpreter, threads)

    @Test
    fun invoke50() = testInvocation(GPType.Interpreter, 50)

    @Test
    fun invoke500() = testInvocation(GPType.Interpreter, 500)

    @Test
    fun invoke100_000() = testInvocation(GPType.Interpreter, 100_000)
}