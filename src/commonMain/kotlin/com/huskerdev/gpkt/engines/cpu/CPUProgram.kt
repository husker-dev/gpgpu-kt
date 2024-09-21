package com.huskerdev.gpkt.engines.cpu

import com.huskerdev.gpkt.Program
import com.huskerdev.gpkt.Source
import com.huskerdev.gpkt.ast.objects.Scope
import com.huskerdev.gpkt.ast.types.Type
import com.huskerdev.gpkt.engines.cpu.objects.ExField
import com.huskerdev.gpkt.engines.cpu.objects.ExScope
import com.huskerdev.gpkt.engines.cpu.objects.ExValue

expect val threads: Int
expect fun runThread(f: () -> Unit): AbstractThread


class CPUProgram(
    val ast: Scope
): Program {
    override fun execute(instances: Int, vararg mapping: Pair<String, Source>) {
        val variables = mapping.associate {
            val array = (it.second as CPUSource).array!!
            it.first to ExField(Type.FLOAT_ARRAY, ExValue(array))
        }.toMutableMap()

        if(threads == 1){
            execPeriod(variables, 0, instances)
        } else if(instances > threads) {
            val instancesPerThread = instances / threads
            val threadList = arrayListOf<AbstractThread>()
            for (i in 0 until threads) {
                val fromIndex = i * instancesPerThread
                threadList += runThread {
                    execPeriod(variables, fromIndex, fromIndex + instancesPerThread)
                }
            }
            threadList.forEach { it.waitEnd() }
        }else {
            val threadList = arrayListOf<AbstractThread>()
            for(i in 0 until instances){
                threadList += runThread {
                    execPeriod(variables, i, i+1)
                }
            }
            threadList.forEach { it.waitEnd() }
        }
    }

    private fun execPeriod(variables: MutableMap<String, ExField>, from: Int, to: Int){
        val scope = ExScope(ast)
        val iteration = ExField(Type.INT, ExValue(null))
        val threadLocalVariables = hashMapOf(
            "__i__" to iteration
        )
        threadLocalVariables.putAll(variables)

        for(step in from until to){
            iteration.value!!.set(step)
            scope.execute(threadLocalVariables)
        }
    }

    override fun dealloc() = Unit
}

interface AbstractThread {
    fun waitEnd()
}