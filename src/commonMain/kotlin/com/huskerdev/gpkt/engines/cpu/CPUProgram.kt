package com.huskerdev.gpkt.engines.cpu

import com.huskerdev.gpkt.Program
import com.huskerdev.gpkt.Source
import com.huskerdev.gpkt.ast.objects.Scope
import com.huskerdev.gpkt.ast.types.Type
import com.huskerdev.gpkt.engines.cpu.objects.ExField
import com.huskerdev.gpkt.engines.cpu.objects.ExScope
import com.huskerdev.gpkt.engines.cpu.objects.ExValue
import com.huskerdev.gpkt.utils.splitThreadInvocation


class CPUProgram(
    val ast: Scope
): Program {
    override fun execute(instances: Int, vararg mapping: Pair<String, Source>) {
        val variables = mapping.associate {
            val array = (it.second as CPUSource).array!!
            it.first to ExField(Type.FLOAT_ARRAY, ExValue(array))
        }.toMutableMap()

        splitThreadInvocation(instances) { from, to ->
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
    }

    override fun dealloc() = Unit
}

