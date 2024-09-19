package com.huskerdev.gpkt.engines.cpu

import com.huskerdev.gpkt.Program
import com.huskerdev.gpkt.Source
import com.huskerdev.gpkt.ast.objects.Scope
import com.huskerdev.gpkt.ast.types.Type
import com.huskerdev.gpkt.engines.cpu.objects.ExField
import com.huskerdev.gpkt.engines.cpu.objects.ExScope
import com.huskerdev.gpkt.engines.cpu.objects.ExValue

class CPUProgram(
    val ast: Scope
): Program {
    private val executable = ExScope(ast, null)

    override fun execute(instances: Int, vararg mapping: Pair<String, Source>) {
        val variables = mapping.associate {
            val array = (it.second as CPUSource).array!!
            it.first to ExField(Type.FLOAT_ARRAY, ExValue(array))
        }.toMutableMap()

        val iteration = ExField(Type.INT, ExValue(null))
        variables["__i"] = iteration

        for(step in 0 until instances){
            iteration.value!!.set(step)
            executable.execute(variables)
        }
    }
}