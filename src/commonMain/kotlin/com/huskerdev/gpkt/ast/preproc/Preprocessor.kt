package com.huskerdev.gpkt.ast.preproc

import com.huskerdev.gpkt.GPContext
import com.huskerdev.gpkt.ast.compilationError

fun processPreprocessor(text: String, context: GPContext?): String{
    val moduleSet = hashSetOf<String>()
    val expanded = StringBuilder()
    expand(text, context, moduleSet, expanded)

    return expanded.toString()
}

private fun expand(text: String, context: GPContext?, set: HashSet<String>, expanded: StringBuilder){
    text.split("\n").forEachIndexed { i, line ->
        val trimLine = line.trim()
        if(trimLine.startsWith("import ")){
            if(!trimLine.endsWith(";"))
                throw compilationError("Expected ';'", i, line.lastIndex, text)

            val modules = trimLine.split("import")[1].split(";")[0].trim().split(",")
            modules.forEach { moduleNamePure ->
                val moduleName = moduleNamePure.trim()
                if(moduleName in set)
                    return@forEach
                set += moduleName
                val moduleGetter = context?.modules?.get(moduleName)
                    ?: throw compilationError("Module '${moduleName}' not found", i, line.lastIndex, text)

                expand(moduleGetter(), context, set, expanded)
            }
        }else if(trimLine.startsWith("//"))
            return@forEachIndexed
        else
            expanded.append(line).append('\n')
    }
}