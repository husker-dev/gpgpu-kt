package test

import com.huskerdev.gpkt.GPEngine
import com.huskerdev.gpkt.GPType
import kotlin.math.ceil
import kotlin.math.floor


fun main(){
    GPType.entries.forEach {
        exec(it)
    }
}

var time = System.nanoTime()
fun reset(text: String) {
    val length = 25
    val left = "_".repeat(floor((length - text.length) / 2.0).toInt())
    val right = "_".repeat(ceil((length - text.length) / 2.0).toInt())
    println("$left $text $right")
    time = System.nanoTime()
}
fun check(text: String, some: FloatArray? = null){
    val period = System.nanoTime() - time
    println("$text:${" ".repeat(15 - text.length)}${period / 1_000_000.0} ms")
    if (some != null)
        println(some.toList().subList(0, 20))
    time = System.nanoTime()
}

fun exec(engineType: GPType){
    reset(engineType.name)

    val engine = GPEngine.create(engineType)!!
    check("Init")

    val program = engine.compile("""
        in float[] arr1, arr2;
        out float[] result;
        
        float a(float v1, float v2){
            if((int)v1 % 2 == 0)
                return v1 + v2;
            else
                return v1 * v2;
        }
        
        void main(int i){
            result[i] = a(arr1[i], arr2[i]);
        }
    """.trimIndent())
    check("Compilation")

    val arr1 = engine.alloc(exampleArray())
    val arr2 = engine.alloc(exampleArray())
    val result = engine.alloc(arr1.length)
    check("Allocation")

    program.execute(
        instances = arr1.length,
        "arr1" to arr1,
        "arr2" to arr2,
        "result" to result
    )
    check("Execution")

    val r = result.read()
    check("Read", r)

    arr1.dealloc()
    arr2.dealloc()
    result.dealloc()
}


// 1 2 3 4 5 6 7 8
fun exampleArray() = FloatArray(10_000_000) { it.toFloat() }