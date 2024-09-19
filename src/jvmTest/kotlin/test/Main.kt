package test

import com.huskerdev.gpkt.GPEngine
import com.huskerdev.gpkt.GPType


fun main(){
    val engine = GPEngine.create(GPType.OpenCL)!!
    println("engine: ${engine.type}")

    val program = engine.compile("""
        in float[] arr1, arr2;
        out float[] result;
        
        float add(float left, float right){
            return left + right;
        }
        
        void main(int i){
            result[i] = add(arr1[i], arr2[i]);
        }
        
    """.trimIndent())

    val arr1 = engine.alloc(exampleArray())
    val arr2 = engine.alloc(exampleArray())
    val result = engine.alloc(arr1.length)

    program.execute(
        "arr1" to arr1,
        "arr2" to arr2,
        "result" to result
    )

    println(result.read().toList().subList(0, 20))
}



// 1 2 3 4 5 6 7 8
fun exampleArray() = FloatArray(10_000_000) { it.toFloat() }