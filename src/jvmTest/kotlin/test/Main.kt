package test

import com.huskerdev.gpkt.GPEngine
import com.huskerdev.gpkt.GPType


// 1 2 3 4 5 6 7 8
fun exampleArray() = FloatArray(10_000_000) { it.toFloat() }


fun main() {
    val engine = GPEngine.create(GPType.CUDA)!!
    println(engine.type)
    val program = engine.compile("""
        in float[] arr1;
        out float[] result;
        
        float sma(float[] data, int from, int period){
            float sum = 0.0;
            for(int i = 0; i < period; i++)
                sum += data[from - period];
            return sum / (float)period;
        }
        
        void main(int i){
            result[i] = sma(arr1, i, 3);
        }
    """.trimIndent())

    val arr1 = engine.alloc(exampleArray())
    val result = engine.alloc(arr1.length)
    program.execute(
        instances = arr1.length,
        "arr1" to arr1,
        "result" to result
    )

    val r = result.read()
    println(r.toList().subList(0, 20))
}