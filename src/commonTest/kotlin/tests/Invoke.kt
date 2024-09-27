package tests

import com.huskerdev.gpkt.GPDevice
import com.huskerdev.gpkt.GPType
import kotlin.test.assertTrue

fun testInvocation(type: GPType, instances: Int){
    val arraySize = instances + 100

    val engine = GPDevice.create(requestedType = arrayOf(type))!!
    val data1 = engine.allocFloat(FloatArray(instances) { it.toFloat() + 1 })
    val data2 = engine.allocFloat(FloatArray(instances) { it.toFloat() + 1 })
    val result = engine.allocFloat(arraySize)

    val program = engine.compile("""
            in float[] data1, data2;
            out float[] result;
            
            void main(int i){
                result[i] = data1[i] + data2[i];
            }
        """.trimIndent())

    val correctResult = FloatArray(arraySize) {
        if (it < instances) (it.toFloat() + 1) * 2 else 0f
    }

    for(i in 0 until 200) {
        program.execute(
            instances,
            "data1" to data1,
            "data2" to data2,
            "result" to result
        )
        assertTrue(correctResult.contentEquals(result.read()))
    }

    data1.dealloc()
    data2.dealloc()
    result.dealloc()
    program.dealloc()
}