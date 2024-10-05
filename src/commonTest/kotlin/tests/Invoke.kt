package tests

import com.huskerdev.gpkt.GPSyncDevice
import com.huskerdev.gpkt.GPType
import kotlin.test.assertTrue

fun testInvocation(type: GPType, instances: Int){
    val arraySize = instances + 100

    val engine = GPSyncDevice.create(requestedType = arrayOf(type))!!
    val data1 = engine.wrapFloats(FloatArray(instances) { it.toFloat() + 1 })
    val data2 = engine.wrapFloats(FloatArray(instances) { it.toFloat() + 1 })
    val result = engine.allocFloats(arraySize)

    val program = engine.compile("""
            extern float[] data1, data2;
            extern float[] result;
            
            void main(int i){
                double r_pi = PI;
                double r_e = E;
            
                double r_abs = abs(-123);
                double r_acos = acos(123.0);
                double r_asin = asin(123.0);
                double r_atan = atan(123.0);
                double r_atan2 = atan2(123.0, 23.0);
                double r_cbrt = cbrt(25.0);
                double r_ceil = ceil(123.0);
                double r_cos = cos(123.0);
                double r_cosh = cosh(123.0);
                double r_exp = exp(123.0);
                double r_expm1 = expm1(123.0);
                double r_floor = floor(123.0);
                double r_hypot = hypot(123.0, 321.0);
                double r_log = log(123.0);
                double r_log10 = log10(123.0);
                double r_max = max(123.0, 1.0);
                double r_min = min(123.0, 1.0);
                double r_pow = pow(123.0, 2);
                double r_round = round(123.0);
                double r_sin = sin(123.0);
                double r_sinh = sinh(123.0);
                double r_sqrt = sqrt(123.0);
                double r_tan = tan(123.0);
                double r_tanh = tanh(123.0);
            
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