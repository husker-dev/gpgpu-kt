package tests

import com.huskerdev.gpkt.GPApiType
import com.huskerdev.gpkt.GPSyncApi
import kotlin.test.assertTrue

fun testInvocation(type: GPApiType, instances: Int){
    val arraySize = instances + 100

    val api = GPSyncApi.getByType(type)
    val context = api.defaultDevice.createContext()
    val data1 = context.wrapFloats(FloatArray(instances) { it.toFloat() + 1 })
    val data2 = context.wrapFloats(FloatArray(instances) { it.toFloat() + 1 })
    val result = context.allocFloats(arraySize)

    val program = context.compile("""
            extern float[] data1, data2;
            extern float[] result;
            
            void main(int i){
                float r_pi = PI;
                float r_e = E;
            
                float r_abs = abs(-123);
                float r_acos = acos(123f);
                float r_asin = asin(123f);
                float r_atan = atan(123f);
                float r_atan2 = atan2(123f, 23f);
                float r_cbrt = cbrt(25f);
                float r_ceil = ceil(123f);
                float r_cos = cos(123f);
                float r_cosh = cosh(123f);
                float r_exp = exp(123f);
                float r_expm1 = expm1(123f);
                float r_floor = floor(123f);
                float r_hypot = hypot(123f, 321f);
                float r_log = log(123f);
                float r_log10 = log10(123f);
                float r_max = max(123f, 1f);
                float r_min = min(123f, 1f);
                float r_pow = pow(123f, 2);
                float r_round = round(123f);
                float r_sin = sin(123f);
                float r_sinh = sinh(123f);
                float r_sqrt = sqrt(123f);
                float r_tan = tan(123f);
                float r_tanh = tanh(123f);
            
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