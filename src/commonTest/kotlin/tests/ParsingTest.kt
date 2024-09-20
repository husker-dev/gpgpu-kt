package tests

import com.huskerdev.gpkt.GPAst
import kotlin.test.Test


class ParsingTest {
    @Test
    fun parsing(){
        GPAst.parse("""
            int a;
            int b;
            int c = 10;
            int d = 10, e;
            int f, g = 10;
            int h = 10, i = 10;
            
            in float[] inputs1, inputs2;
            out float[] outputs1, outputs2;
            
            void emptyFunc(){
            }
            
            float floatFunc(int arg1, float arg2){
                return 1.2;
            }
            
            int intFunc(int arg1, float arg2){
                return 0;
            }
            
            boolean boolFunc(int arg1, float arg2){
                return true;
            }
            
            void main(int index){
                /* ================= *\
                      Expressions
                \* ================= */
                emptyFunc();
                123 + 321;
                123 + intFunc(123, 321.0);
                c = 123 + intFunc(123, 321.0);
            
                /* ================= *\
                         Parse
                \* ================= */
                int iVar1 = (int)123.0;
                int iVar2 = (int)(123.0 + floatFunc(1, 1.1));
                int iVar3 = (int)(float)1;
                
                float fVar1 = (float)123;
                float fVar2 = (float)(123 + intFunc(1, 1.1));
                float fVar3 = (float)(int)1;
            
                /* ================= *\
                       Functions
                \* ================= */
            
                floatFunc(c, 2.2);
                intFunc(d, 1.23);
                
                /* ================= *\
                      If statement
                \* ================= */
                
                if(boolFunc(1, 1.2)){
                    intFunc(d, 1.23);
                }
                
                if(boolFunc(1, 1.2) == true){
                    intFunc(d, 1.23);
                }
                
                if(123 == 321){
                    intFunc(d, 1.23);
                }
                
                if(boolFunc(1, 1.2))
                    intFunc(d, 1.23);
                    
                if(boolFunc(1, 1.2)){
                    intFunc(d, 1.23);
                }else {
                    floatFunc(c, 2.2);
                }
                
                if(boolFunc(1, 1.2)){
                    intFunc(d, 1.23);
                }else
                    floatFunc(c, 2.2);
                
                if(boolFunc(1, 1.2))
                    intFunc(d, 1.23);
                else {
                    floatFunc(c, 2.2);
                }
                
                if(boolFunc(1, 1.2))
                    intFunc(d, 1.23);
                else
                    floatFunc(c, 2.2);
                    
                /* ================= *\
                       For loops
                \* ================= */
                
                for(int it = 0; it < 100; it++){
                    intFunc(d, 1.23);
                }
                
                for(int it = 0; it < 100; it++){
                    intFunc(d, 1.23);
                    break;
                }
                
                for(int it = 0; it < 100; it++){
                    intFunc(d, 1.23);
                    continue;
                }
                
                for(int it = 0; it < 100; it++)
                    intFunc(d, 1.23);
                
                for(int it = 0; it < 100;)
                    intFunc(d, 1.23);
                    
                for(int it = 0;;)
                    intFunc(d, 1.23);
                    
                for(;;)
                    intFunc(d, 1.23);
                    
                for(;c < 100;)
                    intFunc(d, 1.23);
                    
                for(;c < 100;c++)
                    intFunc(d, 1.23);
                    
                /* ================= *\
                      While loops
                \* ================= */
                
                while(boolFunc(1, 1.2)) {
                    intFunc(d, 1.23);
                }
                
                while(boolFunc(1, 1.2)) {
                    intFunc(d, 1.23);
                    break;
                }
                
                while(boolFunc(1, 1.2)) {
                    intFunc(d, 1.23);
                    continue;
                }
                
                while(true) {
                    intFunc(d, 1.23);
                }
                
                while(true)
                    intFunc(d, 1.23);
                
                
            }
        """.trimIndent())
    }

}

