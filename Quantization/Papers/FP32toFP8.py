union FPConvertHelper {
    float value;
    uint32_t data;  //data，value一起变
};

template<typename Dtype, typename Stype, typename Otype>
__device__ __inline__
float QuantizeScalarFloating(
    const Dtype value, const Stype scale, const Otype offset,
    const int exponent, const int mantissa,
    const float clip_min, const float clip_max, 
    const Rounding rounding){
    /**
     * PPQ Quantization Function implementation.
     * This function convert an float value to low-precision float
     */
    FPConvertHelper helper; FPConvertHelper rounding_helper;
    float Unscaled_FP32 = static_cast<float>(value) / scale;  //将value转化为浮点型/缩放因子
    
    helper.value = Unscaled_FP32;
	int32_t exponent_min  = -(1 << (exponent - 1)) + 1;   //FP8 E4M3中指数最小为-7（0000）
    int32_t exponent_max  = (1 << (exponent - 1));        //FP8 E4M3指数最大为8（1111）

    // Following code will process exponent overflow
    /* For FP8 E4M3, the maximum exponent value should be 8.                                  */
    /* The Maximum number of FP8 E4M3 should be (0 1111 111) = 480                            */
    /* We call it as theoretical_maximum, FP8 E4M3 can not represent a number larger than it. */
    uint32_t fp32_sign    = 0;  //符号+
    int32_t fp32_exp      = (exponent_max + 127) << 23;  //FP8的最大指数（8）转换到FP32为10000111（135-127）
    int32_t fp32_mantissa = ~(0x007FFFFF >> mantissa) & 0x007FFFFF;  //FP8的最大尾数转换到FP32
    /*0x007FFFFF:00000000 01111111 11111111 11111111 */
    /*右移3位:    00000000 00001111 11111111 11111111 */
    /*取反:       11111111 11110000 00000000 00000000 */
    /*与：        00000000 01110000 00000000 00000000 */
    helper.data = fp32_sign + fp32_mantissa + fp32_exp;  //FP8最大值转换到FP32为01000011 11110000 00000000 00000000=480
    float theoretical_maximum = helper.value;

    if (Unscaled_FP32 > min(clip_max, theoretical_maximum)) 
        return min(clip_max, theoretical_maximum);
    if (Unscaled_FP32 < max(clip_min, -theoretical_maximum)) 
        return max(clip_min, -theoretical_maximum);
    /*当 Unscaled FP32 数据已经超出 FP8 的表示范围，即 Unscaled FP32 的幅值大于 448，那么直接进行截断，此时为浮点上溢出。*/


    // Code start from here will convert number within fp8 range.
    // Following code will Split float32 into sign, exp, mantissa
    /* IEEE 754 Standard: 1 bit sign, 8 bit exponent, 23 bit mantissa */

    /* In binary 10000000 00000000 00000000 00000000 = 0x80000000 in Hex */
    /* In binary 01111111 10000000 00000000 00000000 = 0x7F800000 in Hex */
    /* In binary 00000000 01111111 11111111 11111111 = 0x007FFFFF in Hex */

    /* Tool: https://www.h-schmidt.net/FloatConverter/IEEE754.html */
    helper.value  = Unscaled_FP32;  //将目标FP32重新赋值
    fp32_sign     = helper.data & 0x80000000;
    fp32_exp      = helper.data & 0x7F800000;
    fp32_mantissa = helper.data & 0x007FFFFF;

    // Following code will process exponent underflow
    /* Float underflow means fp32_exp is smaller than exponent_min          */
    /* Where exponent_min is the minimum exponent value of quantized float. */
    /* For FP8 E4M3, the minimum exponent value should be -7.               */
    /* The Min Subnormal value of FP8 E4M3 should be (0 0000 001) = 2^-9    */
    /* The Min normal value of FP8 E4M3 should be (0 0001 000) = 2^-6       */
	if (((fp32_exp >> 23) - 127) < exponent_min + 1){      //指数<-6 
        // following divide might have some problems
        // but it is the simplest method with very limited error.
        float min_subnormal = 1.0f / (1 << ((1 << (exponent - 1)) + mantissa - 2));  //单精度1/2^9
        return _round2int(Unscaled_FP32 / min_subnormal, rounding) * min_subnormal;  
        //当 Unscaled FP32 数据小于规范化 FP8 能够表达的最小值，此时浮点下溢出，此时我们除以非规范化的 FP8 最小值并取整。
	}

    /* high precision mantissa convert to low precision mantissa requires rounding                         */
    /* Here we apply a tricky method to round mantissa:                                                    */
    /* We create another float, which sign = 0, exponent = 127, mantissa = fp32_mantissa << (23 - mantissa) */
    /* Then we directly round this float to int, result here is what we want, you can prove it by yourself */
    rounding_helper.data = ((fp32_mantissa << (mantissa)) & 0x007FFFFF) + 0x3F800000; //取出FP8不可表示的20位尾数

    uint32_t round_bit = _round2int(rounding_helper.value - 1, rounding); // -1进行四舍五取偶六入

    // process mantissa
    fp32_mantissa = ((fp32_mantissa >> (23 - mantissa)) + round_bit) << (23 - mantissa); //将尾数后20位去掉，加上前面四舍五入的值形成新的3位尾数
    helper.data = fp32_sign + fp32_mantissa + fp32_exp;

    return CLIP<float>(helper.value, clip_min, clip_max);
}