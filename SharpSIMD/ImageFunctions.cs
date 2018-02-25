using System;

namespace SharpSIMD
{
    static unsafe class ImageFunctions
    {
        public delegate void ArrayTransform(byte* dst, int* src, ulong size);

        class Inner
        {
            public Inner()
            {
                code = new SIMDcode(Options.Default,
                _ =>
                {
                    // BGRx to Y, fast but not super accurate
                    // assumes there is no row-padding in the input (a Sytem.Drawing.Bitmap with Format32bppRgb satisfies this requirement)
                    // assumes there is no row-padding in the output (ensure width is multiple of 4 or write it into an array)
                    // assumes the image has at least 16 pixels or is padded up to such a size (64 bytes in, 16 bytes out)
                    // uses approximate Y weights from JPEG in Q7:
                    // Wr = 0.299 -> 38/128 (0.296875)
                    // Wg = 0.587 -> 75/128 (0.5859375)
                    // Wb = 0.114 -> 15/128 (0.1171875)
                    // Q8 doesn't work because pmaddubsw's second operand is *signed* bytes and Wg would be 150
                    // Q7 would not work in general because of the saturating addition, but the maximum sum
                    // that can occur with these weights is (38+75)*255=28815 which fits in a signed word.
                    // Result is rounded by adding 0.5, then truncated.
                    // _mm_packus_epi16 applies unsigned clipping but its input is never out of range.
                    byte rw = 38, gw = 75, bw = 15;
                    var dst = _.arg(0);
                    var src = _.arg(1);
                    var len = _.arg(2);
                    var weight = _._mm_setr_epi8(bw, gw, rw, 0, bw, gw, rw, 0, bw, gw, rw, 0, bw, gw, rw, 0);
                    var rounding = _._mm_set1_epi16(64);
                    _.repeat(len, 16);
                    {
                        var p0 = _._mm_loadu_si128(src);
                        var p1 = _._mm_loadu_si128(src, 16);
                        var p2 = _._mm_loadu_si128(src, 32);
                        var p3 = _._mm_loadu_si128(src, 48);
                        var t0 = _._mm_maddubs_epi16(p0, weight);
                        var t1 = _._mm_maddubs_epi16(p1, weight);
                        var t2 = _._mm_maddubs_epi16(p2, weight);
                        var t3 = _._mm_maddubs_epi16(p3, weight);
                        _.assign(src, _.add(src, 64));
                        t0 = _._mm_hadd_epi16(t0, t1);
                        t1 = _._mm_hadd_epi16(t2, t3);
                        t0 = _._mm_add_epi16(t0, rounding);
                        t1 = _._mm_add_epi16(t1, rounding);
                        t0 = _._mm_srli_epi16(t0, 7);
                        t1 = _._mm_srli_epi16(t1, 7);
                        t0 = _._mm_packus_epi16(t0, t1);
                        _._mm_storeu_si128(dst, t0);
                        _.assign(dst, _.add(dst, 16));
                    }
                    _.endrepeat();
                });
                Rgb32_to_Gray = code.GetDelegate<ArrayTransform>(0);
            }

            SIMDcode code;

            public ArrayTransform Rgb32_to_Gray;
        }

        static Lazy<Inner> inner = new Lazy<Inner>();

        public static ArrayTransform Rgb32_to_Gray
        {
            get { return inner.Value.Rgb32_to_Gray; }
        }
    }
}
