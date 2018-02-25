using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Drawing;

namespace SharpSIMD
{
    unsafe class Program
    {
        static void Main(string[] args)
        {
            var opts = Options.Default;
            opts.AVX2 = false;
            //opts.breakAtFunctionEntry = true;
            SIMDcode s = new SIMDcode(opts, _ =>
            {
                // BGRx to Y, fast but not super accurate
                // using approximate Y weights from JPEG in Q7:
                // Wr = 0.299 -> 38/128 (0.296875)
                // Wg = 0.587 -> 75/128 (0.5859375)
                // Wb = 0.114 -> 15/128 (0.1171875)
                // Q8 would be more complicated because pmaddubsw's second operand is *signed* bytes and Wg would be 150
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
            }, _ =>
            {
                // upscale x2
                var dst = _.arg(0);
                var src = _.arg(1);
                var width = _.arg(2);
                var height = _.arg(3);
                // upscale vertically
                var prev = _._mm_setzero_si128();
                _.repeat(width, 16);
                {
                    var t0 = _._mm_loadu_si128(src);
                    
                    //t3 = 0;
                }
                _.endrepeat();
            });

            Bitmap input = (Bitmap)Bitmap.FromFile(@"C:\Users\Harold\Pictures\Barns_grand_tetons.jpg");
            Bitmap output = new Bitmap(input.Width, input.Height, System.Drawing.Imaging.PixelFormat.Format8bppIndexed);
            var bdin = input.LockBits(new Rectangle(Point.Empty, input.Size), System.Drawing.Imaging.ImageLockMode.ReadOnly, System.Drawing.Imaging.PixelFormat.Format32bppRgb);
            var bdout = output.LockBits(new Rectangle(Point.Empty, input.Size), System.Drawing.Imaging.ImageLockMode.WriteOnly, System.Drawing.Imaging.PixelFormat.Format8bppIndexed);
            System.Drawing.Imaging.ColorPalette pal = output.Palette;
            for (int i = 0; i < 256; i++)
                pal.Entries[i] = Color.FromArgb(255, i, i, i);
            output.Palette = pal;

            var foo = s.GetDelegate<Foo>(0);
            foo((uint*)bdout.Scan0, (uint*)bdin.Scan0, input.Width * input.Height);

            input.UnlockBits(bdin);
            output.UnlockBits(bdout);

            output.Save(@"C:\Users\Harold\Pictures\Barns_grand_tetons_graytest.png", System.Drawing.Imaging.ImageFormat.Png);

            return;
        }

        delegate void Foo(uint* dst, uint* src, int length);
    }
}
