//#define NOCAST

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

#if NOCAST
using __m128 = SharpSIMD.__xmm;
using __m128i = SharpSIMD.__xmm;
using __m128d = SharpSIMD.__xmm;
using __m256 = SharpSIMD.__ymm;
using __m256i = SharpSIMD.__ymm;
using __m256d = SharpSIMD.__ymm;
#endif

namespace SharpSIMD
{
    sealed class SIMDcode
    {
        NativeCode nc;
        int[] functionOffsets;

        public SIMDcode(Options options = null, params Action<IBuilder>[] ctors)
        {
            if (IntPtr.Size != 8)
                throw new Exception("This version of SharpSIMD does not work in 32bit mode.");
            System.Diagnostics.Debug.WriteLine("");
            System.Diagnostics.Debug.WriteLine("Compiling SIMD code block");
            var sw = System.Diagnostics.Stopwatch.StartNew();
            if (options == null)
                options = Options.Default;
            CodeBuffer cb = new CodeBuffer();
            int nextlabel = 1;
            int nextdlabel = -10;
            List<int> funcLabels = new List<int>();
            int f = 0;
            foreach (var ctor in ctors)
            {
                System.Diagnostics.Debug.WriteLine("Function {0}", f);
                f++;
                Builder b = new Builder(options, nextdlabel);
                ctor(b);
                nextdlabel = b.nextdatalabel;
                funcLabels.Add(b.Compile(cb, ref nextlabel));
            }
            Dictionary<int, int> labelPos = new Dictionary<int, int>();
            nc = cb.Save(labelPos);
            functionOffsets = new int[funcLabels.Count];
            for (int i = 0; i < functionOffsets.Length; i++)
                functionOffsets[i] = labelPos[funcLabels[i]];
            sw.Stop();
            System.Diagnostics.Debug.WriteLine("Compiling SIMD code block took: {0}ms", sw.ElapsedMilliseconds);
            System.Diagnostics.Debug.WriteLine("");
            return;
        }

        public T GetDelegate<T>(int index)
        {
            return (T)(object)nc.GetDelegate(typeof(T), functionOffsets[index]);
        }

        sealed class Builder : IBuilder
        {
            Options options;
            int nextvreg = 5;
            int nextvxmm = 0x100005;
            const int m = 0xFFFFF;
            List<Instr> inst = new List<Instr>(16);
            List<Block> data = new List<Block>();
            public int nextdatalabel = -10;

            const int calleeSaveGPRs = 0xf0f8;
            const int calleeSaveXMMs = 0xffc0;
            const int RSP = 4;

            public Builder(Options options, int nextdlabel)
            {
                this.options = options;
                inst.Add(new Instr(Op.nop, 0, 0, 0));
                this.nextdatalabel = nextdlabel;
            }

            internal int Compile(CodeBuffer cb, ref int nextlabel)
            {
                cb.AddData(data);

                int funclabel = nextlabel++;
                cb.OpenBlock(funclabel, options.FunctionAlignment);

                optimize();

                var allocation = allocateRegisters(options);                
                var regs = allocation.Item1;
                int usedGPRs = allocation.Item2;
                int usedXMMs = allocation.Item3;

                // if any YMM register is ever used, there will be a vzeroupper
                bool usedYMM = false;

                lowerFMAs(regs);

                if (options.BreakAtFunctionEntry)
                    cb.Writeb(0xCC);

                int pushOffset = 8;
                // push callee-save GPRs
                System.Diagnostics.Debug.WriteLine("Saving {0} callee-save GPRs.", popcnt16(usedGPRs & calleeSaveGPRs));
                if ((usedGPRs & calleeSaveGPRs) != 0)
                {
                    for (int i = 3; i < 16; i++)
                    {
                        if ((usedGPRs & calleeSaveGPRs & (1 << i)) == 0)
                            continue;
                        push_r64(cb, i);
                        pushOffset += 8;
                    }
                }
                int stackallocsize = 0;
                // save callee-save XMMs (upper half of YMM is volatile)
                System.Diagnostics.Debug.WriteLine("Saving {0} callee-save XMM registers.", popcnt16(usedXMMs & calleeSaveXMMs));
                if ((usedXMMs & calleeSaveXMMs) != 0)
                {
                    int xmmalloc = popcnt16(usedXMMs & calleeSaveXMMs) * 16;
                    stackallocsize = (pushOffset & 8) + xmmalloc;
                    subopt_r_imm(cb, 4, stackallocsize);
                    for (int i = 6; i < 16; i++)
                    {
                        if ((usedXMMs & (1 << i)) == 0)
                            continue;
                        xmmalloc -= 16;
                        movaps_rm_r(cb, new MemoryArg(4, 0, 0, xmmalloc), i);
                    }
                }
                else
                {
                    // extra push for alignment
                    if ((pushOffset & 8) == 8)
                        push_r64(cb, 1);
                    stackallocsize = pushOffset & 8;
                }

                var repeatStack = new Stack<Tuple<int, int, int, Instr>>();

                foreach (var i in inst)
                {
                    uint op = (uint)i.op;
                    if (i.op == Op.stmxcsr)
                    {
                        // stmxcsr is weird, needs to write to memory
                        int offset = -stackallocsize - pushOffset - 12;
                        System.Diagnostics.Debug.Assert(offset >= -128 && offset <= 127);
                        cb.Write(new byte[] { 0xc5, 0xf8, 0xae, 0x5c, 0x24 });
                        cb.Writeb(offset);
                        cb.Writeb(0x8B);
                        ModRMSIB(cb, regs[i.R], new MemoryArg(RSP, 0, 0, offset), null);
                    }
                    else if (i.op.HasFlag(Op.nonvec))
                    {
                        // normal instructions
                        map_select ms = (map_select)((op >> 12) & 3);
                        if (i.op.HasFlag(Op.alugrp))
                        {
                            if (i.Mem == null && i.B != 0)
                            {
                                int a = regs[i.A];
                                int b = regs[i.B];
                                int r = regs[i.R];
                                if (r != 1 && r == b && (i.op & ~Op.W) != Op.sub && (i.op & ~Op.W) != Op.cmp)
                                {
                                    // op b, a (not sub or cmp)
                                    int t = a;
                                    a = b;
                                    b = t;
                                }
                                if (r == a)
                                {
                                    // op r(=a), r/m(=b)
                                    int rex = Rex(i.op.HasFlag(Op.W), a > 7, b > 7);
                                    // technically REX with none of its bits set is not useless, 
                                    // but access to sil/dil/spl/bpl is not used so here a REX with none of its bit set /is/ useless
                                    if (rex != 0x40)
                                        cb.Writeb(rex);
                                    int baseopcode = (byte)op;
                                    // op r, r/m[32/64] form
                                    cb.Writeb(baseopcode + 3);
                                    ModRM(cb, a, b);
                                }
                                else if (i.op == Op.add || i.op == Op.addq)
                                {
                                    // lea r,[a+b]
                                    int rex = Rex(i.op == Op.addq, r > 7, a > 7, b > 7);
                                    if (rex != 0x40)
                                        cb.Writeb(rex);
                                    cb.Writeb(0x8D);
                                    ModRMSIB(cb, r, new MemoryArg(a, b, 1, 0), null);
                                }
                                else
                                    throw new Exception("Same-reg constraint not satisfied");
                            }
                            else if (i.Mem == null)
                            {
                                int a = regs[i.A];
                                int r = regs[i.R];
                                if (r == a)
                                {
                                    // op r/m, imm
                                    int rex = Rex(i.op.HasFlag(Op.W), false, a > 7);
                                    // technically REX with none of its bits set is not useless, 
                                    // but access to sil/dil/spl/bpl is not used so here a REX with none of its bit set /is/ useless
                                    if (rex != 0x40)
                                        cb.Writeb(rex);
                                    int baseopcode = (byte)op;
                                    int regfield = baseopcode >> 3;
                                    if ((sbyte)i.Imm == i.Imm)
                                    {
                                        cb.Writeb(0x83);
                                        ModRM(cb, regfield, r);
                                        cb.Writeb(i.Imm);
                                    }
                                    else
                                    {
                                        cb.Writeb(0x81);
                                        ModRM(cb, regfield, r);
                                        cb.Write(i.Imm);
                                    }
                                }
                                else if (i.op == Op.add || i.op == Op.addq)
                                {
                                    // lea r,[a+imm]
                                    int rex = Rex(i.op == Op.addq, r > 7, a > 7);
                                    if (rex != 0x40)
                                        cb.Writeb(rex);
                                    cb.Writeb(0x8D);
                                    ModRMSIB(cb, r, new MemoryArg(a, 0, 0, i.Imm), null);
                                }
                                else
                                    System.Diagnostics.Debug.Assert(regs[i.R] == regs[i.A], "Same-reg constraint not satisfied");
                            }
                            else
                                throw new NotImplementedException();
                        }
                        else
                        {
                            if (i.Mem == null)
                            {
                                if (i.op.HasFlag(Op.samereg))
                                {
                                    System.Diagnostics.Debug.Assert(regs[i.R] == regs[i.A], "Same-reg constraint not satisfied");
                                    int b = regs[i.B];
                                    int r = regs[i.R];
                                    int rex = Rex(i.op.HasFlag(Op.W), r > 7, b > 7);
                                    if (rex != 0x40)
                                        cb.Writeb(rex);
                                    if (ms == map_select.x0F)
                                        cb.Writeb(0x0F);
                                    cb.Writeb((byte)op);
                                    ModRM(cb, r, b);
                                }
                                else
                                {
                                    int a = regs[i.A];
                                    int r = regs[i.R];
                                    int rex = Rex(i.op.HasFlag(Op.W), r > 7, a > 7);
                                    if (rex != 0x40)
                                        cb.Writeb(rex);
                                    if (ms == map_select.x0F)
                                        cb.Writeb(0x0F);
                                    cb.Writeb((byte)op);
                                    ModRM(cb, r, a);
                                }
                            }
                            else
                                throw new NotImplementedException();
                        }
                    }
                    else if (i.op.HasFlag(Op.special))
                    {
                        if (i.op == Op.ignore ||
                            i.op == Op.nop)
                            continue;
                        if (i.op == Op.ldarg)
                        {
                            // first 4 args are handled by the register allocator, 
                            // so only args that are passed through the stack make it here
                            int r = regs[i.R];
                            cb.Writeb(Rex(true, r > 7, false));
                            cb.Writeb(0x8B);
                            ModRMSIB(cb, r, new MemoryArg(RSP, 0, 0, -stackallocsize - pushOffset - 8 - 8 * i.Imm), null);
                        }
                        else if (i.op == Op.repeat_z)
                        {
                            int loopBody = nextlabel++;
                            int loopCond = nextlabel++;
                            int loopAfter = nextlabel++;
                            int r = regs[i.R];
                            int a = regs[i.A];
                            if (r != a)
                            {
                                // mov r, a
                                cb.Writeb(Rex(true, r > 7, a > 7));
                                cb.Writeb(0x8B);
                                ModRM(cb, r, a);
                            }
                            // test count, count
                            cb.Writeb(Rex(true, a > 7, a > 7));
                            cb.Writeb(0x85);
                            ModRM(cb, a, a);
                            // jz skip
                            cb.Writeb(0x74);
                            cb.SetGoto(loopAfter);
                            cb.OpenBlock(loopBody, options.LoopAlignment);
                            repeatStack.Push(new Tuple<int, int, int, Instr>(loopBody, loopCond, loopAfter, i));
                        }
                        else if (i.op == Op.repeat)
                        {
                            int loopBody = nextlabel++;
                            int loopCond = nextlabel++;
                            int loopAfter = nextlabel++;
                            int r = regs[i.R];
                            int a = regs[i.A];
                            if (r != a)
                            {
                                // mov r, a
                                cb.Writeb(Rex(true, r > 7, a > 7));
                                cb.Writeb(0x8B);
                                ModRM(cb, r, a);
                            }
                            cb.OpenBlock(loopBody, options.LoopAlignment);
                            repeatStack.Push(new Tuple<int, int, int, Instr>(loopBody, loopCond, loopAfter, i));
                        }
                        else if (i.op == Op.endrepeat)
                        {
                            var rep = repeatStack.Pop();
                            int loopBody = rep.Item1;
                            int loopCond = rep.Item2;
                            int loopAfter = rep.Item3;
                            var repeat = rep.Item4;
                            cb.OpenBlock(loopCond);
                            subopt_r_imm(cb, regs[repeat.R], repeat.Imm);
                            cb.Writeb(0x75);
                            cb.SetGoto(loopBody);
                            cb.OpenBlock(loopAfter);
                        }
                        else if ((i.op & ~Op.W) == Op.assign)
                        {
                            int r = regs[i.R];
                            int rm = regs[i.A];
                            if (r == rm)
                                continue;
                            int rex = Rex(i.op.HasFlag(Op.W), r > 7, rm > 7);
                            if (rex != 0x40)
                                cb.Writeb(rex);
                            cb.Writeb(0x8B);
                            cb.Writeb(0xC0 | ((r & 7) << 3) | (rm & 7));
                        }
                        else if (i.op == Op.vassign || i.op == Op.viassign || i.op == Op.vlassign || i.op == Op.vliassign)
                        {
                            int r = regs[i.R];
                            int rm = regs[i.A];
                            if (r == rm)
                                continue;
                            if (i.op == Op.vassign || i.op == Op.vlassign)
                            {
                                // vmovaps
                                vex(cb, 0, i.op == Op.vlassign ? 1 : 0, 0, r, 0, rm);
                                cb.Writeb(0x28);
                                ModRM(cb, r, rm);
                            }
                            else
                            {
                                // vmovdqa
                                vex(cb, 0, i.op == Op.vliassign ? 1 : 0, 0, r, 0, rm, pp.x66);
                                cb.Writeb(0x6F);
                                ModRM(cb, r, rm);
                            }
                        }
                        else
                            throw new NotImplementedException();
                    }
                    else if (i.op.HasFlag(Op.vex))
                    {
                        map_select ms = (map_select)((op >> 12) & 3);
                        pp p = (pp)((op >> 8) & 3);
                        int L = (int)((op >> 23) & 1);
                        int W = (int)((op >> 16) & 1);
                        int vvvvMeaning = (int)((op >> 20) & 3);
                        Op opnl = (Op)(op & ~(uint)Op.L);


                        if (vvvvMeaning == 1)
                        {
                            // NDS
                            if (i.Mem == null)
                            {
                                // XOR is allowed to have bogus operands if they are the same
                                if (opnl == Op.xorps || opnl == Op.xorpd || opnl == Op.pxor)
                                {
                                    System.Diagnostics.Debug.Assert(i.A == i.B || regs.ContainsKey(i.A) && regs.ContainsKey(i.B));
                                    vex(cb, W, L, regs.GetOrDefault(i.A), regs[i.R], 0, regs.GetOrDefault(i.B), p, ms);
                                }
                                else
                                    vex(cb, W, L, regs[i.A], regs[i.R], 0, regs.GetOrDefault(i.B), p, ms);
                                cb.Write((byte)op);
                                ModRM(cb, regs[i.R], regs.GetOrDefault(i.B));
                                if (i.op.HasFlag(Op.imm8))
                                    cb.Writeb(i.Imm);
                                else if (i.op.HasFlag(Op.is4))
                                    cb.Writeb(regs[i.C] << 4);
                            }
                            else if (!i.op.HasFlag(Op.store))
                            {
                                vex(cb, W, L, regs[i.A], regs[i.R], regs.GetOrDefault(i.Mem.index), regs[i.Mem.basereg], p, ms);
                                cb.Write((byte)op);
                                ModRMSIB(cb, regs[i.R], i.Mem, regs);
                                if (i.op.HasFlag(Op.is4))
                                    cb.Writeb(regs[i.C] << 4);
                            }
                            else
                                throw new NotImplementedException();
                        }
                        else if (vvvvMeaning == 2)
                        {
                            // NDD
                            int r = ((int)i.op >> 17) & 7;
                            if (i.Mem == null)
                            {
                                vex(cb, W, L, regs[i.R], 0, 0, regs[i.A], p, ms);
                                cb.Write((byte)op);
                                ModRM(cb, r, regs[i.A]);
                                if (i.op.HasFlag(Op.imm8))
                                    cb.Writeb(i.Imm);
                            }
                            else
                                throw new NotImplementedException();
                        }
                        else if (vvvvMeaning == 0)
                        {
                            // no vvvv
                            if (i.Mem == null)
                            {
                                if ((i.op & ~Op.L) == Op.ptest)
                                {
                                    // ptest is a weirdo with A and B but no R
                                    vex(cb, W, L, 0, regs[i.A], 0, regs[i.B], p, ms);
                                    cb.Write((byte)op);
                                    ModRM(cb, regs[i.A], regs[i.B]);
                                }
                                else
                                {
                                    vex(cb, W, L, 0, regs[i.R], 0, regs[i.A], p, ms);
                                    cb.Write((byte)op);
                                    ModRM(cb, regs[i.R], regs[i.A]);
                                    if (i.op.HasFlag(Op.imm8))
                                        cb.Writeb(i.Imm);
                                }
                            }
                            else if (!i.op.HasFlag(Op.store))
                            {
                                if (i.Mem.basereg == 0)
                                    vex(cb, W, L, 0, regs[i.R], 0, 0, p, ms); // RIP-rel
                                else
                                    vex(cb, W, L, 0, regs[i.R], regs.GetOrDefault(i.Mem.index), regs[i.Mem.basereg], p, ms);
                                cb.Write((byte)op);
                                if (i.Mem.basereg == 0)
                                {
                                    // RIP-rel
                                    // rip-offset is not from the position /directly/ after the RIP-rel offset
                                    // immediates come after the offset field but count towards the RIP from which the offset is measured
                                    // this ofs fixes that
                                    int ofs = 0;
                                    if (i.op.HasFlag(Op.imm8))
                                        ofs = 1;
                                    ModRMSIB(cb, regs[i.R], new MemoryArg(0, 0, 0, ofs), regs);
                                    cb.SetRIPREL(i.Mem.offset);
                                    cb.OpenBlock(nextlabel++);
                                }
                                else
                                    ModRMSIB(cb, regs[i.R], i.Mem, regs);
                                if (i.op.HasFlag(Op.imm8))
                                    cb.Writeb(i.Imm);
                            }
                            else
                            {
                                vex(cb, W, L, 0, regs[i.A], regs.GetOrDefault(i.Mem.index), regs[i.Mem.basereg], p, ms);
                                cb.Write((byte)op);
                                ModRMSIB(cb, regs[i.A], i.Mem, regs);
                                if (i.op.HasFlag(Op.imm8))
                                    cb.Writeb(i.Imm);
                            }
                        }
                        else
                            throw new NotImplementedException();
                    }
                    else
                        throw new NotImplementedException();
                }

                if (usedYMM)
                {
                    cb.Writeb(0xC5);
                    cb.Writeb(0xF8);
                    cb.Writeb(0x77);
                }

                // restore callee-save XMMs
                if ((usedXMMs & calleeSaveXMMs) != 0)
                {
                    int xmmalloc = 0;                    
                    for (int i = 15; i >= 6; i--)
                    {
                        if ((usedXMMs & (1 << i)) == 0)
                            continue;                        
                        movaps_r_rm(cb, i, new MemoryArg(RSP, 0, 0, xmmalloc));
                        xmmalloc += 16;
                    }
                    addopt_r_imm(cb, RSP, stackallocsize);
                }
                else
                {
                    // extra pop for alignment
                    if ((pushOffset & 8) == 8)
                        pop_r64(cb, 1);
                }
                // pop callee-save registers in reverse
                if ((usedGPRs & calleeSaveGPRs) != 0)
                {
                    for (int i = 15; i >= 3; i--)
                    {
                        if ((usedGPRs & calleeSaveGPRs & (1 << i)) == 0)
                            continue;
                        pop_r64(cb, i);
                    }
                }
                
                // ret
                cb.Writeb(0xC3);

                return funclabel;
            }

            static int Rex(bool W, bool R, bool B, bool X = false)
            {
                int r = 0x40;
                if (B)
                    r |= 1;
                if (X)
                    r |= 2;
                if (R)
                    r |= 4;
                if (W)
                    r |= 8;
                return r;
            }

            static void ModRM(CodeBuffer cb, int r, int rm)
            {
                cb.Writeb(0xC0 | ((r & 7) << 3) | (rm & 7));
            }

            static void ModRMSIB(CodeBuffer cb, int r, MemoryArg m, Dictionary<int, int> regs)
            {
                int basereg = regs == null ? m.basereg : regs.GetOrDefault(m.basereg);
                int index = regs == null ? m.index : regs.GetOrDefault(m.index);
                if (basereg == 0)
                {
                    // RIP-rel
                    cb.Writeb(((r & 7) << 3) | 5);
                    cb.Write(m.offset);
                }
                else if (m.scale == 0 && (basereg & 7) != 4)
                {
                    if (m.offset == 0 && (basereg & 7) != 5)
                        cb.Writeb(((r & 7) << 3) | (basereg & 7));
                    else if ((sbyte)m.offset == m.offset)
                    {
                        cb.Writeb(0x40 | ((r & 7) << 3) | (basereg & 7));
                        cb.Write((byte)m.offset);
                    }
                    else
                    {
                        cb.Writeb(0x80 | ((r & 7) << 3) | (basereg & 7));
                        cb.Write(m.offset);
                    }
                }
                else
                {
                    // if scale = 0, we're only here because the basereg is rsp or r12
                    // set the index to "no index"
                    if (m.scale == 0)
                        index = 4;
                    int scale = m.scale >> 1;
                    if (scale == 4)
                        scale = 3;
                    scale <<= 6;
                    if (m.offset == 0 && (basereg & 7) != 5)
                    {
                        cb.Writeb(((r & 7) << 3) | 4);
                        cb.Writeb(((index & 7) << 3) | (basereg & 7) | scale);
                    }
                    else if ((sbyte)m.offset == m.offset)
                    {
                        cb.Writeb(((r & 7) << 3) | 0x44);
                        cb.Writeb(((index & 7) << 3) | (basereg & 7) | scale);
                        cb.Write((byte)m.offset);
                    }
                    else
                    {
                        cb.Writeb(((r & 7) << 3) | 0x84);
                        cb.Writeb(((index & 7) << 3) | (basereg & 7) | scale);
                        cb.Write(m.offset);
                    }
                }
            }

            enum pp
            {
                none,
                x66,
                xF3,
                xF2
            }

            enum map_select
            {
                x0F = 1,
                x0F38 = 2,
                x0F3A = 3
            }

            static void vex(CodeBuffer cb, int W, int L, int vvvv, int r, int X, int rm, pp p = pp.none, map_select m = map_select.x0F)
            {
                if (rm < 8 && X < 8 && m == map_select.x0F && W == 0)
                {
                    // 2-byte VEX
                    cb.Writeb(0xC5);
                    cb.Writeb(((~r & 8) << 4) | (int)p | (L << 2) | ((vvvv ^ 15) << 3));
                }
                else
                {
                    // 3-byte VEX
                    cb.Writeb(0xC4);
                    cb.Writeb(((~r & 8) << 4) | ((~X & 8) << 3) |((~rm & 8) << 2) | (int)m);
                    cb.Writeb((int)p | (L << 2) | ((vvvv ^ 15) << 3) | (W << 7));
                }
            }

            static void push_r64(CodeBuffer cb, int r)
            {
                if (r > 7)
                    cb.Writeb(0x41);
                cb.Writeb(0x50 + (r & 7));
            }

            static void pop_r64(CodeBuffer cb, int r)
            {
                if (r > 7)
                    cb.Writeb(0x41);
                cb.Writeb(0x58 + (r & 7));
            }

            static void subopt_r_imm(CodeBuffer cb, int r, int imm)
            {
                cb.Writeb(0x48 | (r >> 3));
                if (imm == 128)
                {
                    // add -128
                    cb.Writeb(0x83);
                    ModRM(cb, 0, r);
                    cb.Writeb(imm);
                }
                else if ((sbyte)imm == imm)
                {
                    cb.Writeb(0x83);
                    ModRM(cb, 5, r);
                    cb.Writeb(imm);
                }
                else
                {
                    cb.Writeb(0x81);
                    ModRM(cb, 5, r);
                    cb.Write(imm);
                }
            }

            static void addopt_r_imm(CodeBuffer cb, int r, int imm)
            {
                cb.Writeb(0x48 | (r >> 3));
                if (imm == -128)
                {
                    // sub -128
                    cb.Writeb(0x83);
                    ModRM(cb, 5, r);
                    cb.Writeb(imm);
                }
                else if ((sbyte)imm == imm)
                {
                    cb.Writeb(0x83);
                    ModRM(cb, 0, r);
                    cb.Writeb(imm);
                }
                else
                {
                    cb.Writeb(0x81);
                    ModRM(cb, 0, r);
                    cb.Write(imm);
                }
            }

            static void movaps_rm_r(CodeBuffer cb, MemoryArg m, int r)
            {
                vex(cb, 0, 0, 0, r, m.index, m.basereg);
                cb.Writeb(0x29);
                ModRMSIB(cb, r, m, null);
            }

            static void movaps_r_rm(CodeBuffer cb, int r, MemoryArg m)
            {
                vex(cb, 0, 0, 0, r, m.index, m.basereg);
                cb.Writeb(0x28);
                ModRMSIB(cb, r, m, null);
            }

            static int popcnt16(int x)
            {
                x = (x & 0x5555) + ((x & 0xAAAA) >> 1);
                x = (x & 0x3333) + ((x & 0xCCCC) >> 2);
                x = (x & 0x0F0F) + ((x & 0xF0F0) >> 4);
                return (x & 0xFF) + (x >> 8);
            }

            Tuple<Dictionary<int, int>, int, int> allocateRegisters(Options options)
            {
#if DEBUG
                var sw = System.Diagnostics.Stopwatch.StartNew();
#endif

                // temp list to hold dependencies and outputs
                List<int> c = new List<int>(4);

                // start and end points of live ranges
                int[][] start, end;

                int loopCount = 0, xmmmov = 0, nonvecmov = 0;


                do
                {
                    start = new int[2][] { new int[nextvreg + 1], new int[nextvxmm + 1 & 0xfffff] };
                    end = new int[2][] { new int[nextvreg + 1], new int[nextvxmm + 1 & 0xfffff] };

                    // find starts of live ranges
                    Stack<int> live_repeats = new Stack<int>();
                    for (int i = 0; i < inst.Count; i++)
                    {
                        Instr x = inst[i];
                        // for vregs written to by this instruction, 
                        // if the live range has not started yet start it here
                        c.Clear();
                        x.creates(c);
                        foreach (var r in c)
                        {
                            if (start[r >> 20][r & m] == 0)
                                start[r >> 20][r & m] = i;
                        }

                        if (x.op == Op.repeat_z || x.op == Op.repeat)
                            live_repeats.Push(i);
                        else if (x.op == Op.endrepeat)
                        {
                            x.A = live_repeats.Pop();
                            inst[i] = x;
                        }
                    }
                    if (live_repeats.Count != 0)
                        throw new UnbalancedRepeatException("Unbalanced repeat at " + live_repeats.Peek());

                    for (int i = inst.Count - 1; i >= 0; i--)
                    {
                        Instr x = inst[i];
                        c.Clear();
                        if (x.op == Op.endrepeat)
                        {
                            live_repeats.Push(i);
                            int r = inst[x.A].R;
                            end[0][r & m] = Math.Max(i, end[0][r & m]);
                        }
                        else if (x.op == Op.repeat_z || x.op == Op.repeat)
                            live_repeats.Pop();

                        x.consumes(c);
                        foreach (var r in c)
                        {
                            int until = i;
                            // if the live range began outside of an repeat,
                            // extend it until the end of that repeat because it has to survive until the next loop iteration
                            foreach (var endat in live_repeats)
                            {
                                Instr endr = inst[endat];
                                System.Diagnostics.Debug.Assert(endr.op == Op.endrepeat, "endrepeat mismatch");
                                // is the repeat is later than the start of the live range, extend live range until matching endrepeat
                                if (endr.A >= start[r >> 20][r & m])
                                {
                                    until = endat;
                                    break;
                                }
                            }
                            end[r >> 20][r & m] = Math.Max(until, end[r >> 20][r & m]);
                        }
                    }

                    // for values that are never used, no end is set, fix that here
                    // this is necessary to not break the range overlap test later
                    for (int t = 0; t < start.Length; t++)
                    {
                        int s = t << 20;
                        for (int i = 0; i < start[t].Length; i++)
                        {
                            if (start[t][i] != 0 && end[t][i] == 0)
                            {
                                if (options.DiscardedValueIsAnError)
                                {
                                    var pos = (from j in Enumerable.Range(1, inst.Count - 1)
                                               where inst[j].R == i + s
                                               select j + 1).FirstOrDefault();
                                    throw new ValueNotUsedException("Value produced by instruction " + pos + " is never used, if this occurs inside a 'repeat' this may indicate a missing 'assign'");
                                }
                                System.Diagnostics.Debug.WriteLineIf(loopCount == 0,
                                    String.Format("Value produced by instruction {0} is never used.",
                                        (from j in Enumerable.Range(1, inst.Count - 1)
                                         where inst[j].R == i + s
                                         select j + 1).FirstOrDefault()));
                                end[t][i] = start[t][i];
                            }
                        }
                    }

                    // inputs are valid from the start even if the first actual write to them appears later
                    for (int r = 0; r < start.Length; r++)
                        for (int i = 1; i <= 4; i++)
                            start[r][i] = 0;

                    // try to coalesce to prevent copying
                    Dictionary<int, List<int>> preferSameAs = new Dictionary<int, List<int>>();

                    foreach (var i in inst)
                    {
                        Op opnl = i.op & ~Op.L;
                        Op opnw = i.op & ~Op.W;
                        if (opnw == Op.assign ||
                            i.op == Op.vassign ||
                            i.op == Op.viassign ||
                            i.op == Op.vlassign ||
                            i.op == Op.vliassign ||
                            i.op == Op.repeat ||
                            i.op == Op.repeat_z ||
                            (i.op.HasFlag(Op.alugrp) && opnw != Op.add))
                        {
                            preferSameAs.AddM(i.R, i.A);
                            preferSameAs.AddM(i.A, i.R);
                        }
                        else if (opnl >= Op.fmadd_ps && opnl <= Op.fmsubadd_pd)
                        {
                            if (i.Imm == 0)
                            {
                                if (i.R != i.A)
                                {
                                    preferSameAs.AddM(i.R, i.A);
                                    preferSameAs.AddM(i.A, i.R);
                                }
                                if (i.R != i.B)
                                {
                                    preferSameAs.AddM(i.R, i.B);
                                    preferSameAs.AddM(i.B, i.R);
                                }
                            }
                            else if (i.Imm == 2)
                            {
                                if (i.R != i.A)
                                {
                                    preferSameAs.AddM(i.R, i.A);
                                    preferSameAs.AddM(i.A, i.R);
                                }
                            }
                            else
                            {
                                if (i.R != i.B)
                                {
                                    preferSameAs.AddM(i.R, i.B);
                                    preferSameAs.AddM(i.B, i.R);
                                }
                            }
                        }
                    }

                    // allocate registers
                    Dictionary<int, int> mapping = new Dictionary<int, int>();
                    int usedGPRs = 0, usedXMMs = 0;
                    // allocate GPRs
                    // inputs already mapped
                    mapping[1] = 1;
                    mapping[2] = 2;
                    mapping[3] = 8;
                    mapping[4] = 9;
                    mapping[0x100000] = 0; // this is not strictly an input, just a random register for setzero
                    mapping[0x100001] = 0;
                    mapping[0x100002] = 1;
                    mapping[0x100003] = 2;
                    mapping[0x100004] = 3;
                    // some instructions have fixed registers
                    HashSet<int> premapped = new HashSet<int>();
                    foreach (var i in inst)
                    {
                        Op x = i.op & ~Op.L & ~Op.W;
                        switch (x)
                        {
                            case Op.pcmpistri:
                                mapping[i.R] = 1;
                                premapped.Add(i.R);
                                break;
                            case Op.pcmpistrm:
                                mapping[i.R] = 0;
                                premapped.Add(i.R);
                                break;
                            default:
                                break;
                        }
                    }
                    for (int t = 0; t < start.Length; t++)
                    {
                        int s = t << 20;
                        var registers = t == 0 ?
                            // GPRs ordered to prioritize caller-save and non-REX registers
                            new int[] { 0, 1, 2, 8, 9, 5, 3, 7, 10, 11, 14, 15, 5, 13, 12 } :
                            // YMMs ordered to prioritize caller-save registers, the rest doesn't matter
                            new int[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
                        for (int i = 5; i < start[t].Length; i++)
                        {
                            if (start[t][i] == 0 || premapped.Contains(i | s))
                                continue;
                            var conflict = new HashSet<int>(from j in Enumerable.Range(1, start[t].Length - 1)
                                                            where start[t][j] < end[t][i] &&
                                                                  start[t][i] < end[t][j] &&
                                                                  mapping.ContainsKey(j + s)
                                                            select mapping[j + s]);
                            var allowed = from j in registers
                                          where !conflict.Contains(j)
                                          select j;
                            int r = -1;
                            if (preferSameAs.ContainsKey(i + s) &&
                                (from j in preferSameAs[i + s]
                                 select mapping.ContainsKey(j)).Any())
                            {
                                var shouldBeSameAs = new HashSet<int>(from j in preferSameAs[i + s]
                                                                      where mapping.ContainsKey(j)
                                                                      select mapping[j]);
                                shouldBeSameAs.IntersectWith(allowed);
                                if (shouldBeSameAs.Count != 0)
                                    r = shouldBeSameAs.First();
                            }
                            if (r == -1)
                                r = allowed.First();
                            mapping.Add(i | s, r);
                            if (t == 0)
                                usedGPRs |= 1 << r;
                            else
                                usedXMMs |= 1 << r;
                        }
                    }

                    List<int> unsatisfiedFmas = new List<int>();
                    int unsatisfiedNonvec = 0;
                    for (int j = 0; j < inst.Count; j++)
                    {
                        Instr i = inst[j];
                        if ((i.op & ~Op.L) >= Op.fmadd_ps && (i.op & ~Op.L) <= Op.fmsubadd_pd)
                        {
                            // FMA imm indicates which (if any) operand is the mem
                            // 0: no mem
                            // 1: A
                            // 2: B
                            // 3: C
                            int r = mapping[i.R];
                            int m = i.Imm;
                            if (!(
                                // if no mem, pick any version
                                m == 0 && (r == mapping[i.A] || r == mapping[i.B] || r == mapping[i.C]) ||
                                // if mem=1,2: 231 or 132 version
                                (m == 1 || m == 2) && (r == mapping[i.A] || r == mapping[i.C]) ||
                                // if mem=3: 213 version (but can swap operands)
                                i.Imm == 3 && (r == mapping[i.B] || r == mapping[i.A])))
                                unsatisfiedFmas.Add(j);
                        }
                        else if (i.op.HasFlag(Op.nonvec) && i.op.HasFlag(Op.samereg))
                        {
                            if (mapping[i.R] != mapping[i.A])
                                unsatisfiedNonvec++;
                        }
                    }

                    if (unsatisfiedFmas.Count == 0 && unsatisfiedNonvec == 0)
                    {
#if DEBUG
                        sw.Stop();
                        System.Diagnostics.Debug.WriteLine("Register allocation took: {0} re-tries ({1}ms) inserting {2} xmm-movs and {3} GPR-movs", loopCount, sw.ElapsedMilliseconds, xmmmov, nonvecmov);
#endif
                        return new Tuple<Dictionary<int, int>, int, int>(mapping, usedGPRs, usedXMMs);
                    }

                    // insert copies for unsatisfied FMAs so they can be satisfied next time
                    // go backwards to easily preserve indexes
                    for (int i = unsatisfiedFmas.Count - 1; i >= 0; i--)
                    {
                        int index = unsatisfiedFmas[i];
                        Instr x = inst[index];
                        int dst = x.R;
                        int src;
                        switch (x.Imm)
                        {
                            default:
                                throw new Exception("Malformed FMA");
                            case 1:
                                // R = fma(mem, B, C) ->
                                // movaps R, C
                                // fma231 R, B, mem
                                src = inst[index].C;
                                x.A = dst;
                                break;
                            case 0:
                                // R = fma(A, B, C) ->
                                // movaps R, A
                                // fma213 R, B, C
                            case 2:
                                // R = fma(A, mem, C) ->
                                // movaps R, A
                                // fma132 R, C, mem
                            case 3:
                                // R = fma(A, B, mem) ->
                                // movaps R, A
                                // fma213 R, B, mem
                                src = inst[index].A;
                                x.A = dst;
                                break;
                        }
                        inst[index] = x;
                        Op copyop = Op.movaps;
                        if (((int)x.op & 1) == 1)
                            copyop = Op.movapd;
                        if (x.op.HasFlag(Op.L))
                            copyop |= Op.L;
                        inst.Insert(index, new Instr(copyop, dst, src, 0));
                        xmmmov++;
                    }

                    for (int j = inst.Count - 1; j >= 0; j--)
                    {
                        Instr i = inst[j];
                        if (i.op.HasFlag(Op.nonvec) && i.op.HasFlag(Op.samereg))
                        {
                            int iR = i.R, iA = i.A;
                            if (mapping[iR] == mapping[iA])
                                continue;
                            // unsatisfied non-vec, insert mov
                            i.A = iR;
                            inst[j] = i;
                            inst.Insert(j, new Instr(Op.assign | Op.W, iR, iA, 0));
                            nonvecmov++;
                        }
                    }

                    loopCount++;
                } while (true);
            }

            void optimize()
            {
#if DEBUG
                var sw = System.Diagnostics.Stopwatch.StartNew();
#endif
                int mergedLoads = 0;

                int[] usecounts = new int[Math.Max(nextvxmm & m, 6)];
                List<int> t = new List<int>();
                foreach (var i in inst)
                {
                    t.Clear();
                    i.consumes(t);
                    foreach (var r in t)
                    {
                        if (r > m)
                            usecounts[r & m]++;
                    }
                }

                List<int> remove = new List<int>();
                for (int i = 0; i < inst.Count; i++)
                {
                    Instr x = inst[i];
                    switch (x.op & ~Op.L)
                    {
                        default:
                            break;
                        case Op.movapd:
                        case Op.movaps:
                        case Op.movdqa:
                        case Op.movdqu:
                        case Op.movupd:
                        case Op.movups:
                            if (!options.MergeLoadWithUse ||
                                x.Mem == null || 
                                usecounts[x.R & m] != 1)
                                break;
                            // find use
                            for (int j = i + 1; j < inst.Count; j++)
                            {
                                Instr y = inst[j];
                                // stay within basic block
                                if (y.op == Op.repeat_z ||
                                    y.op == Op.endrepeat)
                                    break;
                                t.Clear();
                                y.consumes(t);
                                if (t.Contains(x.R))
                                {
                                    if ((y.op & ~Op.L) >= Op.fmadd_ps && (y.op & ~Op.L) <= Op.fmsubadd_pd)
                                    {
                                        y.Mem = x.Mem;
                                        remove.Add(i);
                                        if (y.A == x.R)
                                        {
                                            y.Imm = 1;
                                            y.A = 0;
                                        }
                                        else if (y.B == x.R)
                                        {
                                            y.Imm = 2;
                                            y.B = 0;
                                        }
                                        else if (y.C == x.R)
                                        {
                                            y.Imm = 3;
                                            y.C = 0;
                                        }
                                        else throw new Exception();
                                        mergedLoads++;
                                    }
                                    else if (y.A == x.R && y.op.HasFlag(Op.merge_mem_A) ||
                                        y.B == x.R && y.op.HasFlag(Op.merge_mem_B))
                                    {
                                        y.Mem = x.Mem;
                                        remove.Add(i);
                                        mergedLoads++;
                                    }
                                    inst[j] = y;
                                    break;
                                }
                            }
                            break;
                    }
                }

                for (int j = remove.Count - 1; j >= 0; j--)
                {
                    inst.RemoveAt(remove[j]);
                }

#if DEBUG
                sw.Stop();
                System.Diagnostics.Debug.WriteLine("Optimized in {0}ms:\r\nMerged {1} loads with uses\r\n", sw.ElapsedMilliseconds, mergedLoads);
#endif

                return;
            }

            void lowerFMAs(Dictionary<int, int> regs)
            {
                Dictionary<Op, Op[]> translate = new Dictionary<Op, Op[]>();
                translate.Add(Op.fmadd_pd, new Op[] { 0, Op.vfmadd231pd, Op.vfmadd132pd, Op.vfmadd213pd });
                translate.Add(Op.fmadd_ps, new Op[] { 0, Op.vfmadd231ps, Op.vfmadd132ps, Op.vfmadd213ps });
                translate.Add(Op.fmadd_sd, new Op[] { 0, Op.vfmadd231sd, Op.vfmadd132sd, Op.vfmadd213sd });
                translate.Add(Op.fmadd_ss, new Op[] { 0, Op.vfmadd231ss, Op.vfmadd132ss, Op.vfmadd213ss });

                for (int j = 0; j < inst.Count; j++)
                {
                    Instr i = inst[j];
                    Instr orig = i;
                    Op[] lookup;
                    if (translate.TryGetValue(i.op & ~Op.L, out lookup))
                    {
                        int r = regs[i.R];
                        if (i.Imm == 0)
                        {
                            // choose form based on which is allowed by register allocation
                            if (r == regs[i.A])
                            {
                                i.op = lookup[2] | (i.op & Op.L);
                                i.A = i.C;
                            }
                            else if (r == regs[i.B])
                            {
                                i.op = lookup[3] | (i.op & Op.L);
                                i.B = i.C;
                            }
                            else if (r == regs[i.C])
                            {
                                i.op = lookup[1] | (i.op & Op.L);
                            }
                            else throw new Exception("Failed to lower FMA due to bad register allocation");
                        }
                        else
                        {
                            i.op = lookup[i.Imm] | (i.op & Op.L);
                            if (i.Imm == 1)
                            {
                                // fma(mem, B, C)
                                // choose from:
                                // fma231 R, B, mem
                                // fma132 R, C, mem
                                int b = regs[i.B];
                                int c = regs[i.C];
                                if (r == c)
                                {
                                    i.op = lookup[1] | (i.op & Op.L);
                                    i.A = i.B;
                                    i.B = 0;
                                    i.C = 0;
                                }
                                else if (r == b)
                                {
                                    i.op = lookup[2] | (i.op & Op.L);
                                    i.A = i.C;
                                    i.B = 0;
                                    i.C = 0;
                                }
                                else
                                    throw new Exception("Failed to lower FMA due to bad register allocation");
                            }
                            else if (i.Imm == 2)
                            {                                
                                //if (r != regs[i.A])
                                    throw new Exception("Failed to lower FMA due to bad register allocation");
                                i.A = i.C;
                            }
                            else if (i.Imm == 3)
                            {
                                // fma(A, B, mem)
                                // there is no choice, it must be fma213 R, B, mem
                                int a = regs[i.A];
                                if (r == a)
                                {
                                    i.op = lookup[3] | (i.op & Op.L);
                                    i.A = i.B;
                                    i.B = 0;
                                    i.C = 0;
                                }
                                else
                                    throw new Exception("Failed to lower FMA due to bad register allocation");
                            }
                        }
                        inst[j] = i;
                    }
                }
            }

            [Serializable]
            public class ValueNotUsedException : Exception
            {
                public ValueNotUsedException() { }
                public ValueNotUsedException(string message) : base(message) { }
                public ValueNotUsedException(string message, Exception inner) : base(message, inner) { }
                protected ValueNotUsedException(
                  System.Runtime.Serialization.SerializationInfo info,
                  System.Runtime.Serialization.StreamingContext context)
                    : base(info, context) { }
            }

            [Serializable]
            public class UnbalancedRepeatException : Exception
            {
                public UnbalancedRepeatException() { }
                public UnbalancedRepeatException(string message) : base(message) { }
                public UnbalancedRepeatException(string message, Exception inner) : base(message, inner) { }
                protected UnbalancedRepeatException(
                  System.Runtime.Serialization.SerializationInfo info,
                  System.Runtime.Serialization.StreamingContext context)
                    : base(info, context) { }
            }

            #region interface implementation

            public scalarReg arg(int index)
            {
                if (index < 4)
                    return new scalarReg(index + 1);
                inst.Add(new Instr(Op.ldarg, nextvreg, 0, 0, 0, index));
                return new scalarReg(nextvreg++);
            }

            public __m128 farg(int index)
            {
                if (index < 4)
                    return new __m128(index + 0x100001);
                inst.Add(new Instr(Op.ldfarg, nextvreg, 0, 0, 0, index));
                return new __m128(nextvxmm++);
            }

            public void assign(__m128 dst, __m128 src)
            {
                inst.Add(new Instr(Op.vassign, dst.r, src.r, 0));
            }

            public void assign(__m256 dst, __m256 src)
            {
                inst.Add(new Instr(Op.vlassign, dst.r, src.r, 0));
            }

            public void assign(scalarReg dst, scalarReg src)
            {
                inst.Add(new Instr(Op.assign | Op.W, dst.r, src.r, 0));
            }

            public void ignore(__m128 r)
            {
                inst.Add(new Instr(Op.ignore, 0, r.r, 0));
            }

#if !NOCAST
            public void assign(__m128i dst, __m128i src)
            {
                inst.Add(new Instr(Op.viassign, dst.r, src.r, 0));
            }

            public void assign(__m128d dst, __m128d src)
            {
                inst.Add(new Instr(Op.vassign, dst.r, src.r, 0));
            }

            public void assign(__m256d dst, __m256d src)
            {
                inst.Add(new Instr(Op.vlassign, dst.r, src.r, 0));
            }

            public void ignore(__m128d r)
            {
                inst.Add(new Instr(Op.ignore, 0, r.r, 0));
            }

            public void ignore(__m128i r)
            {
                inst.Add(new Instr(Op.ignore, 0, r.r, 0));
            }

            public void assign(__m256i dst, __m256i src)
            {
                inst.Add(new Instr(Op.vliassign, dst.r, src.r, 0));
            }

            public void ignore(__m256d r)
            {
                inst.Add(new Instr(Op.ignore, 0, r.r, 0));
            }

            public void ignore(__m256i r)
            {
                inst.Add(new Instr(Op.ignore, 0, r.r, 0));
            }
#endif

            public void ignore(__m256 r)
            {
                inst.Add(new Instr(Op.ignore, 0, r.r, 0));
            }

            public void ignore(scalarReg r)
            {
                inst.Add(new Instr(Op.ignore, 0, r.r, 0));
            }

            public scalarReg repeat_z(scalarReg count, int step)
            {
                inst.Add(new Instr(Op.repeat_z, nextvreg, count.r, 0, 0, step));
                return new scalarReg(nextvreg++);
            }

            public scalarReg repeat(scalarReg count, int step)
            {
                inst.Add(new Instr(Op.repeat, nextvreg, count.r, 0, 0, step));
                return new scalarReg(nextvreg++);
            }

            public void endrepeat()
            {
                inst.Add(new Instr(Op.endrepeat, 0, 0, 0));
            }

            public scalarReg add(scalarReg a, scalarReg b)
            {
                inst.Add(new Instr(Op.addq, nextvreg, a.r, b.r));
                return new scalarReg(nextvreg++);
            }

            public scalarReg add(scalarReg a, int imm)
            {
                inst.Add(new Instr(Op.addq, nextvreg, a.r, 0, 0, imm));
                return new scalarReg(nextvreg++);
            }

            public scalarReg sub(scalarReg a, scalarReg b)
            {
                inst.Add(new Instr(Op.subq, nextvreg, a.r, b.r));
                return new scalarReg(nextvreg++);
            }

            public scalarReg sub(scalarReg a, int imm)
            {
                inst.Add(new Instr(Op.subq, nextvreg, a.r, 0, 0, imm));
                return new scalarReg(nextvreg++);
            }

#if !NOCAST
            __m128d _2op(Op op, __m128d a, __m128d b)
            {
                inst.Add(new Instr(op, nextvxmm, a.r, b.r));
                return new __m128d(nextvxmm++);
            }

            __m128 _2op(Op op, __m128 a, __m128 b, int imm)
            {
                inst.Add(new Instr(op, nextvxmm, a.r, b.r, 0, imm));
                return new __m128(nextvxmm++);
            }

            __m128d _1op(Op op, __m128d a)
            {
                inst.Add(new Instr(op, nextvxmm, a.r, 0));
                return new __m128d(nextvxmm++);
            }

            __m128i _2op(Op op, __m128i a, __m128i b)
            {
                inst.Add(new Instr(op, nextvxmm, a.r, b.r));
                return new __m128i(nextvxmm++);
            }

            __m128i _2op(Op op, __m128i a, __m128i b, int imm)
            {
                inst.Add(new Instr(op, nextvxmm, a.r, b.r, 0, imm));
                return new __m128i(nextvxmm++);
            }

            __m128i _1op(Op op, __m128i a)
            {
                inst.Add(new Instr(op, nextvxmm, a.r, 0));
                return new __m128i(nextvxmm++);
            }

            __m128 _1op(Op op, __m128 a, int imm)
            {
                inst.Add(new Instr(op, nextvxmm, a.r, 0, 0, imm));
                return new __m128(nextvxmm++);
            }

            __m128d _1op(Op op, __m128d a, int imm)
            {
                inst.Add(new Instr(op, nextvxmm, a.r, 0, 0, imm));
                return new __m128d(nextvxmm++);
            }

            __m256d _2op(Op op, __m256d a, __m256d b)
            {
                inst.Add(new Instr(op | Op.L, nextvxmm, a.r, b.r));
                return new __m256d(nextvxmm++);
            }

            __m256d _2op(Op op, __m256d a, __m256d b, int imm)
            {
                inst.Add(new Instr(op | Op.L, nextvxmm, a.r, b.r, 0, imm));
                return new __m256d(nextvxmm++);
            }

            __m256i _2op(Op op, __m256i a, __m256i b)
            {
                inst.Add(new Instr(op | Op.L, nextvxmm, a.r, b.r));
                return new __m256i(nextvxmm++);
            }

            __m256i _1op(Op op, __m256i a, int imm)
            {
                inst.Add(new Instr(op | Op.L, nextvxmm, a.r, 0, 0, imm));
                return new __m256i(nextvxmm++);
            }

            __m256d _1op(Op op, __m256d a, int imm)
            {
                inst.Add(new Instr(op | Op.L, nextvxmm, a.r, 0, 0, imm));
                return new __m256d(nextvxmm++);
            }

            __m256d _1op(Op op, __m256d a)
            {
                inst.Add(new Instr(op | Op.L, nextvxmm, a.r, 0));
                return new __m256d(nextvxmm++);
            }

            __m256i _2op(Op op, __m256i a, __m256i b, int imm)
            {
                inst.Add(new Instr(op | Op.L, nextvxmm, a.r, b.r, 0, imm));
                return new __m256i(nextvxmm++);
            }

            __m256i _1op(Op op, __m256i a)
            {
                inst.Add(new Instr(op | Op.L, nextvxmm, a.r, 0));
                return new __m256i(nextvxmm++);
            }
#endif

            __m128 _2op(Op op, __m128 a, __m128 b)
            {
                inst.Add(new Instr(op, nextvxmm, a.r, b.r));
                return new __m128(nextvxmm++);
            }

            __m128 _1op(Op op, __m128 a)
            {
                inst.Add(new Instr(op, nextvxmm, a.r, 0));
                return new __m128(nextvxmm++);
            }

            __m256 _1op(Op op, __m256 a)
            {
                inst.Add(new Instr(op | Op.L, nextvxmm, a.r, 0));
                return new __m256(nextvxmm++);
            }

            __m256 _1op(Op op, __m256 a, int imm)
            {
                inst.Add(new Instr(op | Op.L, nextvxmm, a.r, 0, 0, imm));
                return new __m256(nextvxmm++);
            }

            __m128d _2op(Op op, __m128d a, __m128d b, int imm)
            {
                inst.Add(new Instr(op, nextvxmm, a.r, b.r, 0, imm));
                return new __m128d(nextvxmm++);
            }

            __m128i _1op(Op op, __m128i a, int imm)
            {
                inst.Add(new Instr(op, nextvxmm, a.r, 0, 0, imm));
                return new __m128i(nextvxmm++);
            }

            __m256 _2op(Op op, __m256 a, __m256 b)
            {
                inst.Add(new Instr(op | Op.L, nextvxmm, a.r, b.r));
                return new __m256(nextvxmm++);
            }

            __m256 _2op(Op op, __m256 a, __m256 b, int imm)
            {
                inst.Add(new Instr(op | Op.L, nextvxmm, a.r, b.r, 0, imm));
                return new __m256(nextvxmm++);
            }

            public __m128 _mm_load_ps(scalarReg baseptr, int offset)
            {
                inst.Add(new Instr(Op.movaps, nextvxmm, 0, 0, 0, 0, new MemoryArg(baseptr.r, 0, 0, offset)));
                return new __m128(nextvxmm++);
            }

            static int convscale(int scale)
            {
                return scale;
            }

            public __m128 _mm_load_ps(scalarReg ptr, scalarReg index, int scale, int offset)
            {
                inst.Add(new Instr(Op.movaps, nextvxmm, 0, 0, 0, 0, new MemoryArg(ptr.r, index.r, convscale(scale), offset)));
                return new __m128(nextvxmm++);
            }

            public __m128 _mm_loadu_ps(scalarReg baseptr, int offset)
            {
                inst.Add(new Instr(Op.movups, nextvxmm, 0, 0, 0, 0, new MemoryArg(baseptr.r, 0, 0, offset)));
                return new __m128(nextvxmm++);
            }

            public __m128 _mm_loadu_ps(scalarReg ptr, scalarReg index, int scale, int offset)
            {
                inst.Add(new Instr(Op.movups, nextvxmm, 0, 0, 0, 0, new MemoryArg(ptr.r, index.r, convscale(scale), offset)));
                return new __m128(nextvxmm++);
            }

            public __m128d _mm_load_pd(scalarReg baseptr, int offset)
            {
                inst.Add(new Instr(Op.movapd, nextvxmm, 0, 0, 0, 0, new MemoryArg(baseptr.r, 0, 0, offset)));
                return new __m128d(nextvxmm++);
            }

            public __m128d _mm_load_pd(scalarReg ptr, scalarReg index, int scale = 1, int offset = 0)
            {
                inst.Add(new Instr(Op.movapd, nextvxmm, 0, 0, 0, 0, new MemoryArg(ptr.r, index.r, convscale(scale), offset)));
                return new __m128d(nextvxmm++);
            }

            public __m128d _mm_loadu_pd(scalarReg baseptr, int offset = 0)
            {
                inst.Add(new Instr(Op.movupd, nextvxmm, 0, 0, 0, 0, new MemoryArg(baseptr.r, 0, 0, offset)));
                return new __m128d(nextvxmm++);
            }

            public __m128d _mm_loadu_pd(scalarReg ptr, scalarReg index, int scale = 1, int offset = 0)
            {
                inst.Add(new Instr(Op.movupd, nextvxmm, 0, 0, 0, 0, new MemoryArg(ptr.r, index.r, convscale(scale), offset)));
                return new __m128d(nextvxmm++);
            }

            public __m256 _mm256_load_ps(scalarReg baseptr, int offset = 0)
            {
                inst.Add(new Instr(Op.movaps | Op.L, nextvxmm, 0, 0, 0, 0, new MemoryArg(baseptr.r, 0, 0, offset)));
                return new __m256(nextvxmm++);
            }

            public __m256 _mm256_load_ps(scalarReg ptr, scalarReg index, int scale = 1, int offset = 0)
            {
                inst.Add(new Instr(Op.movaps | Op.L, nextvxmm, 0, 0, 0, 0, new MemoryArg(ptr.r, index.r, convscale(scale), offset)));
                return new __m256(nextvxmm++);
            }

            public __m128i _mm_load_si128(scalarReg baseptr, int offset)
            {
                inst.Add(new Instr(Op.movdqa, nextvxmm, 0, 0, 0, 0, new MemoryArg(baseptr.r, 0, 0, offset)));
                return new __m128i(nextvxmm++);
            }

            public __m128i _mm_load_si128(scalarReg ptr, scalarReg index, int scale = 1, int offset = 0)
            {
                inst.Add(new Instr(Op.movdqa, nextvxmm, 0, 0, 0, 0, new MemoryArg(ptr.r, index.r, convscale(scale), offset)));
                return new __m128i(nextvxmm++);
            }

            public __m128i _mm_loadu_si128(scalarReg baseptr, int offset)
            {
                inst.Add(new Instr(Op.movdqu, nextvxmm, 0, 0, 0, 0, new MemoryArg(baseptr.r, 0, 0, offset)));
                return new __m128i(nextvxmm++);
            }

            public __m128i _mm_loadu_si128(scalarReg ptr, scalarReg index, int scale = 1, int offset = 0)
            {
                inst.Add(new Instr(Op.movdqu, nextvxmm, 0, 0, 0, 0, new MemoryArg(ptr.r, index.r, convscale(scale), offset)));
                return new __m128i(nextvxmm++);
            }

            public void _mm_store_si128(scalarReg baseptr, __m128i x, int offset)
            {
                inst.Add(new Instr(Op.movdqa_r, 0, x.r, 0, 0, 0, new MemoryArg(baseptr.r, 0, 0, offset)));
            }

            public void _mm_store_si128(scalarReg baseptr, __m128i x, scalarReg index, int scale, int offset)
            {
                inst.Add(new Instr(Op.movdqa_r, 0, x.r, 0, 0, 0, new MemoryArg(baseptr.r, index.r, convscale(scale), offset)));
            }

            public void _mm_storeu_si128(scalarReg baseptr, __m128i x, int offset)
            {
                inst.Add(new Instr(Op.movdqu_r, 0, x.r, 0, 0, 0, new MemoryArg(baseptr.r, 0, 0, offset)));
            }

            public void _mm_storeu_si128(scalarReg baseptr, __m128i x, scalarReg index, int scale, int offset)
            {
                inst.Add(new Instr(Op.movdqu_r, 0, x.r, 0, 0, 0, new MemoryArg(baseptr.r, index.r, convscale(scale), offset)));
            }

            public void _mm_store_ps(scalarReg baseptr, __m128 x, int offset)
            {
                inst.Add(new Instr(Op.movaps_r, 0, x.r, 0, 0, 0, new MemoryArg(baseptr.r, 0, 0, offset)));
            }

            public void _mm_store_ps(scalarReg baseptr, __m128 x, scalarReg index, int scale, int offset)
            {
                inst.Add(new Instr(Op.movaps_r, 0, x.r, 0, 0, 0, new MemoryArg(baseptr.r, index.r, convscale(scale), offset)));
            }

            public void _mm_storeu_ps(scalarReg baseptr, __m128 x, int offset)
            {
                inst.Add(new Instr(Op.movups_r, 0, x.r, 0, 0, 0, new MemoryArg(baseptr.r, 0, 0, offset)));
            }

            public void _mm_storeu_ps(scalarReg baseptr, __m128 x, scalarReg index, int scale, int offset)
            {
                inst.Add(new Instr(Op.movups_r, 0, x.r, 0, 0, 0, new MemoryArg(baseptr.r, index.r, convscale(scale), offset)));
            }

            public void _mm_store_ss(scalarReg baseptr, __m128 x, scalarReg index, int scale, int offset)
            {
                inst.Add(new Instr(Op.movss_r, 0, x.r, 0, 0, 0, new MemoryArg(baseptr.r, index.r, convscale(scale), offset)));
            }

            public void _mm_store_pd(scalarReg baseptr, __m128d x, int offset = 0)
            {
                inst.Add(new Instr(Op.movapd_r, 0, x.r, 0, 0, 0, new MemoryArg(baseptr.r, 0, 0, offset)));
            }

            public void _mm_storeu_pd(scalarReg baseptr, __m128d x, int offset = 0)
            {
                inst.Add(new Instr(Op.movupd_r, 0, x.r, 0, 0, 0, new MemoryArg(baseptr.r, 0, 0, offset)));
            }

            public void _mm256_store_ps(scalarReg baseptr, __m256 x, int offset = 0)
            {
                inst.Add(new Instr(Op.movaps_r | Op.L, 0, x.r, 0, 0, 0, new MemoryArg(baseptr.r, 0, 0, offset)));
            }

            public void _mm256_storeu_ps(scalarReg baseptr, __m256 x, int offset = 0)
            {
                inst.Add(new Instr(Op.movups_r | Op.L, 0, x.r, 0, 0, 0, new MemoryArg(baseptr.r, 0, 0, offset)));
            }

            public void _mm256_store_pd(scalarReg baseptr, __m256d x, int offset = 0)
            {
                inst.Add(new Instr(Op.movapd_r | Op.L, 0, x.r, 0, 0, 0, new MemoryArg(baseptr.r, 0, 0, offset)));
            }

            public void _mm256_storeu_pd(scalarReg baseptr, __m256d x, int offset = 0)
            {
                inst.Add(new Instr(Op.movupd_r | Op.L, 0, x.r, 0, 0, 0, new MemoryArg(baseptr.r, 0, 0, offset)));
            }

            public __m128 _mm_add_ps(__m128 a, __m128 b)
            {
                return _2op(Op.addps, a, b);
            }

            public __m128d _mm_add_pd(__m128d a, __m128d b)
            {
                return _2op(Op.addpd, a, b);
            }

            public __m128 _mm_add_ss(__m128 a, __m128 b)
            {
                return _2op(Op.addss, a, b);
            }

            public __m128d _mm_add_sd(__m128d a, __m128d b)
            {
                return _2op(Op.addsd, a, b);
            }

            public __m128 _mm_and_ps(__m128 a, __m128 b)
            {
                return _2op(Op.andps, a, b);
            }

            public __m128 _mm_fmadd_ps(__m128 a, __m128 b, __m128 c)
            {
                inst.Add(new Instr(Op.fmadd_ps, nextvxmm, a.r, b.r, c.r));
                return new __m128(nextvxmm++);
            }

            public __m128 _mm_fmadd_ss(__m128 a, __m128 b, __m128 c)
            {
                inst.Add(new Instr(Op.fmadd_ss, nextvxmm, a.r, b.r, c.r));
                return new __m128(nextvxmm++);
            }

            public __m128d _mm_fmadd_pd(__m128d a, __m128d b, __m128d c)
            {
                inst.Add(new Instr(Op.fmadd_pd, nextvxmm, a.r, b.r, c.r));
                return new __m128d(nextvxmm++);
            }

            public __m128d _mm_fmadd_sd(__m128d a, __m128d b, __m128d c)
            {
                inst.Add(new Instr(Op.fmadd_sd, nextvxmm, a.r, b.r, c.r));
                return new __m128d(nextvxmm++);
            }

            public __m256 _mm256_fmadd_ps(__m256 a, __m256 b, __m256 c)
            {
                inst.Add(new Instr(Op.fmadd_ps | Op.L, nextvxmm, a.r, b.r, c.r));
                return new __m256(nextvxmm++);
            }

            public __m256 _mm256_fmadd_ss(__m256 a, __m256 b, __m256 c)
            {
                inst.Add(new Instr(Op.fmadd_ss | Op.L, nextvxmm, a.r, b.r, c.r));
                return new __m256(nextvxmm++);
            }

            public __m256d _mm256_fmadd_pd(__m256d a, __m256d b, __m256d c)
            {
                inst.Add(new Instr(Op.fmadd_pd | Op.L, nextvxmm, a.r, b.r, c.r));
                return new __m256d(nextvxmm++);
            }

            public __m256d _mm256_fmadd_sd(__m256d a, __m256d b, __m256d c)
            {
                inst.Add(new Instr(Op.fmadd_sd | Op.L, nextvxmm, a.r, b.r, c.r));
                return new __m256d(nextvxmm++);
            }

            public __m128 _mm_andnot_ps(__m128 a, __m128 b)
            {
                return _2op(Op.andnps, a, b);
            }

            public __m128 _mm_cmp_ps(__m128 a, __m128 b, int mode)
            {
                return _2op(Op.cmpps, a, b, mode);
            }

            public __m128 _mm_cmp_ss(__m128 a, __m128 b, int mode)
            {
                return _2op(Op.cmpss, a, b, mode);
            }

            public __m128 _mm_cmpeq_ps(__m128 a, __m128 b)
            {
                return _mm_cmp_ps(a, b, 0);
            }

            public __m128 _mm_cmpeq_ss(__m128 a, __m128 b)
            {
                return _mm_cmp_ss(a, b, 0);
            }

            public __m128 _mm_cmpge_ps(__m128 a, __m128 b)
            {
                return _mm_cmp_ps(b, a, 2);
            }

            public __m128 _mm_cmpge_ss(__m128 a, __m128 b)
            {
                return _mm_cmp_ss(b, a, 2);
            }

            public __m128 _mm_cmpgt_ps(__m128 a, __m128 b)
            {
                return _mm_cmp_ps(b, a, 1);
            }

            public __m128 _mm_cmpgt_ss(__m128 a, __m128 b)
            {
                return _mm_cmp_ss(b, a, 1);
            }

            public __m128 _mm_cmple_ps(__m128 a, __m128 b)
            {
                return _mm_cmp_ps(a, b, 2);
            }

            public __m128 _mm_cmple_ss(__m128 a, __m128 b)
            {
                return _mm_cmp_ss(a, b, 2);
            }

            public __m128 _mm_cmplt_ps(__m128 a, __m128 b)
            {
                return _mm_cmp_ps(a, b, 1);
            }

            public __m128 _mm_cmplt_ss(__m128 a, __m128 b)
            {
                return _mm_cmp_ss(a, b, 1);
            }

            public __m128 _mm_cmpneq_ps(__m128 a, __m128 b)
            {
                return _mm_cmp_ps(a, b, 4);
            }

            public __m128 _mm_cmpneq_ss(__m128 a, __m128 b)
            {
                return _mm_cmp_ss(a, b, 4);
            }

            public __m128 _mm_cmpnge_ps(__m128 a, __m128 b)
            {
                return _mm_cmp_ps(b, a, 6);
            }

            public __m128 _mm_cmpnge_ss(__m128 a, __m128 b)
            {
                return _mm_cmp_ss(b, a, 6);
            }

            public __m128 _mm_cmpngt_ps(__m128 a, __m128 b)
            {
                return _mm_cmp_ps(b, a, 5);
            }

            public __m128 _mm_cmpngt_ss(__m128 a, __m128 b)
            {
                return _mm_cmp_ss(b, a, 5);
            }

            public __m128 _mm_cmpnle_ps(__m128 a, __m128 b)
            {
                return _mm_cmp_ps(a, b, 6);
            }

            public __m128 _mm_cmpnle_ss(__m128 a, __m128 b)
            {
                return _mm_cmp_ss(a, b, 6);
            }

            public __m128 _mm_cmpnlt_ps(__m128 a, __m128 b)
            {
                return _mm_cmp_ps(a, b, 5);
            }

            public __m128 _mm_cmpnlt_ss(__m128 a, __m128 b)
            {
                return _mm_cmp_ss(a, b, 5);
            }

            public __m128 _mm_cmpord_ps(__m128 a, __m128 b)
            {
                return _mm_cmp_ps(a, b, 7);
            }

            public __m128 _mm_cmpord_ss(__m128 a, __m128 b)
            {
                return _mm_cmp_ss(a, b, 7);
            }

            public __m128 _mm_cmpnord_ps(__m128 a, __m128 b)
            {
                return _mm_cmp_ps(a, b, 3);
            }

            public __m128 _mm_cmpnord_ss(__m128 a, __m128 b)
            {
                return _mm_cmp_ss(a, b, 3);
            }

            public __m128 _mm_cvtsi32_ss(__m128 a, scalarReg b)
            {
                inst.Add(new Instr(Op.cvtsi2ss, nextvxmm, a.r, b.r));
                return new __m128(nextvxmm++);
            }

            public __m128 _mm_cvtsi64_ss(__m128 a, scalarReg b)
            {
                inst.Add(new Instr(Op.cvtsi2ss | Op.W, nextvxmm, a.r, b.r));
                return new __m128(nextvxmm++);
            }

            public scalarReg _mm_cvtss_si32(__m128 a)
            {
                inst.Add(new Instr(Op.cvtss2si, nextvreg, a.r, 0));
                return new scalarReg(nextvreg++);
            }

            public scalarReg _mm_cvtss_si64(__m128 a)
            {
                inst.Add(new Instr(Op.cvtss2si | Op.W, nextvreg, a.r, 0));
                return new scalarReg(nextvreg++);
            }

            public __m128 _mm_div_ps(__m128 a, __m128 b)
            {
                return _2op(Op.divps, a, b);
            }

            public __m128 _mm_div_ss(__m128 a, __m128 b)
            {
                return _2op(Op.divss, a, b);
            }

            public scalarReg _mm_getcsr()
            {
                inst.Add(new Instr(Op.stmxcsr, nextvreg, 0, 0));
                return new scalarReg(nextvreg++);
            }

            public __m128 _mm_load_ss(scalarReg ptr, int offset = 0)
            {
                inst.Add(new Instr(Op.movss, nextvxmm, 0, 0, 0, 0, new MemoryArg(ptr.r, 0, 0, offset)));
                return new __m128(nextvxmm++);
            }

            public __m128 _mm_load_ss(scalarReg ptr, scalarReg index, int scale = 1, int offset = 0)
            {
                inst.Add(new Instr(Op.movss, nextvxmm, 0, 0, 0, 0, new MemoryArg(ptr.r, index.r, convscale(scale), offset)));
                return new __m128(nextvxmm++);
            }

            public __m128 _mm_load1_ps(scalarReg ptr, int offset)
            {
                throw new NotImplementedException();
            }

            public __m128 _mm_max_ps(__m128 a, __m128 b)
            {
                return _2op(Op.maxps, a, b);
            }

            public __m128 _mm_max_ss(__m128 a, __m128 b)
            {
                return _2op(Op.maxss, a, b);
            }

            public __m128 _mm_min_ps(__m128 a, __m128 b)
            {
                return _2op(Op.minps, a, b);
            }

            public __m128 _mm_min_ss(__m128 a, __m128 b)
            {
                return _2op(Op.minss, a, b);
            }

            public __m128 _mm_mov_ss(__m128 a, __m128 b)
            {
                return _2op(Op.movss, a, b);
            }

            public scalarReg _mm_movemask_ps(__m128 a)
            {
                inst.Add(new Instr(Op.movmskps, nextvreg, a.r, 0));
                return new scalarReg(nextvreg++);
            }

            public __m128 _mm_mul_ps(__m128 a, __m128 b)
            {
                return _2op(Op.mulps, a, b);
            }

            public __m128 _mm_mul_ss(__m128 a, __m128 b)
            {
                return _2op(Op.mulss, a, b);
            }

            public __m128 _mm_or_ps(__m128 a, __m128 b)
            {
                return _2op(Op.orps, a, b);
            }

            public __m128 _mm_rcp_ps(__m128 a)
            {
                return _1op(Op.rcpps, a);
            }

            public __m128 _mm_rcp_ss(__m128 a)
            {
                return _1op(Op.rcpss, a);
            }

            public __m128 _mm_rsqrt_ps(__m128 a)
            {
                return _1op(Op.rsqrtps, a);
            }

            public __m128 _mm_rsqrt_ss(__m128 a)
            {
                return _1op(Op.rsqrtss, a);
            }

            public __m128 _mm_setzero_ps()
            {
                inst.Add(new Instr(Op.xorps, nextvxmm, 0x100000, 0x100000));
                return new __m128(nextvxmm++);
            }

            public __m128 _mm_shuffle_ps(__m128 a, __m128 b, int shufmask)
            {
                inst.Add(new Instr(Op.shufps, nextvxmm, a.r, b.r, 0, shufmask));
                return new __m128(nextvxmm++);
            }

            public __m128 _mm_sqrt_ps(__m128 a)
            {
                return _1op(Op.sqrtps, a);
            }

            public __m128 _mm_sqrt_ss(__m128 a)
            {
                return _1op(Op.sqrtss, a);
            }

            public __m128 _mm_sub_ps(__m128 a, __m128 b)
            {
                return _2op(Op.subps, a, b);
            }

            public __m128 _mm_sub_ss(__m128 a, __m128 b)
            {
                return _2op(Op.subss, a, b);
            }

            public __m128 _mm_setr_ps(float e0, float e1, float e2, float e3)
            {
                return _mm_set_ps(e3, e2, e1, e0);
            }

            public __m128 _mm_set_ps(float e3, float e2, float e1, float e0)
            {
                var e3r = BitConverter.GetBytes(e3);
                uint u3 = BitConverter.ToUInt32(e3r, 0);
                var e2r = BitConverter.GetBytes(e2);
                uint u2 = BitConverter.ToUInt32(e2r, 0);
                var e1r = BitConverter.GetBytes(e1);
                uint u1 = BitConverter.ToUInt32(e1r, 0);
                var e0r = BitConverter.GetBytes(e0);
                uint u0 = BitConverter.ToUInt32(e0r, 0);
                if (u3 == 0 && u2 == 0 && u1 == 0 && u0 == 0)
                    return _mm_setzero_ps();
                if (u3 == u2 && u3 == u1 && u3 == u0)
                    return _mm_set1_ps(u3);

                var b = new Block(nextdatalabel, 16);
                b.buffer.AddRange(e0r);
                b.buffer.AddRange(e1r);
                b.buffer.AddRange(e2r);
                b.buffer.AddRange(e3r);
                data.Add(b);

                inst.Add(new Instr(Op.movaps, nextvxmm, 0, 0, 0, 0, new MemoryArg(0, 0, 0, nextdatalabel--)));
                return new __m128(nextvxmm++);
            }

            public __m128 _mm_set1_ps(float e)
            {
                var raw = BitConverter.GetBytes(e);
                uint u = BitConverter.ToUInt32(raw, 0);
                if (u == 0)
                    return _mm_setzero_ps();

                var b = new Block(nextdatalabel, 4);
                b.buffer.AddRange(raw);
                data.Add(b);

                inst.Add(new Instr(Op.vbroadcastss, nextvxmm, 0, 0, 0, 0, new MemoryArg(0, 0, 0, nextdatalabel--)));
                return new __m128(nextvxmm++);
            }

            public __m128 _mm_set_ss(float e)
            {
                var raw = BitConverter.GetBytes(e);
                uint u = BitConverter.ToUInt32(raw, 0);
                if (u == 0)
                    return _mm_setzero_ps();

                var b = new Block(nextdatalabel, 4);
                b.buffer.AddRange(raw);
                data.Add(b);

                inst.Add(new Instr(Op.movss, nextvxmm, 0, 0, 0, 0, new MemoryArg(0, 0, 0, nextdatalabel--)));
                return new __m128(nextvxmm++);
            }

            public __m128 _mm_movehl_ps(__m128 a, __m128 b)
            {
                return _2op(Op.movhlps, a, b);
            }

            public __m128 _mm_movelh_ss(__m128 a, __m128 b)
            {
                return _2op(Op.movlhps, a, b);
            }

            public void _mm_store_ss(scalarReg ptr, __m128 a, int offset = 0)
            {
                inst.Add(new Instr(Op.movss_r, 0, a.r, 0, 0, 0, new MemoryArg(ptr.r, 0, 0, offset)));
            }

            public void _mm_stream_ps(scalarReg ptr, __m128 a, int offset = 0)
            {
                inst.Add(new Instr(Op.movntps, 0, a.r, 0, 0, 0, new MemoryArg(ptr.r, 0, 0, offset)));
            }

            public __m128 _mm_unpackhi_ps(__m128 a, __m128 b)
            {
                return _2op(Op.unpckhps, a, b);
            }

            public __m128 _mm_unpacklo_ps(__m128 a, __m128 b)
            {
                return _2op(Op.unpcklps, a, b);
            }

            public __m128 _mm_xor_ps(__m128 a, __m128 b)
            {
                return _2op(Op.xorps, a, b);
            }

            public __m128i _mm_add_epi16(__m128i a, __m128i b)
            {
                return _2op(Op.paddw, a, b);
            }

            public __m128i _mm_add_epi32(__m128i a, __m128i b)
            {
                return _2op(Op.paddd, a, b);
            }

            public __m128i _mm_add_epi64(__m128i a, __m128i b)
            {
                return _2op(Op.paddq, a, b);
            }

            public __m128i _mm_add_epi8(__m128i a, __m128i b)
            {
                return _2op(Op.paddb, a, b);
            }

            public __m128i _mm_adds_epi16(__m128i a, __m128i b)
            {
                return _2op(Op.paddsw, a, b);
            }

            public __m128i _mm_adds_epi8(__m128i a, __m128i b)
            {
                return _2op(Op.paddsb, a, b);
            }

            public __m128i _mm_adds_epu16(__m128i a, __m128i b)
            {
                return _2op(Op.paddusw, a, b);
            }

            public __m128i _mm_adds_epu8(__m128i a, __m128i b)
            {
                return _2op(Op.paddusb, a, b);
            }

            public __m128d _mm_and_pd(__m128d a, __m128d b)
            {
                return _2op(Op.andpd, a, b);
            }

            public __m128i _mm_and_si128(__m128i a, __m128i b)
            {
                return _2op(Op.pand, a, b);
            }

            public __m128d _mm_andnot_pd(__m128d a, __m128d b)
            {
                return _2op(Op.andnpd, a, b);
            }

            public __m128i _mm_andnot_si128(__m128i a, __m128i b)
            {
                return _2op(Op.pandn, a, b);
            }

            public __m128i _mm_avg_epu16(__m128i a, __m128i b)
            {
                return _2op(Op.pavgw, a, b);
            }

            public __m128i _mm_avg_epu8(__m128i a, __m128i b)
            {
                return _2op(Op.pavgb, a, b);
            }

            public __m128i _mm_bslli_si128(__m128i a, int imm)
            {
                inst.Add(new Instr(Op.pslldq, nextvxmm, a.r, 0, 0, imm));
                return new __m128i(nextvxmm++);
            }

            public __m128i _mm_bsrli_si128(__m128i a, int imm)
            {
                inst.Add(new Instr(Op.psrldq, nextvxmm, a.r, 0, 0, imm));
                return new __m128i(nextvxmm++);
            }

            public __m128 _mm_castpd_ps(__m128d a)
            {
                return new __m128(a.r);
            }

            public __m128i _mm_castpd_si128(__m128d a)
            {
                return new __m128i(a.r);
            }

            public __m128d _mm_castps_pd(__m128 a)
            {
                return new __m128d(a.r);
            }

            public __m128i _mm_castps_si128(__m128 a)
            {
                return new __m128i(a.r);
            }

            public __m128d _mm_castsi128_pd(__m128i a)
            {
                return new __m128d(a.r);
            }

            public __m128 _mm_castsi128_ps(__m128i a)
            {
                return new __m128(a.r);
            }

            public __m128i _mm_cmpeq_epi16(__m128i a, __m128i b)
            {
                return _2op(Op.pcmpeqw, a, b);
            }

            public __m128i _mm_cmpeq_epi32(__m128i a, __m128i b)
            {
                return _2op(Op.pcmpeqd, a, b);
            }

            public __m128i _mm_cmpeq_epi8(__m128i a, __m128i b)
            {
                return _2op(Op.pcmpeqb, a, b);
            }

            public __m128i _mm_cmpgt_epi16(__m128i a, __m128i b)
            {
                return _2op(Op.pcmpgtw, a, b);
            }

            public __m128i _mm_cmpgt_epi32(__m128i a, __m128i b)
            {
                return _2op(Op.pcmpgtd, a, b);
            }

            public __m128i _mm_cmpgt_epi8(__m128i a, __m128i b)
            {
                return _2op(Op.pcmpgtb, a, b);
            }

            public __m128i _mm_cmplt_epi16(__m128i a, __m128i b)
            {
                return _2op(Op.pcmpgtw, b, a);
            }

            public __m128i _mm_cmplt_epi32(__m128i a, __m128i b)
            {
                return _2op(Op.pcmpgtd, b, a);
            }

            public __m128i _mm_cmplt_epi8(__m128i a, __m128i b)
            {
                return _2op(Op.pcmpgtb, b, a);
            }

            public __m128d _mm_cmp_pd(__m128d a, __m128d b, int imm)
            {
                inst.Add(new Instr(Op.cmppd, nextvxmm, a.r, b.r, 0, imm));
                return new __m128d(nextvxmm++);
            }

            public __m128d _mm_cmp_sd(__m128d a, __m128d b, int imm)
            {
                inst.Add(new Instr(Op.cmpsd, nextvxmm, a.r, b.r, 0, imm));
                return new __m128d(nextvxmm++);
            }

            public __m128d _mm_cmpeq_pd(__m128d a, __m128d b)
            {
                return _mm_cmp_pd(a, b, 0);
            }

            public __m128d _mm_cmpeq_sd(__m128d a, __m128d b)
            {
                return _mm_cmp_sd(a, b, 0);
            }

            public __m128d _mm_cmpge_pd(__m128d a, __m128d b)
            {
                return _mm_cmp_pd(b, a, 2);
            }

            public __m128d _mm_cmpge_sd(__m128d a, __m128d b)
            {
                return _mm_cmp_sd(b, a, 2);
            }

            public __m128d _mm_cmpgt_pd(__m128d a, __m128d b)
            {
                return _mm_cmp_pd(b, a, 1);
            }

            public __m128d _mm_cmpgt_sd(__m128d a, __m128d b)
            {
                return _mm_cmp_sd(b, a, 1);
            }

            public __m128d _mm_cmple_pd(__m128d a, __m128d b)
            {
                return _mm_cmp_pd(a, b, 2);
            }

            public __m128d _mm_cmple_sd(__m128d a, __m128d b)
            {
                return _mm_cmp_sd(a, b, 2);
            }

            public __m128d _mm_cmplt_pd(__m128d a, __m128d b)
            {
                return _mm_cmp_pd(a, b, 1);
            }

            public __m128d _mm_cmplt_sd(__m128d a, __m128d b)
            {
                return _mm_cmp_sd(a, b, 1);
            }

            public __m128d _mm_cmpneq_pd(__m128d a, __m128d b)
            {
                return _mm_cmp_pd(a, b, 4);
            }

            public __m128d _mm_cmpneq_sd(__m128d a, __m128d b)
            {
                return _mm_cmp_sd(a, b, 4);
            }

            public __m128d _mm_cmpnge_pd(__m128d a, __m128d b)
            {
                return _mm_cmp_pd(b, a, 6);
            }

            public __m128d _mm_cmpnge_sd(__m128d a, __m128d b)
            {
                return _mm_cmp_sd(b, a, 6);
            }

            public __m128d _mm_cmpngt_pd(__m128d a, __m128d b)
            {
                return _mm_cmp_pd(b, a, 5);
            }

            public __m128d _mm_cmpngt_sd(__m128d a, __m128d b)
            {
                return _mm_cmp_sd(b, a, 5);
            }

            public __m128d _mm_cmpnle_pd(__m128d a, __m128d b)
            {
                return _mm_cmp_pd(a, b, 6);
            }

            public __m128d _mm_cmpnle_sd(__m128d a, __m128d b)
            {
                return _mm_cmp_sd(a, b, 6);
            }

            public __m128d _mm_cmpnlt_pd(__m128d a, __m128d b)
            {
                return _mm_cmp_pd(a, b, 5);
            }

            public __m128d _mm_cmpnlt_sd(__m128d a, __m128d b)
            {
                return _mm_cmp_sd(a, b, 5);
            }

            public __m128d _mm_cmpord_pd(__m128d a, __m128d b)
            {
                return _mm_cmp_pd(a, b, 7);
            }

            public __m128d _mm_cmpord_sd(__m128d a, __m128d b)
            {
                return _mm_cmp_sd(a, b, 7);
            }

            public __m128d _mm_cmpnord_pd(__m128d a, __m128d b)
            {
                return _mm_cmp_pd(a, b, 3);
            }

            public __m128d _mm_cmpnord_sd(__m128d a, __m128d b)
            {
                return _mm_cmp_sd(a, b, 3);
            }

            public scalarReg _mm_cvttss_si32(__m128 a)
            {
                inst.Add(new Instr(Op.cvttss2si, nextvreg, a.r, 0));
                return new scalarReg(nextvreg++);
            }

            public scalarReg _mm_cvttss_si64(__m128 a)
            {
                inst.Add(new Instr(Op.cvttss2si | Op.W, nextvreg, a.r, 0));
                return new scalarReg(nextvreg++);
            }

            public __m128d _mm_cvtepi32_pd(__m128i a)
            {
                var x = _1op(Op.cvtdq2pd, a);
                return new __m128d(x.r);
            }

            public __m128 _mm_cvtepi32_ps(__m128i a)
            {
                var x = _1op(Op.cvtdq2ps, a);
                return new __m128(x.r);
            }

            public __m128i _mm_cvtpd_epi32(__m128d a)
            {
                var x = _1op(Op.cvtpd2dq, a);
                return new __m128i(x.r);
            }

            public __m128 _mm_cvtpd_ps(__m128d a)
            {
                var x = _1op(Op.cvtpd2ps, a);
                return new __m128(x.r);
            }

            public __m128i _mm_cvtps_epi32(__m128 a)
            {
                var x = _1op(Op.cvtps2dq, a);
                return new __m128i(x.r);
            }

            public __m128d _mm_cvtps_pd(__m128 a)
            {
                var x = _1op(Op.cvtps2pd, a);
                return new __m128d(x.r);
            }

            public scalarReg _mm_cvtsd_si32(__m128d a)
            {
                inst.Add(new Instr(Op.cvtsd2si, nextvreg, a.r, 0));
                return new scalarReg(nextvreg++);
            }

            public scalarReg _mm_cvtsd_si64(__m128d a)
            {
                inst.Add(new Instr(Op.cvtsd2si | Op.W, nextvreg, a.r, 0));
                return new scalarReg(nextvreg++);
            }

            public __m128 _mm_cvtsd_ss(__m128 a, __m128d b)
            {
                return _1op(Op.cvtsd2ss, a);
            }

            public scalarReg _mm_cvtsi128_si32(__m128i a)
            {
                inst.Add(new Instr(Op.movd_r, nextvreg, a.r, 0));
                return new scalarReg(nextvreg++);
            }

            public scalarReg _mm_cvtsi128_si64(__m128i a)
            {
                inst.Add(new Instr(Op.movd_r | Op.W, nextvreg, a.r, 0));
                return new scalarReg(nextvreg++);
            }

            public __m128d _mm_cvtsi32_sd(__m128d a, scalarReg b)
            {
                return _1op(Op.cvtsi2sd, a);
            }

            public __m128i _mm_cvtsi32_si128(scalarReg a)
            {
                var x = _1op(Op.movd, new __m128(a.r));
                return new __m128i(x.r);
            }

            public __m128d _mm_cvtsi64_sd(__m128d a, scalarReg b)
            {
                inst.Add(new Instr(Op.cvtsi2sd | Op.W, nextvxmm, a.r, b.r));
                return new __m128d(nextvxmm++);
            }

            public __m128i _mm_cvtsi64_si128(scalarReg b)
            {
                inst.Add(new Instr(Op.movd | Op.W, nextvxmm, b.r, 0));
                return new __m128i(nextvxmm++);
            }

            public __m128d _mm_cvtss_sd(__m128d a, __m128 b)
            {
                inst.Add(new Instr(Op.cvtss2sd, nextvxmm, a.r, b.r));
                return new __m128d(nextvxmm++);
            }

            public __m128i _mm_cvttpd_epi32(__m128d a)
            {
                var x = _1op(Op.cvttpd2dq, a);
                return new __m128i(x.r);
            }

            public __m128i _mm_cvttps_epi32(__m128 a)
            {
                var x = _1op(Op.cvttps2dq, a);
                return new __m128i(x.r);
            }

            public scalarReg _mm_cvttsd_si32(__m128d a)
            {
                inst.Add(new Instr(Op.cvttsd2si, nextvreg, a.r, 0));
                return new scalarReg(nextvreg++);
            }

            public scalarReg _mm_cvttsd_si64(__m128d a)
            {
                inst.Add(new Instr(Op.cvttsd2si | Op.W, nextvreg, a.r, 0));
                return new scalarReg(nextvreg++);
            }

            public __m128d _mm_div_pd(__m128d a, __m128d b)
            {
                return _2op(Op.divpd, a, b);
            }

            public __m128d _mm_div_sd(__m128d a, __m128d b)
            {
                return _2op(Op.divsd, a, b);
            }

            public scalarReg _mm_extract_epi16(__m128i a, int imm)
            {
                inst.Add(new Instr(Op.pextrw, nextvreg, a.r, 0, 0, imm));
                return new scalarReg(nextvreg++);
            }

            public __m128i _mm_insert_epi16(__m128i a, scalarReg v, int imm)
            {
                inst.Add(new Instr(Op.pinsrw, nextvxmm, a.r, v.r, 0, imm));
                return new __m128i(nextvxmm++);
            }

            public __m128d _mm_load_sd(scalarReg ptr, int offset)
            {
                inst.Add(new Instr(Op.movsd, nextvxmm, 0, 0, 0, 0, new MemoryArg(ptr.r, 0, 0, offset)));
                return new __m128d(nextvxmm++);
            }

            public __m128d _mm_load_sd(scalarReg ptr, scalarReg index, int scale = 1, int offset = 0)
            {
                inst.Add(new Instr(Op.movsd, nextvxmm, 0, 0, 0, 0, new MemoryArg(ptr.r, index.r, convscale(scale), offset)));
                return new __m128d(nextvxmm++);
            }

            public __m128d _mm_load1_pd(scalarReg ptr, int offset)
            {
                inst.Add(new Instr(Op.movddup, nextvxmm, 0, 0, 0, 0, new MemoryArg(ptr.r, 0, 0, offset)));
                return new __m128d(nextvxmm++);
            }

            public __m128i _mm_madd_epi16(__m128i a, __m128i b)
            {
                return _2op(Op.pmaddwd, a, b);
            }

            public __m128i _mm_max_epi16(__m128i a, __m128i b)
            {
                return _2op(Op.pmaxsw, a, b);
            }

            public __m128i _mm_max_epu8(__m128i a, __m128i b)
            {
                return _2op(Op.pmaxub, a, b);
            }

            public __m128d _mm_max_pd(__m128d a, __m128d b)
            {
                return _2op(Op.maxpd, a, b);
            }

            public __m128d _mm_max_sd(__m128d a, __m128d b)
            {
                return _2op(Op.maxsd, a, b);
            }

            public __m128i _mm_min_epi16(__m128i a, __m128i b)
            {
                return _2op(Op.pminsw, a, b);
            }

            public __m128i _mm_min_epu8(__m128i a, __m128i b)
            {
                return _2op(Op.pminub, a, b);
            }

            public __m128d _mm_min_pd(__m128d a, __m128d b)
            {
                return _2op(Op.minpd, a, b);
            }

            public __m128d _mm_min_sd(__m128d a, __m128d b)
            {
                return _2op(Op.minsd, a, b);
            }

            public __m128i _mm_mov_epi64(__m128i a)
            {
                return _1op(Op.movd | Op.W, a);
            }

            public __m128i _mm_loadl_epi64(scalarReg ptr, int offset = 0)
            {
                inst.Add(new Instr(Op.movd | Op.W, nextvxmm, 0, 0, 0, 0, new MemoryArg(ptr.r, 0, 0, offset)));
                return new __m128i(nextvxmm++);
            }

            public void _mm_storel_epi64(scalarReg ptr, __m128i value, int offset = 0)
            {
                inst.Add(new Instr(Op.movd_r | Op.W, 0, value.r, 0, 0, 0, new MemoryArg(ptr.r, 0, 0, offset)));
            }

            public __m128d _mm_mov_sd(__m128d a, __m128d b)
            {
                return _2op(Op.movsd, a, b);
            }

            public scalarReg _mm_movemask_epi8(__m128i a)
            {
                inst.Add(new Instr(Op.pmovmskb, nextvreg, a.r, 0));
                return new scalarReg(nextvreg++);
            }

            public scalarReg _mm_movemask_pd(__m128d a)
            {
                inst.Add(new Instr(Op.movmskpd, nextvreg, a.r, 0));
                return new scalarReg(nextvreg++);
            }

            public __m128i _mm_mul_epu32(__m128i a, __m128i b)
            {
                return _2op(Op.pmuludq, a, b);
            }

            public __m128d _mm_mul_pd(__m128d a, __m128d b)
            {
                return _2op(Op.mulpd, a, b);
            }

            public __m128d _mm_mul_sd(__m128d a, __m128d b)
            {
                return _2op(Op.mulsd, a, b);
            }

            public __m128i _mm_mulhi_epi16(__m128i a, __m128i b)
            {
                return _2op(Op.pmulhw, a, b);
            }

            public __m128i _mm_mulhi_epu16(__m128i a, __m128i b)
            {
                return _2op(Op.pmulhuw, a, b);
            }

            public __m128i _mm_mullo_epi16(__m128i a, __m128i b)
            {
                return _2op(Op.pmullw, a, b);
            }

            public __m128d _mm_or_pd(__m128d a, __m128d b)
            {
                return _2op(Op.orpd, a, b);
            }

            public __m128i _mm_or_si128(__m128i a, __m128i b)
            {
                return _2op(Op.por, a, b);
            }

            public __m128i _mm_packs_epi16(__m128i a, __m128i b)
            {
                return _2op(Op.packsswb, a, b);
            }

            public __m128i _mm_packs_epi32(__m128i a, __m128i b)
            {
                return _2op(Op.packssdw, a, b);
            }

            public __m128i _mm_packus_epi16(__m128i a, __m128i b)
            {
                return _2op(Op.packuswb, a, b);
            }

            public __m128i _mm_sad_epu8(__m128i a, __m128i b)
            {
                return _2op(Op.psadbw, a, b);
            }

            public __m128i _mm_set_epi16(short e7, short e6, short e5, short e4, short e3, short e2, short e1, short e0)
            {
                int label = nextdatalabel--;
                Block b = new Block(label, 16);
                b.buffer.AddRange(BitConverter.GetBytes(e0));
                b.buffer.AddRange(BitConverter.GetBytes(e1));
                b.buffer.AddRange(BitConverter.GetBytes(e2));
                b.buffer.AddRange(BitConverter.GetBytes(e3));
                b.buffer.AddRange(BitConverter.GetBytes(e4));
                b.buffer.AddRange(BitConverter.GetBytes(e5));
                b.buffer.AddRange(BitConverter.GetBytes(e6));
                b.buffer.AddRange(BitConverter.GetBytes(e7));
                data.Add(b);

                inst.Add(new Instr(Op.movdqa, nextvxmm, 0, 0, 0, 0, new MemoryArg(0, 0, 0, label)));
                return new __m128i(nextvxmm++);
            }

            public __m128i _mm_set_epi32(int e3, int e2, int e1, int e0)
            {
                int label = nextdatalabel--;
                Block b = new Block(label, 16);
                b.buffer.AddRange(BitConverter.GetBytes(e0));
                b.buffer.AddRange(BitConverter.GetBytes(e1));
                b.buffer.AddRange(BitConverter.GetBytes(e2));
                b.buffer.AddRange(BitConverter.GetBytes(e3));
                data.Add(b);

                inst.Add(new Instr(Op.movdqa, nextvxmm, 0, 0, 0, 0, new MemoryArg(0, 0, 0, label)));
                return new __m128i(nextvxmm++);
            }

            public __m128i _mm_set_epi8(byte e15, byte e14, byte e13, byte e12, byte e11, byte e10, byte e9, byte e8, byte e7, byte e6, byte e5, byte e4, byte e3, byte e2, byte e1, byte e0)
            {
                byte[] d = new byte[] { e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13, e14, e15 };
                int label = nextdatalabel--;
                Block b = new Block(label, 16);
                b.buffer.AddRange(d);
                data.Add(b);

                inst.Add(new Instr(Op.movdqa, nextvxmm, 0, 0, 0, 0, new MemoryArg(0, 0, 0, label)));
                return new __m128i(nextvxmm++);
            }

            public __m128d _mm_set_pd(double e1, double e0)
            {
                int label = nextdatalabel--;
                Block b = new Block(label, 16);
                b.buffer.AddRange(BitConverter.GetBytes(e0));
                b.buffer.AddRange(BitConverter.GetBytes(e1));
                data.Add(b);

                inst.Add(new Instr(Op.movapd, nextvxmm, 0, 0, 0, 0, new MemoryArg(0, 0, 0, label)));
                return new __m128d(nextvxmm++);
            }

            public __m128d _mm_set_sd(double a)
            {
                if (BitConverter.DoubleToInt64Bits(a) == 0)
                    return _mm_setzero_pd();

                int label = nextdatalabel--;
                Block b = new Block(label, 8);
                b.buffer.AddRange(BitConverter.GetBytes(a));
                data.Add(b);

                inst.Add(new Instr(Op.movsd, nextvxmm, 0, 0, 0, 0, new MemoryArg(0, 0, 0, label)));
                return new __m128d(nextvxmm++);
            }

            public __m128i _mm_set1_epi16(short a)
            {
                if (a == 0)
                    return _mm_setzero_si128();
                else if (a == -1)
                    return _2op(Op.pcmpeqb, new __m128i(0x100000), new __m128i(0x100000));

                int label = nextdatalabel--;
                Block b;
                if (options.AVX2)
                {
                    b = new Block(label, 2);
                    b.buffer.AddRange(BitConverter.GetBytes(a));
                    inst.Add(new Instr(Op.vpbroadcastw, nextvxmm, 0, 0, 0, 0, new MemoryArg(0, 0, 0, label)));
                }
                else
                {
                    b = new Block(label, 4);
                    b.buffer.AddRange(BitConverter.GetBytes(a));
                    b.buffer.AddRange(BitConverter.GetBytes(a));
                    inst.Add(new Instr(Op.movd, nextvxmm, 0, 0, 0, 0, new MemoryArg(0, 0, 0, label)));
                    inst.Add(new Instr(Op.pshufd, nextvxmm, nextvxmm, 0));
                }
                data.Add(b);
                return new __m128i(nextvxmm++);
            }

            public __m128i _mm_set1_epi32(int a)
            {
                if (a == 0)
                    return _mm_setzero_si128();
                else if (a == -1)
                    return _2op(Op.pcmpeqb, new __m128i(0x100000), new __m128i(0x100000));

                int label = nextdatalabel--;
                Block b = new Block(label, 4);
                b.buffer.AddRange(BitConverter.GetBytes(a));
                data.Add(b);

                if (options.AVX2)
                    inst.Add(new Instr(Op.vpbroadcastd, nextvxmm, 0, 0, 0, 0, new MemoryArg(0, 0, 0, label)));
                else
                {
                    inst.Add(new Instr(Op.movd, nextvxmm, 0, 0, 0, 0, new MemoryArg(0, 0, 0, label)));
                    inst.Add(new Instr(Op.pshufd, nextvxmm, nextvxmm, 0));
                }
                return new __m128i(nextvxmm++);
            }

            public __m128i _mm_set1_epi64x(long a)
            {
                if (a == 0)
                    return _mm_setzero_si128();
                else if (a == -1)
                    return _2op(Op.pcmpeqb, new __m128i(0x100000), new __m128i(0x100000));

                int label = nextdatalabel--;
                Block b = new Block(label, 8);
                b.buffer.AddRange(BitConverter.GetBytes(a));
                data.Add(b);

                inst.Add(new Instr(Op.vpbroadcastq, nextvxmm, 0, 0, 0, 0, new MemoryArg(0, 0, 0, label)));
                return new __m128i(nextvxmm++);
            }

            public __m128i _mm_set1_epi8(byte a)
            {
                if (a == 0)
                    return _mm_setzero_si128();
                else if (a == 255)
                    return _2op(Op.pcmpeqb, new __m128i(0x100000), new __m128i(0x100000));

                int label = nextdatalabel--;
                Block b = new Block(label, 1);
                b.buffer.Add(a);
                data.Add(b);

                inst.Add(new Instr(Op.vpbroadcastb, nextvxmm, 0, 0, 0, 0, new MemoryArg(0, 0, 0, label)));
                return new __m128i(nextvxmm++);
            }

            public __m128d _mm_set1_pd(double a)
            {
                if (BitConverter.DoubleToInt64Bits(a) == 0)
                    return _mm_setzero_pd();

                int label = nextdatalabel--;
                Block b = new Block(label, 8);
                b.buffer.AddRange(BitConverter.GetBytes(a));
                data.Add(b);

                inst.Add(new Instr(Op.movddup, nextvxmm, 0, 0, 0, 0, new MemoryArg(0, 0, 0, label)));
                return new __m128d(nextvxmm++);
            }

            public __m128i _mm_setr_epi16(short e7, short e6, short e5, short e4, short e3, short e2, short e1, short e0)
            {
                return _mm_set_epi16(e0, e1, e2, e3, e4, e5, e6, e7);
            }

            public __m128i _mm_setr_epi32(int e3, int e2, int e1, int e0)
            {
                return _mm_set_epi32(e0, e1, e2, e3);
            }

            public __m128i _mm_setr_epi8(byte e15, byte e14, byte e13, byte e12, byte e11, byte e10, byte e9, byte e8, byte e7, byte e6, byte e5, byte e4, byte e3, byte e2, byte e1, byte e0)
            {
                return _mm_set_epi8(e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13, e14, e15);
            }

            public __m128d _mm_setr_pd(double e1, double e0)
            {
                return _mm_set_pd(e0, e1);
            }

            public __m128d _mm_setzero_pd()
            {
                inst.Add(new Instr(Op.xorpd, nextvxmm, 0x100000, 0x100000));
                return new __m128d(nextvxmm++);
            }

            public __m128i _mm_setzero_si128()
            {
                inst.Add(new Instr(Op.pxor, nextvxmm, 0x100000, 0x100000));
                return new __m128i(nextvxmm++);
            }

            public __m128i _mm_shuffle_epi32(__m128i a, int imm)
            {
                return _1op(Op.pshufd, a, imm);
            }

            public __m128d _mm_shuffle_pd(__m128d a, __m128d b, int imm)
            {
                return _2op(Op.shufpd, a, b, imm);
            }

            public __m128i _mm_shufflehi_epi16(__m128i a, int imm)
            {
                return _1op(Op.pshufhw, a, imm);
            }

            public __m128i _mm_shufflelo_epi16(__m128i a, int imm)
            {
                return _1op(Op.pshuflw, a, imm);
            }

            public __m128i _mm_sll_epi16(__m128i a, __m128i count)
            {
                return _2op(Op.psllw, a, count);
            }

            public __m128i _mm_sll_epi32(__m128i a, __m128i count)
            {
                return _2op(Op.pslld, a, count);
            }

            public __m128i _mm_sll_epi64(__m128i a, __m128i count)
            {
                return _2op(Op.psllq, a, count);
            }

            public __m128i _mm_slli_epi16(__m128i a, int imm)
            {
                return _1op(Op.psllwi, a, imm);
            }

            public __m128i _mm_slli_epi32(__m128i a, int imm)
            {
                return _1op(Op.pslldi, a, imm);
            }

            public __m128i _mm_slli_epi64(__m128i a, int imm)
            {
                return _1op(Op.psllqi, a, imm);
            }

            public __m128i _mm_slli_si128(__m128i a, int imm)
            {
                return _1op(Op.pslldq, a, imm);
            }

            public __m128d _mm_sqrt_pd(__m128d a, __m128d b)
            {
                return _2op(Op.sqrtpd, a, b);
            }

            public __m128d _mm_sqrt_sd(__m128d a, __m128d b)
            {
                return _2op(Op.sqrtsd, a, b);
            }

            public __m128i _mm_sra_epi16(__m128i a, __m128i count)
            {
                return _2op(Op.psraw, a, count);
            }

            public __m128i _mm_sra_epi32(__m128i a, __m128i count)
            {
                return _2op(Op.psrad, a, count);
            }

            public __m128i _mm_srai_epi16(__m128i a, int imm)
            {
                return _1op(Op.psrawi, a, imm);
            }

            public __m128i _mm_srai_epi32(__m128i a, int imm)
            {
                return _1op(Op.psradi, a, imm);
            }

            public __m128i _mm_srl_epi16(__m128i a, __m128i count)
            {
                return _2op(Op.psrlw, a, count);
            }

            public __m128i _mm_srl_epi32(__m128i a, __m128i count)
            {
                return _2op(Op.psrld, a, count);
            }

            public __m128i _mm_srl_epi64(__m128i a, __m128i count)
            {
                return _2op(Op.psrlq, a, count);
            }

            public __m128i _mm_srli_epi16(__m128i a, int imm)
            {
                return _1op(Op.psrlwi, a, imm);
            }

            public __m128i _mm_srli_epi32(__m128i a, int imm)
            {
                return _1op(Op.psrldi, a, imm);
            }

            public __m128i _mm_srli_epi64(__m128i a, int imm)
            {
                return _1op(Op.psrlqi, a, imm);
            }

            public __m128i _mm_srli_si128(__m128i a, int imm)
            {
                return _1op(Op.psrldq, a, imm);
            }

            public void _mm_store_sd(scalarReg ptr, __m128d a, int offset = 0)
            {
                inst.Add(new Instr(Op.movsd_r, 0, a.r, 0, 0, 0, new MemoryArg(ptr.r, 0, 0, offset)));
            }

            public __m128i _mm_sub_epi16(__m128i a, __m128i b)
            {
                return _2op(Op.psubw, a, b);
            }

            public __m128i _mm_sub_epi32(__m128i a, __m128i b)
            {
                return _2op(Op.psubd, a, b);
            }

            public __m128i _mm_sub_epi64(__m128i a, __m128i b)
            {
                return _2op(Op.psubq, a, b);
            }

            public __m128i _mm_sub_epi8(__m128i a, __m128i b)
            {
                return _2op(Op.psubb, a, b);
            }

            public __m128d _mm_sub_pd(__m128d a, __m128d b)
            {
                return _2op(Op.subpd, a, b);
            }

            public __m128d _mm_sub_sd(__m128d a, __m128d b)
            {
                return _2op(Op.subsd, a, b);
            }

            public __m128i _mm_subs_epi16(__m128i a, __m128i b)
            {
                return _2op(Op.psubsw, a, b);
            }

            public __m128i _mm_subs_epi8(__m128i a, __m128i b)
            {
                return _2op(Op.psubsb, a, b);
            }

            public __m128i _mm_subs_epu16(__m128i a, __m128i b)
            {
                return _2op(Op.psubusw, a, b);
            }

            public __m128i _mm_subs_epu8(__m128i a, __m128i b)
            {
                return _2op(Op.psubusb, a, b);
            }

            public __m128i _mm_unpackhi_epi16(__m128i a, __m128i b)
            {
                return _2op(Op.punpckhwd, a, b);
            }

            public __m128i _mm_unpackhi_epi32(__m128i a, __m128i b)
            {
                return _2op(Op.punpckhdq, a, b);
            }

            public __m128i _mm_unpackhi_epi64(__m128i a, __m128i b)
            {
                return _2op(Op.punpckhqdq, a, b);
            }

            public __m128i _mm_unpackhi_epi8(__m128i a, __m128i b)
            {
                return _2op(Op.punpckhbw, a, b);
            }

            public __m128d _mm_unpackhi_pd(__m128d a, __m128d b)
            {
                return _2op(Op.unpckhpd, a, b);
            }

            public __m128i _mm_unpacklo_epi16(__m128i a, __m128i b)
            {
                return _2op(Op.punpcklwd, a, b);
            }

            public __m128i _mm_unpacklo_epi32(__m128i a, __m128i b)
            {
                return _2op(Op.punpckldq, a, b);
            }

            public __m128i _mm_unpacklo_epi64(__m128i a, __m128i b)
            {
                return _2op(Op.punpcklqdq, a, b);
            }

            public __m128i _mm_unpacklo_epi8(__m128i a, __m128i b)
            {
                return _2op(Op.punpcklbw, a, b);
            }

            public __m128d _mm_unpacklo_pd(__m128d a, __m128d b)
            {
                return _2op(Op.unpcklpd, a, b);
            }

            public __m128d _mm_xor_pd(__m128d a, __m128d b)
            {
                return _2op(Op.xorpd, a, b);
            }

            public __m128i _mm_xor_si128(__m128i a, __m128i b)
            {
                return _2op(Op.pxor, a, b);
            }

            public __m128d _mm_addsub_pd(__m128d a, __m128d b)
            {
                return _2op(Op.addsubpd, a, b);
            }

            public __m128 _mm_addsub_ps(__m128 a, __m128 b)
            {
                return _2op(Op.addsubps, a, b);
            }

            public __m128d _mm_hadd_pd(__m128d a, __m128d b)
            {
                return _2op(Op.haddpd, a, b);
            }

            public __m128 _mm_hadd_ps(__m128 a, __m128 b)
            {
                return _2op(Op.haddps, a, b);
            }

            public __m128d _mm_hsub_pd(__m128d a, __m128d b)
            {
                return _2op(Op.hsubpd, a, b);
            }

            public __m128 _mm_hsub_ps(__m128 a, __m128 b)
            {
                return _2op(Op.hsubps, a, b);
            }

            public __m128i _mm_lddqu_si128(scalarReg ptr, int offset = 0)
            {
                inst.Add(new Instr(Op.lddqu, nextvxmm, 0, 0, 0, 0, new MemoryArg(ptr.r, 0, 0, offset)));
                return new __m128i(nextvxmm++);
            }

            public __m128d _mm_loaddup_pd(scalarReg ptr, int offset = 0)
            {
                inst.Add(new Instr(Op.movddup, nextvxmm, 0, 0, 0, 0, new MemoryArg(ptr.r, 0, 0, offset)));
                return new __m128d(nextvxmm++);
            }

            public __m128d _mm_movedup_pd(__m128d a)
            {
                return _1op(Op.movddup, a);
            }

            public __m128 _mm_movehdup_ps(__m128 a)
            {
                return _1op(Op.movshdup, a);
            }

            public __m128 _mm_moveldup_ps(__m128 a)
            {
                return _1op(Op.movsldup, a);
            }

            public __m128i _mm_abs_epi16(__m128i a)
            {
                return _1op(Op.pabsw, a);
            }

            public __m128i _mm_abs_epi32(__m128i a)
            {
                return _1op(Op.pabsd, a);
            }

            public __m128i _mm_abs_epi8(__m128i a)
            {
                return _1op(Op.pabsb, a);
            }

            public __m128i _mm_alignr_epi8(__m128i a, __m128i b, int count)
            {
                return _2op(Op.palignr, a, b, count);
            }

            public __m128i _mm_hadd_epi16(__m128i a, __m128i b)
            {
                return _2op(Op.phaddw, a, b);
            }

            public __m128i _mm_hadd_epi32(__m128i a, __m128i b)
            {
                return _2op(Op.phaddd, a, b);
            }

            public __m128i _mm_hadds_epi16(__m128i a, __m128i b)
            {
                return _2op(Op.phaddsw, a, b);
            }

            public __m128i _mm_hsub_epi16(__m128i a, __m128i b)
            {
                return _2op(Op.phsubw, a, b);
            }

            public __m128i _mm_hsub_epi32(__m128i a, __m128i b)
            {
                return _2op(Op.phsubd, a, b);
            }

            public __m128i _mm_hsubs_epi16(__m128i a, __m128i b)
            {
                return _2op(Op.phsubsw, a, b);
            }

            public __m128i _mm_maddubs_epi16(__m128i a, __m128i b)
            {
                return _2op(Op.pmaddubsw, a, b);
            }

            public __m128i _mm_mulhrs_epi16(__m128i a, __m128i b)
            {
                return _2op(Op.pmulhrsw, a, b);
            }

            public __m128i _mm_shuffle_epi8(__m128i a, __m128i b)
            {
                return _2op(Op.pshufb, a, b);
            }

            public __m128i _mm_sign_epi16(__m128i a)
            {
                return _1op(Op.psignw, a);
            }

            public __m128i _mm_sign_epi32(__m128i a)
            {
                return _1op(Op.psignd, a);
            }

            public __m128i _mm_sign_epi8(__m128i a)
            {
                return _1op(Op.psignb, a);
            }

            public __m128i _mm_blend_epi16(__m128i a, __m128i b, int imm)
            {
                return _2op(Op.pblendw, a, b, imm);
            }

            public __m128d _mm_blend_pd(__m128d a, __m128d b, int imm)
            {
                return _2op(Op.blendpd, a, b, imm);
            }

            public __m128 _mm_blend_ps(__m128 a, __m128 b, int imm)
            {
                return _2op(Op.blendps, a, b, imm);
            }

            public __m128i _mm_blendv_epi8(__m128i a, __m128i b, __m128i mask)
            {
                inst.Add(new Instr(Op.pblendvb, nextvxmm, a.r, b.r, mask.r));
                return new __m128i(nextvxmm++);
            }

            public __m128d _mm_blendv_pd(__m128d a, __m128d b, __m128d mask)
            {
                inst.Add(new Instr(Op.blendvpd, nextvxmm, a.r, b.r, mask.r));
                return new __m128d(nextvxmm++);
            }

            public __m128 _mm_blendv_ps(__m128 a, __m128 b, __m128 mask)
            {
                inst.Add(new Instr(Op.blendvps, nextvxmm, a.r, b.r, mask.r));
                return new __m128(nextvxmm++);
            }

            public __m128d _mm_ceil_pd(__m128d a)
            {
                return _1op(Op.roundpd, a, 2);
            }

            public __m128 _mm_ceil_ps(__m128 a)
            {
                return _1op(Op.roundps, a, 2);
            }

            public __m128d _mm_ceil_sd(__m128d a)
            {
                return _1op(Op.roundsd, a, 2);
            }

            public __m128 _mm_ceil_ss(__m128 a)
            {
                return _1op(Op.roundss, a, 2);
            }

            public __m128i _mm_cmpeq_epi64(__m128i a, __m128i b)
            {
                return _2op(Op.pcmpeqq, a, b);
            }

            public __m128i _mm_cvtepi16_epi32(__m128i a)
            {
                return _1op(Op.pmovsxwd, a);
            }

            public __m128i _mm_cvtepi16_epi64(__m128i a)
            {
                return _1op(Op.pmovsxwq, a);
            }

            public __m128i _mm_cvtepi32_epi64(__m128i a)
            {
                return _1op(Op.pmovsxdq, a);
            }

            public __m128i _mm_cvtepi8_epi16(__m128i a)
            {
                return _1op(Op.pmovsxbw, a);
            }

            public __m128i _mm_cvtepi8_epi32(__m128i a)
            {
                return _1op(Op.pmovsxbd, a);
            }

            public __m128i _mm_cvtepi8_epi64(__m128i a)
            {
                return _1op(Op.pmovsxbq, a);
            }

            public __m128i _mm_cvtepu16_epi32(__m128i a)
            {
                return _1op(Op.pmovzxwd, a);
            }

            public __m128i _mm_cvtepu16_epi64(__m128i a)
            {
                return _1op(Op.pmovzxwq, a);
            }

            public __m128i _mm_cvtepu32_epi64(__m128i a)
            {
                return _1op(Op.pmovzxdq, a);
            }

            public __m128i _mm_cvtepu8_epi16(__m128i a)
            {
                return _1op(Op.pmovzxbw, a);
            }

            public __m128i _mm_cvtepu8_epi32(__m128i a)
            {
                return _1op(Op.pmovzxbd, a);
            }

            public __m128i _mm_cvtepu8_epi64(__m128i a)
            {
                return _1op(Op.pmovzxbq, a);
            }

            public __m128d _mm_dp_pd(__m128d a, __m128d b, int imm)
            {
                return _2op(Op.dppd, a, b, imm);
            }

            public __m128 _mm_dp_ps(__m128 a, __m128 b, int imm)
            {
                return _2op(Op.dpps, a, b, imm);
            }

            public scalarReg _mm_extract_epi32(__m128i a, int imm)
            {
                var x = _1op(Op.pextrd, a, imm);
                return new scalarReg(x.r);
            }

            public scalarReg _mm_extract_epi64(__m128i a, int imm)
            {
                var x = _1op(Op.pextrq, a, imm);
                return new scalarReg(x.r);
            }

            public scalarReg _mm_extract_ep8(__m128i a, int imm)
            {
                var x = _1op(Op.pextrb, a, imm);
                return new scalarReg(x.r);
            }

            public scalarReg _mm_extract_ps(__m128 a, int imm)
            {
                var x = _1op(Op.extractps, a, imm);
                return new scalarReg(x.r);
            }

            public void _mm_extract_ps(__m128 a, int imm, scalarReg ptr, int offset = 0)
            {
                inst.Add(new Instr(Op.extractps | Op.store, 0, a.r, 0, 0, imm, new MemoryArg(ptr.r, 0, 0, offset)));
            }

            public __m128d _mm_floor_pd(__m128d a)
            {
                return _1op(Op.roundpd, a, 1);
            }

            public __m128 _mm_floor_ps(__m128 a)
            {
                return _1op(Op.roundps, a, 1);
            }

            public __m128d _mm_floor_sd(__m128d a)
            {
                return _1op(Op.roundsd, a, 1);
            }

            public __m128 _mm_floor_ss(__m128 a)
            {
                return _1op(Op.roundss, a, 1);
            }

            public __m128i _mm_insert_epi32(__m128i a, scalarReg b, int imm)
            {
                return _2op(Op.pinsrd, a, new __m128i(b.r), imm);
            }

            public __m128i _mm_insert_epi64(__m128i a, scalarReg b, int imm)
            {
                return _2op(Op.pinsrq, a, new __m128i(b.r), imm);
            }

            public __m128i _mm_insert_epi8(__m128i a, scalarReg b, int imm)
            {
                return _2op(Op.pinsrb, a, new __m128i(b.r), imm);
            }

            public __m128 _mm_insert_ps(__m128 a, scalarReg b, int imm)
            {
                return _2op(Op.insertps, a, new __m128(b.r), imm);
            }

            public __m128 _mm_insert_ps(__m128 a, scalarReg ptr, int offset, int imm)
            {
                inst.Add(new Instr(Op.insertps, nextvxmm, a.r, 0, 0, imm, new MemoryArg(ptr.r, 0, 0, offset)));
                return new __m128(nextvxmm++);
            }

            public __m128i _mm_max_epi32(__m128i a, __m128i b)
            {
                return _2op(Op.pmaxsd, a, b);
            }

            public __m128i _mm_max_epi8(__m128i a, __m128i b)
            {
                return _2op(Op.pmaxsb, a, b);
            }

            public __m128i _mm_max_epu16(__m128i a, __m128i b)
            {
                return _2op(Op.pmaxuw, a, b);
            }

            public __m128i _mm_min_epi32(__m128i a, __m128i b)
            {
                return _2op(Op.pminsd, a, b);
            }

            public __m128i _mm_min_epi8(__m128i a, __m128i b)
            {
                return _2op(Op.pminsb, a, b);
            }

            public __m128i _mm_min_epu16(__m128i a, __m128i b)
            {
                return _2op(Op.pminuw, a, b);
            }

            public __m128i _mm_min_epu32(__m128i a, __m128i b)
            {
                return _2op(Op.pminud, a, b);
            }

            public __m128i _mm_minpos_epu16(__m128i a)
            {
                return _1op(Op.phminposuw, a);
            }

            public __m128i _mm_mpsadbw_epu8(__m128i a, __m128i b, int imm)
            {
                return _2op(Op.mpsadbw, a, b, imm);
            }

            public __m128i _mm_mul_epi32(__m128i a, __m128i b)
            {
                return _2op(Op.pmuldq, a, b);
            }

            public __m128i _mm_mullo_epi32(__m128i a, __m128i b)
            {
                return _2op(Op.pmulld, a, b);
            }

            public __m128i _mm_packus_epi32(__m128i a, __m128i b)
            {
                return _2op(Op.packusdw, a, b);
            }

            public __m128d _mm_round_pd(__m128d a, RoundingMode mode)
            {
                return _1op(Op.roundpd, a, (int)mode);
            }

            public __m128 _mm_round_ps(__m128 a, RoundingMode mode)
            {
                return _1op(Op.roundps, a, (int)mode);
            }

            public __m128d _mm_round_sd(__m128d a, RoundingMode mode)
            {
                return _1op(Op.roundsd, a, (int)mode);
            }

            public __m128 _mm_round_ss(__m128 a, RoundingMode mode)
            {
                return _1op(Op.roundss, a, (int)mode);
            }

            public __m128i _mm_stream_load_si128(scalarReg ptr, int offset = 0)
            {
                inst.Add(new Instr(Op.movntdqa, nextvxmm, 0, 0, 0, 0, new MemoryArg(ptr.r, 0, 0, offset)));
                return new __m128i(nextvxmm++);
            }

            public __m128i _mm_stream_load_si128(scalarReg ptr, scalarReg index, int scale = 1, int offset = 0)
            {
                inst.Add(new Instr(Op.movntdqa, nextvxmm, 0, 0, 0, 0, new MemoryArg(ptr.r, index.r, convscale(scale), offset)));
                return new __m128i(nextvxmm++);
            }

            public scalarReg _mm_test_all_ones(__m128i a)
            {
                var t = _mm_cmpeq_epi32(a, a);
                inst.Add(new Instr(Op.ptest, 0, t.r, t.r));
                inst.Add(new Instr(Op.setc, nextvreg, nextvreg, 0));
                inst.Add(new Instr(Op.movzx_bd, nextvreg, nextvreg, 0));
                return new scalarReg(nextvreg++);
            }

            public scalarReg _mm_test_all_zeros(__m128i a, __m128i mask)
            {
                inst.Add(new Instr(Op.ptest, 0, a.r, mask.r));
                inst.Add(new Instr(Op.setz, nextvreg, nextvreg, 0));
                inst.Add(new Instr(Op.movzx_bd, nextvreg, nextvreg, 0));
                return new scalarReg(nextvreg++);
            }

            public scalarReg _mm_test_mix_ones_zeros(__m128i a, __m128i mask)
            {
                inst.Add(new Instr(Op.ptest, 0, a.r, mask.r));
                inst.Add(new Instr(Op.seta, nextvreg, nextvreg, 0));
                inst.Add(new Instr(Op.movzx_bd, nextvreg, nextvreg, 0));
                return new scalarReg(nextvreg++);
            }

            public scalarReg _mm_testc_si128(__m128i a, __m128i b)
            {
                inst.Add(new Instr(Op.ptest, 0, a.r, b.r));
                inst.Add(new Instr(Op.setc, nextvreg, nextvreg, 0));
                inst.Add(new Instr(Op.movzx_bd, nextvreg, nextvreg, 0));
                return new scalarReg(nextvreg++);
            }

            public scalarReg _mm_testnzc_si128(__m128i a, __m128i b)
            {
                return _mm_test_mix_ones_zeros(a, b);
            }

            public scalarReg _mm_testz_si128(__m128i a, __m128i b)
            {
                return _mm_test_all_zeros(a, b);
            }

            scalarReg _mm_cmpistrcc(__m128i a, __m128i b, SIDD mode, Op setcc)
            {
                int tempreg = nextvreg++;
                int tempreg2 = nextvreg++;
                int resreg = nextvreg++;
                inst.Add(new Instr(Op.assign | Op.W, tempreg, 1, 0));
                inst.Add(new Instr(Op.pcmpistri, tempreg2, a.r, b.r, 0, (int)mode));
                inst.Add(new Instr(Op.assign | Op.W, 1, tempreg, 0));
                inst.Add(new Instr(setcc, resreg, resreg, 0));
                inst.Add(new Instr(Op.movzx_bd, resreg, resreg, 0));
                return new scalarReg(resreg);
            }

            public scalarReg _mm_cmpistra(__m128i a, __m128i b, SIDD mode)
            {
                return _mm_cmpistrcc(a, b, mode, Op.seta);
            }

            public scalarReg _mm_cmpistrc(__m128i a, __m128i b, SIDD mode)
            {
                return _mm_cmpistrcc(a, b, mode, Op.setc);
            }

            public scalarReg _mm_cmpistri(__m128i a, __m128i b, SIDD mode)
            {
                int tempreg = nextvreg++;
                int resreg = nextvreg++;
                inst.Add(new Instr(Op.assign | Op.W, tempreg, 1, 0));
                inst.Add(new Instr(Op.pcmpistri, nextvreg, a.r, b.r, 0, (int)mode));
                inst.Add(new Instr(Op.assign, resreg, nextvreg++, 0));
                inst.Add(new Instr(Op.assign | Op.W, 1, tempreg, 0));
                return new scalarReg(resreg);
            }

            public scalarReg _mm_cmpistro(__m128i a, __m128i b, SIDD mode)
            {
                return _mm_cmpistrcc(a, b, mode, Op.seto);
            }

            public scalarReg _mm_cmpistrs(__m128i a, __m128i b, SIDD mode)
            {
                return _mm_cmpistrcc(a, b, mode, Op.sets);
            }

            public scalarReg _mm_cmpistrz(__m128i a, __m128i b, SIDD mode)
            {
                return _mm_cmpistrcc(a, b, mode, Op.setz);
            }

            public __m128i _mm_cmpistrm(__m128i a, __m128i b, SIDD mode)
            {
                return _2op(Op.pcmpistrm, a, b, (int)mode);
            }

            public __m128i _mm_cmpgt_epi64(__m128i a, __m128i b)
            {
                return _2op(Op.pcmpgtq, a, b);
            }

            public __m256 _mm256_add_ps(__m256 a, __m256 b)
            {
                return _2op(Op.addps, a, b);
            }

            public __m256d _mm256_add_pd(__m256d a, __m256d b)
            {
                return _2op(Op.addpd, a, b);
            }

            public __m256d _mm256_addsub_pd(__m256d a, __m256d b)
            {
                return _2op(Op.addsubpd, a, b);
            }

            public __m256 _mm256_addsub_ps(__m256 a, __m256 b)
            {
                return _2op(Op.addsubps, a, b);
            }

            public __m256d _mm256_and_pd(__m256d a, __m256d b)
            {
                return _2op(Op.andpd, a, b);
            }

            public __m256 _mm256_and_ps(__m256 a, __m256 b)
            {
                return _2op(Op.andps, a, b);
            }

            public __m256d _mm256_andnot_pd(__m256d a, __m256d b)
            {
                return _2op(Op.andnpd, a, b);
            }

            public __m256 _mm256_andnot_ps(__m256 a, __m256 b)
            {
                return _2op(Op.andnps, a, b);
            }

            public __m256d _mm256_blend_pd(__m256d a, __m256d b, int imm)
            {
                return _2op(Op.blendpd, a, b, imm);
            }

            public __m256 _mm256_blend_ps(__m256 a, __m256 b, int imm)
            {
                return _2op(Op.blendps, a, b, imm);
            }

            public __m256d _mm256_blendv_pd(__m256d a, __m256d b, __m256d mask)
            {
                inst.Add(new Instr(Op.blendvpd | Op.L, nextvxmm, a.r, b.r, mask.r));
                return new __m256d(nextvxmm++);
            }

            public __m256 _mm256_blendv_ps(__m256 a, __m256 b, __m256 mask)
            {
                inst.Add(new Instr(Op.blendvps | Op.L, nextvxmm, a.r, b.r, mask.r));
                return new __m256(nextvxmm++);
            }

            public __m256 _mm256_castpd_ps(__m256d a)
            {
                return new __m256(a.r);
            }

            public __m256 _mm256_castsi256_ps(__m256i a)
            {
                return new __m256(a.r);
            }

            public __m256d _mm256_castps_pd(__m256 a)
            {
                return new __m256d(a.r);
            }

            public __m256d _mm256_castsi256_pd(__m256i a)
            {
                return new __m256d(a.r);
            }

            public __m256i _mm256_castps_si256(__m256 a)
            {
                return new __m256i(a.r);
            }

            public __m256i _mm256_castpd_si256(__m256d a)
            {
                return new __m256i(a.r);
            }

            public __m256 _mm256_castps128_ps256(__m128 a)
            {
                return new __m256(a.r);
            }

            public __m256d _mm256_castpd128_pd256(__m128d a)
            {
                return new __m256d(a.r);
            }

            public __m256i _mm256_castsi128_si256(__m128i a)
            {
                return new __m256i(a.r);
            }

            public __m128 _mm256_castps256_ps128(__m256 a)
            {
                return new __m128(a.r);
            }

            public __m128d _mm256_castpd256_pd128(__m256d a)
            {
                return new __m128d(a.r);
            }

            public __m128i _mm256_castsi256_si128(__m256i a)
            {
                return new __m128i(a.r);
            }

            public __m256d _mm256_ceil_pd(__m256d a)
            {
                return _1op(Op.roundpd, a, 2);
            }

            public __m256 _mm256_ceil_ps(__m256 a)
            {
                return _1op(Op.roundps, a, 2);
            }

            public __m256d _mm256_cmp_pd(__m256d a, __m256d b, int imm)
            {
                return _2op(Op.cmppd, a, b, imm);
            }

            public __m256 _mm256_cmp_ps(__m256 a, __m256 b, int imm)
            {
                return _2op(Op.cmpps, a, b, imm);
            }

            public __m256d _mm256_cvtepi32_pd(__m128i a)
            {
                return _1op(Op.cvtdq2pd, new __m256d(a.r), 2);
            }

            public __m256 _mm256_cvtepi32_ps(__m256i a)
            {
                return _1op(Op.cvtdq2ps, new __m256(a.r), 2);
            }

            public __m128i _mm256_cvtpd_epi32(__m256d a)
            {
                var x = _1op(Op.cvtpd2dq, a);
                return new __m128i(x.r);
            }

            public __m256i _mm256_cvtps_epi32(__m256 a)
            {
                return _1op(Op.cvtps2dq, new __m256i(a.r), 2);
            }

            public __m256d _mm256_cvtps_pd(__m128 a)
            {
                return _1op(Op.cvtps2pd, new __m256d(a.r), 2);
            }

            public __m256d _mm256_div_pd(__m256d a, __m256d b)
            {
                return _2op(Op.divpd, a, b);
            }

            public __m256 _mm256_div_ps(__m256 a, __m256 b)
            {
                return _2op(Op.divps, a, b);
            }

            public __m256 _mm256_dp_ps(__m256 a, __m256 b, int imm)
            {
                return _2op(Op.dpps, a, b, imm);
            }

            public scalarReg _mm256_extract_epi16(__m256i a, int index)
            {
                inst.Add(new Instr(Op.pextrw | Op.L, nextvreg, a.r, 0, 0, index));
                return new scalarReg(nextvreg++);
            }

            public scalarReg _mm256_extract_epi32(__m256i a, int index)
            {
                inst.Add(new Instr(Op.pextrd | Op.L, nextvreg, a.r, 0, 0, index));
                return new scalarReg(nextvreg++);
            }

            public scalarReg _mm256_extract_epi64(__m256i a, int index)
            {
                inst.Add(new Instr(Op.pextrq | Op.L, nextvreg, a.r, 0, 0, index));
                return new scalarReg(nextvreg++);
            }

            public scalarReg _mm256_extract_epi8(__m256i a, int index)
            {
                inst.Add(new Instr(Op.pextrb | Op.L, nextvreg, a.r, 0, 0, index));
                return new scalarReg(nextvreg++);
            }

            public __m128d _mm256_extractf128_pd(__m256d a, int index)
            {
                var x = _1op(Op.vextractf128, a, index);
                return new __m128d(x.r);
            }

            public __m128 _mm256_extractf128_ps(__m256 a, int index)
            {
                var x = _1op(Op.vextractf128, a, index);
                return new __m128(x.r);
            }

            public __m128i _mm256_extractf128_si256(__m256i a, int index)
            {
                var x = _1op(Op.vextractf128, a, index);
                return new __m128i(x.r);
            }

            public __m256d _mm256_floor_pd(__m256d a)
            {
                return _1op(Op.roundpd, a, 1);
            }

            public __m256 _mm256_floor_ps(__m256 a)
            {
                return _1op(Op.roundps, a, 1);
            }

            public __m256d _mm256_hadd_pd(__m256d a, __m256d b)
            {
                return _2op(Op.haddpd, a, b);
            }

            public __m256 _mm256_hadd_ps(__m256 a, __m256 b)
            {
                return _2op(Op.haddps, a, b);
            }

            public __m256d _mm256_hsub_pd(__m256d a, __m256d b)
            {
                return _2op(Op.hsubpd, a, b);
            }

            public __m256 _mm256_hsub_ps(__m256 a, __m256 b)
            {
                return _2op(Op.haddps, a, b);
            }

            public __m256i _mm256_insert_epi16(__m256i a, scalarReg v, int index)
            {
                inst.Add(new Instr(Op.pinsrw | Op.L, nextvxmm, a.r, v.r, 0, index));
                return new __m256i(nextvxmm++);
            }

            public __m256i _mm256_insert_epi32(__m256i a, scalarReg v, int index)
            {
                inst.Add(new Instr(Op.pinsrd | Op.L, nextvxmm, a.r, v.r, 0, index));
                return new __m256i(nextvxmm++);
            }

            public __m256i _mm256_insert_epi64(__m256i a, scalarReg v, int index)
            {
                inst.Add(new Instr(Op.pinsrq | Op.L, nextvxmm, a.r, v.r, 0, index));
                return new __m256i(nextvxmm++);
            }

            public __m256i _mm256_insert_epi8(__m256i a, scalarReg v, int index)
            {
                inst.Add(new Instr(Op.pinsrb | Op.L, nextvxmm, a.r, v.r, 0, index));
                return new __m256i(nextvxmm++);
            }

            public __m256d _mm256_insertf128_pd(__m256d a, __m128d b, int index)
            {
                return _2op(Op.vinsertf128, a, new __m256d(b.r), index);
            }

            public __m256 _mm256_insertf128_ps(__m256 a, __m128 b, int index)
            {
                return _2op(Op.vinsertf128, a, new __m256(b.r), index);
            }

            public __m256i _mm256_insertf128_si256(__m256i a, __m128i b, int index)
            {
                return _2op(Op.vinsertf128, a, new __m256i(b.r), index);
            }

            public __m256i _mm256_lddqu_si256(scalarReg ptr, int offset = 0)
            {
                inst.Add(new Instr(Op.lddqu | Op.L, nextvxmm, 0, 0, 0, 0, new MemoryArg(ptr.r, 0, 0, offset)));
                return new __m256i(nextvxmm++);
            }

            public __m256d _mm256_load_pd(scalarReg ptr, int offset = 0)
            {
                inst.Add(new Instr(Op.movapd | Op.L, nextvxmm, 0, 0, 0, 0, new MemoryArg(ptr.r, 0, 0, offset)));
                return new __m256d(nextvxmm++);
            }

            public __m256d _mm256_load_pd(scalarReg ptr, scalarReg index, int scale = 1, int offset = 0)
            {
                inst.Add(new Instr(Op.movapd | Op.L, nextvxmm, 0, 0, 0, 0, new MemoryArg(ptr.r, index.r, scale, offset)));
                return new __m256d(nextvxmm++);
            }

            public __m256i _mm256_load_si256(scalarReg ptr, int offset = 0)
            {
                inst.Add(new Instr(Op.movdqa | Op.L, nextvxmm, 0, 0, 0, 0, new MemoryArg(ptr.r, 0, 0, offset)));
                return new __m256i(nextvxmm++);
            }

            public __m256i _mm256_load_si256(scalarReg ptr, scalarReg index, int scale = 1, int offset = 0)
            {
                inst.Add(new Instr(Op.movdqa | Op.L, nextvxmm, 0, 0, 0, 0, new MemoryArg(ptr.r, index.r, scale, offset)));
                return new __m256i(nextvxmm++);
            }

            public __m256d _mm256_loadu_pd(scalarReg ptr, int offset = 0)
            {
                inst.Add(new Instr(Op.movupd | Op.L, nextvxmm, 0, 0, 0, 0, new MemoryArg(ptr.r, 0, 0, offset)));
                return new __m256d(nextvxmm++);
            }

            public __m256d _mm256_loadu_pd(scalarReg ptr, scalarReg index, int scale = 1, int offset = 0)
            {
                inst.Add(new Instr(Op.movupd | Op.L, nextvxmm, 0, 0, 0, 0, new MemoryArg(ptr.r, index.r, scale, offset)));
                return new __m256d(nextvxmm++);
            }

            public __m256 _mm256_loadu_ps(scalarReg ptr, int offset = 0)
            {
                inst.Add(new Instr(Op.movups | Op.L, nextvxmm, 0, 0, 0, 0, new MemoryArg(ptr.r, 0, 0, offset)));
                return new __m256(nextvxmm++);
            }

            public __m256 _mm256_loadu_ps(scalarReg ptr, scalarReg index, int scale = 1, int offset = 0)
            {
                inst.Add(new Instr(Op.movups | Op.L, nextvxmm, 0, 0, 0, 0, new MemoryArg(ptr.r, index.r, scale, offset)));
                return new __m256(nextvxmm++);
            }

            public __m256i _mm256_loadu_si256(scalarReg ptr, int offset = 0)
            {
                inst.Add(new Instr(Op.movdqu | Op.L, nextvxmm, 0, 0, 0, 0, new MemoryArg(ptr.r, 0, 0, offset)));
                return new __m256i(nextvxmm++);
            }

            public __m256i _mm256_loadu_si256(scalarReg ptr, scalarReg index, int scale = 1, int offset = 0)
            {
                inst.Add(new Instr(Op.movdqu | Op.L, nextvxmm, 0, 0, 0, 0, new MemoryArg(ptr.r, index.r, scale, offset)));
                return new __m256i(nextvxmm++);
            }

            public __m256d _mm256_max_pd(__m256d a, __m256d b)
            {
                return _2op(Op.maxpd, a, b);
            }

            public __m256 _mm256_max_ps(__m256 a, __m256 b)
            {
                return _2op(Op.maxps, a, b);
            }

            public __m256d _mm256_min_pd(__m256d a, __m256d b)
            {
                return _2op(Op.minpd, a, b);
            }

            public __m256 _mm256_min_ps(__m256 a, __m256 b)
            {
                return _2op(Op.minps, a, b);
            }

            public __m256d _mm256_movedup_pd(__m256d a)
            {
                return _1op(Op.movddup, a);
            }

            public __m256 _mm256_movehdup_ps(__m256 a)
            {
                return _1op(Op.movshdup, a);
            }

            public __m256 _mm256_moveldup_ps(__m256 a)
            {
                return _1op(Op.movsldup, a);
            }

            public scalarReg _mm256_movemask_pd(__m256d a)
            {
                inst.Add(new Instr(Op.movmskpd | Op.L, nextvreg, a.r, 0));
                return new scalarReg(nextvreg++);
            }

            public scalarReg _mm256_movemask_ps(__m256 a)
            {
                inst.Add(new Instr(Op.movmskps | Op.L, nextvreg, a.r, 0));
                return new scalarReg(nextvreg++);
            }

            public __m256d _mm256_mul_pd(__m256d a, __m256d b)
            {
                return _2op(Op.mulpd, a, b);
            }

            public __m256 _mm256_mul_ps(__m256 a, __m256 b)
            {
                return _2op(Op.mulps, a, b);
            }

            public __m256d _mm256_or_pd(__m256d a, __m256d b)
            {
                return _2op(Op.orpd, a, b);
            }

            public __m256 _mm256_or_ps(__m256 a, __m256 b)
            {
                return _2op(Op.orps, a, b);
            }

            public __m256d _mm256_permute_pd(__m256d a, int imm)
            {
                return _1op(Op.vpermilpdi, a, imm);
            }

            public __m256 _mm256_permute_ps(__m256 a, int imm)
            {
                return _1op(Op.vpermilpsi, a, imm);
            }

            public __m128d _mm_permute_pd(__m128d a, int imm)
            {
                return _1op(Op.vpermilpdi, a, imm);
            }

            public __m128 _mm_permute_ps(__m128 a, int imm)
            {
                return _1op(Op.vpermilpsi, a, imm);
            }

            public __m256d _mm256_permutevar_pd(__m256d a, __m256d b)
            {
                return _2op(Op.vpermilpd, a, b);
            }

            public __m256 _mm256_permutevar_ps(__m256 a, __m256 b)
            {
                return _2op(Op.vpermilps, a, b);
            }

            public __m128d _mm_permutevar_pd(__m128d a, __m128d b)
            {
                return _2op(Op.vpermilpd, a, b);
            }

            public __m128 _mm_permutevar_ps(__m128 a, __m128 b)
            {
                return _2op(Op.vpermilps, a, b);
            }

            public __m256d _mm256_permute2f128_pd(__m256d a, __m256d b, int imm)
            {
                return _2op(Op.vperm2f128, a, b, imm);
            }

            public __m256 _mm256_permute2f128_ps(__m256 a, __m256 b, int imm)
            {
                return _2op(Op.vperm2f128, a, b, imm);
            }

            public __m256i _mm256_permute2f128_si256(__m256i a, __m256i b, int imm)
            {
                return _2op(Op.vperm2f128, a, b, imm);
            }

            public __m256 _mm256_rcp_ps(__m256 a)
            {
                return _1op(Op.rcpps, a);
            }

            public __m256d _mm256_round_pd(__m256d a, SIDD mode)
            {
                return _1op(Op.roundpd, a, (int)mode);
            }

            public __m256 _mm256_round_ps(__m256 a, SIDD mode)
            {
                return _1op(Op.roundps, a, (int)mode);
            }

            public __m256 _mm256_rsqrt_ps(__m256 a)
            {
                return _1op(Op.rsqrtps, a);
            }

            public __m256i _mm256_set_epi16(short e15, short e14, short e13, short e12, short e11, short e10, short e9, short e8, short e7, short e6, short e5, short e4, short e3, short e2, short e1, short e0)
            {
                short[] d = { e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13, e14, e15 };
                int label = nextdatalabel--;
                Block b = new Block(label, 32);
                for (int i = 0; i < d.Length; i++)
                {
                    b.buffer.Add((byte)d[i]);
                    b.buffer.Add((byte)(d[i] >> 8));
                }
                data.Add(b);

                inst.Add(new Instr(Op.movdqa | Op.L, nextvxmm, 0, 0, 0, 0, new MemoryArg(0, 0, 0, label)));
                return new __m256i(nextvxmm++);
            }

            public __m256i _mm256_set_epi32(int e7, int e6, int e5, int e4, int e3, int e2, int e1, int e0)
            {
                int[] d = { e0, e1, e2, e3, e4, e5, e6, e7 };
                int label = nextdatalabel--;
                Block b = new Block(label, 32);
                for (int i = 0; i < d.Length; i++)
                    b.buffer.AddRange(BitConverter.GetBytes(d[i]));
                data.Add(b);

                inst.Add(new Instr(Op.movdqa | Op.L, nextvxmm, 0, 0, 0, 0, new MemoryArg(0, 0, 0, label)));
                return new __m256i(nextvxmm++);
            }

            public __m256i _mm256_set_epi64x(long e3, long e2, long e1, long e0)
            {
                long[] d = { e0, e1, e2, e3 };
                int label = nextdatalabel--;
                Block b = new Block(label, 32);
                for (int i = 0; i < d.Length; i++)
                    b.buffer.AddRange(BitConverter.GetBytes(d[i]));
                data.Add(b);

                inst.Add(new Instr(Op.movdqa | Op.L, nextvxmm, 0, 0, 0, 0, new MemoryArg(0, 0, 0, label)));
                return new __m256i(nextvxmm++);
            }

            public __m256i _mm256_set_epi8(byte e31, byte e30, byte e29, byte e28, byte e27, byte e26, byte e25, byte e24, byte e23, byte e22, byte e21, byte e20, byte e19, byte e18, byte e17, byte e16, byte e15, byte e14, byte e13, byte e12, byte e11, byte e10, byte e9, byte e8, byte e7, byte e6, byte e5, byte e4, byte e3, byte e2, byte e1, byte e0)
            {
                byte[] d = { e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13, e14, e15,
                           e16, e17, e18, e19, e20, e21, e22, e23, e24, e25, e26, e27, e28, e29, e30, e31};
                int label = nextdatalabel--;
                Block b = new Block(label, 32);
                b.buffer.AddRange(d);
                data.Add(b);

                inst.Add(new Instr(Op.movdqa | Op.L, nextvxmm, 0, 0, 0, 0, new MemoryArg(0, 0, 0, label)));
                return new __m256i(nextvxmm++);
            }

            public __m256 _mm256_set_m128(__m128 hi, __m128 lo)
            {
                return _2op(Op.vinsertf128, new __m256(lo.r), new __m256(hi.r), 1);
            }

            public __m256d _mm256_set_m128d(__m128d hi, __m128d lo)
            {
                return _2op(Op.vinsertf128, new __m256d(lo.r), new __m256d(hi.r), 1);
            }

            public __m256i _mm256_set_m128i(__m128i hi, __m128i lo)
            {
                return _2op(Op.vinsertf128, new __m256i(lo.r), new __m256i(hi.r), 1);
            }

            public __m256d _mm256_set_pd(double e3, double e2, double e1, double e0)
            {
                long[] d = 
                { 
                    BitConverter.DoubleToInt64Bits(e0),
                    BitConverter.DoubleToInt64Bits(e1),
                    BitConverter.DoubleToInt64Bits(e2),
                    BitConverter.DoubleToInt64Bits(e3)
                };

                int label = nextdatalabel--;
                Block b = new Block(label, 32);
                for (int i = 0; i < d.Length; i++)
                    b.buffer.AddRange(BitConverter.GetBytes(d[i]));
                data.Add(b);

                inst.Add(new Instr(Op.movapd | Op.L, nextvxmm, 0, 0, 0, 0, new MemoryArg(0, 0, 0, label)));
                return new __m256d(nextvxmm++);
            }

            public __m256 _mm256_set_ps(float e7, float e6, float e5, float e4, float e3, float e2, float e1, float e0)
            {
                float[] d = new float[] { e0, e1, e2, e3, e4, e5, e6, e7 };

                int label = nextdatalabel--;
                Block b = new Block(label, 32);
                for (int i = 0; i < d.Length; i++)
                    b.buffer.AddRange(BitConverter.GetBytes(d[i]));
                data.Add(b);

                inst.Add(new Instr(Op.movaps | Op.L, nextvxmm, 0, 0, 0, 0, new MemoryArg(0, 0, 0, label)));
                return new __m256(nextvxmm++);
            }

            public __m256i _mm256_set1_epi16(short a)
            {
                if (a == 0)
                    return _mm256_setzero_si256();
                else if (a == -1)
                    return _2op(Op.pcmpeqb, new __m256i(0x100000), new __m256i(0x100000));

                int label = nextdatalabel--;
                Block b = new Block(label, 2);
                b.buffer.AddRange(BitConverter.GetBytes(a));
                data.Add(b);

                inst.Add(new Instr(Op.vpbroadcastw | Op.L, nextvxmm, 0, 0, 0, 0, new MemoryArg(0, 0, 0, label)));
                return new __m256i(nextvxmm++);
            }

            public __m256i _mm256_set1_epi32(int a)
            {
                if (a == 0)
                    return _mm256_setzero_si256();
                else if (a == -1)
                    return _2op(Op.pcmpeqb, new __m256i(0x100000), new __m256i(0x100000));

                int label = nextdatalabel--;
                Block b = new Block(label, 4);
                b.buffer.AddRange(BitConverter.GetBytes(a));
                data.Add(b);

                inst.Add(new Instr(Op.vpbroadcastd | Op.L, nextvxmm, 0, 0, 0, 0, new MemoryArg(0, 0, 0, label)));
                return new __m256i(nextvxmm++);
            }

            public __m256i _mm256_set1_epi64x(long a)
            {
                if (a == 0)
                    return _mm256_setzero_si256();
                else if (a == -1)
                    return _2op(Op.pcmpeqb, new __m256i(0x100000), new __m256i(0x100000));

                int label = nextdatalabel--;
                Block b = new Block(label, 8);
                b.buffer.AddRange(BitConverter.GetBytes(a));
                data.Add(b);

                inst.Add(new Instr(Op.vpbroadcastq | Op.L, nextvxmm, 0, 0, 0, 0, new MemoryArg(0, 0, 0, label)));
                return new __m256i(nextvxmm++);
            }

            public __m256i _mm256_set1_epi8(byte a)
            {
                if (a == 0)
                    return _mm256_setzero_si256();
                else if (a == 255)
                    return _2op(Op.pcmpeqb, new __m256i(0x100000), new __m256i(0x100000));

                int label = nextdatalabel--;
                Block b = new Block(label, 1);
                b.buffer.AddRange(BitConverter.GetBytes(a));
                data.Add(b);

                inst.Add(new Instr(Op.vpbroadcastb | Op.L, nextvxmm, 0, 0, 0, 0, new MemoryArg(0, 0, 0, label)));
                return new __m256i(nextvxmm++);
            }

            public __m256d _mm256_set1_pd(double a)
            {
                if (BitConverter.DoubleToInt64Bits(a) == 0)
                    return _mm256_setzero_pd();

                int label = nextdatalabel--;
                Block b = new Block(label, 8);
                b.buffer.AddRange(BitConverter.GetBytes(a));
                data.Add(b);

                inst.Add(new Instr(Op.vbroadcastsd | Op.L, nextvxmm, 0, 0, 0, 0, new MemoryArg(0, 0, 0, label)));
                return new __m256d(nextvxmm++);
            }

            public __m256 _mm256_set1_ps(float a)
            {
                if (BitConverter.DoubleToInt64Bits(a) == 0)
                    return _mm256_setzero_ps();

                int label = nextdatalabel--;
                Block b = new Block(label, 4);
                b.buffer.AddRange(BitConverter.GetBytes(a));
                data.Add(b);

                inst.Add(new Instr(Op.vbroadcastss | Op.L, nextvxmm, 0, 0, 0, 0, new MemoryArg(0, 0, 0, label)));
                return new __m256(nextvxmm++);
            }

            public __m256i _mm256_setr_epi16(short e15, short e14, short e13, short e12, short e11, short e10, short e9, short e8, short e7, short e6, short e5, short e4, short e3, short e2, short e1, short e0)
            {
                return _mm256_set_epi16(e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13, e14, e15);
            }

            public __m256i _mm256_setr_epi32(int e7, int e6, int e5, int e4, int e3, int e2, int e1, int e0)
            {
                return _mm256_set_epi32(e0, e1, e2, e3, e4, e5, e6, e7);
            }

            public __m256i _mm256_setr_epi64x(long e3, long e2, long e1, long e0)
            {
                return _mm256_set_epi64x(e0, e1, e2, e3);
            }

            public __m256i _mm256_setr_epi8(byte e31, byte e30, byte e29, byte e28, byte e27, byte e26, byte e25, byte e24, byte e23, byte e22, byte e21, byte e20, byte e19, byte e18, byte e17, byte e16, byte e15, byte e14, byte e13, byte e12, byte e11, byte e10, byte e9, byte e8, byte e7, byte e6, byte e5, byte e4, byte e3, byte e2, byte e1, byte e0)
            {
                return _mm256_set_epi8(e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13, e14, e15, e16, e17, e18, e19, e20, e21, e22, e23, e24, e25, e26, e27, e28, e29, e30, e31);
            }

            public __m256d _mm256_setr_pd(double e3, double e2, double e1, double e0)
            {
                return _mm256_set_pd(e0, e1, e2, e3);
            }

            public __m256 _mm256_setr_ps(float e7, float e6, float e5, float e4, float e3, float e2, float e1, float e0)
            {
                return _mm256_set_ps(e0, e1, e2, e3, e4, e5, e6, e7);
            }

            public __m256 _mm256_setzero_ps()
            {
                return _2op(Op.xorps, new __m256(0x100000), new __m256(0x100000));
            }

            public __m256d _mm256_setzero_pd()
            {
                return _2op(Op.xorpd, new __m256d(0x100000), new __m256d(0x100000));
            }

            public __m256i _mm256_setzero_si256()
            {
                return _2op(Op.pxor, new __m256i(0x100000), new __m256i(0x100000));
            }

            public __m256d _mm256_shuffle_pd(__m256d a, __m256d b, int imm)
            {
                return _2op(Op.shufpd, a, b, imm);
            }

            public __m256 _mm256_shuffle_ps(__m256 a, __m256 b, int imm)
            {
                return _2op(Op.shufps, a, b, imm);
            }

            public __m256d _mm256_sqrt_pd(__m256d a)
            {
                return _1op(Op.sqrtpd, a);
            }

            public __m256 _mm256_sqrt_ps(__m256 a)
            {
                return _1op(Op.sqrtps, a);
            }

            public void _mm256_stream_pd(scalarReg ptr, __m256d a, int offset = 0)
            {
                inst.Add(new Instr(Op.movntpd, nextvxmm, 0, 0, 0, 0, new MemoryArg(ptr.r, 0, 0, offset)));
            }

            public void _mm256_stream_ps(scalarReg ptr, __m256 a, int offset = 0)
            {
                inst.Add(new Instr(Op.movntps, nextvxmm, 0, 0, 0, 0, new MemoryArg(ptr.r, 0, 0, offset)));
            }

            public __m256d _mm256_sub_pd(__m256d a, __m256d b)
            {
                return _2op(Op.subpd, a, b);
            }

            public __m256 _mm256_sub_ps(__m256 a, __m256 b)
            {
                return _2op(Op.subps, a, b);
            }

            public __m256d _mm256_unpackhi_pd(__m256d a, __m256d b)
            {
                return _2op(Op.unpckhpd, a, b);
            }

            public __m256 _mm256_unpackhi_ps(__m256 a, __m256 b)
            {
                return _2op(Op.unpckhps, a, b);
            }

            public __m256d _mm256_unpacklo_pd(__m256d a, __m256d b)
            {
                return _2op(Op.unpcklpd, a, b);
            }

            public __m256 _mm256_unpacklo_ps(__m256 a, __m256 b)
            {
                return _2op(Op.unpcklps, a, b);
            }

            public __m256d _mm256_xor_pd(__m256d a, __m256d b)
            {
                return _2op(Op.xorpd, a, b);
            }

            public __m256 _mm256_xor_ps(__m256 a, __m256 b)
            {
                return _2op(Op.xorps, a, b);
            }

            public void _mm256_zeroall()
            {
                inst.Add(new Instr(Op.vzeroall, 0, 0, 0));
            }

            public void _mm256_zeroupper()
            {
                inst.Add(new Instr(Op.vzeroupper, 0, 0, 0));
            }

            public __m256i _mm256_abs_epi16(__m256i a)
            {
                return _1op(Op.pabsw, a);
            }

            public __m256i _mm256_abs_epi32(__m256i a)
            {
                return _1op(Op.pabsd, a);
            }

            public __m256i _mm256_abs_epi8(__m256i a)
            {
                return _1op(Op.pabsb, a);
            }

            public __m256i _mm256_add_epi16(__m256i a, __m256i b)
            {
                return _2op(Op.paddw, a, b);
            }

            public __m256i _mm256_add_epi32(__m256i a, __m256i b)
            {
                return _2op(Op.paddd, a, b);
            }

            public __m256i _mm256_add_epi64(__m256i a, __m256i b)
            {
                return _2op(Op.paddq, a, b);
            }

            public __m256i _mm256_add_epi8(__m256i a, __m256i b)
            {
                return _2op(Op.paddb, a, b);
            }

            public __m256i _mm256_adds_epi16(__m256i a, __m256i b)
            {
                return _2op(Op.paddsw, a, b);
            }

            public __m256i _mm256_adds_epi8(__m256i a, __m256i b)
            {
                return _2op(Op.paddsb, a, b);
            }

            public __m256i _mm256_adds_epu16(__m256i a, __m256i b)
            {
                return _2op(Op.paddusw, a, b);
            }

            public __m256i _mm256_adds_epu8(__m256i a, __m256i b)
            {
                return _2op(Op.paddusb, a, b);
            }

            public __m256i _mm256_alignr_epi8(__m256i a, __m256i b, int count)
            {
                return _2op(Op.palignr, a, b, count);
            }

            public __m256i _mm256_and_si256(__m256i a, __m256i b)
            {
                return _2op(Op.pand, a, b);
            }

            public __m256i _mm256_andnot_si256(__m256i a, __m256i b)
            {
                return _2op(Op.pandn, a, b);
            }

            public __m256i _mm256_avg_epu16(__m256i a, __m256i b)
            {
                return _2op(Op.pavgw, a, b);
            }

            public __m256i _mm256_avg_epu8(__m256i a, __m256i b)
            {
                return _2op(Op.pavgw, a, b);
            }

            public __m256i _mm256_blend_epi16(__m256i a, __m256i b, int imm8)
            {
                return _2op(Op.pblendw, a, b, imm8);
            }

            public __m128i _mm_blend_epi32(__m128i a, __m128i b, int imm8)
            {
                return _2op(Op.vpblendd, a, b, imm8);
            }

            public __m256i _mm256_blend_epi32(__m256i a, __m256i b, int imm8)
            {
                return _2op(Op.vpblendd, a, b, imm8);
            }

            public __m256i _mm256_blendv_epi8(__m256i a, __m256i b, __m256i mask)
            {
                inst.Add(new Instr(Op.pblendvb | Op.L, nextvxmm, a.r, b.r, mask.r));
                return new __m256i(nextvxmm++);
            }

            public __m128i _mm_broadcastb_epi8(__m128i a)
            {
                return _1op(Op.vpbroadcastb, a);
            }

            public __m256i _mm256_broadcastb_epi8(__m128i a)
            {
                return _1op(Op.vpbroadcastb, new __m256i(a.r));
            }

            public __m128i _mm_broadcastd_epi32(__m128i a)
            {
                return _1op(Op.vpbroadcastd, a);
            }

            public __m256i _mm256_broadcastd_epi32(__m128i a)
            {
                return _1op(Op.vpbroadcastd, new __m256i(a.r));
            }

            public __m128i _mm_broadcastq_epi64(__m128i a)
            {
                return _1op(Op.vpbroadcastq, a);
            }

            public __m256i _mm256_broadcastq_epi64(__m128i a)
            {
                return _1op(Op.vpbroadcastq, new __m256i(a.r));
            }

            public __m128d _mm_broadcastsd_pd(__m128d a)
            {
                return _1op(Op.vbroadcastsd, a);
            }

            public __m256d _mm256_broadcastsd_pd(__m128d a)
            {
                return _1op(Op.vbroadcastsd, new __m256d(a.r));
            }

            public __m256i _mm_broadcastsi128_si256(__m128i a)
            {
                return _1op(Op.vpbroadcasti128, new __m256i(a.r));
            }

            public __m256i _mm256_broadcastsi128_si256(__m128i a)
            {
                return _1op(Op.vpbroadcasti128, new __m256i(a.r));
            }

            public __m128 _mm_broadcastss_ps(__m128 a)
            {
                return _1op(Op.vbroadcastss, a);
            }

            public __m256 _mm256_broadcastss_ps(__m128 a)
            {
                return _1op(Op.vbroadcastss, new __m256(a.r));
            }

            public __m128i _mm_broadcastw_epi16(__m128i a)
            {
                return _1op(Op.vpbroadcastw, a);
            }

            public __m256i _mm256_broadcastw_epi16(__m128i a)
            {
                return _1op(Op.vpbroadcastw, new __m256i(a.r));
            }

            public __m256i _mm256_bslli_epi128(__m256i a, int imm8)
            {
                return _1op(Op.pslldq, a, imm8);
            }

            public __m256i _mm256_bsrli_epi128(__m256i a, int imm8)
            {
                return _1op(Op.psrldq, a, imm8);
            }

            public __m256i _mm256_cmpeq_epi16(__m256i a, __m256i b)
            {
                return _2op(Op.pcmpeqw, a, b);
            }

            public __m256i _mm256_cmpeq_epi32(__m256i a, __m256i b)
            {
                return _2op(Op.pcmpeqd, a, b);
            }

            public __m256i _mm256_cmpeq_epi64(__m256i a, __m256i b)
            {
                return _2op(Op.pcmpeqq, a, b);
            }

            public __m256i _mm256_cmpeq_epi8(__m256i a, __m256i b)
            {
                return _2op(Op.pcmpeqb, a, b);
            }

            public __m256i _mm256_cmpgt_epi16(__m256i a, __m256i b)
            {
                return _2op(Op.pcmpgtw, a, b);
            }

            public __m256i _mm256_cmpgt_epi32(__m256i a, __m256i b)
            {
                return _2op(Op.pcmpgtd, a, b);
            }

            public __m256i _mm256_cmpgt_epi64(__m256i a, __m256i b)
            {
                return _2op(Op.pcmpgtq, a, b);
            }

            public __m256i _mm256_cmpgt_epi8(__m256i a, __m256i b)
            {
                return _2op(Op.pcmpgtb, a, b);
            }

            public __m256i _mm256_cvtepi16_epi32(__m128i a)
            {
                return _1op(Op.pmovsxwd, new __m256i(a.r));
            }

            public __m256i _mm256_cvtepi16_epi64(__m128i a)
            {
                return _1op(Op.pmovsxwq, new __m256i(a.r));
            }

            public __m256i _mm256_cvtepi32_epi64(__m128i a)
            {
                return _1op(Op.pmovsxdq, new __m256i(a.r));
            }

            public __m256i _mm256_cvtepi8_epi16(__m128i a)
            {
                return _1op(Op.pmovsxbw, new __m256i(a.r));
            }

            public __m256i _mm256_cvtepi8_epi32(__m128i a)
            {
                return _1op(Op.pmovsxbd, new __m256i(a.r));
            }

            public __m256i _mm256_cvtepi8_epi64(__m128i a)
            {
                return _1op(Op.pmovsxbq, new __m256i(a.r));
            }

            public __m256i _mm256_cvtepu16_epi32(__m128i a)
            {
                return _1op(Op.pmovzxwd, new __m256i(a.r));
            }

            public __m256i _mm256_cvtepu16_epi64(__m128i a)
            {
                return _1op(Op.pmovzxwq, new __m256i(a.r));
            }

            public __m256i _mm256_cvtepu32_epi64(__m128i a)
            {
                return _1op(Op.pmovzxdq, new __m256i(a.r));
            }

            public __m256i _mm256_cvtepu8_epi16(__m128i a)
            {
                return _1op(Op.pmovzxbw, new __m256i(a.r));
            }

            public __m256i _mm256_cvtepu8_epi32(__m128i a)
            {
                return _1op(Op.pmovzxbd, new __m256i(a.r));
            }

            public __m256i _mm256_cvtepu8_epi64(__m128i a)
            {
                return _1op(Op.pmovzxbq, new __m256i(a.r));
            }

            public __m128i _mm256_extracti128_si256(__m256i a, int imm8)
            {
                var x = _1op(Op.vextracti128, a, imm8);
                return new __m128i(x.r);
            }

            public __m256i _mm256_hadd_epi16(__m256i a, __m256i b)
            {
                return _2op(Op.phaddw, a, b);
            }

            public __m256i _mm256_hadd_epi32(__m256i a, __m256i b)
            {
                return _2op(Op.phaddd, a, b);
            }

            public __m256i _mm256_hadds_epi16(__m256i a, __m256i b)
            {
                return _2op(Op.phaddsw, a, b);
            }

            public __m256i _mm256_hsub_epi16(__m256i a, __m256i b)
            {
                return _2op(Op.phsubw, a, b);
            }

            public __m256i _mm256_hsub_epi32(__m256i a, __m256i b)
            {
                return _2op(Op.phsubd, a, b);
            }

            public __m256i _mm256_hsubs_epi16(__m256i a, __m256i b)
            {
                return _2op(Op.phsubw, a, b);
            }

            public __m256i _mm256_inserti128_si256(__m256i a, __m128i b, int imm8)
            {
                return _2op(Op.vinserti128, a, new __m256i(b.r), imm8);
            }

            public __m256i _mm256_madd_epi16(__m256i a, __m256i b)
            {
                return _2op(Op.pmaddwd, a, b);
            }

            public __m256i _mm256_maddubs_epi16(__m256i a, __m256i b)
            {
                return _2op(Op.pmaddubsw, a, b);
            }

            public __m256i _mm256_max_epi16(__m256i a, __m256i b)
            {
                return _2op(Op.pmaxsw, a, b);
            }

            public __m256i _mm256_max_epi32(__m256i a, __m256i b)
            {
                return _2op(Op.pmaxsd, a, b);
            }

            public __m256i _mm256_max_epi8(__m256i a, __m256i b)
            {
                return _2op(Op.pmaxsb, a, b);
            }

            public __m256i _mm256_max_epu16(__m256i a, __m256i b)
            {
                return _2op(Op.pmaxuw, a, b);
            }

            public __m256i _mm256_max_epu32(__m256i a, __m256i b)
            {
                return _2op(Op.pmaxud, a, b);
            }

            public __m256i _mm256_max_epu8(__m256i a, __m256i b)
            {
                return _2op(Op.pmaxub, a, b);
            }

            public __m256i _mm256_min_epi16(__m256i a, __m256i b)
            {
                return _2op(Op.pminsw, a, b);
            }

            public __m256i _mm256_min_epi32(__m256i a, __m256i b)
            {
                return _2op(Op.pminsd, a, b);
            }

            public __m256i _mm256_min_epi8(__m256i a, __m256i b)
            {
                return _2op(Op.pminsb, a, b);
            }

            public __m256i _mm256_min_epu16(__m256i a, __m256i b)
            {
                return _2op(Op.pminuw, a, b);
            }

            public __m256i _mm256_min_epu32(__m256i a, __m256i b)
            {
                return _2op(Op.pminud, a, b);
            }

            public __m256i _mm256_min_epu8(__m256i a, __m256i b)
            {
                return _2op(Op.pminub, a, b);
            }

            public scalarReg _mm256_movemask_epi8(__m256i a)
            {
                inst.Add(new Instr(Op.pmovmskb, nextvreg, a.r, 0));
                return new scalarReg(nextvreg++);
            }

            public __m256i _mm256_mpsadbw_epu8(__m256i a, __m256i b, int imm8)
            {
                return _2op(Op.mpsadbw, a, b, imm8);
            }

            public __m256i _mm256_mul_epi32(__m256i a, __m256i b)
            {
                return _2op(Op.pmuldq, a, b);
            }

            public __m256i _mm256_mul_epu32(__m256i a, __m256i b)
            {
                return _2op(Op.pmuludq, a, b);
            }

            public __m256i _mm256_mulhi_epi16(__m256i a, __m256i b)
            {
                return _2op(Op.pmulhw, a, b);
            }

            public __m256i _mm256_mulhi_epu16(__m256i a, __m256i b)
            {
                return _2op(Op.pmulhuw, a, b);
            }

            public __m256i _mm256_mulhrs_epi16(__m256i a, __m256i b)
            {
                return _2op(Op.pmulhrsw, a, b);
            }

            public __m256i _mm256_mullo_epi16(__m256i a, __m256i b)
            {
                return _2op(Op.pmullw, a, b);
            }

            public __m256i _mm256_mullo_epi32(__m256i a, __m256i b)
            {
                return _2op(Op.pmulld, a, b);
            }

            public __m256i _mm256_or_si256(__m256i a, __m256i b)
            {
                return _2op(Op.por, a, b);
            }

            public __m256i _mm256_packs_epi16(__m256i a, __m256i b)
            {
                return _2op(Op.packsswb, a, b);
            }

            public __m256i _mm256_packs_epi32(__m256i a, __m256i b)
            {
                return _2op(Op.packssdw, a, b);
            }

            public __m256i _mm256_packus_epi16(__m256i a, __m256i b)
            {
                return _2op(Op.packuswb, a, b);
            }

            public __m256i _mm256_packus_epi32(__m256i a, __m256i b)
            {
                return _2op(Op.packusdw, a, b);
            }

            public __m256i _mm256_sad_epu8(__m256i a, __m256i b)
            {
                return _2op(Op.psadbw, a, b);
            }

            public __m256i _mm256_shuffle_epi32(__m256i a, int imm8)
            {
                return _1op(Op.pshufd, a, imm8);
            }

            public __m256i _mm256_shuffle_epi8(__m256i a, __m256i b)
            {
                return _2op(Op.pshufb, a, b);
            }

            public __m256i _mm256_shufflehi_epi16(__m256i a, int imm8)
            {
                return _1op(Op.pshufhw, a, imm8);
            }

            public __m256i _mm256_shufflelo_epi16(__m256i a, int imm8)
            {
                return _1op(Op.pshuflw, a, imm8);
            }

            public __m256i _mm256_sign_epi16(__m256i a, __m256i b)
            {
                return _1op(Op.psignw, a);
            }

            public __m256i _mm256_sign_epi32(__m256i a, __m256i b)
            {
                return _1op(Op.psignd, a);
            }

            public __m256i _mm256_sign_epi8(__m256i a, __m256i b)
            {
                return _1op(Op.psignb, a);
            }

            public __m256i _mm256_sll_epi16(__m256i a, __m128i count)
            {
                return _2op(Op.psllw, a, new __m256i(count.r));
            }

            public __m256i _mm256_sll_epi32(__m256i a, __m128i count)
            {
                return _2op(Op.pslld, a, new __m256i(count.r));
            }

            public __m256i _mm256_sll_epi64(__m256i a, __m128i count)
            {
                return _2op(Op.psllq, a, new __m256i(count.r));
            }

            public __m256i _mm256_slli_epi16(__m256i a, int imm8)
            {
                return _1op(Op.psllwi, a, imm8);
            }

            public __m256i _mm256_slli_epi32(__m256i a, int imm8)
            {
                return _1op(Op.pslldi, a, imm8);
            }

            public __m256i _mm256_slli_epi64(__m256i a, int imm8)
            {
                return _1op(Op.psllqi, a, imm8);
            }

            public __m256i _mm256_slli_si256(__m256i a, int imm8)
            {
                return _1op(Op.pslldq, a, imm8);
            }

            public __m128i _mm_sllv_epi32(__m128i a, __m128i count)
            {
                return _2op(Op.vpsllvd, a, count);
            }

            public __m256i _mm256_sllv_epi32(__m256i a, __m256i count)
            {
                return _2op(Op.vpsllvd, a, count);
            }

            public __m128i _mm_sllv_epi64(__m128i a, __m128i count)
            {
                return _2op(Op.vpsllvq, a, count);
            }

            public __m256i _mm256_sllv_epi64(__m256i a, __m256i count)
            {
                return _2op(Op.vpsllvq, a, count);
            }

            public __m256i _mm256_sra_epi16(__m256i a, __m128i count)
            {
                return _2op(Op.psraw, a, new __m256i(count.r));
            }

            public __m256i _mm256_sra_epi32(__m256i a, __m128i count)
            {
                return _2op(Op.psrad, a, new __m256i(count.r));
            }

            public __m256i _mm256_srai_epi16(__m256i a, int imm8)
            {
                return _1op(Op.psrawi, a, imm8);
            }

            public __m256i _mm256_srai_epi32(__m256i a, int imm8)
            {
                return _1op(Op.psradi, a, imm8);
            }

            public __m128i _mm_srav_epi32(__m128i a, __m128i count)
            {
                return _2op(Op.vpsravd, a, count);
            }

            public __m256i _mm256_srav_epi32(__m256i a, __m256i count)
            {
                return _2op(Op.vpsravd, a, count);
            }

            public __m256i _mm256_srl_epi16(__m256i a, __m128i count)
            {
                return _2op(Op.psrlw, a, new __m256i(count.r));
            }

            public __m256i _mm256_srl_epi32(__m256i a, __m128i count)
            {
                return _2op(Op.psrld, a, new __m256i(count.r));
            }

            public __m256i _mm256_srl_epi64(__m256i a, __m128i count)
            {
                return _2op(Op.psrlq, a, new __m256i(count.r));
            }

            public __m256i _mm256_srli_epi16(__m256i a, int imm8)
            {
                return _1op(Op.psrlwi, a, imm8);
            }

            public __m256i _mm256_srli_epi32(__m256i a, int imm8)
            {
                return _1op(Op.psrldi, a, imm8);
            }

            public __m256i _mm256_srli_epi64(__m256i a, int imm8)
            {
                return _1op(Op.psrlqi, a, imm8);
            }

            public __m256i _mm256_srli_si256(__m256i a, int imm8)
            {
                return _1op(Op.psrldq, a, imm8);
            }

            public __m128i _mm_srlv_epi32(__m128i a, __m128i count)
            {
                return _2op(Op.vpsrlvd, a, count);
            }

            public __m256i _mm256_srlv_epi32(__m256i a, __m256i count)
            {
                return _2op(Op.vpsrlvd, a, count);
            }

            public __m128i _mm_srlv_epi64(__m128i a, __m128i count)
            {
                return _2op(Op.vpsrlvd, a, count);
            }

            public __m256i _mm256_srlv_epi64(__m256i a, __m256i count)
            {
                return _2op(Op.vpsrlvd, a, count);
            }

            public __m256i _mm256_stream_load_si256(scalarReg ptr, int offset = 0)
            {
                inst.Add(new Instr(Op.movntdqa | Op.L, nextvxmm, 0, 0, 0, 0, new MemoryArg(ptr.r, 0, 0, offset)));
                return new __m256i(nextvxmm++);
            }

            public __m256i _mm256_sub_epi16(__m256i a, __m256i b)
            {
                return _2op(Op.psubw, a, b);
            }

            public __m256i _mm256_sub_epi32(__m256i a, __m256i b)
            {
                return _2op(Op.psubd, a, b);
            }

            public __m256i _mm256_sub_epi64(__m256i a, __m256i b)
            {
                return _2op(Op.psubq, a, b);
            }

            public __m256i _mm256_sub_epi8(__m256i a, __m256i b)
            {
                return _2op(Op.psubb, a, b);
            }

            public __m256i _mm256_subs_epi16(__m256i a, __m256i b)
            {
                return _2op(Op.psubsw, a, b);
            }

            public __m256i _mm256_subs_epi8(__m256i a, __m256i b)
            {
                return _2op(Op.psubsb, a, b);
            }

            public __m256i _mm256_subs_epu16(__m256i a, __m256i b)
            {
                return _2op(Op.psubusw, a, b);
            }

            public __m256i _mm256_subs_epu8(__m256i a, __m256i b)
            {
                return _2op(Op.psubusb, a, b);
            }

            public __m256i _mm256_unpackhi_epi16(__m256i a, __m256i b)
            {
                return _2op(Op.punpckhwd, a, b);
            }

            public __m256i _mm256_unpackhi_epi32(__m256i a, __m256i b)
            {
                return _2op(Op.punpckhdq, a, b);
            }

            public __m256i _mm256_unpackhi_epi64(__m256i a, __m256i b)
            {
                return _2op(Op.punpckhqdq, a, b);
            }

            public __m256i _mm256_unpackhi_epi8(__m256i a, __m256i b)
            {
                return _2op(Op.punpckhbw, a, b);
            }

            public __m256i _mm256_unpacklo_epi16(__m256i a, __m256i b)
            {
                return _2op(Op.punpcklwd, a, b);
            }

            public __m256i _mm256_unpacklo_epi32(__m256i a, __m256i b)
            {
                return _2op(Op.punpckldq, a, b);
            }

            public __m256i _mm256_unpacklo_epi64(__m256i a, __m256i b)
            {
                return _2op(Op.punpcklqdq, a, b);
            }

            public __m256i _mm256_unpacklo_epi8(__m256i a, __m256i b)
            {
                return _2op(Op.punpcklbw, a, b);
            }

            public __m256i _mm256_xor_si256(__m256i a, __m256i b)
            {
                return _2op(Op.pxor, a, b);
            }

            #endregion
        }

        struct Instr
        {
            public Op op;
            public int R, A, B, C;
            public int Imm;
            public MemoryArg Mem;

            public Instr(Op op, int r, int a, int b, int c = 0, int imm = 0, MemoryArg mem = null)
            {
                this.op = op;
                R = r;
                A = a;
                B = b;
                C = c;
                Imm = imm;
                Mem = mem;
            }

            public void creates(List<int> c)
            {
                if (op == Op.ignore ||
                    op == Op.endrepeat ||
                    R == 0)
                    return;
                else
                    c.Add(R);
            }

            public void consumes(List<int> c)
            {
                if (op == Op.repeat_z || op == Op.repeat)
                {
                    c.Add(R);
                    c.Add(A);
                }
                else if (op == Op.endrepeat)
                    return;
                else
                {
                    if (A != 0)
                        c.Add(A);
                    if (B != 0)
                        c.Add(B);
                    if (C != 0)
                        c.Add(C);
                    if (Mem != null)
                    {
                        if (Mem.basereg != 0)
                            c.Add(Mem.basereg);
                        if (Mem.index != 0)
                            c.Add(Mem.index);
                    }
                    int vvvvMeaning = ((int)op >> 20) & 3;
                    if (vvvvMeaning == 3)
                        c.Add(R);
                }
            }
        }

        class MemoryArg
        {
            public int basereg, index, scale, offset;

            public MemoryArg(int basereg, int index, int scale, int offset)
            {
                this.basereg = basereg;
                this.index = index;
                this.scale = scale;
                this.offset = offset;
            }

            public override string ToString()
            {
                if (basereg == 0 && index == 0)
                    return string.Format("[{0}]", offset);
                else if (basereg == 0)
                    return string.Format("[{0} * {1} + {2}]", index, 1 << scale, offset);
                else
                    return string.Format("[{0} + {1} * {2} + {3}]", basereg, index, 1 << scale, offset);
            }
        }

        enum Op : uint
        {
            special = 1u << 31,
            L = 1u << 30,
            store = 1u << 29,
            merge_mem_R = 1u << 28,
            merge_mem_A = 1u << 27,
            merge_mem_B = 1u << 26,
            is4 = 1u << 25,
            vex = 1u << 24,
            nonvec = 1u << 23,
            alugrp = 1u << 24,
            samereg = 1u << 25,
            imm8 = 1u << 22,
            NDS = vex | (1u << 20),
            NDD = vex | (2u << 20),
            DDS = NDS, // DDS has the same encoding as NDS, the destructiveness is special-cased in the register allocator
            x66 = 0x0100,
            xF3 = 0x0200,
            xF2 = 0x0300,
            x0F = 0x1000,
            x0F38 = 0x2000,
            x0F3A = 0x3000,
            W = 0x10000,

            nop = special | 0x90,

            ldarg = special | 1,
            ldfarg = special | 2,
            repeat_z = special | 16,
            repeat = special | 17,
            endrepeat = special | 18,
            ignore = special | 0x20,
            assign = special | 0x21,
            vassign = special | 0x22,
            viassign = special | 0x23,
            vlassign = special | 0x24,
            vliassign = special | 0x25,

            add = nonvec | alugrp | 0x00,
            addq = add | W,
            sub = nonvec | samereg | alugrp | 0x28,
            subq = sub | W,
            xor = nonvec | samereg | alugrp | 0x30,
            xorq = xor | W,
            cmp = nonvec | samereg | alugrp | 0x38,
            cmpq = cmp | W,
            movzx_bd = nonvec | x0F | 0xB6,
            seto = nonvec | x0F | 0x90,
            setno = nonvec | x0F | 0x91,
            setc = nonvec | x0F | 0x92,
            setnc = nonvec | x0F | 0x93,
            setz = nonvec | x0F | 0x94,
            setnz = nonvec | x0F | 0x95,
            setna = nonvec | x0F | 0x96,
            seta = nonvec | x0F | 0x97,
            sets = nonvec | x0F | 0x98,
            setns = nonvec | x0F | 0x99,
            mov_r_r = nonvec | 0x8B,

            fmadd_ps = special | 0x1000,
            fmadd_pd = special | 0x1001,
            fmadd_ss = special | 0x1002,
            fmadd_sd = special | 0x1003,
            fmsub_ps = special | 0x1004,
            fmsub_pd = special | 0x1005,
            fmsub_ss = special | 0x1006,
            fmsub_sd = special | 0x1007,
            fnmadd_ps = special | 0x1008,
            fnmadd_pd = special | 0x1009,
            fnmadd_ss = special | 0x100A,
            fnmadd_sd = special | 0x100B,
            fnmsub_ps = special | 0x100C,
            fnmsub_pd = special | 0x100D,
            fnmsub_ss = special | 0x100E,
            fnmsub_sd = special | 0x100F,
            fmaddsub_ps = special | 0x1010,
            fmaddsub_pd = special | 0x1011,
            fmsubadd_ps = special | 0x1012,
            fmsubadd_pd = special | 0x1013,

            // format:
            // /r mm pp OP
            addps = NDS | x0F | 0x58 | merge_mem_B,
            addpd = NDS | x66 | x0F | 0x58 | merge_mem_B,
            addss = NDS | xF3 | x0F | 0x58 | merge_mem_B,
            addsd = NDS | xF2 | x0F | 0x58 | merge_mem_B,
            addsubpd = NDS | x66 | x0F | 0xD0 | merge_mem_B,
            addsubps = NDS | xF2 | x0F | 0xD0 | merge_mem_B,
            andps = NDS | x0F | 0x54 | merge_mem_B,
            andpd = NDS | x66 | x0F | 0x54 | merge_mem_B,
            andnps = NDS | x0F | 0x55 | merge_mem_B,
            andnpd = NDS | x66 | x0F | 0x55 | merge_mem_B,
            blendps = NDS | x66 | x0F3A | 0x0C | imm8 | merge_mem_B,
            blendpd = NDS | x66 | x0F3A | 0x0D | imm8 | merge_mem_B,
            blendvps = NDS | x66 | x0F3A | 0x4A | is4 | merge_mem_B,
            blendvpd = NDS | x66 | x0F3A | 0x4B | is4 | merge_mem_B,
            cmpps = NDS | x0F | 0xC2 | imm8 | merge_mem_B,
            cmppd = NDS | x66 | x0F | 0xC2 | imm8 | merge_mem_B,
            cmpss = NDS | xF3 | x0F | 0xC2 | imm8 | merge_mem_B,
            cmpsd = NDS | xF2 | x0F | 0xC2 | imm8 | merge_mem_B,
            comiss = vex | x0F | 0x2F | merge_mem_B,
            comisd = vex | x66 | x0F | 0x2F | merge_mem_B,
            cvtdq2ps = vex | x0F | 0x5B | merge_mem_A,
            cvtdq2pd = vex | xF3 | x0F | 0x5B | merge_mem_A,
            cvtpd2dq = vex | xF2 | x0F | 0xE6 | merge_mem_A,
            cvtpd2ps = vex | x66 | x0F | 0x5A | merge_mem_A,
            cvtps2dq = vex | x66 | x0F | 0x5B | merge_mem_A,
            cvtps2pd = vex | x0F | 0x5A | merge_mem_A,
            cvtsd2si = vex | xF2 | x0F | 0x2D | merge_mem_A,
            cvtsd2ss = NDS | xF2 | x0F | 0x5A | merge_mem_B,
            cvtsi2sd = NDS | xF2 | x0F | 0x2A | merge_mem_A,
            cvtsi2ss = NDS | xF3 | x0F | 0x2A | merge_mem_A,
            cvtss2sd = NDS | xF3 | x0F | 0x5A | merge_mem_B,
            cvtss2si = vex | xF3 | x0F | 0x2D | merge_mem_A,
            cvttpd2dq = vex | x66 | x0F | 0xE6 | merge_mem_A,
            cvttps2dq = vex | xF3 | x0F | 0x5B | merge_mem_A,
            cvttsd2si = vex | xF2 | x0F | 0x2C | merge_mem_A,
            cvttss2si = vex | xF3 | x0F | 0x2C | merge_mem_A,
            divps = NDS | x0F | 0x5E | merge_mem_B,
            divpd = NDS | x66 | x0F | 0x5E | merge_mem_B,
            divss = NDS | xF3 | x0F | 0x5E | merge_mem_B,
            divsd = NDS | xF2 | x0F | 0x5E | merge_mem_B,
            dpps = NDS | x66 | x0F3A | 0x40 | imm8 | merge_mem_B,
            dppd = NDS | x66 | x0F3A | 0x41 | imm8 | merge_mem_B,
            extractps = vex | x66 | x0F3A | 0x17 | imm8 | merge_mem_R,
            haddps = NDS | xF2 | x0F | 0x7C | merge_mem_B,
            haddpd = NDS | x66 | x0F | 0x7C | merge_mem_B,
            hsubps = NDS | xF2 | x0F | 0x7D | merge_mem_B,
            hsubpd = NDS | x66 | x0F | 0x7D | merge_mem_B,
            insertps = vex | x66 | x0F3A | 0x21 | imm8 | merge_mem_B,
            lddqu = vex | xF2 | x0F | 0xF0,
            maskmovdqu = vex | x66 | x0F | 0xF7 | store,
            maxps = NDS | x0F | 0x5F | merge_mem_B,
            maxpd = NDS | x66 | x0F | 0x5F | merge_mem_B,
            maxss = NDS | xF3 | x0F | 0x5F | merge_mem_B,
            maxsd = NDS | xF2 | x0F | 0x5F | merge_mem_B,
            minps = NDS | x0F | 0x5D | merge_mem_B,
            minpd = NDS | x66 | x0F | 0x5D | merge_mem_B,
            minss = NDS | xF3 | x0F | 0x5D | merge_mem_B,
            minsd = NDS | xF2 | x0F | 0x5D | merge_mem_B,
            movaps = vex | x0F | 0x28,
            movaps_r = vex | x0F | 0x29 | store,
            movapd = vex | x66 | x0F | 0x28,
            movapd_r = vex | x66 | x0F | 0x29 | store,
            movd = vex | x66 | x0F | 0x6E | merge_mem_A,
            movd_r = vex | x66 | x0F | 0x7E | store,
            movddup = vex | xF2 | x0F | 0x12 | merge_mem_A,
            movdqa = vex | x66 | x0F | 0x6F,
            movdqa_r = vex | x66 | x0F | 0x7F | store,
            movdqu = vex | xF3 | x0F | 0x6F,
            movdqu_r = vex | xF3 | x0F | 0x7F | store,
            movhlps = NDS | x0F | 0x12,
            movlhps = NDS | x0F | 0x16,
            movhps = NDS | x0F | 0x16,
            movhps_r = vex | x0F | 0x17 | store,
            movhpd = NDS | x66 | x0F | 0x16,
            movhpd_r = vex | x66 | x0F | 0x17 | store,
            movlps = NDS | x0F | 0x12,
            movlps_r = vex | x0F | 0x13 | store,
            movlpd = NDS | x66 | x0F | 0x12,
            movlpd_r = vex | x66 | x0F | 0x13 | store,
            movntdqa = vex | x66 | x0F38 | 0x2A,
            movntps = vex | x0F | 0x2B | store,
            movntpd = vex | x66 | x0F | 0x2B | store,
            movmskps = vex | x0F | 0x50,
            movmskpd = vex | x66 | x0F | 0x50,
            movsldup = vex | xF3 | x0F | 0x12,
            movshdup = vex | xF3 | x0F | 0x16,
            movsd = NDS | xF2 | x0F | 0x10,
            movsd_r = vex | xF2 | x0F | 0x11,
            movss = NDS | xF3 | x0F | 0x10,
            movss_r = vex | xF3 | x0F | 0x11,
            movups = vex | x0F | 0x10,
            movups_r = vex | x0F | 0x11 | store,
            movupd = vex | x66 | x0F | 0x10,
            movupd_r = vex | x66 | x0F | 0x11 | store,
            mpsadbw = NDS | x66 | x0F3A | 0x42 | merge_mem_B,
            mulps = NDS | x0F | 0x59 | merge_mem_B,
            mulpd = NDS | x66 | x0F | 0x59 | merge_mem_B,
            mulss = NDS | xF3 | x0F | 0x59 | merge_mem_B,
            mulsd = NDS | xF2 | x0F | 0x59 | merge_mem_B,
            orps = NDS | x0F | 0x56 | merge_mem_B,
            orpd = NDS | x66 | x0F | 0x56 | merge_mem_B,
            pabsb = vex | x66 | x0F38 | 0x1C | merge_mem_A,
            pabsw = vex | x66 | x0F38 | 0x1D | merge_mem_A,
            pabsd = vex | x66 | x0F38 | 0x1E | merge_mem_A,
            packsswb = NDS | x66 | x0F | 0x63 | merge_mem_B,
            packssdw = NDS | x66 | x0F | 0x6B | merge_mem_B,
            packusdw = NDS | x66 | x0F38 | 0x2B | merge_mem_B,
            packuswb = NDS | x66 | x0F | 0x67 | merge_mem_B,
            paddb = NDS | x66 | x0F | 0xFC | merge_mem_B,
            paddw = NDS | x66 | x0F | 0xFD | merge_mem_B,
            paddd = NDS | x66 | x0F | 0xFE | merge_mem_B,
            paddq = NDS | x66 | x0F | 0xD4 | merge_mem_B,
            paddsb = NDS | x66 | x0F | 0xEC | merge_mem_B,
            paddsw = NDS | x66 | x0F | 0xED | merge_mem_B,
            paddusb = NDS | x66 | x0F | 0xDC | merge_mem_B,
            paddusw = NDS | x66 | x0F | 0xDD | merge_mem_B,
            palignr = NDS | x66 | x0F3A | 0x0F | imm8 | merge_mem_B,
            pand = NDS | x66 | x0F | 0xDB | merge_mem_B,
            pandn = NDS | x66 | x0F | 0xDF | merge_mem_B,
            pavgb = NDS | x66 | x0F | 0xE0 | merge_mem_B,
            pavgw = NDS | x66 | x0F | 0xE3 | merge_mem_B,
            pblendvb = NDS | x66 | x0F3A | 0x4C | is4 | merge_mem_B,
            pblendw = NDS | x66 | x0F3A | 0x0E | imm8 | merge_mem_B,
            pmulqdq = NDS | x66 | x0F3A | 0x44 | imm8 | merge_mem_B,
            pcmpeqb = NDS | x66 | x0F | 0x74 | merge_mem_B,
            pcmpeqw = NDS | x66 | x0F | 0x75 | merge_mem_B,
            pcmpeqd = NDS | x66 | x0F | 0x76 | merge_mem_B,
            pcmpeqq = NDS | x66 | x0F38 | 0x29 | merge_mem_B,
            pcmpestri = vex | x66 | x0F3A | 0x61 | imm8 | merge_mem_B,
            pcmpestrm = vex | x66 | x0F3A | 0x60 | imm8 | merge_mem_B,
            pcmpgtb = NDS | x66 | x0F | 0x64 | merge_mem_B,
            pcmpgtw = NDS | x66 | x0F | 0x65 | merge_mem_B,
            pcmpgtd = NDS | x66 | x0F | 0x66 | merge_mem_B,
            pcmpgtq = NDS | x66 | x0F38 | 0x37 | merge_mem_B,
            pcmpistri = vex | x66 | x0F3A | 0x63 | imm8 | merge_mem_B,
            pcmpistrm = vex | x66 | x0F3A | 0x62 | imm8 | merge_mem_B,
            pextrb = vex | x66 | x0F3A | 0x14 | imm8 | merge_mem_R,
            pextrd = vex | x66 | x0F3A | 0x16 | imm8 | merge_mem_R,
            pextrq = vex | W | x66 | x0F3A | 0x16 | imm8 | merge_mem_R,
            pextrw = vex | x66 | x0F | 0xC5 | imm8,
            pextrw2 = vex | x66 | x0F3A | 0x15 | imm8 | merge_mem_B,
            phaddw = NDS | x66 | x0F38 | 0x01 | merge_mem_B,
            phaddd = NDS | x66 | x0F38 | 0x02 | merge_mem_B,
            phaddsw = NDS | x66 | x0F38 | 0x03 | merge_mem_B,
            phminposuw = vex | x66 | x0F38 | 0x41 | merge_mem_A,
            phsubw = NDS | x66 | x0F38 | 0x05 | merge_mem_B,
            phsubd = NDS | x66 | x0F38 | 0x06 | merge_mem_B,
            phsubsw = NDS | x66 | x0F38 | 0x07 | merge_mem_B,
            pinsrb = NDS | x66 | x0F3A | 0x20 | imm8 | merge_mem_B,
            pinsrd = NDS | x66 | x0F3A | 0x22 | imm8 | merge_mem_B,
            pinsrq = NDS | W | x66 | x0F3A | 0x22 | imm8 | merge_mem_B,
            pinsrw = NDS | x66 | x0F | 0xC4 | imm8 | merge_mem_B,
            pmaddubsw = NDS | x66 | x0F38 | 0x04 | merge_mem_B,
            pmaddwd = NDS | x66 | x0F | 0xF5 | merge_mem_B,
            pmaxsb = NDS | x66 | x0F38 | 0x3C | merge_mem_B,
            pmaxsd = NDS | x66 | x0F38 | 0x3D | merge_mem_B,
            pmaxsw = NDS | x66 | x0F | 0xEE | merge_mem_B,
            pmaxub = NDS | x66 | x0F | 0xDE | merge_mem_B,
            pmaxud = NDS | x66 | x0F38 | 0x3F | merge_mem_B,
            pmaxuw = NDS | x66 | x0F38 | 0x3E | merge_mem_B,
            pminsb = NDS | x66 | x0F38 | 0x38 | merge_mem_B,
            pminsd = NDS | x66 | x0F38 | 0x39 | merge_mem_B,
            pminsw = NDS | x66 | x0F | 0xEA | merge_mem_B,
            pminub = NDS | x66 | x0F | 0xDA | merge_mem_B,
            pminud = NDS | x66 | x0F38 | 0x3B | merge_mem_B,
            pminuw = NDS | x66 | x0F38 | 0x3A | merge_mem_B,
            pmovmskb = vex | x66 | x0F | 0xD7,
            pmovsxbw = vex | x66 | x0F38 | 0x20 | merge_mem_A,
            pmovsxbd = vex | x66 | x0F38 | 0x21 | merge_mem_A,
            pmovsxbq = vex | x66 | x0F38 | 0x22 | merge_mem_A,
            pmovsxwd = vex | x66 | x0F38 | 0x23 | merge_mem_A,
            pmovsxwq = vex | x66 | x0F38 | 0x24 | merge_mem_A,
            pmovsxdq = vex | x66 | x0F38 | 0x25 | merge_mem_A,
            pmovzxbw = vex | x66 | x0F38 | 0x30 | merge_mem_A,
            pmovzxbd = vex | x66 | x0F38 | 0x31 | merge_mem_A,
            pmovzxbq = vex | x66 | x0F38 | 0x32 | merge_mem_A,
            pmovzxwd = vex | x66 | x0F38 | 0x33 | merge_mem_A,
            pmovzxwq = vex | x66 | x0F38 | 0x34 | merge_mem_A,
            pmovzxdq = vex | x66 | x0F38 | 0x35 | merge_mem_A,
            pmuldq = NDS | x66 | x0F38 | 0x28 | merge_mem_B,
            pmulhrsw = NDS | x66 | x0F38 | 0x0B | merge_mem_B,
            pmulhuw = NDS | x66 | x0F | 0xE4 | merge_mem_B,
            pmulhw = NDS | x66 | x0F | 0xE5 | merge_mem_B,
            pmulld = NDS | x66 | x0F38 | 0x40 | merge_mem_B,
            pmullw = NDS | x66 | x0F | 0xD5 | merge_mem_B,
            pmuludq = NDS | x66 | x0F | 0xF4 | merge_mem_B,
            por = NDS | x66 | x0F | 0xEB | merge_mem_B,
            psadbw = NDS | x66 | x0F | 0xF6 | merge_mem_B,
            pshufb = NDS | x66 | x0F38 | 0x00 | merge_mem_B,
            pshufd = vex | x66 | x0F | 0x70 | imm8 | merge_mem_A,
            pshufhw = vex | xF3 | x0F | 0x70 | imm8 | merge_mem_A,
            pshuflw = vex | xF2 | x0F | 0x70 | imm8 | merge_mem_A,
            psignb = NDS | x66 | x0F38 | 0x08 | merge_mem_B,
            psignw = NDS | x66 | x0F38 | 0x09 | merge_mem_B,
            psignd = NDS | x66 | x0F38 | 0x0A | merge_mem_B,
            pslldq = NDD | x66 | x0F | 0x73 | (7 << 17) | imm8,
            psllw = NDS | x66 | x0F | 0xF1 | merge_mem_A,
            psllwi = NDD | x66 | x0F | 0x71 | (6 << 17) | imm8,
            pslld = NDS | x66 | x0F | 0xF2 | merge_mem_A,
            pslldi = NDD | x66 | x0F | 0x72 | (6 << 17) | imm8,
            psllq = NDS | x66 | x0F | 0xF3 | merge_mem_A,
            psllqi = NDD | x66 | x0F | 0x73 | (6 << 17) | imm8,
            psraw = NDS | x66 | x0F | 0xE1 | merge_mem_A,
            psrawi = NDD | x66 | x0F | 0x71 | (4 << 17) | imm8,
            psrad = NDS | x66 | x0F | 0xE2 | merge_mem_A,
            psradi = NDD | x66 | x0F | 0x72 | (4 << 17) | imm8,
            psrldq = NDD | x66 | x0F | 0x73 | (3 << 17) | imm8,
            psrlw = NDS | x66 | x0F | 0xD1 | merge_mem_A,
            psrlwi = NDD | x66 | x0F | 0x71 | (2 << 17) | imm8,
            psrld = NDS | x66 | x0F | 0xD2 | merge_mem_A,
            psrldi = NDD | x66 | x0F | 0x72 | (2 << 17) | imm8,
            psrlq = NDS | x66 | x0F | 0xD3 | merge_mem_A,
            psrlqi = NDD | x66 | x0F | 0x73 | (2 << 17) | imm8,
            psubb = NDS | x66 | x0F | 0xF8 | merge_mem_B,
            psubw = NDS | x66 | x0F | 0xF9 | merge_mem_B,
            psubd = NDS | x66 | x0F | 0xFA | merge_mem_B,
            psubq = NDS | x66 | x0F | 0xFB | merge_mem_B,
            psubsb = NDS | x66 | x0F | 0xE8 | merge_mem_B,
            psubsw = NDS | x66 | x0F | 0xE9 | merge_mem_B,
            psubusb = NDS | x66 | x0F | 0xD8 | merge_mem_B,
            psubusw = NDS | x66 | x0F | 0xD9 | merge_mem_B,
            ptest = vex | x66 | x0F38 | 0x17, // could merge_mem_B but codegen is annoying
            punpckhbw = NDS | x66 | x0F | 0x68 | merge_mem_B,
            punpckhwd = NDS | x66 | x0F | 0x69 | merge_mem_B,
            punpckhdq = NDS | x66 | x0F | 0x6A | merge_mem_B,
            punpckhqdq = NDS | x66 | x0F | 0x6D | merge_mem_B,
            punpcklbw = NDS | x66 | x0F | 0x60 | merge_mem_B,
            punpcklwd = NDS | x66 | x0F | 0x61 | merge_mem_B,
            punpckldq = NDS | x66 | x0F | 0x62 | merge_mem_B,
            punpcklqdq = NDS | x66 | x0F | 0x6C | merge_mem_B,
            pxor = NDS | x66 | x0F | 0xEF | merge_mem_B,
            rcpps = vex | x0F | 0x53 | merge_mem_A,
            rcpss = vex | xF3 | x0F | 0x53 | merge_mem_B,
            roundps = vex | x66 | x0F3A | 0x08 | imm8 | merge_mem_A,
            roundpd = vex | x66 | x0F3A | 0x09 | imm8 | merge_mem_A,
            roundss = vex | x66 | x0F3A | 0x0A | imm8 | merge_mem_B,
            roundsd = vex | x66 | x0F3A | 0x0B | imm8 | merge_mem_B,
            rsqrtps = vex | x0F | 0x52 | merge_mem_A,
            rsqrtss = vex | xF3 | x0F | 0x52 | merge_mem_B,
            shufps = NDS | x0F | 0xC6 | imm8 | merge_mem_B,
            shufpd = NDS | x66 | x0F | 0xC6 | imm8 | merge_mem_B,
            sqrtps = vex | x0F | 0x51 | merge_mem_A,
            sqrtpd = vex | x66 | x0F | 0x51 | merge_mem_A,
            sqrtss = vex | xF3 | x0F | 0x51 | merge_mem_B,
            sqrtsd = vex | xF2 | x0F | 0x51 | merge_mem_B,
            stmxcsr = vex | x0F | 0xAE,
            subps = NDS | x0F | 0x5C | merge_mem_B,
            subpd = NDS | x66 | x0F | 0x5C | merge_mem_B,
            subss = NDS | xF3 | x0F | 0x5C | merge_mem_B,
            subsd = NDS | xF2 | x0F | 0x5C | merge_mem_B,
            ucomiss = vex | x0F | 0x2E | merge_mem_B,
            ucomisd = vex | x66 | x0F | 0x2E | merge_mem_B,
            unpckhps = NDS | x0F | 0x15 | merge_mem_B,
            unpckhpd = NDS | x66 | x0F | 0x15 | merge_mem_B,
            unpcklps = NDS | x0F | 0x14 | merge_mem_B,
            unpcklpd = NDS | x66 | x0F | 0x14 | merge_mem_B,
            vbroadcastss = vex | x66 | x0F38 | 0x18 | merge_mem_B,
            vbroadcastsd = vex | x66 | x0F38 | 0x19 | merge_mem_B,
            vbroadcastf128 = vex | x66 | x0F38 | 0x1A,
            vcvtph2ps = vex | x66 | x0F38 | 0x13 | merge_mem_B,
            vcvtps2ph = vex | x66 | x0F3A | 0x1D | imm8 | merge_mem_R,
            vextractf128 = vex | x66 | x0F3A | 0x19 | imm8 | merge_mem_R,
            vextracti128 = vex | x66 | x0F3A | 0x39 | imm8 | merge_mem_R,
            vfmadd132pd = DDS | W | x66 | x0F38 | 0x98 | merge_mem_B,
            vfmadd213pd = DDS | W | x66 | x0F38 | 0xA8 | merge_mem_B,
            vfmadd231pd = DDS | W | x66 | x0F38 | 0xB8 | merge_mem_B,
            vfmadd132ps = DDS | x66 | x0F38 | 0x98 | merge_mem_B,
            vfmadd213ps = DDS | x66 | x0F38 | 0xA8 | merge_mem_B,
            vfmadd231ps = DDS | x66 | x0F38 | 0xB8 | merge_mem_B,
            vfmadd132sd = DDS | W | x66 | x0F38 | 0x99 | merge_mem_B,
            vfmadd213sd = DDS | W | x66 | x0F38 | 0xA9 | merge_mem_B,
            vfmadd231sd = DDS | W | x66 | x0F38 | 0xB9 | merge_mem_B,
            vfmadd132ss = DDS | x66 | x0F38 | 0x99 | merge_mem_B,
            vfmadd213ss = DDS | x66 | x0F38 | 0xA9 | merge_mem_B,
            vfmadd231ss = DDS | x66 | x0F38 | 0xB9 | merge_mem_B,
            vfmaddsub132pd = DDS | W | x66 | x0F38 | 0x96 | merge_mem_B,
            vfmaddsub213pd = DDS | W | x66 | x0F38 | 0xA6 | merge_mem_B,
            vfmaddsub231pd = DDS | W | x66 | x0F38 | 0xB6 | merge_mem_B,
            vfmaddsub132ps = DDS | x66 | x0F38 | 0x96 | merge_mem_B,
            vfmaddsub213ps = DDS | x66 | x0F38 | 0xA6 | merge_mem_B,
            vfmaddsub231ps = DDS | x66 | x0F38 | 0xB6 | merge_mem_B,
            vfmsubadd132pd = DDS | W | x66 | x0F38 | 0x97 | merge_mem_B,
            vfmsubadd213pd = DDS | W | x66 | x0F38 | 0xA7 | merge_mem_B,
            vfmsubadd231pd = DDS | W | x66 | x0F38 | 0xB7 | merge_mem_B,
            vfmsubadd132ps = DDS | x66 | x0F38 | 0x97 | merge_mem_B,
            vfmsubadd213ps = DDS | x66 | x0F38 | 0xA7 | merge_mem_B,
            vfmsubadd231ps = DDS | x66 | x0F38 | 0xB7 | merge_mem_B,
            vfmsub132pd = DDS | W | x66 | x0F38 | 0x9A | merge_mem_B,
            vfmsub213pd = DDS | W | x66 | x0F38 | 0xAA | merge_mem_B,
            vfmsub231pd = DDS | W | x66 | x0F38 | 0xBA | merge_mem_B,
            vfmsub132ps = DDS | x66 | x0F38 | 0x9A | merge_mem_B,
            vfmsub213ps = DDS | x66 | x0F38 | 0xAA | merge_mem_B,
            vfmsub231ps = DDS | x66 | x0F38 | 0xBA | merge_mem_B,
            vfnmadd132pd = DDS | W | x66 | x0F38 | 0x9C | merge_mem_B,
            vfnmadd213pd = DDS | W | x66 | x0F38 | 0xAC | merge_mem_B,
            vfnmadd231pd = DDS | W | x66 | x0F38 | 0xBC | merge_mem_B,
            vfnmadd132ps = DDS | x66 | x0F38 | 0x9C | merge_mem_B,
            vfnmadd213ps = DDS | x66 | x0F38 | 0xAC | merge_mem_B,
            vfnmadd231ps = DDS | x66 | x0F38 | 0xBC | merge_mem_B,
            vfnmsub132pd = DDS | W | x66 | x0F38 | 0x9E | merge_mem_B,
            vfnmsub213pd = DDS | W | x66 | x0F38 | 0xAE | merge_mem_B,
            vfnmsub231pd = DDS | W | x66 | x0F38 | 0xBE | merge_mem_B,
            vfnmsub132ps = DDS | x66 | x0F38 | 0x9E | merge_mem_B,
            vfnmsub213ps = DDS | x66 | x0F38 | 0xAE | merge_mem_B,
            vfnmsub231ps = DDS | x66 | x0F38 | 0xBE | merge_mem_B,
            vgatherdps = DDS | x66 | x0F38 | 0x92,
            vgatherqps = DDS | x66 | x0F38 | 0x93,
            vgatherdpd = DDS | W | x66 | x0F38 | 0x92,
            vgatherqpd = DDS | W | x66 | x0F38 | 0x93,
            vgatherdd = DDS | x66 | x0F38 | 0x90,
            vgatherqd = DDS | x66 | x0F38 | 0x91,
            vgatherdq = DDS | W | x66 | x0F38 | 0x90,
            vgatherqq = DDS | W | x66 | x0F38 | 0x91,
            vinsertf128 = NDS | x66 | x0F3A | 0x18 | imm8 | merge_mem_B,
            vinserti128 = NDS | x66 | x0F3A | 0x38 | imm8 | merge_mem_B,
            vmaskmovps = NDS | x66 | x0F38 | 0x2C,
            vmaskmovpd = NDS | x66 | x0F38 | 0x2D,
            vmaskmovps_r = NDS | x66 | x0F38 | 0x2E,
            vmaskmovpd_r = NDS | x66 | x0F38 | 0x2F,
            vpblendd = NDS | x66 | x0F3A | 0x02 | imm8 | merge_mem_B,
            vpbroadcastb = vex | x66 | x0F38 | 0x78 | merge_mem_A,
            vpbroadcastw = vex | x66 | x0F38 | 0x79 | merge_mem_A,
            vpbroadcastd = vex | x66 | x0F38 | 0x58 | merge_mem_A,
            vpbroadcastq = vex | x66 | x0F38 | 0x59 | merge_mem_A,
            vpbroadcasti128 = vex | x66 | x0F38 | 0x5A,
            vpermd = NDS | x66 | x0F38 | 0x36 | merge_mem_B,
            vpermpd = vex | W | x66 | x0F3A | 0x01 | imm8 | merge_mem_A,
            vpermps = NDS | x66 | x0F38 | 0x16 | merge_mem_B,
            vpermq = vex | W | x66 | x0F3A | 0x00 | imm8 | merge_mem_A,
            vperm2i128 = NDS | x66 | x0F3A | 0x46 | imm8 | merge_mem_B,
            vpermilpd = NDS | x66 | x0F38 | 0x0D | merge_mem_B,
            vpermilpdi = vex | x66 | x0F3A | 0x05 | imm8 | merge_mem_A,
            vpermilps = NDS | x66 | x0F38 | 0x0C | merge_mem_B,
            vpermilpsi = vex | x66 | x0F3A | 0x04 | imm8 | merge_mem_A,
            vperm2f128 = NDS | x66 | x0F3A | 0x06 | imm8 | merge_mem_B,
            vpmaskmovd = NDS | x66 | x0F38 | 0x8C,
            vpmaskmovq = NDS | W | x66 | x0F38 | 0x8C,
            vpmaskmovd_r = NDS | x66 | x0F38 | 0x8E,
            vpmaskmovq_r = NDS | W | x66 | x0F38 | 0x8E,
            vpsllvd = NDS | x66 | x0F38 | 0x47 | merge_mem_B,
            vpsllvq = NDS | W | x66 | x0F38 | 0x47 | merge_mem_B,
            vpsravd = NDS | x66 | x0F38 | 0x46 | merge_mem_B,
            vpsrlvd = NDS | x66 | x0F38 | 0x45 | merge_mem_B,
            vpsrlvq = NDS | W | x66 | x0F38 | 0x45 | merge_mem_B,
            vptestps = vex | x66 | x0F38 | 0x0E | merge_mem_B,
            vptestpd = vex | x66 | x0F38 | 0x0F | merge_mem_B,
            vzeroall = vex | L | x0F | 0x77,
            vzeroupper = vex | x0F | 0x77,
            xorps = NDS | x0F | 0x57 | merge_mem_B,
            xorpd = NDS | x66 | x0F | 0x57 | merge_mem_B,
        }
    }

    interface IBuilder
    {
        // special
        scalarReg arg(int index);
        __m128 farg(int index);
        void assign(__m128 dst, __m128 src);
        void assign(__m256 dst, __m256 src);
        void assign(scalarReg dst, scalarReg src);
        void ignore(__m128 r);
        void ignore(__m256 r);
        void ignore(scalarReg r);
#if !NOCAST
        void ignore(__m128d r);
        void ignore(__m128i r);
        void ignore(__m256d r);
        void ignore(__m256i r);
        void assign(__m128i dst, __m128i src);
        void assign(__m128d dst, __m128d src);
        void assign(__m256i dst, __m256i src);
        void assign(__m256d dst, __m256d src);
#endif
        
        scalarReg repeat_z(scalarReg count, int step);
        scalarReg repeat(scalarReg count, int step);
        void endrepeat();

        scalarReg add(scalarReg a, scalarReg b);
        scalarReg add(scalarReg a, int imm);
        scalarReg sub(scalarReg a, scalarReg b);
        scalarReg sub(scalarReg a, int imm);

        // SSE
        __m128 _mm_add_ps(__m128 a, __m128 b);
        __m128 _mm_add_ss(__m128 a, __m128 b);
        __m128 _mm_and_ps(__m128 a, __m128 b);
        __m128 _mm_andnot_ps(__m128 a, __m128 b);
        __m128 _mm_cmpeq_ps(__m128 a, __m128 b);
        __m128 _mm_cmpeq_ss(__m128 a, __m128 b);
        __m128 _mm_cmpge_ps(__m128 a, __m128 b);
        __m128 _mm_cmpge_ss(__m128 a, __m128 b);
        __m128 _mm_cmpgt_ps(__m128 a, __m128 b);
        __m128 _mm_cmpgt_ss(__m128 a, __m128 b);
        __m128 _mm_cmple_ps(__m128 a, __m128 b);
        __m128 _mm_cmple_ss(__m128 a, __m128 b);
        __m128 _mm_cmplt_ps(__m128 a, __m128 b);
        __m128 _mm_cmplt_ss(__m128 a, __m128 b);
        __m128 _mm_cmpneq_ps(__m128 a, __m128 b);
        __m128 _mm_cmpneq_ss(__m128 a, __m128 b);
        __m128 _mm_cmpnge_ps(__m128 a, __m128 b);
        __m128 _mm_cmpnge_ss(__m128 a, __m128 b);
        __m128 _mm_cmpngt_ps(__m128 a, __m128 b);
        __m128 _mm_cmpngt_ss(__m128 a, __m128 b);
        __m128 _mm_cmpnle_ps(__m128 a, __m128 b);
        __m128 _mm_cmpnle_ss(__m128 a, __m128 b);
        __m128 _mm_cmpnlt_ps(__m128 a, __m128 b);
        __m128 _mm_cmpnlt_ss(__m128 a, __m128 b);
        __m128 _mm_cmpord_ps(__m128 a, __m128 b);
        __m128 _mm_cmpord_ss(__m128 a, __m128 b);
        __m128 _mm_cmpnord_ps(__m128 a, __m128 b);
        __m128 _mm_cmpnord_ss(__m128 a, __m128 b);
        //scalarReg _mm_comieq_ss(__m128 a, __m128 b);
        //scalarReg _mm_comige_ss(__m128 a, __m128 b);
        //scalarReg _mm_comigt_ss(__m128 a, __m128 b);
        //scalarReg _mm_comile_ss(__m128 a, __m128 b);
        //scalarReg _mm_comilt_ss(__m128 a, __m128 b);
        //scalarReg _mm_comineq_ss(__m128 a, __m128 b);
        __m128 _mm_cvtsi32_ss(__m128 a, scalarReg b);
        __m128 _mm_cvtsi64_ss(__m128 a, scalarReg b);
        scalarReg _mm_cvtss_si32(__m128 a);
        scalarReg _mm_cvtss_si64(__m128 a);
        scalarReg _mm_cvttss_si32(__m128 a);
        scalarReg _mm_cvttss_si64(__m128 a);
        __m128 _mm_div_ps(__m128 a, __m128 b);
        __m128 _mm_div_ss(__m128 a, __m128 b);
        scalarReg _mm_getcsr();
        __m128 _mm_load_ps(scalarReg ptr, int offset = 0);
        __m128 _mm_load_ps(scalarReg ptr, scalarReg index, int scale = 1, int offset = 0);
        __m128 _mm_load_ss(scalarReg ptr, int offset);
        __m128 _mm_load_ss(scalarReg ptr, scalarReg index, int scale = 1, int offset = 0);
        __m128 _mm_load1_ps(scalarReg ptr, int offset);
        __m128 _mm_loadu_ps(scalarReg ptr, int offset = 0);
        __m128 _mm_loadu_ps(scalarReg ptr, scalarReg index, int scale = 1, int offset = 0);
        __m128 _mm_max_ps(__m128 a, __m128 b);
        __m128 _mm_max_ss(__m128 a, __m128 b);
        __m128 _mm_min_ps(__m128 a, __m128 b);
        __m128 _mm_min_ss(__m128 a, __m128 b);
        __m128 _mm_mov_ss(__m128 a, __m128 b);
        __m128 _mm_movehl_ps(__m128 a, __m128 b);
        __m128 _mm_movelh_ss(__m128 a, __m128 b);
        scalarReg _mm_movemask_ps(__m128 a);
        __m128 _mm_mul_ps(__m128 a, __m128 b);
        __m128 _mm_mul_ss(__m128 a, __m128 b);
        __m128 _mm_or_ps(__m128 a, __m128 b);
        __m128 _mm_rcp_ps(__m128 a);
        __m128 _mm_rcp_ss(__m128 a);
        __m128 _mm_rsqrt_ps(__m128 a);
        __m128 _mm_rsqrt_ss(__m128 a);
        __m128 _mm_set_ps(float e3, float e2, float e1, float e0);
        __m128 _mm_setr_ps(float e0, float e1, float e2, float e3);
        __m128 _mm_set1_ps(float a);
        __m128 _mm_set_ss(float a);
        __m128 _mm_setzero_ps();
        __m128 _mm_shuffle_ps(__m128 a, __m128 b, int shufmask);
        __m128 _mm_sqrt_ps(__m128 a);
        __m128 _mm_sqrt_ss(__m128 a);
        __m128 _mm_sub_ps(__m128 a, __m128 b);
        __m128 _mm_sub_ss(__m128 a, __m128 b);
        void _mm_store_ps(scalarReg ptr, __m128 a, int offset = 0);
        void _mm_store_ps(scalarReg ptr, __m128 a, scalarReg index, int scale, int offset);
        void _mm_storeu_ps(scalarReg ptr, __m128 a, int offset = 0);
        void _mm_storeu_ps(scalarReg ptr, __m128 a, scalarReg index, int scale, int offset);
        void _mm_store_ss(scalarReg ptr, __m128 a, int offset = 0);
        void _mm_store_ss(scalarReg ptr, __m128 a, scalarReg index, int scale, int offset);
        void _mm_stream_ps(scalarReg ptr, __m128 a, int offset = 0);
        //scalarReg _mm_ucomieq_ss(__m128 a, __m128 b);
        //scalarReg _mm_ucomige_ss(__m128 a, __m128 b);
        //scalarReg _mm_ucomigt_ss(__m128 a, __m128 b);
        //scalarReg _mm_ucomile_ss(__m128 a, __m128 b);
        //scalarReg _mm_ucomilt_ss(__m128 a, __m128 b);
        //scalarReg _mm_ucomineq_ss(__m128 a, __m128 b);
        //__m128 _mm_undefined_ps();
        __m128 _mm_unpackhi_ps(__m128 a, __m128 b);
        __m128 _mm_unpacklo_ps(__m128 a, __m128 b);
        __m128 _mm_xor_ps(__m128 a, __m128 b);

        // SSE2
        __m128i _mm_add_epi16(__m128i a, __m128i b);
        __m128i _mm_add_epi32(__m128i a, __m128i b);
        __m128i _mm_add_epi64(__m128i a, __m128i b);
        __m128i _mm_add_epi8(__m128i a, __m128i b);
        __m128d _mm_add_pd(__m128d a, __m128d b);
        __m128d _mm_add_sd(__m128d a, __m128d b);
        __m128i _mm_adds_epi16(__m128i a, __m128i b);
        __m128i _mm_adds_epi8(__m128i a, __m128i b);
        __m128i _mm_adds_epu16(__m128i a, __m128i b);
        __m128i _mm_adds_epu8(__m128i a, __m128i b);
        __m128d _mm_and_pd(__m128d a, __m128d b);
        __m128i _mm_and_si128(__m128i a, __m128i b);
        __m128d _mm_andnot_pd(__m128d a, __m128d b);
        __m128i _mm_andnot_si128(__m128i a, __m128i b);
        __m128i _mm_avg_epu16(__m128i a, __m128i b);
        __m128i _mm_avg_epu8(__m128i a, __m128i b);
        __m128i _mm_bslli_si128(__m128i a, int imm);
        __m128i _mm_bsrli_si128(__m128i a, int imm);
        __m128 _mm_castpd_ps(__m128d a);
        __m128i _mm_castpd_si128(__m128d a);
        __m128d _mm_castps_pd(__m128 a);
        __m128i _mm_castps_si128(__m128 a);
        __m128d _mm_castsi128_pd(__m128i a);
        __m128 _mm_castsi128_ps(__m128i a);
        //void _mm_clflush(scalarReg ptr);
        __m128i _mm_cmpeq_epi16(__m128i a, __m128i b);
        __m128i _mm_cmpeq_epi32(__m128i a, __m128i b);
        __m128i _mm_cmpeq_epi8(__m128i a, __m128i b);
        __m128i _mm_cmpgt_epi16(__m128i a, __m128i b);
        __m128i _mm_cmpgt_epi32(__m128i a, __m128i b);
        __m128i _mm_cmpgt_epi8(__m128i a, __m128i b);
        __m128i _mm_cmplt_epi16(__m128i a, __m128i b);
        __m128i _mm_cmplt_epi32(__m128i a, __m128i b);
        __m128i _mm_cmplt_epi8(__m128i a, __m128i b);
        __m128d _mm_cmp_pd(__m128d a, __m128d b, int imm);
        __m128d _mm_cmp_sd(__m128d a, __m128d b, int imm);
        __m128d _mm_cmpeq_pd(__m128d a, __m128d b);
        __m128d _mm_cmpeq_sd(__m128d a, __m128d b);
        __m128d _mm_cmpge_pd(__m128d a, __m128d b);
        __m128d _mm_cmpge_sd(__m128d a, __m128d b);
        __m128d _mm_cmpgt_pd(__m128d a, __m128d b);
        __m128d _mm_cmpgt_sd(__m128d a, __m128d b);
        __m128d _mm_cmple_pd(__m128d a, __m128d b);
        __m128d _mm_cmple_sd(__m128d a, __m128d b);
        __m128d _mm_cmplt_pd(__m128d a, __m128d b);
        __m128d _mm_cmplt_sd(__m128d a, __m128d b);
        __m128d _mm_cmpneq_pd(__m128d a, __m128d b);
        __m128d _mm_cmpneq_sd(__m128d a, __m128d b);
        __m128d _mm_cmpnge_pd(__m128d a, __m128d b);
        __m128d _mm_cmpnge_sd(__m128d a, __m128d b);
        __m128d _mm_cmpngt_pd(__m128d a, __m128d b);
        __m128d _mm_cmpngt_sd(__m128d a, __m128d b);
        __m128d _mm_cmpnle_pd(__m128d a, __m128d b);
        __m128d _mm_cmpnle_sd(__m128d a, __m128d b);
        __m128d _mm_cmpnlt_pd(__m128d a, __m128d b);
        __m128d _mm_cmpnlt_sd(__m128d a, __m128d b);
        __m128d _mm_cmpord_pd(__m128d a, __m128d b);
        __m128d _mm_cmpord_sd(__m128d a, __m128d b);
        __m128d _mm_cmpnord_pd(__m128d a, __m128d b);
        __m128d _mm_cmpnord_sd(__m128d a, __m128d b);
        //scalarReg _mm_comieq_sd(__m128d a, __m128d b);
        //scalarReg _mm_comige_sd(__m128d a, __m128d b);
        //scalarReg _mm_comigt_sd(__m128d a, __m128d b);
        //scalarReg _mm_comile_sd(__m128d a, __m128d b);
        //scalarReg _mm_comineq_sd(__m128d a, __m128d b);
        __m128d _mm_cvtepi32_pd(__m128i a);
        __m128 _mm_cvtepi32_ps(__m128i a);
        __m128i _mm_cvtpd_epi32(__m128d a);
        __m128 _mm_cvtpd_ps(__m128d a);
        __m128i _mm_cvtps_epi32(__m128 a);
        __m128d _mm_cvtps_pd(__m128 a);
        scalarReg _mm_cvtsd_si32(__m128d a);
        scalarReg _mm_cvtsd_si64(__m128d a);
        __m128 _mm_cvtsd_ss(__m128 a, __m128d b);
        scalarReg _mm_cvtsi128_si32(__m128i a);
        scalarReg _mm_cvtsi128_si64(__m128i a);
        __m128d _mm_cvtsi32_sd(__m128d a, scalarReg b);
        __m128i _mm_cvtsi32_si128(scalarReg b);
        __m128d _mm_cvtsi64_sd(__m128d a, scalarReg b);
        __m128i _mm_cvtsi64_si128(scalarReg b);
        __m128d _mm_cvtss_sd(__m128d a, __m128 b);
        __m128i _mm_cvttpd_epi32(__m128d a);
        __m128i _mm_cvttps_epi32(__m128 a);
        scalarReg _mm_cvttsd_si32(__m128d a);
        scalarReg _mm_cvttsd_si64(__m128d a);
        __m128d _mm_div_pd(__m128d a, __m128d b);
        __m128d _mm_div_sd(__m128d a, __m128d b);
        scalarReg _mm_extract_epi16(__m128i a, int imm);
        __m128i _mm_insert_epi16(__m128i a, scalarReg v, int imm);
        //void _mm_lfence();
        __m128i _mm_load_si128(scalarReg ptr, int offset = 0);
        __m128i _mm_load_si128(scalarReg ptr, scalarReg index, int scale = 1, int offset = 0);
        __m128i _mm_loadu_si128(scalarReg ptr, int offset = 0);
        __m128i _mm_loadu_si128(scalarReg ptr, scalarReg index, int scale = 1, int offset = 0);
        __m128d _mm_load_pd(scalarReg ptr, int offset = 0);
        __m128d _mm_load_pd(scalarReg ptr, scalarReg index, int scale = 1, int offset = 0);
        __m128d _mm_load_sd(scalarReg ptr, int offset);
        __m128d _mm_load_sd(scalarReg ptr, scalarReg index, int scale = 1, int offset = 0);
        __m128d _mm_load1_pd(scalarReg ptr, int offset);
        __m128d _mm_loadu_pd(scalarReg ptr, int offset = 0);
        __m128d _mm_loadu_pd(scalarReg ptr, scalarReg index, int scale = 1, int offset = 0);
        __m128i _mm_madd_epi16(__m128i a, __m128i b);
        //void _mm_maskmoveu_si128(__m128i a, __m128i mask, scalarReg ptr);
        __m128i _mm_max_epi16(__m128i a, __m128i b);
        __m128i _mm_max_epu8(__m128i a, __m128i b);
        __m128d _mm_max_pd(__m128d a, __m128d b);
        __m128d _mm_max_sd(__m128d a, __m128d b);
        //void _mm_mfence();
        __m128i _mm_min_epi16(__m128i a, __m128i b);
        __m128i _mm_min_epu8(__m128i a, __m128i b);
        __m128d _mm_min_pd(__m128d a, __m128d b);
        __m128d _mm_min_sd(__m128d a, __m128d b);
        __m128i _mm_mov_epi64(__m128i a);
        __m128d _mm_mov_sd(__m128d a, __m128d b);
        scalarReg _mm_movemask_epi8(__m128i a);
        scalarReg _mm_movemask_pd(__m128d a);
        __m128i _mm_mul_epu32(__m128i a, __m128i b);
        __m128d _mm_mul_pd(__m128d a, __m128d b);
        __m128d _mm_mul_sd(__m128d a, __m128d b);
        __m128i _mm_mulhi_epi16(__m128i a, __m128i b);
        __m128i _mm_mulhi_epu16(__m128i a, __m128i b);
        __m128i _mm_mullo_epi16(__m128i a, __m128i b);
        __m128d _mm_or_pd(__m128d a, __m128d b);
        __m128i _mm_or_si128(__m128i a, __m128i b);
        __m128i _mm_packs_epi16(__m128i a, __m128i b);
        __m128i _mm_packs_epi32(__m128i a, __m128i b);
        __m128i _mm_packus_epi16(__m128i a, __m128i b);
        // _mm_pause()
        __m128i _mm_sad_epu8(__m128i a, __m128i b);
        __m128i _mm_set_epi16(short e7, short e6, short e5, short e4, short e3, short e2, short e1, short e0);
        __m128i _mm_set_epi32(int e3, int e2, int e1, int e0);
        __m128i _mm_set_epi8(byte e15, byte e14, byte e13, byte e12,
            byte e11, byte e10, byte e9, byte e8,
            byte e7, byte e6, byte e5, byte e4,
            byte e3, byte e2, byte e1, byte e0);
        __m128d _mm_set_pd(double e1, double e0);
        __m128d _mm_set_sd(double a);
        __m128i _mm_set1_epi16(short a);
        __m128i _mm_set1_epi32(int a);
        __m128i _mm_set1_epi64x(long a);
        __m128i _mm_set1_epi8(byte a);
        __m128d _mm_set1_pd(double a);
        __m128i _mm_setr_epi16(short e7, short e6, short e5, short e4, short e3, short e2, short e1, short e0);
        __m128i _mm_setr_epi32(int e3, int e2, int e1, int e0);
        __m128i _mm_setr_epi8(byte e15, byte e14, byte e13, byte e12,
            byte e11, byte e10, byte e9, byte e8,
            byte e7, byte e6, byte e5, byte e4,
            byte e3, byte e2, byte e1, byte e0);
        __m128d _mm_setr_pd(double e1, double e0);
        __m128d _mm_setzero_pd();
        __m128i _mm_setzero_si128();
        __m128i _mm_shuffle_epi32(__m128i a, int imm);
        __m128d _mm_shuffle_pd(__m128d a, __m128d b, int imm);
        __m128i _mm_shufflehi_epi16(__m128i a, int imm);
        __m128i _mm_shufflelo_epi16(__m128i a, int imm);
        __m128i _mm_sll_epi16(__m128i a, __m128i count);
        __m128i _mm_sll_epi32(__m128i a, __m128i count);
        __m128i _mm_sll_epi64(__m128i a, __m128i count);
        __m128i _mm_slli_epi16(__m128i a, int imm);
        __m128i _mm_slli_epi32(__m128i a, int imm);
        __m128i _mm_slli_epi64(__m128i a, int imm);
        __m128i _mm_slli_si128(__m128i a, int imm);
        __m128d _mm_sqrt_pd(__m128d a, __m128d b);
        __m128d _mm_sqrt_sd(__m128d a, __m128d b);
        __m128i _mm_sra_epi16(__m128i a, __m128i count);
        __m128i _mm_sra_epi32(__m128i a, __m128i count);
        __m128i _mm_srai_epi16(__m128i a, int imm);
        __m128i _mm_srai_epi32(__m128i a, int imm);
        __m128i _mm_srl_epi16(__m128i a, __m128i count);
        __m128i _mm_srl_epi32(__m128i a, __m128i count);
        __m128i _mm_srl_epi64(__m128i a, __m128i count);
        __m128i _mm_srli_epi16(__m128i a, int imm);
        __m128i _mm_srli_epi32(__m128i a, int imm);
        __m128i _mm_srli_epi64(__m128i a, int imm);
        __m128i _mm_srli_si128(__m128i a, int imm);
        void _mm_store_pd(scalarReg ptr, __m128d a, int offset = 0);
        void _mm_store_sd(scalarReg ptr, __m128d a, int offset = 0);
        void _mm_storeu_pd(scalarReg ptr, __m128d a, int offset = 0);
        void _mm_store_si128(scalarReg ptr, __m128i x, int offset = 0);
        void _mm_store_si128(scalarReg ptr, __m128i a, scalarReg index, int scale = 1, int offset = 0);
        void _mm_storeu_si128(scalarReg ptr, __m128i x, int offset = 0);
        void _mm_storeu_si128(scalarReg ptr, __m128i a, scalarReg index, int scale = 1, int offset = 0);
        __m128i _mm_sub_epi16(__m128i a, __m128i b);
        __m128i _mm_sub_epi32(__m128i a, __m128i b);
        __m128i _mm_sub_epi64(__m128i a, __m128i b);
        __m128i _mm_sub_epi8(__m128i a, __m128i b);
        __m128d _mm_sub_pd(__m128d a, __m128d b);
        __m128d _mm_sub_sd(__m128d a, __m128d b);
        __m128i _mm_subs_epi16(__m128i a, __m128i b);
        __m128i _mm_subs_epi8(__m128i a, __m128i b);
        __m128i _mm_subs_epu16(__m128i a, __m128i b);
        __m128i _mm_subs_epu8(__m128i a, __m128i b);
        // ucomieq_sd
        // ucomige_sd
        // ucomigt_sd
        // ucomile_sd
        // ucomilt_sd
        // ucomineq_sd
        // undefined
        __m128i _mm_unpackhi_epi16(__m128i a, __m128i b);
        __m128i _mm_unpackhi_epi32(__m128i a, __m128i b);
        __m128i _mm_unpackhi_epi64(__m128i a, __m128i b);
        __m128i _mm_unpackhi_epi8(__m128i a, __m128i b);
        __m128d _mm_unpackhi_pd(__m128d a, __m128d b);
        __m128i _mm_unpacklo_epi16(__m128i a, __m128i b);
        __m128i _mm_unpacklo_epi32(__m128i a, __m128i b);
        __m128i _mm_unpacklo_epi64(__m128i a, __m128i b);
        __m128i _mm_unpacklo_epi8(__m128i a, __m128i b);
        __m128d _mm_unpacklo_pd(__m128d a, __m128d b);
        __m128d _mm_xor_pd(__m128d a, __m128d b);
        __m128i _mm_xor_si128(__m128i a, __m128i b);
        __m128i _mm_loadl_epi64(scalarReg ptr, int offset = 0);
        void _mm_storel_epi64(scalarReg ptr, __m128i val, int offset = 0);

        // SSE3
        __m128d _mm_addsub_pd(__m128d a, __m128d b);
        __m128 _mm_addsub_ps(__m128 a, __m128 b);
        __m128d _mm_hadd_pd(__m128d a, __m128d b);
        __m128 _mm_hadd_ps(__m128 a, __m128 b);
        __m128d _mm_hsub_pd(__m128d a, __m128d b);
        __m128 _mm_hsub_ps(__m128 a, __m128 b);
        __m128i _mm_lddqu_si128(scalarReg ptr, int offset = 0);
        __m128d _mm_loaddup_pd(scalarReg ptr, int offset = 0);
        __m128d _mm_movedup_pd(__m128d a);
        __m128 _mm_movehdup_ps(__m128 a);
        __m128 _mm_moveldup_ps(__m128 a);

        // SSSE3
        __m128i _mm_abs_epi16(__m128i a);
        __m128i _mm_abs_epi32(__m128i a);
        __m128i _mm_abs_epi8(__m128i a);
        __m128i _mm_alignr_epi8(__m128i a, __m128i b, int count);
        __m128i _mm_hadd_epi16(__m128i a, __m128i b);
        __m128i _mm_hadd_epi32(__m128i a, __m128i b);
        __m128i _mm_hadds_epi16(__m128i a, __m128i b);
        __m128i _mm_hsub_epi16(__m128i a, __m128i b);
        __m128i _mm_hsub_epi32(__m128i a, __m128i b);
        __m128i _mm_hsubs_epi16(__m128i a, __m128i b);
        __m128i _mm_maddubs_epi16(__m128i a, __m128i b);
        __m128i _mm_mulhrs_epi16(__m128i a, __m128i b);
        __m128i _mm_shuffle_epi8(__m128i a, __m128i b);
        __m128i _mm_sign_epi16(__m128i a);
        __m128i _mm_sign_epi32(__m128i a);
        __m128i _mm_sign_epi8(__m128i a);

        // SSE4.1
        __m128i _mm_blend_epi16(__m128i a, __m128i b, int imm);
        __m128d _mm_blend_pd(__m128d a, __m128d b, int imm);
        __m128 _mm_blend_ps(__m128 a, __m128 b, int imm);
        __m128i _mm_blendv_epi8(__m128i a, __m128i b, __m128i mask);
        __m128d _mm_blendv_pd(__m128d a, __m128d b, __m128d mask);
        __m128 _mm_blendv_ps(__m128 a, __m128 b, __m128 mask);
        __m128d _mm_ceil_pd(__m128d a);
        __m128 _mm_ceil_ps(__m128 a);
        __m128d _mm_ceil_sd(__m128d a);
        __m128 _mm_ceil_ss(__m128 a);
        __m128i _mm_cmpeq_epi64(__m128i a, __m128i b);
        __m128i _mm_cvtepi16_epi32(__m128i a);
        __m128i _mm_cvtepi16_epi64(__m128i a);
        __m128i _mm_cvtepi32_epi64(__m128i a);
        __m128i _mm_cvtepi8_epi16(__m128i a);
        __m128i _mm_cvtepi8_epi32(__m128i a);
        __m128i _mm_cvtepi8_epi64(__m128i a);
        __m128i _mm_cvtepu16_epi32(__m128i a);
        __m128i _mm_cvtepu16_epi64(__m128i a);
        __m128i _mm_cvtepu32_epi64(__m128i a);
        __m128i _mm_cvtepu8_epi16(__m128i a);
        __m128i _mm_cvtepu8_epi32(__m128i a);
        __m128i _mm_cvtepu8_epi64(__m128i a);
        __m128d _mm_dp_pd(__m128d a, __m128d b, int imm);
        __m128 _mm_dp_ps(__m128 a, __m128 b, int imm);
        scalarReg _mm_extract_epi32(__m128i a, int imm);
        scalarReg _mm_extract_epi64(__m128i a, int imm);
        scalarReg _mm_extract_ep8(__m128i a, int imm);
        scalarReg _mm_extract_ps(__m128 a, int imm);
        void _mm_extract_ps(__m128 a, int imm, scalarReg ptr, int offset = 0);
        __m128d _mm_floor_pd(__m128d a);
        __m128 _mm_floor_ps(__m128 a);
        __m128d _mm_floor_sd(__m128d a);
        __m128 _mm_floor_ss(__m128 a);
        __m128i _mm_insert_epi32(__m128i a, scalarReg b, int imm);
        __m128i _mm_insert_epi64(__m128i a, scalarReg b, int imm);
        __m128i _mm_insert_epi8(__m128i a, scalarReg b, int imm);
        __m128 _mm_insert_ps(__m128 a, scalarReg b, int imm);
        __m128 _mm_insert_ps(__m128 a, scalarReg ptr, int offset, int imm);
        __m128i _mm_max_epi32(__m128i a, __m128i b);
        __m128i _mm_max_epi8(__m128i a, __m128i b);
        __m128i _mm_max_epu16(__m128i a, __m128i b);
        __m128i _mm_min_epi32(__m128i a, __m128i b);
        __m128i _mm_min_epi8(__m128i a, __m128i b);
        __m128i _mm_min_epu16(__m128i a, __m128i b);
        __m128i _mm_min_epu32(__m128i a, __m128i b);
        __m128i _mm_minpos_epu16(__m128i a);
        __m128i _mm_mpsadbw_epu8(__m128i a, __m128i b, int imm);
        __m128i _mm_mul_epi32(__m128i a, __m128i b);
        __m128i _mm_mullo_epi32(__m128i a, __m128i b);
        __m128i _mm_packus_epi32(__m128i a, __m128i b);
        __m128d _mm_round_pd(__m128d a, RoundingMode mode);
        __m128 _mm_round_ps(__m128 a, RoundingMode mode);
        __m128d _mm_round_sd(__m128d a, RoundingMode mode);
        __m128 _mm_round_ss(__m128 a, RoundingMode mode);
        __m128i _mm_stream_load_si128(scalarReg ptr, int offset = 0);
        __m128i _mm_stream_load_si128(scalarReg ptr, scalarReg index, int scale = 1, int offset = 0);
        scalarReg _mm_test_all_ones(__m128i a);
        scalarReg _mm_test_all_zeros(__m128i a, __m128i mask);
        scalarReg _mm_test_mix_ones_zeros(__m128i a, __m128i mask);
        scalarReg _mm_testc_si128(__m128i a, __m128i b);
        scalarReg _mm_testnzc_si128(__m128i a, __m128i b);
        scalarReg _mm_testz_si128(__m128i a, __m128i b);

        // SSE 4.2
        scalarReg _mm_cmpistra(__m128i a, __m128i b, SIDD mode);
        scalarReg _mm_cmpistrc(__m128i a, __m128i b, SIDD mode);
        scalarReg _mm_cmpistri(__m128i a, __m128i b, SIDD mode);
        scalarReg _mm_cmpistro(__m128i a, __m128i b, SIDD mode);
        scalarReg _mm_cmpistrs(__m128i a, __m128i b, SIDD mode);
        scalarReg _mm_cmpistrz(__m128i a, __m128i b, SIDD mode);
        __m128i _mm_cmpistrm(__m128i a, __m128i b, SIDD mode);
        __m128i _mm_cmpgt_epi64(__m128i a, __m128i b);
        //scalarReg _mm_crc32_u32(scalarReg crc, scalarReg v);
        //scalarReg _mm_crc32_u64(scalarReg crc, scalarReg v);
        //scalarReg _mm_crc32_u8(scalarReg crc, scalarReg v);

        // AVX
        __m256d _mm256_add_pd(__m256d a, __m256d b);
        __m256 _mm256_add_ps(__m256 a, __m256 b);
        __m256d _mm256_addsub_pd(__m256d a, __m256d b);
        __m256 _mm256_addsub_ps(__m256 a, __m256 b);
        __m256d _mm256_and_pd(__m256d a, __m256d b);
        __m256 _mm256_and_ps(__m256 a, __m256 b);
        __m256d _mm256_andnot_pd(__m256d a, __m256d b);
        __m256 _mm256_andnot_ps(__m256 a, __m256 b);
        __m256d _mm256_blend_pd(__m256d a, __m256d b, int imm);
        __m256 _mm256_blend_ps(__m256 a, __m256 b, int imm);
        __m256d _mm256_blendv_pd(__m256d a, __m256d b, __m256d mask);
        __m256 _mm256_blendv_ps(__m256 a, __m256 b, __m256 mask);
        __m256 _mm256_castpd_ps(__m256d a);
        __m256 _mm256_castsi256_ps(__m256i a);
        __m256d _mm256_castps_pd(__m256 a);
        __m256d _mm256_castsi256_pd(__m256i a);
        __m256i _mm256_castps_si256(__m256 a);
        __m256i _mm256_castpd_si256(__m256d a);
        __m256 _mm256_castps128_ps256(__m128 a);
        __m256d _mm256_castpd128_pd256(__m128d a);
        __m256i _mm256_castsi128_si256(__m128i a);
        __m128 _mm256_castps256_ps128(__m256 a);
        __m128d _mm256_castpd256_pd128(__m256d a);
        __m128i _mm256_castsi256_si128(__m256i a);
        __m256d _mm256_ceil_pd(__m256d a);
        __m256 _mm256_ceil_ps(__m256 a);
        __m256d _mm256_cmp_pd(__m256d a, __m256d b, int imm);
        __m256 _mm256_cmp_ps(__m256 a, __m256 b, int imm);
        __m256d _mm256_cvtepi32_pd(__m128i a);
        __m256 _mm256_cvtepi32_ps(__m256i a);
        __m128i _mm256_cvtpd_epi32(__m256d a);
        __m256i _mm256_cvtps_epi32(__m256 a);
        __m256d _mm256_cvtps_pd(__m128 a);
        __m256d _mm256_div_pd(__m256d a, __m256d b);
        __m256 _mm256_div_ps(__m256 a, __m256 b);
        __m256 _mm256_dp_ps(__m256 a, __m256 b, int imm);
        scalarReg _mm256_extract_epi16(__m256i a, int index);
        scalarReg _mm256_extract_epi32(__m256i a, int index);
        scalarReg _mm256_extract_epi64(__m256i a, int index);
        scalarReg _mm256_extract_epi8(__m256i a, int index);
        __m128d _mm256_extractf128_pd(__m256d a, int index);
        __m128 _mm256_extractf128_ps(__m256 a, int index);
        __m128i _mm256_extractf128_si256(__m256i a, int index);
        __m256d _mm256_floor_pd(__m256d a);
        __m256 _mm256_floor_ps(__m256 a);
        __m256d _mm256_hadd_pd(__m256d a, __m256d b);
        __m256 _mm256_hadd_ps(__m256 a, __m256 b);
        __m256d _mm256_hsub_pd(__m256d a, __m256d b);
        __m256 _mm256_hsub_ps(__m256 a, __m256 b);
        __m256i _mm256_insert_epi16(__m256i a, scalarReg v, int index);
        __m256i _mm256_insert_epi32(__m256i a, scalarReg v, int index);
        __m256i _mm256_insert_epi64(__m256i a, scalarReg v, int index);
        __m256i _mm256_insert_epi8(__m256i a, scalarReg v, int index);
        __m256d _mm256_insertf128_pd(__m256d a, __m128d b, int index);
        __m256 _mm256_insertf128_ps(__m256 a, __m128 b, int index);
        __m256i _mm256_insertf128_si256(__m256i a, __m128i b, int index);
        __m256i _mm256_lddqu_si256(scalarReg ptr, int offset = 0);
        __m256d _mm256_load_pd(scalarReg ptr, int offset = 0);
        __m256d _mm256_load_pd(scalarReg ptr, scalarReg index, int scale = 1, int offset = 0);
        __m256 _mm256_load_ps(scalarReg ptr, int offset = 0);
        __m256 _mm256_load_ps(scalarReg ptr, scalarReg index, int scale = 1, int offset = 0);
        __m256i _mm256_load_si256(scalarReg ptr, int offset = 0);
        __m256i _mm256_load_si256(scalarReg ptr, scalarReg index, int scale = 1, int offset = 0);
        __m256d _mm256_loadu_pd(scalarReg ptr, int offset = 0);
        __m256d _mm256_loadu_pd(scalarReg ptr, scalarReg index, int scale = 1, int offset = 0);
        __m256 _mm256_loadu_ps(scalarReg ptr, int offset = 0);
        __m256 _mm256_loadu_ps(scalarReg ptr, scalarReg index, int scale = 1, int offset = 0);
        __m256i _mm256_loadu_si256(scalarReg ptr, int offset = 0);
        __m256i _mm256_loadu_si256(scalarReg ptr, scalarReg index, int scale = 1, int offset = 0);
        // maskload/maskstore
        __m256d _mm256_max_pd(__m256d a, __m256d b);
        __m256 _mm256_max_ps(__m256 a, __m256 b);
        __m256d _mm256_min_pd(__m256d a, __m256d b);
        __m256 _mm256_min_ps(__m256 a, __m256 b);
        __m256d _mm256_movedup_pd(__m256d a);
        __m256 _mm256_movehdup_ps(__m256 a);
        __m256 _mm256_moveldup_ps(__m256 a);
        scalarReg _mm256_movemask_pd(__m256d a);
        scalarReg _mm256_movemask_ps(__m256 a);
        __m256d _mm256_mul_pd(__m256d a, __m256d b);
        __m256 _mm256_mul_ps(__m256 a, __m256 b);
        __m256d _mm256_or_pd(__m256d a, __m256d b);
        __m256 _mm256_or_ps(__m256 a, __m256 b);
        __m256d _mm256_permute_pd(__m256d a, int imm);
        __m256 _mm256_permute_ps(__m256 a, int imm);
        __m128d _mm_permute_pd(__m128d a, int imm);
        __m128 _mm_permute_ps(__m128 a, int imm);
        __m256d _mm256_permutevar_pd(__m256d a, __m256d b);
        __m256 _mm256_permutevar_ps(__m256 a, __m256 b);
        __m128d _mm_permutevar_pd(__m128d a, __m128d b);
        __m128 _mm_permutevar_ps(__m128 a, __m128 b);
        __m256d _mm256_permute2f128_pd(__m256d a, __m256d b, int imm);
        __m256 _mm256_permute2f128_ps(__m256 a, __m256 b, int imm);
        __m256i _mm256_permute2f128_si256(__m256i a, __m256i b, int imm);
        __m256 _mm256_rcp_ps(__m256 a);
        __m256d _mm256_round_pd(__m256d a, SIDD mode);
        __m256 _mm256_round_ps(__m256 a, SIDD mode);
        __m256 _mm256_rsqrt_ps(__m256 a);
        __m256i _mm256_set_epi16(short e15, short e14, short e13, short e12, short e11, short e10, short e9, short e8, short e7, short e6, short e5, short e4, short e3, short e2, short e1, short e0);
        __m256i _mm256_set_epi32(int e7, int e6, int e5, int e4, int e3, int e2, int e1, int e0);
        __m256i _mm256_set_epi64x(long e3, long e2, long e1, long e0);
        __m256i _mm256_set_epi8(byte e31, byte e30, byte e29, byte e28, byte e27, byte e26, byte e25, byte e24, byte e23, byte e22, byte e21, byte e20, byte e19, byte e18, byte e17, byte e16, byte e15, byte e14, byte e13, byte e12, byte e11, byte e10, byte e9, byte e8, byte e7, byte e6, byte e5, byte e4, byte e3, byte e2, byte e1, byte e0);
        __m256 _mm256_set_m128(__m128 hi, __m128 lo);
        __m256d _mm256_set_m128d(__m128d hi, __m128d lo);
        __m256i _mm256_set_m128i(__m128i hi, __m128i lo);
        __m256d _mm256_set_pd(double e3, double e2, double e1, double e0);
        __m256 _mm256_set_ps(float e7, float e6, float e5, float e4, float e3, float e2, float e1, float e0);
        __m256i _mm256_set1_epi16(short a);
        __m256i _mm256_set1_epi32(int a);
        __m256i _mm256_set1_epi64x(long a);
        __m256i _mm256_set1_epi8(byte a);
        __m256d _mm256_set1_pd(double a);
        __m256 _mm256_set1_ps(float a);
        __m256i _mm256_setr_epi16(short e15, short e14, short e13, short e12, short e11, short e10, short e9, short e8, short e7, short e6, short e5, short e4, short e3, short e2, short e1, short e0);
        __m256i _mm256_setr_epi32(int e7, int e6, int e5, int e4, int e3, int e2, int e1, int e0);
        __m256i _mm256_setr_epi64x(long e3, long e2, long e1, long e0);
        __m256i _mm256_setr_epi8(byte e31, byte e30, byte e29, byte e28, byte e27, byte e26, byte e25, byte e24, byte e23, byte e22, byte e21, byte e20, byte e19, byte e18, byte e17, byte e16, byte e15, byte e14, byte e13, byte e12, byte e11, byte e10, byte e9, byte e8, byte e7, byte e6, byte e5, byte e4, byte e3, byte e2, byte e1, byte e0);
        __m256d _mm256_setr_pd(double e3, double e2, double e1, double e0);
        __m256 _mm256_setr_ps(float e7, float e6, float e5, float e4, float e3, float e2, float e1, float e0);
        __m256 _mm256_setzero_ps();
        __m256d _mm256_setzero_pd();
        __m256i _mm256_setzero_si256();
        __m256d _mm256_shuffle_pd(__m256d a, __m256d b, int imm);
        __m256 _mm256_shuffle_ps(__m256 a, __m256 b, int imm);
        __m256d _mm256_sqrt_pd(__m256d a);
        __m256 _mm256_sqrt_ps(__m256 a);
        void _mm256_store_ps(scalarReg ptr, __m256 a, int offset = 0);
        void _mm256_storeu_ps(scalarReg ptr, __m256 a, int offset = 0);
        void _mm256_store_pd(scalarReg ptr, __m256d a, int offset = 0);
        void _mm256_storeu_pd(scalarReg ptr, __m256d a, int offset = 0);
        void _mm256_stream_pd(scalarReg ptr, __m256d a, int offset = 0);
        void _mm256_stream_ps(scalarReg ptr, __m256 a, int offset = 0);
        __m256d _mm256_sub_pd(__m256d a, __m256d b);
        __m256 _mm256_sub_ps(__m256 a, __m256 b);
        // vptest
        // undefined
        __m256d _mm256_unpackhi_pd(__m256d a, __m256d b);
        __m256 _mm256_unpackhi_ps(__m256 a, __m256 b);
        __m256d _mm256_unpacklo_pd(__m256d a, __m256d b);
        __m256 _mm256_unpacklo_ps(__m256 a, __m256 b);
        __m256d _mm256_xor_pd(__m256d a, __m256d b);
        __m256 _mm256_xor_ps(__m256 a, __m256 b);
        void _mm256_zeroall();
        void _mm256_zeroupper();
        //__m256d _mm256_zextpd128_pd256(__m128d a);
        //__m256 _mm256_zextps128_ps256(__m128 a);
        //__m256i _mm256_zextsi128_si256(__m128i a);

        // AVX 2
        __m256i _mm256_abs_epi16(__m256i a);
        __m256i _mm256_abs_epi32(__m256i a);
        __m256i _mm256_abs_epi8(__m256i a);
        __m256i _mm256_add_epi16(__m256i a, __m256i b);
        __m256i _mm256_add_epi32(__m256i a, __m256i b);
        __m256i _mm256_add_epi64(__m256i a, __m256i b);
        __m256i _mm256_add_epi8(__m256i a, __m256i b);
        __m256i _mm256_adds_epi16(__m256i a, __m256i b);
        __m256i _mm256_adds_epi8(__m256i a, __m256i b);
        __m256i _mm256_adds_epu16(__m256i a, __m256i b);
        __m256i _mm256_adds_epu8(__m256i a, __m256i b);
        __m256i _mm256_alignr_epi8(__m256i a, __m256i b, int count);
        __m256i _mm256_and_si256(__m256i a, __m256i b);
        __m256i _mm256_andnot_si256(__m256i a, __m256i b);
        __m256i _mm256_avg_epu16(__m256i a, __m256i b);
        __m256i _mm256_avg_epu8(__m256i a, __m256i b);
        __m256i _mm256_blend_epi16(__m256i a, __m256i b, int imm8);
        __m128i _mm_blend_epi32(__m128i a, __m128i b, int imm8);
        __m256i _mm256_blend_epi32(__m256i a, __m256i b, int imm8);
        __m256i _mm256_blendv_epi8(__m256i a, __m256i b, __m256i mask);
        __m128i _mm_broadcastb_epi8(__m128i a);
        __m256i _mm256_broadcastb_epi8(__m128i a);
        __m128i _mm_broadcastd_epi32(__m128i a);
        __m256i _mm256_broadcastd_epi32(__m128i a);
        __m128i _mm_broadcastq_epi64(__m128i a);
        __m256i _mm256_broadcastq_epi64(__m128i a);
        __m128d _mm_broadcastsd_pd(__m128d a);
        __m256d _mm256_broadcastsd_pd(__m128d a);
        __m256i _mm_broadcastsi128_si256(__m128i a);
        __m256i _mm256_broadcastsi128_si256(__m128i a);
        __m128 _mm_broadcastss_ps(__m128 a);
        __m256 _mm256_broadcastss_ps(__m128 a);
        __m128i _mm_broadcastw_epi16(__m128i a);
        __m256i _mm256_broadcastw_epi16(__m128i a);
        __m256i _mm256_bslli_epi128(__m256i a, int imm8);
        __m256i _mm256_bsrli_epi128(__m256i a, int imm8);
        __m256i _mm256_cmpeq_epi16(__m256i a, __m256i b);
        __m256i _mm256_cmpeq_epi32(__m256i a, __m256i b);
        __m256i _mm256_cmpeq_epi64(__m256i a, __m256i b);
        __m256i _mm256_cmpeq_epi8(__m256i a, __m256i b);
        __m256i _mm256_cmpgt_epi16(__m256i a, __m256i b);
        __m256i _mm256_cmpgt_epi32(__m256i a, __m256i b);
        __m256i _mm256_cmpgt_epi64(__m256i a, __m256i b);
        __m256i _mm256_cmpgt_epi8(__m256i a, __m256i b);
        __m256i _mm256_cvtepi16_epi32(__m128i a);
        __m256i _mm256_cvtepi16_epi64(__m128i a);
        __m256i _mm256_cvtepi32_epi64(__m128i a);
        __m256i _mm256_cvtepi8_epi16(__m128i a);
        __m256i _mm256_cvtepi8_epi32(__m128i a);
        __m256i _mm256_cvtepi8_epi64(__m128i a);
        __m256i _mm256_cvtepu16_epi32(__m128i a);
        __m256i _mm256_cvtepu16_epi64(__m128i a);
        __m256i _mm256_cvtepu32_epi64(__m128i a);
        __m256i _mm256_cvtepu8_epi16(__m128i a);
        __m256i _mm256_cvtepu8_epi32(__m128i a);
        __m256i _mm256_cvtepu8_epi64(__m128i a);
        __m128i _mm256_extracti128_si256(__m256i a, int imm8);
        __m256i _mm256_hadd_epi16(__m256i a, __m256i b);
        __m256i _mm256_hadd_epi32(__m256i a, __m256i b);
        __m256i _mm256_hadds_epi16(__m256i a, __m256i b);
        __m256i _mm256_hsub_epi16(__m256i a, __m256i b);
        __m256i _mm256_hsub_epi32(__m256i a, __m256i b);
        __m256i _mm256_hsubs_epi16(__m256i a, __m256i b);
        // mask-gather
        __m256i _mm256_inserti128_si256(__m256i a, __m128i b, int imm8);
        __m256i _mm256_madd_epi16(__m256i a, __m256i b);
        __m256i _mm256_maddubs_epi16(__m256i a, __m256i b);
        // mask load
        // mask store
        __m256i _mm256_max_epi16(__m256i a, __m256i b);
        __m256i _mm256_max_epi32(__m256i a, __m256i b);
        __m256i _mm256_max_epi8(__m256i a, __m256i b);
        __m256i _mm256_max_epu16(__m256i a, __m256i b);
        __m256i _mm256_max_epu32(__m256i a, __m256i b);
        __m256i _mm256_max_epu8(__m256i a, __m256i b);
        __m256i _mm256_min_epi16(__m256i a, __m256i b);
        __m256i _mm256_min_epi32(__m256i a, __m256i b);
        __m256i _mm256_min_epi8(__m256i a, __m256i b);
        __m256i _mm256_min_epu16(__m256i a, __m256i b);
        __m256i _mm256_min_epu32(__m256i a, __m256i b);
        __m256i _mm256_min_epu8(__m256i a, __m256i b);
        scalarReg _mm256_movemask_epi8(__m256i a);
        __m256i _mm256_mpsadbw_epu8(__m256i a, __m256i b, int imm8);
        __m256i _mm256_mul_epi32(__m256i a, __m256i b);
        __m256i _mm256_mul_epu32(__m256i a, __m256i b);
        __m256i _mm256_mulhi_epi16(__m256i a, __m256i b);
        __m256i _mm256_mulhi_epu16(__m256i a, __m256i b);
        __m256i _mm256_mulhrs_epi16(__m256i a, __m256i b);
        __m256i _mm256_mullo_epi16(__m256i a, __m256i b);
        __m256i _mm256_mullo_epi32(__m256i a, __m256i b);
        __m256i _mm256_or_si256(__m256i a, __m256i b);
        __m256i _mm256_packs_epi16(__m256i a, __m256i b);
        __m256i _mm256_packs_epi32(__m256i a, __m256i b);
        __m256i _mm256_packus_epi16(__m256i a, __m256i b);
        __m256i _mm256_packus_epi32(__m256i a, __m256i b);
        // permute
        __m256i _mm256_sad_epu8(__m256i a, __m256i b);
        __m256i _mm256_shuffle_epi32(__m256i a, int imm8);
        __m256i _mm256_shuffle_epi8(__m256i a, __m256i b);
        __m256i _mm256_shufflehi_epi16(__m256i a, int imm8);
        __m256i _mm256_shufflelo_epi16(__m256i a, int imm8);
        __m256i _mm256_sign_epi16(__m256i a, __m256i b);
        __m256i _mm256_sign_epi32(__m256i a, __m256i b);
        __m256i _mm256_sign_epi8(__m256i a, __m256i b);
        __m256i _mm256_sll_epi16(__m256i a, __m128i count);
        __m256i _mm256_sll_epi32(__m256i a, __m128i count);
        __m256i _mm256_sll_epi64(__m256i a, __m128i count);
        __m256i _mm256_slli_epi16(__m256i a, int imm8);
        __m256i _mm256_slli_epi32(__m256i a, int imm8);
        __m256i _mm256_slli_epi64(__m256i a, int imm8);
        __m256i _mm256_slli_si256(__m256i a, int imm8);
        __m128i _mm_sllv_epi32(__m128i a, __m128i count);
        __m256i _mm256_sllv_epi32(__m256i a, __m256i count);
        __m128i _mm_sllv_epi64(__m128i a, __m128i count);
        __m256i _mm256_sllv_epi64(__m256i a, __m256i count);
        __m256i _mm256_sra_epi16(__m256i a, __m128i count);
        __m256i _mm256_sra_epi32(__m256i a, __m128i count);
        __m256i _mm256_srai_epi16(__m256i a, int imm8);
        __m256i _mm256_srai_epi32(__m256i a, int imm8);
        __m128i _mm_srav_epi32(__m128i a, __m128i count);
        __m256i _mm256_srav_epi32(__m256i a, __m256i count);
        __m256i _mm256_srl_epi16(__m256i a, __m128i count);
        __m256i _mm256_srl_epi32(__m256i a, __m128i count);
        __m256i _mm256_srl_epi64(__m256i a, __m128i count);
        __m256i _mm256_srli_epi16(__m256i a, int imm8);
        __m256i _mm256_srli_epi32(__m256i a, int imm8);
        __m256i _mm256_srli_epi64(__m256i a, int imm8);
        __m256i _mm256_srli_si256(__m256i a, int imm8);
        __m128i _mm_srlv_epi32(__m128i a, __m128i count);
        __m256i _mm256_srlv_epi32(__m256i a, __m256i count);
        __m128i _mm_srlv_epi64(__m128i a, __m128i count);
        __m256i _mm256_srlv_epi64(__m256i a, __m256i count);
        __m256i _mm256_stream_load_si256(scalarReg ptr, int offset = 0);
        __m256i _mm256_sub_epi16(__m256i a, __m256i b);
        __m256i _mm256_sub_epi32(__m256i a, __m256i b);
        __m256i _mm256_sub_epi64(__m256i a, __m256i b);
        __m256i _mm256_sub_epi8(__m256i a, __m256i b);
        __m256i _mm256_subs_epi16(__m256i a, __m256i b);
        __m256i _mm256_subs_epi8(__m256i a, __m256i b);
        __m256i _mm256_subs_epu16(__m256i a, __m256i b);
        __m256i _mm256_subs_epu8(__m256i a, __m256i b);
        __m256i _mm256_unpackhi_epi16(__m256i a, __m256i b);
        __m256i _mm256_unpackhi_epi32(__m256i a, __m256i b);
        __m256i _mm256_unpackhi_epi64(__m256i a, __m256i b);
        __m256i _mm256_unpackhi_epi8(__m256i a, __m256i b);
        __m256i _mm256_unpacklo_epi16(__m256i a, __m256i b);
        __m256i _mm256_unpacklo_epi32(__m256i a, __m256i b);
        __m256i _mm256_unpacklo_epi64(__m256i a, __m256i b);
        __m256i _mm256_unpacklo_epi8(__m256i a, __m256i b);
        __m256i _mm256_xor_si256(__m256i a, __m256i b);

        // FMA
        __m128 _mm_fmadd_ps(__m128 a, __m128 b, __m128 c);
        __m128 _mm_fmadd_ss(__m128 a, __m128 b, __m128 c);
        __m128d _mm_fmadd_pd(__m128d a, __m128d b, __m128d c);
        __m128d _mm_fmadd_sd(__m128d a, __m128d b, __m128d c);
        __m256 _mm256_fmadd_ps(__m256 a, __m256 b, __m256 c);
        __m256 _mm256_fmadd_ss(__m256 a, __m256 b, __m256 c);
        __m256d _mm256_fmadd_pd(__m256d a, __m256d b, __m256d c);
        __m256d _mm256_fmadd_sd(__m256d a, __m256d b, __m256d c);
    }

    static class Exts
    {
        public static TV GetOrDefault<TK, TV>(this Dictionary<TK, TV> d, TK key)
        {
            TV res;
            d.TryGetValue(key, out res);
            return res;
        }

        public static void AddM(this Dictionary<int, List<int>> d, int key, int item)
        {
            List<int> l;
            if (!d.TryGetValue(key, out l))
            {
                l = new List<int>(1);
                d.Add(key, l);
            }
            l.Add(item);
        }
    }

#if NOCAST
    struct __xmm
    {
        public readonly int r;

        public __xmm(int r)
        {
            this.r = r;
            System.Diagnostics.Debug.Assert(r >= 0x100000, "Register has the wrong type");
        }

        public override string ToString()
        {
            return "xmm" + (r & 0xFFFFF);
        }
    }

    struct __ymm
    {
        public readonly int r;

        public __ymm(int r)
        {
            this.r = r;
            System.Diagnostics.Debug.Assert(r >= 0x100000, "Register has the wrong type");
        }

        public override string ToString()
        {
            return "ymm" + (r & 0xFFFFF);
        }
    }
#endif

#if !NOCAST
    struct __m128
    {
        public readonly int r;

        public __m128(int r)
        {
            this.r = r;
            System.Diagnostics.Debug.Assert(r >= 0x100000, "Register has the wrong type");
        }

        public override string ToString()
        {
            return "xmm" + (r & 0xFFFFF);
        }
    }

    struct __m128d
    {
        public readonly int r;

        public __m128d(int r)
        {
            this.r = r;
            System.Diagnostics.Debug.Assert(r >= 0x100000, "Register has the wrong type");
        }

        public override string ToString()
        {
            return "xmm" + (r & 0xFFFFF);
        }
    }

    struct __m128i
    {
        public readonly int r;

        public __m128i(int r)
        {
            this.r = r;
            System.Diagnostics.Debug.Assert(r >= 0x100000, "Register has the wrong type");
        }

        public override string ToString()
        {
            return "xmm" + (r & 0xFFFFF);
        }
    }

    struct __m256
    {
        public readonly int r;

        public __m256(int r)
        {
            this.r = r;
            System.Diagnostics.Debug.Assert(r >= 0x100000, "Register has the wrong type");
        }

        public override string ToString()
        {
            return "ymm" + (r & 0xFFFFF);
        }
    }

    struct __m256d
    {
        public readonly int r;

        public __m256d(int r)
        {
            this.r = r;
            System.Diagnostics.Debug.Assert(r >= 0x100000, "Register has the wrong type");
        }

        public override string ToString()
        {
            return "ymm" + (r & 0xFFFFF);
        }
    }

    struct __m256i
    {
        public readonly int r;

        public __m256i(int r)
        {
            this.r = r;
            System.Diagnostics.Debug.Assert(r >= 0x100000, "Register has the wrong type");
        }

        public override string ToString()
        {
            return "ymm" + (r & 0xFFFFF);
        }
    }
#endif

    struct scalarReg
    {
        public readonly int r;

        public scalarReg(int r)
        {
            this.r = r;
            System.Diagnostics.Debug.Assert(r < 0x100000, "Register has the wrong type");
        }

        public override string ToString()
        {
            return "v" + r;
        }
    }

    [Flags]
    enum RoundingMode
    {
        _MM_FROUND_TO_NEAREST_INT = 0x00,
        _MM_FROUND_TO_NEG_INF = 0x01,
        _MM_FROUND_TO_POS_INF = 0x02,
        _MM_FROUND_TO_ZERO = 0x03,
        _MM_FROUND_CUR_DIRECTION = 0x04,
        _MM_FROUND_RAISE_EXC = 0x00,
        _MM_FROUND_NO_EXC = 0x08,
        _MM_FROUND_NINT = (_MM_FROUND_RAISE_EXC | _MM_FROUND_TO_NEAREST_INT),
        _MM_FROUND_FLOOR = (_MM_FROUND_RAISE_EXC | _MM_FROUND_TO_NEG_INF),
        _MM_FROUND_CEIL = (_MM_FROUND_RAISE_EXC | _MM_FROUND_TO_POS_INF),
        _MM_FROUND_TRUNC = (_MM_FROUND_RAISE_EXC | _MM_FROUND_TO_ZERO),
        _MM_FROUND_RINT = (_MM_FROUND_RAISE_EXC | _MM_FROUND_CUR_DIRECTION),
        _MM_FROUND_NEARBYINT = (_MM_FROUND_NO_EXC | _MM_FROUND_CUR_DIRECTION)
    }

    [Flags]
    enum SIDD
    {
        UBYTE_OPS = 0,
        UWORD_OPS = 1,
        SBYTE_OPS = 2,
        SWORD_OPS = 3,
        CMP_EQUAL_ANY = 0,
        CMP_RANGES = 4,
        CMP_EQUAL_EACH = 8,
        CMP_EQUAL_ORDERED = 0xC,
        POSITIVE_POLARITY = 0,
        NEGATIVE_POLARITY = 0x10,
        MASKED_POSITIVE_POLARITY = 0x20,
        MASKED_NEGATIVE_POLARITY = 0x30,
        LEAST_SIGNIFICANT = 0,
        MOST_SIGNIFICANT = 0x40,
        BIT_MASK = 0,
        UNIT_MASK = 0x40,
    }

    sealed class Options
    {
        public int FunctionAlignment = 16;
        public int LoopAlignment = 1;
        public bool DiscardedValueIsAnError = true;
        public bool MergeLoadWithUse = true;
        public bool DetectBlend = true;
        public bool BreakAtFunctionEntry = false;
        public bool AVX2 = true;
        

        public static Options Default
        {
            get { return new Options(); }
        }
    }
}
