using System;
using System.Linq;
using System.Collections.Generic;

namespace SharpSIMD
{
    sealed class Block
    {
        public readonly List<byte> buffer;
        public readonly int alignment;
        public readonly int label;
        public int EndWithLabel;
        public int DataLabelRef;

        public Block(int label, int alignment = 1)
        {
            buffer = new List<byte>(4);
            this.alignment = alignment;
            this.label = label;
        }
    }

    sealed class CodeBuffer
    {
        List<Block> dblocks;
        List<Block> blocks;

        Block last
        {
            get { return blocks[blocks.Count - 1]; }
        }

        public CodeBuffer()
        {
            blocks = new List<Block>(4);
            blocks.Add(new Block(-1));
            dblocks = new List<Block>(0);
        }

        public void OpenBlock(int label, int alignment = 1)
        {
            blocks.Add(new Block(label, alignment));
        }

        public void SetGoto(int label)
        {
            last.EndWithLabel = label;
        }

        public void SetRIPREL(int label)
        {
            last.DataLabelRef = label;
        }

        public void AddData(IEnumerable<Block> b)
        {
            dblocks.AddRange(b);
        }

        public void Write(byte x)
        {
            last.buffer.Add(x);
        }

        public void Writeb(int x)
        {
            last.buffer.Add(unchecked((byte)x));
        }

        public void Write(int x)
        {
            unchecked
            {
                last.buffer.Add((byte)x);
                last.buffer.Add((byte)(x >> 8));
                last.buffer.Add((byte)(x >> 16));
                last.buffer.Add((byte)(x >> 24));
            }
        }

        public void Write(uint x)
        {
            unchecked
            {
                last.buffer.Add((byte)x);
                last.buffer.Add((byte)(x >> 8));
                last.buffer.Add((byte)(x >> 16));
                last.buffer.Add((byte)(x >> 24));
            }
        }

        public void Write(byte[] x)
        {
            last.buffer.AddRange(x);
        }

        public void Align16()
        {
            while ((last.buffer.Count & 15) != 0)
                last.buffer.Add(0xCC);
        }

        List<byte> relaxAndConcat(Dictionary<int, int> labelPos)
        {
            int dsize = 0;
            foreach (var b in dblocks)
            {
                dsize = (dsize + b.alignment - 1) & -b.alignment;
                dsize += b.buffer.Count;
            }

            List<byte> result = new List<byte>((from b in blocks
                                                select b.buffer.Count + 4).Sum() + dsize);
            HashSet<int> isLong = new HashSet<int>();
            bool isGood = false;
            do
            {
                // get label positions
                labelPos.Clear();
                int pos = dsize;
                foreach (var b in blocks)
                {
                    pos = (pos + b.alignment - 1) & -b.alignment;
                    labelPos.Add(b.label, pos);
                    if (b.EndWithLabel == 0)
                        pos += b.buffer.Count;
                    else if (isLong.Contains(b.label))
                        pos += b.buffer.Count + 4;
                    else
                        pos += b.buffer.Count + 1;
                }
                if (isGood)
                    break;
                // make out-of-range label ref into long label ref if there is one
                pos = dsize;
                isGood = true;
                foreach (var b in blocks)
                {
                    pos = (pos + b.alignment - 1) & -b.alignment;
                    if (b.EndWithLabel == 0)
                        pos += b.buffer.Count;
                    else if (isLong.Contains(b.label))
                        pos += b.buffer.Count + 4;
                    else
                    {
                        pos += b.buffer.Count + 1;
                        int goTo = labelPos[b.EndWithLabel];
                        int dist = goTo - pos;
                        if (dist < -128 || dist > 127)
                        {
                            // change into long form
                            isLong.Add(b.label);
                            int oldopcode = b.buffer[b.buffer.Count - 1];
                            b.buffer.RemoveAt(b.buffer.Count - 1);
                            if (oldopcode >= 0x70 && oldopcode <= 0x7F)
                            {
                                b.buffer.Add(0x0F);
                                b.buffer.Add((byte)(oldopcode + 0x10));
                            }
                            else if (oldopcode == 0xEB)
                                b.buffer.Add(0xE9);
                            else
                                throw new NotImplementedException();
                            isGood = false;
                            break;
                        }
                    }
                }
            } while (true);

            // serialize and put in offsets
            {
                int pos = 0;
                foreach (var b in dblocks)
                {
                    int npos = (pos + b.alignment - 1) & -b.alignment;
                    if (pos != npos)
                    {
                        for (int i = 0; i < npos - pos; i++)
                            result.Add(0xCD);
                    }
                    pos = npos;
                    labelPos[b.label] = pos;
                    pos += b.buffer.Count;
                    result.AddRange(b.buffer);
                }

                pos = dsize;
                foreach (var b in blocks)
                {
                    int npos = (pos + b.alignment - 1) & -b.alignment;
                    if (pos != npos)
                        result.AddRange(getNop(npos - pos));
                    pos = npos;
                    result.AddRange(b.buffer);
                    pos += b.buffer.Count;
                    if (b.EndWithLabel != 0)
                    {
                        if (isLong.Contains(b.EndWithLabel))
                        {
                            pos += 4;
                            int dist = labelPos[b.EndWithLabel] - pos;
                            result.AddRange(BitConverter.GetBytes(dist));
                        }
                        else
                        {
                            pos++;
                            int dist = labelPos[b.EndWithLabel] - pos;
                            if (dist < -128 || dist > 127)
                                throw new Exception();
                            result.Add(unchecked((byte)dist));
                        }
                    }
                    else if (b.DataLabelRef != 0)
                    {
                        int ofs = result[result.Count - 4];
                        int vrip = pos + ofs;
                        int dist = labelPos[b.DataLabelRef] - vrip;
                        result[result.Count - 4] = (byte)dist;
                        result[result.Count - 3] = (byte)(dist >> 8);
                        result[result.Count - 2] = (byte)(dist >> 16);
                        result[result.Count - 1] = (byte)(dist >> 24);
                    }
                }
            }

            return result;
        }

        public NativeCode Save(Dictionary<int, int> labelPos)
        {
            dblocks.Sort((b0, b1) => -b0.alignment.CompareTo(b1.alignment));
            labelPos.Clear();
            var relaxed = relaxAndConcat(labelPos);

#if DEBUG
            string tostring = string.Join("", from x in relaxed
                                              select x.ToString("X2"));
            Console.WriteLine(tostring);
#endif

            var res = new NativeCode(relaxed);
            return res;
        }

        static byte[] getNop(int size)
        {
            if (size <= 10)
            {
                switch (size)
                {
                    case 0:
                        return new byte[0];
                    case 1:
                        return new byte[] { 0x90 };
                    case 2:
                        return new byte[] { 0x66, 0x90 };
                    case 3:
                        return new byte[] { 0x0F, 0x1F, 0 };
                    case 4:
                        return new byte[] { 0x0F, 0x1F, 0x40, 0 };
                    case 5:
                        return new byte[] { 0x0F, 0x1F, 0x44, 0, 0 };
                    case 6:
                        return new byte[] { 0x66, 0x0F, 0x1F, 0x44, 0, 0 };
                    case 7:
                        return new byte[] { 0x0F, 0x1F, 0x80, 0, 0, 0, 0 };
                    case 8:
                        return new byte[] { 0x0F, 0x1F, 0x84, 0, 0, 0, 0, 0 };
                    case 9:
                        return new byte[] { 0x66, 0x0f, 0x1f, 0x84, 0, 0, 0, 0, 0 };
                    case 10:
                        return new byte[] { 0x66, 0x2E, 0x0f, 0x1f, 0x84, 0, 0, 0, 0, 0 };
                    default:
                        // impossible, just to satisfy the compiler
                        throw new Exception();
                }
            }
            else
            {
                byte[] res = new byte[size];
                byte[] nop10 = new byte[] { 0x66, 0x2E, 0x0f, 0x1f, 0x84, 0, 0, 0, 0, 0 };
                int pos;
                for (pos = 0; pos < res.Length - 9; pos += 10)
                    Array.Copy(nop10, 0, res, pos, 10);
                byte[] tail = getNop(res.Length % 10);
                Array.Copy(tail, 0, res, pos, tail.Length);
                return res;
            }
        }
    }
}
