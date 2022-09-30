#!/usr/bin/env python

from chdrft.cmds import CmdsList
from chdrft.main import app
from chdrft.utils.cmdify import ActionHandler
from chdrft.utils.misc import Attributize
import chdrft.utils.misc as cmisc
import glog
import chdrft.utils.Z as Z
import numpy as np
from gnuradio import gr, blocks, filter
from gnuradio.filter import firdes
from chdrft.display.ui import GraphHelper
from chdrft.algo.viterbi import ConvViterbi, ConvEnc
import chdrft.algo.viterbi as viterbi
from chdrft.utils.swig import swig
import math
from gnuradio import digital
digital.pfb_clock_sync_ccf

try:
  m = swig.opa_math_common_swig
  c = swig.opa_crypto_swig
except:
  pass

global flags, cache
flags = None
cache = None

g_d = cmisc.Attr()
g_d.LRIT_RS_K = 223
g_d.LRIT_RS_N = 255
g_d.LRIT_VCDU_SIZE = 892
g_d.LRIT_CVCDU_SIZE = g_d.LRIT_VCDU_SIZE * g_d.LRIT_RS_N // g_d.LRIT_RS_K
g_d.LRIT_STRIDE = g_d.LRIT_CVCDU_SIZE + 4  # 4 for sync dword
g_d.LRIT_WHITEN_POLY = [1, 0, 0, 1, 0, 1, 0, 1, 1]  # 1,3,7,8
g_d.LRIT_WHITEN_INIT = [1] * 8
g_d.LRIT_SYNC = Z.np.array(Z.Format(Z.binascii.a2b_hex(b'1ACFFC1D')).bitlist(bitorder_le=0).v)
g_d.LRIT_POLY_A = [1, 1, 1, 1, 0, 0, 1]
g_d.LRIT_POLY_B = [1, 0, 1, 1, 0, 1, 1]
g_d.LRIT_TO_DUAL = [
    0x00, 0x7b, 0xaf, 0xd4, 0x99, 0xe2, 0x36, 0x4d, 0xfa, 0x81, 0x55, 0x2e, 0x63, 0x18, 0xcc, 0xb7,
    0x86, 0xfd, 0x29, 0x52, 0x1f, 0x64, 0xb0, 0xcb, 0x7c, 0x07, 0xd3, 0xa8, 0xe5, 0x9e, 0x4a, 0x31,
    0xec, 0x97, 0x43, 0x38, 0x75, 0x0e, 0xda, 0xa1, 0x16, 0x6d, 0xb9, 0xc2, 0x8f, 0xf4, 0x20, 0x5b,
    0x6a, 0x11, 0xc5, 0xbe, 0xf3, 0x88, 0x5c, 0x27, 0x90, 0xeb, 0x3f, 0x44, 0x09, 0x72, 0xa6, 0xdd,
    0xef, 0x94, 0x40, 0x3b, 0x76, 0x0d, 0xd9, 0xa2, 0x15, 0x6e, 0xba, 0xc1, 0x8c, 0xf7, 0x23, 0x58,
    0x69, 0x12, 0xc6, 0xbd, 0xf0, 0x8b, 0x5f, 0x24, 0x93, 0xe8, 0x3c, 0x47, 0x0a, 0x71, 0xa5, 0xde,
    0x03, 0x78, 0xac, 0xd7, 0x9a, 0xe1, 0x35, 0x4e, 0xf9, 0x82, 0x56, 0x2d, 0x60, 0x1b, 0xcf, 0xb4,
    0x85, 0xfe, 0x2a, 0x51, 0x1c, 0x67, 0xb3, 0xc8, 0x7f, 0x04, 0xd0, 0xab, 0xe6, 0x9d, 0x49, 0x32,
    0x8d, 0xf6, 0x22, 0x59, 0x14, 0x6f, 0xbb, 0xc0, 0x77, 0x0c, 0xd8, 0xa3, 0xee, 0x95, 0x41, 0x3a,
    0x0b, 0x70, 0xa4, 0xdf, 0x92, 0xe9, 0x3d, 0x46, 0xf1, 0x8a, 0x5e, 0x25, 0x68, 0x13, 0xc7, 0xbc,
    0x61, 0x1a, 0xce, 0xb5, 0xf8, 0x83, 0x57, 0x2c, 0x9b, 0xe0, 0x34, 0x4f, 0x02, 0x79, 0xad, 0xd6,
    0xe7, 0x9c, 0x48, 0x33, 0x7e, 0x05, 0xd1, 0xaa, 0x1d, 0x66, 0xb2, 0xc9, 0x84, 0xff, 0x2b, 0x50,
    0x62, 0x19, 0xcd, 0xb6, 0xfb, 0x80, 0x54, 0x2f, 0x98, 0xe3, 0x37, 0x4c, 0x01, 0x7a, 0xae, 0xd5,
    0xe4, 0x9f, 0x4b, 0x30, 0x7d, 0x06, 0xd2, 0xa9, 0x1e, 0x65, 0xb1, 0xca, 0x87, 0xfc, 0x28, 0x53,
    0x8e, 0xf5, 0x21, 0x5a, 0x17, 0x6c, 0xb8, 0xc3, 0x74, 0x0f, 0xdb, 0xa0, 0xed, 0x96, 0x42, 0x39,
    0x08, 0x73, 0xa7, 0xdc, 0x91, 0xea, 0x3e, 0x45, 0xf2, 0x89, 0x5d, 0x26, 0x6b, 0x10, 0xc4, 0xbf
]

g_d.LRIT_FROM_DUAL = [
    0x00, 0xcc, 0xac, 0x60, 0x79, 0xb5, 0xd5, 0x19, 0xf0, 0x3c, 0x5c, 0x90, 0x89, 0x45, 0x25, 0xe9,
    0xfd, 0x31, 0x51, 0x9d, 0x84, 0x48, 0x28, 0xe4, 0x0d, 0xc1, 0xa1, 0x6d, 0x74, 0xb8, 0xd8, 0x14,
    0x2e, 0xe2, 0x82, 0x4e, 0x57, 0x9b, 0xfb, 0x37, 0xde, 0x12, 0x72, 0xbe, 0xa7, 0x6b, 0x0b, 0xc7,
    0xd3, 0x1f, 0x7f, 0xb3, 0xaa, 0x66, 0x06, 0xca, 0x23, 0xef, 0x8f, 0x43, 0x5a, 0x96, 0xf6, 0x3a,
    0x42, 0x8e, 0xee, 0x22, 0x3b, 0xf7, 0x97, 0x5b, 0xb2, 0x7e, 0x1e, 0xd2, 0xcb, 0x07, 0x67, 0xab,
    0xbf, 0x73, 0x13, 0xdf, 0xc6, 0x0a, 0x6a, 0xa6, 0x4f, 0x83, 0xe3, 0x2f, 0x36, 0xfa, 0x9a, 0x56,
    0x6c, 0xa0, 0xc0, 0x0c, 0x15, 0xd9, 0xb9, 0x75, 0x9c, 0x50, 0x30, 0xfc, 0xe5, 0x29, 0x49, 0x85,
    0x91, 0x5d, 0x3d, 0xf1, 0xe8, 0x24, 0x44, 0x88, 0x61, 0xad, 0xcd, 0x01, 0x18, 0xd4, 0xb4, 0x78,
    0xc5, 0x09, 0x69, 0xa5, 0xbc, 0x70, 0x10, 0xdc, 0x35, 0xf9, 0x99, 0x55, 0x4c, 0x80, 0xe0, 0x2c,
    0x38, 0xf4, 0x94, 0x58, 0x41, 0x8d, 0xed, 0x21, 0xc8, 0x04, 0x64, 0xa8, 0xb1, 0x7d, 0x1d, 0xd1,
    0xeb, 0x27, 0x47, 0x8b, 0x92, 0x5e, 0x3e, 0xf2, 0x1b, 0xd7, 0xb7, 0x7b, 0x62, 0xae, 0xce, 0x02,
    0x16, 0xda, 0xba, 0x76, 0x6f, 0xa3, 0xc3, 0x0f, 0xe6, 0x2a, 0x4a, 0x86, 0x9f, 0x53, 0x33, 0xff,
    0x87, 0x4b, 0x2b, 0xe7, 0xfe, 0x32, 0x52, 0x9e, 0x77, 0xbb, 0xdb, 0x17, 0x0e, 0xc2, 0xa2, 0x6e,
    0x7a, 0xb6, 0xd6, 0x1a, 0x03, 0xcf, 0xaf, 0x63, 0x8a, 0x46, 0x26, 0xea, 0xf3, 0x3f, 0x5f, 0x93,
    0xa9, 0x65, 0x05, 0xc9, 0xd0, 0x1c, 0x7c, 0xb0, 0x59, 0x95, 0xf5, 0x39, 0x20, 0xec, 0x8c, 0x40,
    0x54, 0x98, 0xf8, 0x34, 0x2d, 0xe1, 0x81, 0x4d, 0xa4, 0x68, 0x08, 0xc4, 0xdd, 0x11, 0x71, 0xbd
]


def args(parser):
  clist = CmdsList()
  parser.add_argument('--n', type=int, default=10)
  parser.add_argument('--outfile', type=str)
  parser.add_argument('--origfile', type=str)
  parser.add_argument('--test-sync', action='store_true')
  parser.add_argument('--inputfile', type=str)
  ActionHandler.Prepare(parser, clist.lst, global_action=1)


class Interleaver:

  def __init__(self, order=None, grp=None, stride=None):
    self.order = order
    self.grp = grp
    self.stride = stride
    self.order_size = len(self.order)
    self.iorder = [0] * self.order_size
    for i, v in enumerate(self.order):
      self.iorder[v] = i

    self.blk_size = stride * self.order_size

  def proc(self, bitlist, rev=0):
    assert len(bitlist) % self.blk_size == 0
    res = [0] * self.blk_size
    pos = 0
    for blkpos in range(0, len(bitlist), self.blk_size):
      for i in range(0, self.stride, self.grp):
        for j in range(self.order_size):
          jp = self.iorder[j]
          for k in range(self.grp):
            curp = blkpos + jp * self.stride + i + k
            if rev:
              res[curp] = bitlist[pos]
            else:
              res[pos] = bitlist[curp]
            pos += 1
    return res


class SigType(Z.Enum):
  BPSK = 0
  QPSK = 1


class BaseBlock:

  def __init__(self, rev=0, name=None, in_sig=None, out_sig=None, sig=None):
    self.rev = rev
    self.name = name
    if rev: self.name = 'i_' + self.name
    if in_sig is None: in_sig = sig
    if out_sig is None: out_sig = sig
    if rev: in_sig, out_sig = out_sig, in_sig
    self.in_sig = cmisc.to_list(in_sig)
    self.out_sig = cmisc.to_list(out_sig)

  def feed(self, data, rev=None):
    if rev is None: rev = self.rev
    self._feed(data, rev)

  def _feed(self, data, rev):
    assert 0

  def get(self, n=None):
    assert 0

  def remaining(self):
    return 0

  def finalize(self):
    pass

  def to_gnuradio_block(self, **kwargs):

    a = self

    class blk(gr.basic_block):

      def __init__(self):
        #gr.basic_block.__init__(self, name="opa_xpsk", in_sig=[np.complex64], out_sig=[np.float64])
        gr.basic_block.__init__(self, name=a.name, in_sig=a.in_sig, out_sig=a.out_sig)
        self.sys = a

      def forecast(self, noutput_items, ninput_items_required):
        for i in range(len(ninput_items_required)):
          if self.sys.remaining() >= noutput_items:
            ninput_items_required[i] = 0
          else:
            ninput_items_required[i] = noutput_items

      def stop(self):
        print('STOOP ', self.name(), self.sys.remaining())
        self.sys.finalize()
        return super().stop()

      def general_work(self, input_items, output_items):
        try:
          in0 = input_items[0]
          out0 = output_items[0]

          self.sys.feed(in0)

          self.consume(0, len(in0))
          res = self.sys.get(len(out0))

          #if len(in0):
          #  self.consume(0, len(in0))
          #res = self.sys.get(len(out0))

          out0[:len(res)] = res
          self.produce(0, len(res))
          return gr.WORK_CALLED_PRODUCE
        except:
          Z.tb.print_exc()
          raise

    return blk()


class EasyBlock(BaseBlock):

  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self.data = None
    self.stats = []
    self.output = []
    self.need_more = False
    self.ts = 0
    self.new_ts = 0
    self.tot = None
    self.cur_output = None

  def get_buf(self, want, consume_ratio=None, consume_num=None):
    self.ts = self.new_ts
    if len(self.data) < want:
      self.need_more = True
      return
    cur = self.data[:want]

    if consume_num is not None: consume = consume_num
    elif consume_ratio is None: consume = want
    else: consume = int(want * consume_ratio)
    self.consume(consume)

    return cur

  def consume(self, n):
    self.data = self.data[n:]
    self.new_ts += n

  def unconsume(self, data):
    self.data = np.concatenate([data, self.data])
    self.new_ts -= len(self.data)

  def get_buf_ts(self, from_ts, to_ts, consume_to_ts):
    self.ts = self.new_ts
    if len(self.data) + self.ts < to_ts:
      self.need_more = True
      return
    assert from_ts >= self.ts and consume_to_ts >= self.ts, f'{from_ts} {to_ts} {self.ts} {consume_to_ts}'
    cur = self.data[from_ts - self.ts:to_ts - self.ts]
    self.consume(consume_to_ts - self.ts)
    return cur

  def _feed(self, data, rev, max_iter=-1):
    if self.data is None:
      self.data = np.array(data)
      self.tot = np.array(data)
    else:
      self.data = np.concatenate([self.data, data])
      self.tot = np.concatenate([self.tot, data])

    self.need_more = False
    while not self.need_more:
      self.cur_output = []
      if rev: res = self.iproc()
      else: res = self.proc()

      self.output.extend(self.cur_output)
      if res is not None:
        self.output.extend(res)
      max_iter -= 1
      if max_iter == 0: break

  def get(self, n=None):

    if n is None: n = len(self.output)
    else: n = min(n, len(self.output))
    res = self.output[:n]
    self.output = self.output[n:]
    return res

  def remaining(self):
    return len(self.output)

  def proc(self):
    assert 0


class ChainBlock(EasyBlock):

  def __init__(self, blocks, blk_size=512):
    self.blk_size = blk_size
    self.blocks = blocks
    super().__init__(name='chain', in_sig=blocks[0].in_sig, out_sig=blocks[-1].out_sig)

  def proc(self):
    cur = self.get_buf(self.blk_size)
    if cur is None: return
    for i, blk in enumerate(self.blocks):
      blk.feed(cur)
      cur = blk.get()
      if not cur: break
    return cur


class RSBlock(EasyBlock):

  def __init__(self, rs, **kwargs):
    super().__init__(name='rs', sig=np.int32, **kwargs)
    self.rs = rs
    self.nk = rs.n() - rs.k()
    self.rs_out_blk_size = self.rs.q() * self.rs.n()
    self.rs_in_blk_size = self.rs.q() * self.rs.k()
    self.rs_check_sym_blk_size = self.rs.q() * self.nk
    self.encode_blocks = []
    self.decode_blocks = []

    self.in_blk_size = self.rs_in_blk_size
    self.out_blk_size = self.rs_out_blk_size

  def proc(self):
    cur = self.get_buf(self.in_blk_size)
    if cur is None: return
    blks = Z.Format(cur.tolist()).bucket(self.rs_in_blk_size).v

    res = []
    check_syms = []
    for blk in blks:
      self.encode_blocks.append(np.array(blk))

      mx = m.pack_vector_gfq_u32(self.rs.field(), m.v_u32(blk))
      c = self.rs.encode(mx)
      cx = m.unpack_vector_gfq_u32(self.rs.field(), c)
      res.extend(cx[self.rs_check_sym_blk_size:])
      check_syms.extend(cx[:self.rs_check_sym_blk_size])

    res.extend(check_syms)
    res = self.interleaver.proc(res)
    return res

  def iproc(self):
    cur = self.get_buf(self.out_blk_size)
    if cur is None: return
    res = self.iproc_do_one(cur)
    assert res is not None
    return res.msg

  def iproc_do_one(self, cmsg, pos=0):

    cmsg = self.interleaver.proc(cmsg.tolist(), rev=1)

    syms_end = self.interleaver.order_size * self.rs_in_blk_size
    syms_groups = Z.Format(cmsg[:syms_end]).bucket(self.rs_in_blk_size).v
    check_syms_groups = Z.Format(cmsg[syms_end:]).bucket(self.rs_check_sym_blk_size).v

    errs = 0
    msg = []
    for syms, check_syms in zip(syms_groups, check_syms_groups):
      c = check_syms + syms

      cx = m.pack_vector_gfq_u32(self.rs.field(), m.v_u32(c))
      mx = self.rs.decode(cx)
      if not mx: return None
      mm = list(m.unpack_vector_gfq_u32(self.rs.field(), mx))
      self.decode_blocks.append(mm)
      errs += self.rs.nerrs()
      msg.extend(mm)
    glog.info(f'Iproc pos={pos}, errs={errs}')
    return cmisc.Attr(errs=errs, msg=msg, pos=pos)


class SyncBlock(EasyBlock):

  def __init__(self, sync_word, blk_size, **kwargs):
    super().__init__(name='sync', sig=np.int32, **kwargs)
    self.sync_word = sync_word
    self.blk_size = blk_size
    self.status = Status.ACQUISITION

  def proc(self):
    cur = self.get_buf(self.blk_size, consume_num=self.blk_size)
    if cur is None: return
    self.output.extend(self.sync_word)
    self.output.extend(cur)

  def iproc(self):
    if self.status == Status.ACQUISITION:
      print('acq')
      return self.iproc_acq()
    else:
      return self.iproc_track()

  def iproc_acq(self):
    cur = self.get_buf(len(self.sync_word), consume_num=0)
    if cur is None: return
    if np.all(cur == self.sync_word):
      self.status = Status.TRACKING
      return
    self.consume(1)

  def iproc_track(self):
    cur = self.get_buf(len(self.sync_word) + self.blk_size)
    if cur is None: return
    nsw = len(self.sync_word)
    sync = cur[:nsw]
    if np.all(sync != self.sync_word):
      print('lost sync')
      assert 0
    return cur[nsw:]


class HysteresisRT:

  def __init__(self, low, high, tlow, thigh, high_start=0):
    self.low = low
    self.high = high
    self.tlow = tlow
    self.thigh = thigh
    self.cur = [low, high][high_start]

  def proc(self, v):
    if self.cur == self.low and v > self.thigh:
      self.cur = self.high
    elif self.cur == self.high and v < self.tlow:
      self.cur = self.low
    return self.cur


class CCSDS(EasyBlock):

  def __init__(self, rs, whiten_seq, **kwargs):
    super().__init__(name='ccsds', sig=np.int32, **kwargs)
    self.rs = rs
    self.whiten_seq = whiten_seq
    self.out_blk_size = g_d.LRIT_CVCDU_SIZE * 8

  def proc(self):
    assert 0
    cur = self.get_buf(self.blk_size)
    if cur is None: return
    self.output.extend(sync_word)
    self.output.extend(cur)

  def iproc(self):
    cur = self.get_buf(self.out_blk_size)
    if cur is None: return
    print(bytes(Z.Format(cur).bin2byte().v))
    cur = self.whiten_seq ^ cur
    b = Z.Format(cur).bin2byte().v
    I = 4

    nb = []
    for x in b:
      nb.append(g_d.LRIT_FROM_DUAL[x])
    b = nb

    for i in range(I):
      tmp = b[i::I]
      cx = list(map(self.rs.field().import_base, tmp))
      mx = self.rs.decode(m.v_poly_u32(cx))
      mxx = list(map(self.rs.field().export_base, mx))
      assert mxx
      print('DO HAVE', len(mxx))
      self.output.extend(mxx)


class WhitenerBlock(EasyBlock):

  def __init__(self, seq, **kwargs):
    super().__init__(name='whiten', sig=np.int32, **kwargs)
    self.seq = np.array(seq)

  def proc(self):
    cur = self.get_buf(len(self.seq))
    if cur is None: return
    return cur ^ self.seq


class TimingSyncBlock(EasyBlock):

  def __init__(
      self,
      resamp_ratio=None,
      sps_base=None,
      sps_max_dev=0.1,
      sig=np.complex64,
      alpha=1 - 1e-6,
      **kwargs,
  ):
    super().__init__(name='timing_sync', sig=sig, **kwargs)
    self.resamp_ratio = resamp_ratio
    self.sps = SimpleIIR(alpha=alpha, v=sps_base, max_dev=sps_max_dev)
    self.diff_hyst = HysteresisRT(0, 1, 0.4, 0.6, 0)
    self.t = 0
    self.last = None

  def proc(self):
    diff = self.t - self.new_ts
    double = 0
    skip = 0
    if self.diff_hyst.proc(abs(diff)):
      if diff < 0:
        double = 1
        print('GOT DOUBLE')
      else:
        skip = 1
        print('GOT SKIP')

    target_num = self.sps.get() * self.resamp_ratio
    want = int(round(target_num))
    b = self.get_buf(want + skip - double)
    #print(skip, double, self.ts, self.sps.get(), diff, self.new_ts - self.ts, abs(diff), want)
    self.stats.append(
        cmisc.Attr(
            ts=self.ts,
            double=double,
            skip=skip,
            diff=diff,
            target_num=target_num,
        )
    )
    if b is None: return
    if skip: b = b[1:]
    if double:
      assert self.last is not None
      b = np.concatenate(([self.last], b))

    self.t += target_num
    self.last = b[-1]
    return b


class HeadBlock(EasyBlock):

  def __init__(self, nskip, sig=np.int32, **kwargs):
    super().__init__(name='head', sig=sig, **kwargs)
    self.nskip = nskip
    self.first = 1

  def proc(self):
    if self.first:
      cur = self.get_buf(self.nskip)
      if cur is None: return
      self.first = 0
      return []

    return self.get_buf(1)


class ConvolutionalBlock(EasyBlock):

  def __init__(self, conv_enc, **kwargs):
    super().__init__(name='conv', in_sig=np.int32, out_sig=np.float32, **kwargs)
    self.conv_enc = conv_enc
    self.status = Status.ACQUISITION
    self.dec = ConvViterbi(self.conv_enc, collapse_depth=20)
    self.acq_data = []

  def proc(self):
    cur = self.get_buf(self.conv_enc.k)
    if cur is None: return
    return self.conv_enc.next(cur)

  def iproc(self):
    if self.status == Status.ACQUISITION:
      return self.iproc_acq()
    else:
      return self.iproc_track()

  def iproc_acq(self):
    cur = self.get_buf(self.dec.collapse_depth * self.conv_enc.n)
    if cur is None: return
    self.acq_data.extend(cur)
    align = ConvViterbi.GuessAlign(
        self.acq_data, self.conv_enc, collapse_depth=self.dec.collapse_depth
    )
    if align is None: return
    print('Choosing align ', align)
    acq_data = self.acq_data[align:]
    self.unconsume(acq_data)
    self.status = Status.TRACKING

  def iproc_track(self):
    cur = self.get_buf(self.conv_enc.n)
    if cur is None: return
    return self.dec.feed(cur)


class SignalBuilder:

  def __init__(self, *args, **kwargs):
    self.r = Z.Range1D(*args, is_int=1)
    self.tb = np.zeros(len(self.r), **kwargs)

  def add(self, pos, data):
    r2 = self.r.make_new(pos, pos + len(data))
    r2 = r2.intersection(self.r)
    self.tb[(r2 - self.r.low).window] += data[(r2 - pos).window]

  def get(self):
    return self.tb

  def extract(self, pos, data):
    r2 = self.r.make_new(pos, pos + len(data))
    r2 = r2.intersection(self.r)
    return data[(r2 - pos).window]


class DynamicNumpyArray:

  def __init__(self, n):
    self.n = n
    self.buf = np.zeros((n,))
    self.p = 0

  def add(self, pos, data):
    assert pos + len(data) < self.p + self.n
    assert self.p <= pos

    diff = pos - self.p
    self.buf[diff:diff + len(data)] += data
    res = list(self.buf[:diff])
    if diff > 0:
      self.buf[:-diff] = self.buf[diff:]
      self.buf[-diff:] = 0
      self.p = pos
    return res


class PulsedSignalBuilder:

  def __init__(self, pulse, upsample_ratio):
    self.pulse = pulse
    self.upsample_ratio = upsample_ratio
    self.data = DynamicNumpyArray(2 * len(pulse))

  def add(self, pos, data):
    p = self.pulse * data
    px = round((pos * self.upsample_ratio) % self.upsample_ratio)
    sp = int(pos) + px // self.upsample_ratio
    return self.data.add(sp, p[px::self.upsample_ratio])

  def flush(self):
    return self.data.buf[:len(self.pulse)]


def build_signal(data, pulse, sps, typ=SigType.BPSK):
  assert isinstance(sps, int)
  pulse = np.array(pulse)
  off = sps // 2
  st = len(pulse) / 2
  #st, off=0, 0
  sb = SignalBuilder(st - off, st + off + sps * len(data))

  if typ == SigType.BPSK:
    data = data * 2 - 1
  else:
    assert 0

  for i, x in enumerate(data):
    sb.add(i * sps, pulse * x)

  return sb.get()


def build_signal_robust(data, pulse, sps, upsample_ratio, typ=SigType.BPSK):
  pulse = np.array(pulse)
  sb = PulsedSignalBuilder(pulse, upsample_ratio)

  if typ == SigType.BPSK:
    data = data * 2 - 1
  else:
    assert 0

  for i, x in enumerate(data):
    x = sb.add(i * sps, x)
    for v in x:
      yield v
  for v in sb.flush():
    yield v


class SimpleIIR:

  def __init__(self, alpha=0.9, minv=None, maxv=None, max_dev=None, v=np.nan, debug=0):
    self.minv = minv
    self.maxv = maxv
    self.alpha = alpha
    self.v = v
    self.debug = debug
    self.tb = []

    if max_dev is not None:
      self.minv = v - v * max_dev
      self.maxv = v + v * max_dev

  def reset(self, v):
    self.v = v

  def get_or(self, default):
    if self.has_value(): return self.v
    return default

  def get(self):
    assert self.has_value()
    return self.v

  def has_value(self):
    return not np.any(np.isnan(self.v))

  def push(self, x):
    if self.minv is not None: x = max(x, self.minv)
    if self.maxv is not None: x = min(x, self.maxv)
    if self.debug: self.tb.append(x)

    if self.has_value():
      self.v = np.polyadd(np.polymul(self.alpha, self.v), np.polymul((1 - self.alpha), x))
      if not isinstance(x, np.ndarray): self.v = self.v[0]
    else:
      self.v = x
    return self.v


def make_graph(g=None, edges=None, node=None, chain=None):
  if g is None:
    g = Z.nx.DiGraph()
  if edges is not None:
    g.add_edges_from_edges(edges)
  if node is not None:
    g.add_node(node)
  if chain:
    chain_normed = []
    for e in chain:
      if isinstance(e, (list, np.ndarray)): chain_normed.append(tuple(e))
      else: chain_normed.append(e)

    for i in range(len(chain_normed) - 1):
      g.add_edge(chain_normed[i], chain_normed[i + 1])
  assert g is not None
  return g


def run_block(g, src_data=None, out_type=np.complex64):

  src = None
  if src_data is not None:
    src = blocks.vector_source_c(src_data)

  tb = gr.top_block()
  snk = None
  if out_type == np.complex64:
    snk = blocks.vector_sink_c()
  elif out_type == np.int32:
    snk = blocks.vector_sink_i()
  elif out_type == np.float32:
    snk = blocks.vector_sink_f()
  elif out_type == np.bool8:
    snk = blocks.vector_sink_b()
  else:
    assert 0

  node_mapper = {}
  for node in g.nodes():
    if isinstance(node, BaseBlock):
      node_mapper[node] = node.to_gnuradio_block()
    elif isinstance(node, tuple):
      if isinstance(node[0], (np.int32, int)):
        node_mapper[node] = blocks.vector_source_i(node)
      elif isinstance(node[0], np.float32):
        node_mapper[node] = blocks.vector_source_f(node)
      elif isinstance(node[0], (np.complex64, np.complex128)):
        node = tuple(map(complex, node))
        node_mapper[node] = blocks.vector_source_c(node)
      elif isinstance(node[0], complex):
        node_mapper[node] = blocks.vector_source_c(node)
      elif isinstance(node[0], np.bool8):
        node_mapper[node] = blocks.vector_source_b(node)
      else:
        assert 0, type(node[0])

    else:
      node_mapper[node] = node

  for u, v in g.edges():
    tb.connect(node_mapper[u], node_mapper[v])

  for n in g.nodes():
    if g.in_degree(n) == 0 and src is not None:
      tb.connect(src, node_mapper[n])
    if g.out_degree(n) == 0:
      tb.connect(node_mapper[n], snk)

  tb.run()
  return np.array(snk.data())


class Status(Z.Enum):
  INIT = 0
  ACQUISITION = 1
  TRACKING = 2
  DONE = 3


class DbHelper:

  def ampl2db(x):
    return 20 * np.log10(np.abs(x))

  def var2db(x):
    return 10 * np.log10(np.abs(x))

  def db2ampl(x):
    return 10**(x / 20)

  def db2var(x):
    return 10**(x / 10)

  def sig_power(x):
    return DbHelper.ampl2db(np.linalg.norm(x) / len(x)**0.5)


def get_sum_distrib(std, n):
  return Z.stats.norm(loc=0, scale=std * (n**0.5))


class DopplerTracker:

  def __init__(self, acq=None, track=None):
    self.acq = acq
    self.track = track

  def get_space(self, cur=None, dt=None):
    if cur is None: return self.get_acq()
    return self.get_track(cur=cur, dt=dt)

  def get_acq(self):
    return self.acq

  def get_track(self, cur=None, dt=None):
    if isinstance(self.track, np.ndarray):
      return cur + self.track
    return cur + np.arange(
        -self.track.max_doppler_rate * dt, self.track.max_doppler_rate * dt, self.track.doppler_step
    )


class OpaXpsk(EasyBlock):

  def __init__(self, conf, **kwargs):
    super().__init__(name='xpsk', in_sig=np.complex64, out_sig=np.float32, **kwargs)
    self.conf = conf
    self.status = Status.INIT
    self.signal_data = Z.Attributize(power=0)
    self.pulse_shape = np.array(self.conf.pulse_shape)
    self.pulse_len = len(self.pulse_shape)
    self.half_pulse_len = self.pulse_len // 2
    self.sps_int = round(self.conf.sps)
    assert abs(self.conf.sps - self.sps_int) < 0.01, 'Your sps should be very close to an integer'
    self.pulse_t_track = np.arange(self.pulse_len +
                                   self.conf.jitter * 2) * (2j * np.pi / self.conf.sample_rate)

    self.acq_proc_size = int(
        2 * self.conf.sps * (1 + self.conf.n_acq_symbol) + len(self.pulse_shape)
    )
    self.pulse_t_acq = np.arange(self.acq_proc_size) * (2j * np.pi / self.conf.sample_rate)
    self.stats = []
    self.tracking_data = None
    self.acq_data = None
    self.need_more = None
    self.fixed_sps_mode = self.conf.get('fixed_sps_mode', 0)
    self.doppler_tracker = DopplerTracker(track=self.conf.track_doppler, acq=self.conf.acq_doppler)

  def ts_to_s(self, ts):
    return ts / self.conf.sample_rate

  def get_doppler_shift(self, sig, doppler):
    t_sig = np.arange(len(sig)) * (2j * np.pi / self.conf.sample_rate)
    return sig * np.exp(t_sig * doppler)

  def proc(self):
    nstatus = self.process()
    if nstatus is not None and nstatus != self.status:
      self.transition(nstatus)

  def process(self):
    if self.status == Status.INIT:
      return Status.ACQUISITION
    elif self.status == Status.ACQUISITION:
      acq_data = self.do_acq()
      if acq_data:
        self.acq_data = acq_data
        return Status.TRACKING
    elif self.status == Status.TRACKING:
      cur = self.do_track()
      if cur is None:
        return
      if not self.is_cnd_ok(cur):
        print('LOST ACQUISITION >> srn=', cur)
        if self.conf.get('fail_on_lost_track', 0):
          assert 0
        if not self.conf.get('ignore_on_lost_track', 0):
          return Status.ACQUISITION
    else:
      assert 0

  def transition(self, nstatus):
    self.status = nstatus
    if nstatus == Status.ACQUISITION:
      pass
    elif nstatus == Status.TRACKING:
      print('Start tracking at ', self.ts, self.new_ts)
      self.tracking_data = self.acq_data
      self.tracking_data.sps = self.conf.sps
      self.tracking_data.bits = []
      self.tracking_data.b0_phase_track = []

      if not self.fixed_sps_mode:
        self.sps_track.reset(self.conf.sps)
    else:
      assert 0

  def analyse_chip(self, freq, sig, acq=1, dbg=0, clear_sig_type2=0):
    if acq:
      doppler_pulse = self.get_doppler_shift(self.pulse_shape, -freq)
      doppler_angles = self.pulse_t_acq * freq
      shift_seq = np.exp(doppler_angles) * sig
      correl = Z.signal.correlate(shift_seq, self.pulse_shape, mode='valid')
      mod = len(correl) % self.conf.sps
      if mod != 0: correl = correl[:-mod]
      correl_max = np.max(np.abs(correl))
      m = np.reshape(correl, (-1, self.conf.sps))
      mquant, = Z.mquantiles(np.abs(m), prob=[0.1], axis=0)

      phase0 = np.argmax(mquant)
      phase = phase0
      qmax = mquant[phase0]

      skip = self.conf.skip_ratio * self.conf.sps
      startpos = phase + skip
      endpos = min(len(sig), phase + self.conf.sps * self.conf.n_acq_symbol - skip)
      rebuild_sig = SignalBuilder(startpos, endpos, dtype=np.complex128)

      unwrapped_phases = np.unwrap(np.angle(correl[phase::self.conf.sps]))
      b0_phase = unwrapped_phases[-1]
      phases = np.abs(np.diff(unwrapped_phases))
      rebuild_sig.add(phase, self.pulse_shape)
      cur = self.pulse_shape
      sign = 1
      data = [sign]
      for i in range(self.conf.n_acq_symbol - 1):
        if phases[i] > np.pi / 2: sign *= -1
        data.append(sign)
        rebuild_sig.add(phase + (i + 1) * self.conf.sps, sign * cur)

      typ2_sig = None
      if clear_sig_type2:
        typ2_sigbuilder = SignalBuilder(startpos, endpos, dtype=np.complex128)
        print(self.conf.sps * self.conf.n_acq_symbol, self.half_pulse_len)

        for i in range(self.conf.n_acq_symbol):
          cursig = Z.opa_struct.Range1D_int(phase + i * self.conf.sps, n=self.conf.sps).extract(sig)
          orth_sig = Z.geo_utils.make_orth_c(cursig, doppler_pulse)
          typ2_sigbuilder.add(phase + i * self.conf.sps, orth_sig)
        typ2_sig = typ2_sigbuilder.get()

      f = rebuild_sig.get()
      f = f / np.linalg.norm(f)
      s = rebuild_sig.extract(0, shift_seq)
      snorm = s / np.linalg.norm(s)
      signal_db = DbHelper.sig_power(s)

      sig_ampl = DbHelper.db2ampl(signal_db)
      distrib = get_sum_distrib(sig_ampl, len(self.conf.pulse_shape))
      prob_detection = 1 - (1 - distrib.cdf(qmax)) / 2

      noise = Z.geo_utils.make_orth_c(s, f)
      noise_db = DbHelper.sig_power(noise)

      if 0 and dbg:
        g = GraphHelper(run_in_jupyter=0)
        x = g.create_plot(plots=[])
        y = g.create_plot(plots=[])
        x.add_plot(np.abs(f), color='r')
        x.add_plot(np.abs(snorm), color='g')
        x.add_plot(np.abs(noise), color='b')
        y.add_plot(np.angle(f), color='r')
        y.add_plot(np.angle(snorm), color='g')
        g.run()

      res = cmisc.Attr(
          snr=signal_db - noise_db,
          noise_db=noise_db,
          signal_db=signal_db,
          data=data,
          phase=phase0,
          doppler=freq,
          doppler_phase=doppler_angles[phase0],
          qmax=qmax,
          correl_max=correl_max,
          typ2_sig=typ2_sig,
          prob_detection=prob_detection,
          b0_phase=SimpleIIR(alpha=self.conf.iir_b0_alpha,
                             v=b0_phase)  # TODO: use all bits to compute this
      )
      if dbg:
        res.dbg = cmisc.Attr(f=f, s=s, noise=noise, shift_seq=shift_seq, m=m, mquant=mquant)

      return res
    else:
      doppler_phase = self.tracking_data.doppler_phase + 2j * np.pi * self.dt * self.tracking_data.doppler
      doppler_angles = self.pulse_t_track * freq + doppler_phase
      shift_seq = np.exp(doppler_angles) * sig
      correl = Z.signal.correlate(shift_seq, self.pulse_shape, mode='valid')
      assert len(correl) == self.conf.jitter * 2 + 1
      mod = len(correl) % self.conf.sps
      signal_db = DbHelper.sig_power(shift_seq)
      noise_db = self.tracking_data.noise_db
      phase0 = np.argmax(np.abs(correl))
      ang_at = np.angle(correl[phase0])
      qmax = np.abs(correl[phase0])
      #print(phase0, np.abs(correl), self.sps_track.get())

      b0_phase = self.tracking_data.b0_phase.get()
      phase_diff = Z.dsp_utils.norm_angle(ang_at - b0_phase)

      noise_std = DbHelper.db2ampl(noise_db)
      correl_noise_std = np.sum(np.abs(self.pulse_shape)) * noise_std
      #print(correl_noise_var, abs(correl[phase0]))

      sig_ampl = DbHelper.db2ampl(signal_db)
      distrib = get_sum_distrib(sig_ampl, len(self.conf.pulse_shape))
      prob_detection = 1 - (1 - distrib.cdf(qmax)) / 2

      p = prob_detection
      nb0_phase = ang_at
      if abs(phase_diff) < np.pi / 2:
        p = 1 - p
      else:
        nb0_phase += np.pi
      nb0_phase = Z.dsp_utils.norm_angle_from(b0_phase, nb0_phase)

      return cmisc.Attr(
          snr=signal_db - noise_db,
          ang_at=ang_at,
          noise_db=noise_db,
          signal_db=signal_db,
          prob_detection=prob_detection,
          p=p,
          phase_diff=phase_diff,
          phase=phase0,
          b0_phase=b0_phase,
          nb0_phase=nb0_phase,
          doppler=freq,
          doppler_phase=doppler_angles[phase0],
      )

  def analyse_chip_doppler_space(self, data, doppler_space, **kwargs):
    cnds = []
    for doppler_shift in doppler_space:
      cnds.append(self.analyse_chip(doppler_shift, data, **kwargs))
    return max(cnds, key=lambda x: x.prob_detection)

  def is_cnd_ok(self, best):
    if self.conf.SNR_THRESH is not None and best.snr > self.conf.SNR_THRESH:
      return 1

    if self.conf.PROB_THRESH is not None and 1 - best.prob_detection < self.conf.PROB_THRESH:
      return 1
    return 0

  def do_acq(self):
    proc_ratio = 2
    want_data = self.acq_proc_size
    #cur = self.get_buf(want_data, 1 / proc_ratio)
    cur = self.get_buf(want_data, consume_num=0)
    if cur is None: return

    doppler_space = self.doppler_tracker.get_space()

    best = self.analyse_chip_doppler_space(cur, doppler_space)
    self.consume(best.phase + self.conf.sps * self.conf.n_acq_symbol - self.conf.jitter)
    self.stats.append(dict(type='acq', best=best))
    print('TRACK BEST ', best)
    best.chip_phase = self.ts + best.phase
    best.chip_start = self.ts
    if self.is_cnd_ok(best): return best

  def do_track(self):
    last_chip_phase = self.tracking_data.chip_phase
    while True:
      chip_start = last_chip_phase + self.tracking_data.sps - self.conf.jitter
      if chip_start >= self.new_ts:
        break
      last_chip_phase += self.sps_int

    chip_end = chip_start + self.pulse_len + 2 * self.conf.jitter
    consume_ts = chip_start + self.tracking_data.sps - self.conf.jitter
    cur = self.get_buf_ts(chip_start, chip_end, consume_ts)
    if cur is None:
      return

    self.dt = self.ts_to_s(chip_start - self.tracking_data.chip_phase)
    doppler_space = self.doppler_tracker.get_space(self.tracking_data.doppler, self.dt)
    best = self.analyse_chip_doppler_space(cur, doppler_space, acq=0)
    if self.conf.get('debug_output', 0):
      print('ANALYSE >> ', len(self.tracking_data.bits), best)

    nphase = chip_start + best.phase
    if not self.fixed_sps_mode:
      #print(self.sps_track.get(), nphase - self.tracking_data.chip_phase)

      self.sps_track.push(nphase - last_chip_phase)
      best.sps_cur = nphase - last_chip_phase
      best.sps_track = self.sps_track.get()
    best.ts = nphase

    # small timing corrections are made with SyncBlock
    self.tracking_data.chip_phase = last_chip_phase + self.sps_int
    self.tracking_data.doppler = best.doppler
    self.tracking_data.doppler_phase = best.doppler_phase

    self.stats.append(dict(type='track', best=best))
    self.tracking_data.bits.append(best.p)
    self.tracking_data.b0_phase_track.append(self.tracking_data.b0_phase.push(best.nb0_phase))
    self.cur_output.append(best)

    return best

  def finalize(self):
    if self.tracking_data:
      bits = np.array(self.tracking_data.bits)
      self.tracking_data.processed_bits = np.abs(np.diff(np.unwrap(np.angle(bits))))


def test(ctx):
  psk_freq = 1691e6
  psk_bw = 660e3

  bpsk_rate = 293883
  sps = 8

  if flags.inputfile == 'opensat_data':
    sample_rate = 1250000
    center_freq = 1691e6
    data = Z.DataFile(
        cmisc.path_here(
            #'/home/benoit/programmation/dsp/data/gqrx_20161202_145551_1691000000_1250000_fc_segment0.raw'
            '/home/benoit/programmation/dsp/data/opensat.c64'
        ),
        typ='complex64',
        samp_rate=sample_rate
    )
    ds = data.to_ds()
    ds = ds[sps * 5000:]
  else:
    sample_rate = 8.738133e6
    center_freq = 1692e6
    data = Z.DataFile(
        cmisc.path_here('../../../dsp/data/goes_small.dat'), typ='complex8', samp_rate=sample_rate
    )
    ds = data.to_ds()
    ds = ds.apply('y / 128')
  xt = np.exp(ds.x * 1j * 2 * Z.pi * -(psk_freq - center_freq))

  ds = ds.apply('y * xt')

  ds = ds[:sample_rate]
  sample_rate_proc = bpsk_rate * sps
  pulse_shape = np.array(
      firdes.root_raised_cosine(1, sample_rate_proc, bpsk_rate, 0.5,
                                int(sps * 15) + 1)
  )

  taps = firdes.low_pass(1.0, sample_rate, psk_bw * 0.7, psk_bw / 2, firdes.WIN_HAMMING, 6.76)
  fil = filter.fir_filter_ccc(1, taps)
  resampler = filter.mmse_resampler_cc(0, sample_rate / sample_rate_proc)
  print(sample_rate / sample_rate_proc)
  tmp = ds.y.astype(np.complex64)
  #tmp = list(map(complex, ds.y))

  ny = run_block(make_graph(chain=[tuple(ds.y), fil, resampler]), out_type=np.complex64)
  ds = ds.make_new('prepare', y=ny, samp_rate=sample_rate_proc)
  print(ds.y.shape)

  if 0:
    water = Z.DataOp.FreqWaterfall_DS(ds, 2**13)

    resx = water.new_y(Z.dsp_utils.g_fft_helper.map(water.y, quant=0.01))
    g = GraphHelper(run_in_jupyter=0)
    g.create_plot(images=[resx.transpose()])
    g.create_plot(plots=[np.abs(ds.y)])
    g.create_plot(plots=[np.unwrap(np.angle(ds.y))])
    g.run()
    return

  if 0:
    correl = Z.signal.correlate(ds.y, pulse_shape, mode='valid')
    g = GraphHelper(run_in_jupyter=0)
    g.create_plot(plots=[np.abs(correl)])
    g.create_plot(plots=[np.angle(correl)])
    g.run()
    return

  conf = cmisc.Attributize(
      sample_rate=sample_rate_proc,
      sps=sps,
      symbol_rate=bpsk_rate,
      pulse_shape=pulse_shape,
      n_acq_symbol=30,
      max_doppler_shift=0,
      doppler_space_size=1,
      SNR_THRESH=5,
  )

  if flags.test_sync:
    # WONT WORK, convcode !
    sig_sync = build_signal(g_d.LRIT_SYNC, pulse_shape, sps)
    g = GraphHelper(run_in_jupyter=0)
    x = g.create_plot(plots=[])

    correl = Z.signal.correlate(sig_sync, ds.y, mode='valid')
    cabs = np.abs(correl)
    obj = x.add_plot(cabs)
    g.run()
    return

  ds.to_file('/tmp/data.filter.out', typ='complex64')
  xpsk = OpaXpsk(conf)
  print(len(ds.y))
  xpsk.feed(ds.y[:10 * 2 * 8000 * sps])
  if 0:
    g = GraphHelper(run_in_jupyter=0)
    g.create_plot(plots=[np.array(xpsk.tracking_data.b0_phase_track)])
    g.run()
    return

  print(len(xpsk.output))

  res = dict(output=xpsk.output, conf=conf)
  Z.pickle.dump(res, open('/tmp/data.pickle', 'wb'))


def plot_correl(ctx):
  a = np.array(Z.pickle.load(open('/tmp/data.pickle', 'rb'))['output'])
  a = a - np.mean(a)

  b = Z.DataFile('/tmp/data.bin', typ=np.int8).to_ds()
  b = b.apply('y / 128').mean().y
  g = GraphHelper(run_in_jupyter=0)
  x = g.create_plot(plots=[])
  correl = Z.signal.correlate(b, a, mode='valid')
  cabs = np.abs(correl)
  obj = x.add_plot(cabs)
  g.run()


def test2(ctx):
  orig_sps = 32
  sps = 8
  bpsk_rate = 32e3
  samp_rate_orig = orig_sps * bpsk_rate
  samp_rate = sps * bpsk_rate

  pulse_shape = firdes.root_raised_cosine(1, samp_rate, bpsk_rate, 1, int(sps * 14) + 1)

  ds = Z.DataFile('/tmp/abc.bin', samp_rate=samp_rate, typ='complex64').to_ds()
  if 0:
    g = GraphHelper(run_in_jupyter=0)
    x = g.create_plot(plots=[])
    cp = Z.ColorPool()

    for shift in (0, 0.5, 1):
      if 1:
        resampler = filter.mmse_resampler_cc(shift, orig_sps / sps)
        ds_v = ds.make_new(
            'prepare',
            y=run_block(make_graph(node=resampler), ds.y[:10000])[145:290],
            samp_rate=samp_rate
        )
      else:
        ds_v = ds.make_new('prepare', y=ds.y[:20000], samp_rate=samp_rate)

      correl = Z.signal.correlate(pulse_shape, ds_v.y, mode='valid')
      if 1:
        cabs = np.abs(correl)
      else:
        cabs = np.abs(ds.y[:20000])
      obj = x.add_plot(cabs, symbol='o', color=cp.get_rgb() / 255)
      break
    g.run()
    return

  else:
    resampler = filter.mmse_resampler_cc(0, orig_sps / sps)
    ds = ds.make_new('prepare', y=run_block(make_graph(node=resampler), ds.y), samp_rate=samp_rate)

  conf = cmisc.Attributize(
      sample_rate=samp_rate,
      sps=sps,
      symbol_rate=bpsk_rate,
      pulse_shape=pulse_shape,
      n_acq_symbol=20,
      max_doppler_shift=0,
      doppler_space_size=1,
      SNR_THRESH=10,
  )

  enc = get_enc()
  sol_bits = Z.DataFile('/tmp/out1.dat', typ=np.int32).to_ds().y
  orig_bits = Z.DataFile('/tmp/orig1.dat', typ=np.int32).to_ds().y
  gen_data0 = enc.get_response(orig_bits)
  print(sol_bits[:100])

  xpsk = OpaXpsk(conf)
  xpsk.feed(ds.y, max_iter=1000)

  bits_soft = np.array(xpsk.tracking_data.bits)
  bits_hard = bits_soft >= 0.5

  print(Z.min_substr_diff(sol_bits, bits_hard, mod_req=2), len(bits_hard))

  #bits_soft = np.clip(10.5*(bits_soft - 0.5) + 0.5, 0, 1)
  #print(Z.min_substr_diff(gen_data, bits_hard[:200]))
  gen_data = enc.get_response(orig_bits[:58 // 2])

  if 0:
    print(bits_soft)
    bits = list(bits_soft >= 0.5)
    print(bits)
    s2 = ''.join(map(lambda x: str(int(x)), bits))
    s3 = ''.join(map(lambda x: str(int(not x)), bits))
    s1 = ''.join(map(str, sol_bits))
    print(s1.find(s2))
    print(s1.find(s3))

  #for tgt in (bits_hard, bits_soft):
  for tgt in (bits_soft,):
    print()
    print('GOOG run ')
    res = list(ConvViterbi.Solve(tgt, enc, collapse_depth=50, fix_align=None))
    res = np.ravel(res)
    print(res)
    print(Z.min_substr_diff(orig_bits, res), len(res))


class LRIT:

  def __init__(self):
    init_state = m.Poly_u32(m.cvar.PR_GF2, m.v_u32(g_d.LRIT_WHITEN_INIT))
    poly = m.Poly_u32(m.cvar.PR_GF2, m.v_u32(g_d.LRIT_WHITEN_POLY))
    lfsr = c.LFSR_u32(m.cvar.GF2, init_state, poly)
    self.whiten_seq = [lfsr.get_next_non_galois2() for _ in range(g_d.LRIT_CVCDU_SIZE * 8)]

    chk_pw, gamma_pw = 112, 11
    pr_gf2 = m.cvar.PR_GF2
    px = m.cvar.PR_GF2.import_base(0x187)
    self.gf_256 = m.GF_q_u32(m.cvar.GF2, px)
    assert self.gf_256.is_prim_elem(self.gf_256.theta())
    self.gf_256.set_prim_elem(self.gf_256.theta())
    self.rs = m.RSCode_u32(self.gf_256, g_d.LRIT_RS_N, g_d.LRIT_RS_K)
    gamma = self.gf_256.faste(self.gf_256.theta(), gamma_pw)
    self.rs.advanced_init(gamma, chk_pw)

    self.interleaver_order = (0, 1, 2, 3)

    mx = [[g_d.LRIT_POLY_A, g_d.LRIT_POLY_B]]
    my = None
    self.conv_enc = ConvEnc(mx, my)

    self.rs_block = RSBlock(self.rs)
    self.irs_block = RSBlock(self.rs)

    self.conv_block = ConvolutionalBlock(self.conv_enc)
    self.iconv_block = ConvolutionalBlock(self.conv_enc, rev=1)

    self.ccsds = CCSDS(self.rs, self.whiten_seq)
    self.iccsds = CCSDS(self.rs, self.whiten_seq, rev=1)
    self.isync = SyncBlock(g_d.LRIT_SYNC, g_d.LRIT_CVCDU_SIZE * 8, rev=1)


def gen_data(ctx):

  assert ctx.n
  enc = LRIT().conv_enc
  np.random.seed(2)
  data = np.random.randint(2, size=ctx.n)
  res = []
  for b in data:
    res.extend(enc.next([b]))
  Z.Dataset(data).to_file(ctx.origfile, 'int32')
  Z.Dataset(res).to_file(ctx.outfile, 'int32')


def gen1(ctx):
  enc = LRIT().conv_enc
  np.random.seed(2)
  data = np.random.randint(2, size=20)
  res = []
  for b in data:
    x = enc.next([b])
    print(x)
    for v in x:
      res.append((v, 0.7))
  #res = np.ravel(res)
  res = Z.Format(res).bucket(2).v[3:]
  #res= Z.Format(res[1:-1]).bucket(2).v
  print(res)

  dec = ConvViterbi(enc, collapse_depth=10)
  for o in res:
    dec.setup_next(o)
    print(dec.states.keys(), dec.pos)
    print(
        'likelyhood',
        max(dec.states[dec.pos].states.values(), key=lambda x: x.likelyhood).likelyhood
    )

  got = []
  for k, v in dec.output.items():
    if v.action is not None: got.append(v.action[0])

    print(k, v.action)

  print('ans', list(data))
  print('got', got)
  print(enc.n, enc.k)


def test_enc(ctx):
  enc = LRIT().conv_enc
  np.random.seed(2)
  data = np.random.randint(2, size=200)

  res = enc.get_response(data)
  #res[100] ^= 1
  #res[50] ^= 1
  #res[150] ^= 1
  res = (res - 0.5) / 3.1 + 0.5

  enc.reset()
  states = []
  for i in range(len(data)):
    states.append(viterbi.State((enc.il, enc.ol)).key)
    enc.next(data[i])
  states.append(viterbi.State((enc.il, enc.ol)).key)

  dec = list(ConvViterbi.Solve(res, enc, collapse_depth=10, fix_align=0, ans_states=states))
  dec = np.ravel(dec)
  print(dec)
  print(data)

  print(Z.min_substr_diff(data, dec), len(dec))


def test_full_fec(ctx):
  np.random.seed(1)
  lrit = LRIT()

  data = np.random.randint(2, size=g_d.LRIT_VCDU_SIZE * 8 * 10).astype(np.int32)
  print(len(data))

  g = make_graph(
      chain=[
          tuple(data),
          lrit.rs_block,
          lrit.conv_block,
          HeadBlock(g_d.LRIT_CVCDU_SIZE * 8 * 2 - 40, sig=np.float32),
          lrit.iconv_block,
          lrit.irs_block,
      ]
  )
  #g = make_graph(chain=[tuple(data), lrit.rs_block, HeadBlock(g_d.LRIT_CVCDU_SIZE*8-10), lrit.irs_block])
  #g = make_graph(chain=[tuple(data), lrit.rs_block, HeadBlock(0), lrit.irs_block])
  #g = make_graph(chain=[tuple(data), lrit.rs_block])
  if 0:
    data = np.random.randint(2, size=10000).astype(np.int32)
    g = make_graph(
        chain=[tuple(data), lrit.conv_block,
               HeadBlock(2, sig=np.float32), lrit.iconv_block]
    )

  res = run_block(g, out_type=np.int32)
  print(len(res), len(data))

  print(Z.min_substr_diff(data, res, mod_req=g_d.LRIT_VCDU_SIZE * 8), len(data))
  Z.create_kernel()


def test_fec(ctx):
  init_state = [1] * 8
  poly = [1, 0, 0, 1, 0, 1, 0, 1, 1]  # 1,3,7,8
  init_state = m.Poly_u32(m.cvar.PR_GF2, m.v_u32(init_state))
  poly = m.Poly_u32(m.cvar.PR_GF2, m.v_u32(poly))
  lfsr = c.LFSR_u32(m.cvar.GF2, init_state, poly)
  lst = [lfsr.get_next_non_galois2() for _ in range(256)]
  print(Z.Format(lst).bit().v)
  seq = Z.Format('01001000000011101100000010011010').bit2list().v
  assert lfsr.init_from_seq(m.cvar.GF2, seq)
  print(lfsr.get_poly().to_vec())


def test_lrit(ctx):
  conf = Z.pickle.load(open('/tmp/data.pickle', 'rb'))
  ds = Z.DataFile('/tmp/data.filter.out', typ='complex64', samp_rate=conf.sample_rate).to_ds()
  xpsk = OpaXpsk(conf)
  g = make_graph(chain=[tuple(ds.y[:10000]), xpsk])
  res = run_block(g, out_type=np.float32)


def try_dec_lrit(output):
  lrit = LRIT()
  res = list(ConvViterbi.Solve(output, lrit.conv_enc, collapse_depth=30))
  pos = 0
  while True:
    diff, bestpos = Z.min_substr_diff(res[pos:], g_d.LRIT_SYNC)
    print('GOT BEST ', diff, bestpos, len(res[pos:]))
    if diff != 0: break
    pos += bestpos + 1

  if 1:
    res = Z.pickle.dump(res, open('/tmp/data.unconv.pickle', 'wb'))
    return

  Z.create_kernel()
  return
  u = lrit.irs_block.feed(res)
  print(lrit.irs_block.decode_blocks)
  Z.create_kernel()


def test_dec_lrit(ctx):
  u = Z.pickle.load(open('/tmp/data.pickle', 'rb'))
  output = u['output']
  conf = u['conf']
  output = np.array(output)
  if 0:
    output = np.abs(output - 0.5)
    g = GraphHelper(run_in_jupyter=0)
    g.create_plot(plots=[output])
    g.run()
    return
  output = np.clip((output - 0.5) * 3.9 + 0.5, 0, 1)
  output = 1 - output
  print(len(output))
  try_dec_lrit(output[30000:70000])
  return


def test_dec_lrit2(ctx):
  ds = Z.DataFile('/tmp/data.bin', typ=np.int8).to_ds()
  ds = ds.apply('y / 128')
  output = (ds.y + 1) / 2
  #output = np.array(output)
  output = np.clip((output - 0.5) * 1.5 + 0.5, 0, 1)
  try_dec_lrit(output)


def test_dec_lrit3(ctx):
  lrit = LRIT()
  res = Z.pickle.load(open('/tmp/data.unconv.pickle', 'rb'))
  res = (np.array(res)).tolist()
  #print(Z.min_substr_diff(res, g_d.LRIT_SYNC))
  #return
  tmp = res[4508:]
  print('have', len(tmp))
  print(len(tmp))
  #tmp = np.array(tmp, dtype=np.int32)
  #lrit.ccsds.feed(tmp, rev=1)
  ny = run_block(make_graph(chain=[tuple(tmp), lrit.isync, lrit.iccsds]), out_type=np.int32)


def test_rs(ctx):
  data = b'404aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa0c5148ac8c4a863cd92807ffb898a146e1a2fc86df766125ff25751c58536030'

  data = b'421eb7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b78973b2624c1e105a46de19bde6c827976547dd109344a09bbd9b24744fdf6cd3'

  data = Z.binascii.a2b_hex(data)

  chk_pw, gamma_pw = 112, 11
  u = Z.swig_unsafe.opa_wrapper_swig.CReedSolomon(8, 16, chk_pw, gamma_pw, 0, 1, 0, 1)
  res, mx = u.Decode(data)
  print(res, mx)
  res, mx = u.Decode(data)
  print(res, mx)

  lrit = LRIT()
  cx = list(map(lrit.irs_block.rs.field().import_base, data))
  mx = lrit.irs_block.rs.decode(m.v_poly_u32(cx))
  mxx = list(map(lrit.irs_block.rs.field().export_base, mx))
  print(mxx)


def test_resamp(ctx):
  g = GraphHelper(run_in_jupyter=0)
  x = g.create_plot(plots=[])
  cp = Z.ColorPool()

  f = 100
  sample_rate = 2 * f
  nper = 5
  sig = np.sin(2 * np.pi * np.linspace(0, nper, sample_rate * nper), dtype=np.float32)

  for shift in np.linspace(0, 1, 10):
    resampler = filter.mmse_resampler_ff(shift, sample_rate / 8)
    resampler2 = filter.mmse_resampler_ff(shift, 1)
    y = run_block(make_graph(chain=(sig, resampler, resampler2)), out_type=np.float32)
    print(y)

    obj = x.add_plot(y, symbol='o', pen=None, symbolPen={'color': cp.get_rgb() / 255})
  g.run()
  return


def test_sync(ctx):
  a = np.arange(1000, dtype=np.float32)
  blk = TimingSyncBlock(resamp_ratio=4, sps_base=4.1, sig=np.float32)
  y = run_block(make_graph(chain=(a, blk)), out_type=np.float32)
  g = GraphHelper(run_in_jupyter=0)
  x = g.create_plot(plots=[y])
  g.run()


def gen_bpsk(ctx):
  np.random.seed(3)
  data = np.random.randint(2, size=ctx.n)
  samp_rate = 1e6
  bpsk_rate = 1e5 * 1.0001
  sps = samp_rate / bpsk_rate
  upsample_ratio = 1
  pulse_shape_0 = firdes.root_raised_cosine(
      1, samp_rate * upsample_ratio, bpsk_rate, 0.5,
      int(upsample_ratio * sps * 50) + 1
  )
  y = build_signal_robust(data, pulse_shape_0, sps, upsample_ratio)
  yx = build_signal(data, pulse_shape_0, round(sps))
  y = np.array(list(y))
  pulse_shape = firdes.root_raised_cosine(1, samp_rate, bpsk_rate, 0.5, int(sps * 15) + 1)

  rem = len(pulse_shape_0) // 2 - len(pulse_shape) // 2
  y = y[rem:-rem]

  if 0:
    g = GraphHelper(run_in_jupyter=0)
    x = g.create_plot()
    obj = x.add_plot(y, symbol='o')
    #obj = x.add_plot(pulse_shape, symbol='o')
    obj = x.add_plot(yx)
    g.run()
    return

  conf = cmisc.Attributize(
      sample_rate=samp_rate,
      sps=round(sps),
      symbol_rate=bpsk_rate,
      pulse_shape=pulse_shape,
      n_acq_symbol=20,
      max_doppler_shift=0,
      doppler_space_size=1,
      SNR_THRESH=10,
  )

  y0 = y
  y = y.astype(np.complex64)
  print(len(y))
  tsync = TimingSyncBlock(resamp_ratio=upsample_ratio, sps_base=sps, sig=np.complex64)
  xpsk = OpaXpsk(conf, fixed_sps_mode=0)
  xpsk.sps_track = tsync.sps

  #y = run_block(make_graph(chain=(y, tsync)), out_type=np.complex64)
  #y = run_block(make_graph(chain=(y, tsync, xpsk)), out_type=np.float32)

  xblk = ChainBlock([xpsk], blk_size=int(sps * 100))
  #xblk = ChainBlock([tsync, xpsk], blk_size=int(sps*100))
  y = run_block(make_graph(chain=(y, xblk)), out_type=np.float32)
  #y = run_block(make_graph(chain=(y, xpsk)), out_type=np.float32)

  y = np.where(y > 0.5, 1, 0)
  correl = np.abs(Z.dsp_utils.compute_normed_correl(data, y))
  alx = np.argmax(correl)

  print(xpsk.sps_track.tb)
  print(Z.min_substr_diff(data.tolist(), y.tolist()), len(y))
  print(Z.min_substr_diff(data.tolist(), (1 - y).tolist()), len(y))
  print(xpsk.sps_track.get())

  if 1:
    g = GraphHelper(run_in_jupyter=0)
    x = g.create_plot()
    x.add_plot(correl, symbol='o')
    x.add_plot(Z.dsp_utils.compute_normed_correl(data, y), symbol='o')
    x = g.create_plot()
    x.add_plot(y)
    x.add_plot(data[alx:])
    x.add_plot(1 - y + 3)
    x.add_plot(data[alx:] + 3)
    g.run()
    return

  if 0:
    g = GraphHelper(run_in_jupyter=0)
    x = g.create_plot()
    obj = x.add_plot(y + 2, symbol='o')
    obj = x.add_plot(data, symbol='o')
    g.run()


def main():
  ctx = Attributize()
  ActionHandler.Run(ctx)


app()
