#!/usr/bin/env python

from chdrft.cmds import CmdsList
from chdrft.main import app
from chdrft.utils.cmdify import ActionHandler
from chdrft.utils.misc import Attributize
import chdrft.utils.misc as cmisc
import glog
import chdrft.utils.Z as Z
import numpy as np

global flags, cache
flags = None
cache = None

dsp_swig = Z.swig.opa_dsp_swig
Satellites = dsp_swig.Satellites.GetSingleton()

kGPS_CHIP_FREQ = 1023000
kGPS_CHIPS_PER_PRN = 1023
kGPS_PRN_FREQ = kGPS_CHIP_FREQ // kGPS_CHIPS_PER_PRN
kGPS_SAMPLE_PER_CHIP = 4
kGPS_SAMPLE_FREQ = kGPS_SAMPLE_PER_CHIP * kGPS_CHIP_FREQ
kGPS_SAMPLE_PER_PRN = kGPS_CHIPS_PER_PRN * kGPS_SAMPLE_PER_CHIP
kGPS_GCODE_FREQ = 50
kGPS_FILTER_MATCH_N_TILE = 20
kGPS_NUM_PRN_PERIOD_PER_GBIT = kGPS_PRN_FREQ // kGPS_GCODE_FREQ

kBINARY_SD = 1  # standard deviation 1 for of {-1, 1}, equiprobable
kSIGNAL_POWER_NOSAT_TRACK = Z.stats.norm.ppf(
    1 - 1e-5
) * kGPS_CHIPS_PER_PRN**0.5 * kGPS_SAMPLE_PER_CHIP * (kBINARY_SD * 1.5)

kSIGNAL_POWER_NOSAT_ACQ = kSIGNAL_POWER_NOSAT_TRACK * kGPS_FILTER_MATCH_N_TILE


def args(parser):
  clist = CmdsList().add(test)
  ActionHandler.Prepare(parser, clist.lst, global_action=1)


class Status(Z.Enum):
  INIT = 0
  ACQUISITION = 1
  TRACKING = 2


def get_l1ca(sid):
  b = Satellites.get(sid)
  lst = []
  for i in range(dsp_swig.PRN_PERIOD):
    lst.append(b.get_l1(i))
  return np.array(lst)


class OpaGpsSatellite:

  def __init__(self, sid, sats):
    self.seq_track = np.repeat(get_l1ca(sid), kGPS_SAMPLE_PER_CHIP) * 2 - 1
    self.seq_acq = np.tile(self.seq_track, kGPS_FILTER_MATCH_N_TILE)
    self.sid = sid
    self.data = np.array([])
    self.status = Status.INIT
    self.sats = sats
    self.signal_data = Z.Attributize(power=0)
    self.pts_acq = np.arange(len(self.seq_acq)) * (2j * np.pi / kGPS_SAMPLE_FREQ)
    self.pts_track = np.arange(len(self.seq_track)) * (2j * np.pi / kGPS_SAMPLE_FREQ)
    self.stats = []
    self.ts = 0
    self.new_ts = 0
    self.tracking_data = None
    self.acq_data = None

  def get_buf(self, want, consume_ratio):
    self.ts = self.new_ts
    if len(self.data) < want: return
    cur = self.data[:want]
    consume = int(want * consume_ratio)
    self.data = self.data[consume:]
    self.new_ts += consume
    return cur

  def get_buf_ts(self, from_ts, to_ts, consume_to_ts):
    self.ts = self.new_ts
    if len(self.data) + self.ts < to_ts: return
    assert from_ts >= self.ts, f'{from_ts} {to_ts} {self.ts}'
    cur = self.data[from_ts - self.ts:to_ts - self.ts]
    self.data = self.data[consume_to_ts - self.ts:]
    self.new_ts = consume_to_ts

    return cur

  def feed(self, data):
    self.data = np.concatenate([self.data, data])
    while True:
      nstatus = self.process()
      if nstatus is None or nstatus == self.status:
        break
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
      if not self.do_track():
        pass
        #return Status.ACQUISITION
    else:
      assert 0

  def transition(self, nstatus):
    self.status = nstatus
    if nstatus == Status.ACQUISITION:
      pass
    elif nstatus == Status.TRACKING:
      self.tracking_data = Z.Attributize(
          doppler=self.acq_data.doppler,
          power=self.acq_data.power,
          chip_phase=self.acq_data.chip_phase,
          bits=[],
      )
    else:
      assert 0

  def analyse_chip(self, freq, data, acq=1):
    if acq:
      pts, seq = self.pts_acq, self.seq_acq
    else:
      pts, seq = self.pts_track, self.seq_track
    shift_seq = np.exp(pts * freq) * seq
    correl = Z.signal.correlate(data, shift_seq, mode='valid')
    correl_abs = np.abs(correl)
    phase = np.argmax(correl_abs)
    power = correl_abs[phase]
    ipower = correl[phase]
    return power, freq, phase, ipower

  def analyse_chip_doppler_space(self, data, doppler_space, **kwargs):
    cnds = []
    for doppler_shift in doppler_space:
      cnds.append(self.analyse_chip(doppler_shift, data, **kwargs))
    return max(cnds)

  def do_acq(self):
    want_data = kGPS_SAMPLE_PER_PRN * kGPS_FILTER_MATCH_N_TILE * 3
    cur = self.get_buf(want_data, 0.5)
    if cur is None: return

    thresh = Z.stats.mstats.mquantiles(np.abs(cur), 0.1)
    cur = cur / thresh

    max_doppler = 2000
    doppler_space = np.linspace(-max_doppler, max_doppler, 100)
    power, freq, phase, ipower = self.analyse_chip_doppler_space(cur, doppler_space)

    self.stats.append(dict(type='acq', power=power, freq=freq))
    print('ACQ ok', np.abs(thresh), self.sid, power, self.ts, phase, freq, kSIGNAL_POWER_NOSAT_ACQ)
    if power > kSIGNAL_POWER_NOSAT_ACQ:
      return Z.Attributize(doppler=freq, power=power, chip_phase=self.ts + phase)

  def do_track(self):
    prn_clock_jitter = 10
    chip_start = self.tracking_data.chip_phase + kGPS_SAMPLE_PER_PRN - prn_clock_jitter
    while True:
      if chip_start >= self.new_ts:
        break
      chip_start += kGPS_SAMPLE_PER_PRN

    chip_end = chip_start + kGPS_SAMPLE_PER_PRN + 2 * prn_clock_jitter
    consume_ts = chip_end - 3 * prn_clock_jitter
    cur = self.get_buf_ts(chip_start, chip_end, consume_ts)
    if cur is None: return
    thresh = Z.stats.mstats.mquantiles(np.abs(cur), 0.1)
    cur = cur / thresh

    max_doppler_rate_hz_per_sec = 2000
    max_doppler_shift = max_doppler_rate_hz_per_sec / kGPS_PRN_FREQ
    doppler_space = np.linspace(-max_doppler_shift, max_doppler_shift, 10) + self.tracking_data.doppler

    power, freq, phase, ipower = self.analyse_chip_doppler_space(cur, doppler_space, acq=0)
    self.tracking_data.doppler = freq
    self.tracking_data.chip_phase = chip_start + phase
    self.stats.append(dict(type='track', power=power, freq=freq, phase=phase + chip_start, ipower=ipower))

    self.tracking_data.bits.append(ipower)
    print('Track: ', power, kSIGNAL_POWER_NOSAT_TRACK)
    if power < kSIGNAL_POWER_NOSAT_TRACK:
      raise 'Lost power'
      return Status.ACQUISITION

  def finalize(self):
    if self.tracking_data:
      bits = np.array(self.tracking_data.bits)
      self.tracking_data.processed_bits = np.abs(np.diff(np.unwrap(np.angle(bits))))

class OpaGps:

  def __init__(self, allowed_sats=None, dump_filename=None):
    self.output_q = []
    self.sats = {}
    self.dump_filename = dump_filename

    for i in range(Satellites.nsats()):
      if allowed_sats and i + 1 not in allowed_sats: continue
      self.sats[i] = OpaGpsSatellite(i, self.sats)

  def feed(self, data):
    for sat in self.sats.values():
      sat.feed(data)

  def get(self, n):
    n = min(n, len(self.output_q))
    res = self.output_q[:n]
    self.output_q = self.output_q[n:]
    return res

  @property
  def stats(self):
    res = []
    for num, sat in self.sats.items():
      for stat in sat.stats:
        stat['satid'] = num + 1
        res.append(stat)
    return res

  def finalize(self):
    res = {}
    for num, sat in self.sats.items():
      tmp = {}
      sat.finalize()
      for kw in Z.cmisc.to_list('tracking_data acq_data stats'):
        tmp[kw] = getattr(sat, kw)
      res[num]=tmp
    with  open(self.dump_filename, 'wb') as f:
      Z.pickle.dump(res, f)



def test(ctx):

  d = Z.pickle.load(open('/tmp/tmp.pickle', 'rb'))
  code = d['code']
  data = d['data']

  if 1:
    gps = OpaGps(allowed_sats=[27])
  else:
    gps = OpaGps()

  #for i in range(len(data) // kGPS_SAMPLE_PER_PRN):
  for i in range(100):
    cur = data[:kGPS_SAMPLE_PER_PRN]
    data = data[kGPS_SAMPLE_PER_PRN:]
    gps.feed(cur)

  Z.pprint(gps.stats)

  bits = np.array(gps.sats[26].tracking_data.bits)
  res = np.abs(np.diff(np.unwrap(np.angle(bits))))
  Z.plt.plot(res)
  Z.plt.show()

  return
  vals, sample_times = Z.DataOp.SolveMatchedFilter(data, code, min_jitter=4)
  angles = Z.np.diff(Z.np.unwrap(Z.np.angle(vals)))

  angles = Z.np.abs(Z.np.diff(Z.np.unwrap(Z.np.angle(vals))))
  clock_bps = Z.DataOp.PulseToClock(Z.DataOp.CleanBinaryDataSimple(angles))
  vals, sample_times = Z.DataOp.RetrieveBinaryData(clock_bps, clean=0)
  print(vals.astype(int))
  print(len(vals))


def test_gnuradio(ctx):
  from gnuradio import gr
  import opa.gps

  from gnuradio import blocks
  d = Z.pickle.load(open('/tmp/tmp.pickle', 'rb'))
  ofile = '/tmp/res.pickle'
  args = dict(allowed_sats=[27], dump_filename=ofile)

  src_data = d['data']
  src = blocks.vector_source_c(src_data)
  blk = opa.gps(str(args))
  snk = blocks.vector_sink_c()
  tb = gr.top_block()
  tb.connect(src, blk)
  tb.connect(blk, snk)
  tb.run()
  result_data = snk.data()

  res = Z.pickle.load(open(ofile, 'rb'))
  Z.pprint(res)


def try_message(content):
  for i in range(2):
    print('TRY ', i)
    if 1: content = list(1-np.array(content))


    tmp =''.join(map(str, content))
    for x in Z.Format(tmp).bucket(300, 'x').v:
      print(x)
    subframe_length = 300
    lst = Z.defaultdict(set)
    for x in Z.re.finditer('10001011', tmp):
      a ,b = x.start() // subframe_length, x.start()%subframe_length
      lst[b].add(a)

    cnds = []
    for k, v in lst.items():
      for i in range(0, len(tmp), subframe_length):
        if i//subframe_length not in v: break
      else:
        cnds.append(k)

    for cnd in cnds:
      tx = tmp[cnd:]
      for i, x in enumerate(Z.Format(tx).bucket(1500, 'x').v):
        print('ON FRAME ', i)
        for j, y in enumerate(Z.Format(x).bucket(300, 'x').v):
          print('\tON SUBFRAME ', j)
          for k, z in enumerate(Z.Format(y).bucket(30, 'x').v):
            print('\t\t', k, z)



def decode_bits(ctx):
  d = Z.pickle.load(open('/tmp/useless', 'rb'))
  assert len(d) == 1
  dd = next(iter(d.values()))
  bits = dd['tracking_data'].processed_bits
  track_stats = []
  for stat in dd['stats']:
    if stat['type'] == 'track' and 'power' in stat:
      track_stats.append(stat['power'])

  cx = Z.DataOp.CleanBinaryDataSimple(bits)
  cx = Z.DataOp.PulseToClock(cx)
  vals, spt =Z.DataOp.RetrieveBinaryData(cx)
  try_message(vals.astype(int))

  #Z.plt.plot(cx)
  #Z.plt.show()



def analyse_sats(ctx):
  fil = '/tmp/output2.bin'
  fil = '/tmp/test2.bin'
  f = Z.DataFile(filename=fil, typ=Z.np.complex64)
  #f = utils.DataFile(filename='/home/benoit/gqrx_20190127_214038_1577575000_8000000_fc.raw', typ=Z.np.complex64)
  content = f.col(0)
  content = content[1000:]
  print(len(content))
  gps = OpaGps(allowed_sats=None, dump_filename='/tmp/useless')
  try:
    BLK = 1024
    for i in range(0, len(content), BLK):
      gps.feed(content[i:i+BLK])
  finally:
    gps.finalize()


def main():
  ctx = Attributize()
  ActionHandler.Run(ctx)


app()
