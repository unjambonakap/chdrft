#!/usr/bin/env python

from chdrft.cmds import CmdsList
from chdrft.main import app
from chdrft.utils.cmdify import ActionHandler
from chdrft.utils.misc import Attributize
import chdrft.utils.misc as cmisc
import chdrft.utils.Z as Z
import chdrft.utils.K as K
import numpy as np
from chdrft.dsp.base import SignalBuilder, OpaXpsk
import chdrft.dsp.base as opa_dsp_base

global flags, cache
flags = None
cache = None

dsp_swig = Z.swig.opa_dsp_swig
Satellites = dsp_swig.Satellites.GetSingleton()

kGPS_CHIP_FREQ = 1023000
kGPS_CHIPS_PER_PRN = 1023
kGPS_PRN_FREQ = kGPS_CHIP_FREQ // kGPS_CHIPS_PER_PRN
kGPS_SAMPLES_PER_CHIP = 4
kGPS_SAMPLE_FREQ = kGPS_SAMPLES_PER_CHIP * kGPS_CHIP_FREQ
kGPS_SAMPLES_PER_PRN = kGPS_CHIPS_PER_PRN * kGPS_SAMPLES_PER_CHIP
kGPS_GCODE_FREQ = 50
kGPS_FILTER_MATCH_N_TILE = 20
kGPS_NUM_PRN_PERIOD_PER_GBIT = kGPS_PRN_FREQ // kGPS_GCODE_FREQ
kGPS_CENTER_FREQ = 1575420000

kBINARY_SD = 1  # standard deviation 1 for of {-1, 1}, equiprobable
kSIGNAL_POWER_NOSAT_TRACK = Z.stats.norm.ppf(
    1 - 1e-5
) * kGPS_CHIPS_PER_PRN**0.5 * kGPS_SAMPLES_PER_CHIP * (kBINARY_SD * 1.5)

kSIGNAL_POWER_NOSAT_ACQ = kSIGNAL_POWER_NOSAT_TRACK * kGPS_FILTER_MATCH_N_TILE


def args(parser):
  clist = CmdsList().add(test)
  parser.add_argument('--outfile')
  parser.add_argument('--infile')
  parser.add_argument('--nsyms', type=int)
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


def get_l1ca_pulse(sid, samples_per_chip):
  return np.repeat(get_l1ca(sid), samples_per_chip) * 2 - 1


class OpaGpsSatellite:

  def __init__(self, sid, sats=None, conf=None):
    if conf is None:
      conf = cmisc.Attr(
          samples_per_chip=kGPS_SAMPLES_PER_CHIP,
          samples_freq=kGPS_SAMPLE_FREQ,
          match_n_tile=kGPS_FILTER_MATCH_N_TILE,
          prn_freq=kGPS_PRN_FREQ,
      )
    self.conf = conf

    self.seq_track = get_l1ca_pulse(sid, self.conf.sample_per_chip)
    self.seq_acq = np.tile(self.seq_track, self.conf.match_n_tile)
    self.sid = sid
    self.data = np.array([])
    self.status = Status.INIT
    self.sats = sats
    self.signal_data = Z.Attributize(power=0)
    self.pts_acq = np.arange(len(self.seq_acq)) * (2j * np.pi / self.conf.samples_freq)
    self.pts_track = np.arange(len(self.seq_track)) * (2j * np.pi / self.conf.samples_freq)
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
    want_data = self.conf.samples_per_prn * self.conf.match_n_tile * 3
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
    chip_start = self.tracking_data.chip_phase + self.conf.samples_per_prn - prn_clock_jitter
    while True:
      if chip_start >= self.new_ts:
        break
      chip_start += self.conf.samples_per_prn

    chip_end = chip_start + self.conf.samples_per_prn + 2 * prn_clock_jitter
    consume_ts = chip_end - 3 * prn_clock_jitter
    cur = self.get_buf_ts(chip_start, chip_end, consume_ts)
    if cur is None: return
    thresh = Z.stats.mstats.mquantiles(np.abs(cur), 0.1)
    cur = cur / thresh

    max_doppler_rate_hz_per_sec = 2000
    max_doppler_shift = max_doppler_rate_hz_per_sec / self.conf.prn_freq
    doppler_space = np.linspace(
        -max_doppler_shift, max_doppler_shift, 10
    ) + self.tracking_data.doppler

    power, freq, phase, ipower = self.analyse_chip_doppler_space(cur, doppler_space, acq=0)
    self.tracking_data.doppler = freq
    self.tracking_data.chip_phase = chip_start + phase
    self.stats.append(
        dict(type='track', power=power, freq=freq, phase=phase + chip_start, ipower=ipower)
    )

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
      res[num] = tmp
    with open(self.dump_filename, 'wb') as f:
      Z.pickle.dump(res, f)


def test(ctx):

  d = Z.pickle.load(open('/tmp/tmp.pickle', 'rb'))
  code = d['code']
  data = d['data']

  if 1:
    gps = OpaGps(allowed_sats=[27])
  else:
    gps = OpaGps()

  #for i in range(len(data) // kGPS_SAMPLES_PER_PRN):
  for i in range(100):
    cur = data[:kGPS_SAMPLES_PER_PRN]
    data = data[kGPS_SAMPLES_PER_PRN:]
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
    if 1: content = list(1 - np.array(content))

    tmp = ''.join(map(str, content))
    for x in Z.Format(tmp).bucket(300, 'x').v:
      print(x)
    subframe_length = 300
    lst = Z.defaultdict(set)
    for x in Z.re.finditer('10001011', tmp):
      a, b = x.start() // subframe_length, x.start() % subframe_length
      lst[b].add(a)

    cnds = []
    for k, v in lst.items():
      for i in range(0, len(tmp), subframe_length):
        if i // subframe_length not in v: break
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
  vals, spt = Z.DataOp.RetrieveBinaryData(cx)
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
      gps.feed(content[i:i + BLK])
  finally:
    gps.finalize()


def test_pulse_shape_gps(ctx):
  # proof that the pulse shape of gps is very similar to unlimited bw bpsk
  # 10.23mhz bpsk signal sent on a bw of 20.46mhz (number 2=cutoff freq in firdes.low_pass)
  upsample = 100
  target_interval = upsample * 100
  pulse = [1] * upsample
  sg = SignalBuilder(0, target_interval)
  sg.add(target_interval // 2, pulse)
  npulse = sg.get()
  taps = K.firdes.low_pass(1.0, upsample, 2, 3, K.firdes.WIN_HAMMING, 6.76)

  g = K.GraphHelper(run_in_jupyter=0)
  x = g.create_plot(plots=[])
  y = g.create_plot(plots=[])
  x.add_plot(npulse, color='r')
  x.add_plot(np.convolve(npulse, taps), color='g')
  y.add_plot(taps, color='r')
  g.run()


def generate_test_bpsk(ctx):
  samples_per_chip = 5
  chips_per_prn = 1023
  ndata = 100
  sb = SignalBuilder(0, samples_per_chip * (ndata + 1) * chips_per_prn, dtype=np.complex128)
  sats = list(range(4))
  nsats = len(sats)

  powers_db = [-0, -20, -30, -30]
  #powers_db = [-0, 0, 0, 0]
  noise_db = -60

  prns = [get_l1ca_pulse(sid, samples_per_chip) for sid in sats]
  offsets = [0.0, 0.1, 0.5, 0.8]
  dopplers = [10000, 1000, -1000, 0]
  phase_start = [0, 6.3, 1.7, 1.3]
  nsats = 4

  data = cmisc.Attr()
  data.orig_data = []
  data.samples_per_chip = samples_per_chip
  data.chips_per_prn = chips_per_prn
  data.prn_freq = kGPS_PRN_FREQ
  data.sample_rate = samples_per_chip * chips_per_prn * data.prn_freq

  prn_ts = np.arange(samples_per_chip * chips_per_prn) / data.sample_rate

  for i in range(nsats):
    sig_data = cmisc.Attr()
    prn = prns[i]
    power_db = powers_db[i]
    offset = int(offsets[i] * samples_per_chip * chips_per_prn)
    prn_pw = Z.DataOp.Power(prn)
    pulse = prn * (10**(power_db / 20) / prn_pw)
    bin_data = np.random.randint(2, size=ndata)
    cur_data = 2 * bin_data - 1
    phase = phase_start[i]
    doppler = dopplers[i]

    cur_sb = sb.make_new()
    cur_sb.add_multiple(offset, cur_data, pulse, len(pulse))
    cur_sb.doppler_shift(doppler, data.sample_rate)
    sb.add_peer(cur_sb)

    sig_data.prn = prn
    sig_data.power_db = power_db
    sig_data.offset = offset
    sig_data.sat = sats[i]
    sig_data.phase = phase
    sig_data.doppler = doppler
    sig_data.bin_data = bin_data
    sig_data.signal = cur_sb.get()
    sig_data.sb = cur_sb
    data.orig_data.append(sig_data)

    print(power_db, Z.DataOp.Power_Db(sig_data.signal), offset)

    if 0:
      g = K.GraphHelper(run_in_jupyter=0)
      x = g.create_plot(plots=[np.angle(cur_sb.get())])
      g.run()
      assert 0

  res = sb.get()
  if noise_db is not None:
    data.noiseless = np.array(res)
    noise = np.random.normal(scale=10**(noise_db / 20), size=len(res))
    print(noise_db, Z.DataOp.Power_Db(noise))
    data.noise = noise
    res += noise

  print(len(res))
  data.signal = res
  Z.pickle.dump(data, open(ctx.outfile, 'wb'))


class IncrementalLeastSquares:

  def __init__(self, dtype=np.complex128):
    self.r = 0
    self.dtype = dtype
    self.m = np.zeros((self.r, self.r), dtype=self.dtype)
    self.tb = []

  def add(self, v):
    self.r += 1
    self.tb.append(v)
    nm = np.zeros((self.r, self.r), dtype=self.dtype)
    nm[:-1, :-1] = self.m
    for i in range(self.r):
      nm[-1, i] = np.dot(v, np.conj(self.tb[i]))
      nm[i, -1] = np.conj(nm[-1, i])

    self.m = nm

  def compute(self, y):
    print('mat is >> ', self.m)
    q = np.linalg.inv(self.m)
    b = []
    for v in self.tb:
      b.append(np.dot(np.conj(v), y))
    res = np.matmul(q, b)
    val = np.matmul(res, self.tb)
    print(Z.DataOp.Power_Db(y))
    print(Z.DataOp.Power_Db(val))
    print('>>>XUXU ', Z.DataOp.Power_Db(val - y))
    return res, val


def solve_least_squares(tb, y):
  ls = IncrementalLeastSquares()
  for v in tb:
    #v = v / np.dot(v, np.conj(v)) ** 0.5
    ls.add(v)
  return ls.compute(y)


def process_bits_xpsk(p, cur_ctx, nbits, freq, prevs, start_pos, orig_data=None):

  print('\nNew processing on sig db: ', Z.DataOp.Power_Db(cur_ctx.cleaned_sig))
  r = p.analyse_chip(freq, cur_ctx.cleaned_sig[:p.acq_proc_size], start_pos)
  p.init_tracking_data(r)
  print(r)
  print(start_pos)
  sym_phases = list(r.sym_phases)

  start_phase = r.phase
  nstart_pos = start_pos + start_phase

  cur_phase = start_phase + p.conf.sps * len(sym_phases)
  for i in range(nbits):
    cx = cur_ctx.cleaned_sig[cur_phase:cur_phase + p.conf.sps]
    nr = p.analyse_chip(freq, cx, start_pos + cur_phase, acq=0)
    sym_phases.append(nr.sym_phase)
    cur_phase += p.conf.sps

  bits = Z.DataOp.GetBPSKBits(sym_phases)
  syms = 2 * bits - 1
  #syms = np.exp(1j * np.array(sym_phases))
  print('NBITS >> ', len(syms))

  sb = SignalBuilder(nstart_pos, n=p.conf.sps * len(syms), dtype=np.complex128)
  sb.add_multiple(nstart_pos, syms, p.conf.pulse_shape, p.conf.sps)
  sb.doppler_shift(freq, p.conf.sample_rate)

  tb = []
  #sz=3000
  #if cur_ctx.i!=1: sz=-1
  for prev in prevs:
    #tb.append(sb.extract_peer(prev.orig_data.sb)[:sz])
    tb.append(sb.extract_peer(prev.sb))

  rebuilt = sb.get()
  tb.append(rebuilt)
  orig = sb.extract(0, cur_ctx.orig_sig)
  vec, sig_space = solve_least_squares(tb, orig)
  print('LEAST SQURE SOL ', vec)

  print('GOOOT', Z.DataOp.Power_Db(orig - sig_space))
  nsig = orig - sig_space

  print(Z.min_substr_diff(orig_data.bin_data, bits))
  print(Z.min_substr_diff(orig_data.bin_data, 1 ^ bits))

  #rebuilt = sb.extract_peer(orig_data.sb)
  if cur_ctx.i == -1:
    sbx = SignalBuilder(0, n=p.conf.sps, dtype=np.complex128)
    sbx.add(0, p.conf.pulse_shape)
    sbx.doppler_shift(freq, p.conf.sample_rate)
    g = K.GraphHelper(run_in_jupyter=0)
    x = g.create_plot(plots=[np.angle(rebuilt) + np.angle(sb.extract_peer(orig_data.sb))])
    #x = g.create_plot(plots=[np.angle(sb.extract_peer(orig_data.sb))])
    #x = g.create_plot(plots=[np.log10(np.abs(Z.signal.correlate(cur_ctx.cleaned_sig, sbx.get())))])
    x = g.create_plot(plots=[np.log10(np.abs(Z.signal.correlate(cur_ctx.orig_sig, sbx.get())))])
    x = g.create_plot(plots=[np.log10(np.abs(Z.signal.correlate(orig_data.sb.get(), sbx.get())))])
    x = g.create_plot(plots=[np.log10(np.abs(Z.signal.correlate(cur_ctx.cleaned_sig, sbx.get())))])
    x = g.create_plot(plots=[np.log10(np.abs(Z.signal.correlate(sb.get(), sbx.get())))])
    g.run()
    return

  print(Z.DataOp.GetBPSKBits(sym_phases))
  print(np.abs(np.diff(np.unwrap(sym_phases))))
  return cmisc.Attr(
      nsig=nsig,
      start_phase=start_phase,
      rebuilt=rebuilt,
      bits=bits,
      sb=sb,
      orig_data=orig_data,
      nstart_pos=nstart_pos
  )


def real_decode(ctx):
  d = Z.pickle.load(open(ctx.infile, 'rb'))
  samples_per_prn = d.samples_per_chip * d.chips_per_prn

  pl = []
  conf = cmisc.Attributize(
      sample_rate=d.sample_rate,
      sps=d.samples_per_chip * d.chips_per_prn,
      symbol_rate=d.prn_freq,
      n_acq_symbol=20,
      max_doppler_shift=0,
      doppler_space_size=1,
      SNR_THRESH=10,
      jitter=0,
  )
  start_pos = samples_per_prn
  cx = d.signal

  cur_ctx = cmisc.Attr(orig_sig=cx, cleaned_sig=cx[start_pos:])
  prevs = []
  for i in range(len(d.orig_data)):
    if i == -1:
      print('goot')
      g = K.GraphHelper(run_in_jupyter=0)
      x = g.create_plot(plots=[np.abs(cx), np.abs(d.signal[start_pos:start_pos + len(cx)])])
      g.run()
      return
    conf.pulse_shape = get_l1ca_pulse(d.orig_data[i].sat, d.samples_per_chip)
    p = OpaXpsk(conf)
    cur_ctx.i = i
    dd = process_bits_xpsk(
        p, cur_ctx, 20 - 2 * i, d.orig_data[i].doppler, prevs, start_pos, d.orig_data[i]
    )
    cur_ctx.cleaned_sig = dd.nsig
    prevs.append(dd)
    start_pos = dd.nstart_pos

    bits = dd.bits >= 0.5
    print(dd.bits, len(bits))
    print(Z.min_substr_diff(d.orig_data[i].bin_data, bits))
    print(Z.min_substr_diff(d.orig_data[i].bin_data, 1 ^ bits))
    print(d.orig_data[i].bin_data)
    print()

  #d2 = process_bits_xpsk(p, d1.nsig[samples_per_prn:], 5)
  print(Z.DataOp.Power_Db(cx))
  return

  for i in range(1, 6):
    conf.pulse_shape = get_l1ca_pulse(i, d.samples_per_chip)
    p = OpaXpsk(conf)
    r = p.analyse_chip(d.orig_data[i].doppler, cx)
    print(r)
    break


def test_correl(ctx):
  a = get_l1ca_pulse(0, 5)
  b = np.concatenate((a, a))
  g = K.GraphHelper(run_in_jupyter=0)
  x = g.create_plot(plots=[np.log10(np.abs(Z.signal.correlate(a, b)))])
  x = g.create_plot(plots=[np.abs(Z.signal.correlate(a, b))])

  sb = SignalBuilder(0, n=2 * len(a), dtype=np.complex128)
  sb.add_multiple(0, [1, -1], a, len(a))
  x = g.create_plot(plots=[np.abs(Z.signal.correlate(a, sb.get(), method='direct'))])
  g.run()


def test_gps_4msps(ctx):
  fil = '/home/benoit/programmation/dsp/data/gps_4msps_c64.dat'
  ds = Z.DataFile(fil, typ='complex64').to_ds()
  #ds = ds[:2000000]

  ds = ds[1000:]

  orig_center_freq = 1575.42e6
  orig_sample_rate = 4e6
  samples_per_chip = 4
  doppler_step = 20

  mp = {
      1: (5062,),
      4: (3874,),
      5: (625,),
      9: (1187,),
      29: (1437,),
  }

  res = []
  nds = None
  for sat_num in mp.keys():
    conf = cmisc.Attr()
    conf.acq_doppler = mp[sat_num]
    conf.track_doppler = cmisc.Attr(
        max_doppler_rate=8,
        doppler_step=1,
    )
    conf.track_doppler = np.linspace(-1, 1, 4)
    conf.update(ctx)
    nsyms = ctx.nsyms

    tmp = proc_ds(ds, orig_center_freq, orig_sample_rate, samples_per_chip, sat_num, nsyms, conf)
    nds = tmp.ds
    res.append(cmisc.Attr(sat_num=sat_num, output=tmp.p.output, conf=tmp.p.conf))
  return cmisc.Attr(ds=nds, lst=res, orig_ds=ds)


def test_seti(ctx):
  data = Z.DataFile('/home/benoit/programmation/dsp/data/test.dat', typ='complex8')
  ds = data.to_ds()
  ds = ds[:2000000]

  orig_center_freq = 1575e6
  orig_sample_rate = 8.7381e6
  samples_per_chip = 8

  sat_num = 26
  conf = cmisc.Attr()
  conf.acq_doppler = (1057,)
  conf.track_doppler = cmisc.Attr(
      max_doppler_rate=8,
      doppler_step=1,
  )
  conf.update(ctx)
  nsyms = ctx.nsyms

  p = proc_ds(ds, orig_center_freq, orig_sample_rate, samples_per_chip, sat_num, nsyms, conf)
  return p


def proc_ds(
    ds,
    orig_center_freq,
    orig_sample_rate,
    samples_per_chip,
    sat_num,
    nsyms=10,
    alpha_tsync=1,
    proc=1,
    kwargs={},
):

  chips_per_prn = kGPS_CHIPS_PER_PRN
  prn_freq = kGPS_PRN_FREQ
  bpsk_rate = kGPS_CHIP_FREQ
  sample_rate = samples_per_chip * chips_per_prn * prn_freq
  center_freq = kGPS_CENTER_FREQ
  l1ca_repeat = 1
  acq_sym = 5

  symbol_rate = prn_freq // l1ca_repeat
  conf = cmisc.Attributize(
      sample_rate=sample_rate,
      sps=samples_per_chip * chips_per_prn * l1ca_repeat,
      symbol_rate=symbol_rate,
      n_acq_symbol=acq_sym,
      max_doppler_shift=0,
      doppler_space_size=1,
      SNR_THRESH=10,
      skip_ratio=0,
      jitter=5,
      pulse_shape=list(get_l1ca_pulse(sat_num, samples_per_chip)) * l1ca_repeat,
      PROB_THRESH=1e-7,
      fixed_sps_mode=0,
  )

  ds = ds[:2 * conf.sps * nsyms]

  conf.update(kwargs)

  from gnuradio.filter import firdes
  from gnuradio import filter

  taps = firdes.low_pass(
      1.0, orig_sample_rate, bpsk_rate * 1.1, bpsk_rate * 0.5, firdes.WIN_HAMMING, 6.76
  )
  fil = filter.freq_xlating_fir_filter_ccc(
      1, taps, (center_freq - orig_center_freq), orig_sample_rate
  )

  resampler = filter.mmse_resampler_cc(0, orig_sample_rate / sample_rate)
  tmp = ds.y.astype(np.complex64)
  #tmp = list(map(complex, ds.y))

  if 1:
    ny = opa_dsp_base.run_block(
        opa_dsp_base.make_graph(chain=[tuple(ds.y), fil, resampler]), out_type=np.complex64
    )
  else:
    ny = opa_dsp_base.run_block(
        opa_dsp_base.make_graph(chain=[tuple(ds.y), resampler]), out_type=np.complex64
    )
  nds = ds.make_new('prepare', y=ny, samp_rate=sample_rate)

  def analyse1(p, doppler_space, sig):
    tb = []
    sig = sig[:p.acq_proc_size]
    print(len(doppler_space))
    for doppler in doppler_space:
      r = p.analyse_chip(doppler, sig)
      tb.append(r)
    mx = max(tb, key=lambda x: x.snr)
    mx2 = max(tb, key=lambda x: x.prob_detection)
    return cmisc.Attr(by_snr=mx, by_prob=mx2)

  #doppler_space = np.arange(-2000, 2000, 300)

  sig = nds.y[:conf.sps * nsyms]
  p = OpaXpsk(conf)
  if not proc:
    return cmisc.Attr(ds=nds, p=p, analyse1=analyse1)

  tsync = opa_dsp_base.TimingSyncBlock(
      resamp_ratio=1, sps_base=conf.sps, sig=np.complex64, alpha=alpha_tsync
  )
  p.sps_track = tsync.sps
  blk = opa_dsp_base.ChainBlock([tsync, p], blk_size=conf.sps)
  print('START >> ', len(sig), sat_num)
  blk.feed(sig)
  return cmisc.Attr(ds=nds, p=p, tsync=tsync, blk=blk)


#1 6.982685490120044e-05 5062.5
#4 7.495027931625486e-10 3875.0
#5 3.018314105318609e-07 625.0
#9 0.00013229216830179524 1187.5
#29 4.773959005888173e-14 1437.5


def main():
  ctx = Attributize()
  ActionHandler.Run(ctx)


app()
