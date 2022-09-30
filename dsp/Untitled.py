#!/usr/bin/env python
# coding: utf-8

# In[2]:


from chdrft.config.env import g_env
import chdrft.utils.K as K
g_env.set_qt5(1, 1)
get_ipython().run_line_magic('gui', 'qt5')
get_ipython().run_line_magic('matplotlib', 'qt')
init_jupyter()
import chdrft.dsp.gps as opa_gps
import chdrft.dsp.base as opa_dsp_base


# In[26]:



mp = {1: -8776.445,
 4: 3779.1567,
 5: 1027.5333,
 6: 5734.2007,
 11: 9263.816,
 18: 4503.002,
 19: -4757.2646,
 22: 8770.7705,
 29: 1274.0911,
 31: -6734.1743,
     }


# In[31]:


oconf = cmisc.Attr(nsyms=400, PROB_THRESH=3e-3, ignore_on_lost_track=1, jitter=20)

fil = '/home/benoit/programmation/dsp/data/gps_4msps_c64.dat'
ds0 = Z.DataFile(fil, typ='complex64').to_ds()
#ds = ds[:2000000]

ds0 = ds0[1000:]
ds = ds0

orig_center_freq = 1575.42e6
orig_sample_rate = 4e6
samples_per_chip = 4
doppler_step = 20


# In[ ]:


sat_num = 13
doppler = mp[sat_num]
acq_doppler = np.linspace(doppler - 200, doppler + 200, 50)

conf = cmisc.Attr(
    iir_b0_alpha=0.1,
    #acq_doppler = (625,),
    acq_doppler=acq_doppler,
    track_doppler=np.linspace(-1, 1, 4)[::-1],
    n_acq_symbol=1,
)
conf.track_doppler = np.array([
    0,
])
conf.update(oconf)

nsyms = 30
proc=1
tmp = opa_gps.proc_ds(
    ds,
    orig_center_freq,
    orig_sample_rate,
    samples_per_chip,
    sat_num,
    nsyms,
    alpha_tsync=1 - 1e-4,
    proc=proc,
    kwargs=conf,
)


# In[27]:


for sat_num, doppler in mp.items():
  if sat_num != 31 : continue
  acq_doppler = np.linspace(doppler - 200, doppler + 200, 50)
  tmp = opa_gps.proc_ds(
      ds,
      orig_center_freq,
      orig_sample_rate,
      samples_per_chip,
      sat_num,
      nsyms,
      alpha_tsync=1 - 1e-4,
      proc=proc,
      kwargs=conf,
  )
  res = tmp.analyse1(tmp.p, acq_doppler, tmp.ds.y)
  print('\n\n', sat_num)
  Z.pprint(res)
  


# In[33]:


print(len(ds.y) / tmp.p.conf.sample_rate)


# In[58]:


print(conf.track_doppler)


# In[29]:





# In[21]:


res = opa_gps.test_gps_4msps()


# In[4]:


print('la')
target = res.lst[1]
df = Z.pd.DataFrame.from_records(target.output)



# In[82]:


df = Z.pd.DataFrame.from_records(tmp.blk.output)
dfsync = Z.pd.DataFrame.from_records(tmp.tsync.stats)
t = df['ts'] / tmp.p.conf.sample_rate
x = df['ts'] 


# In[48]:


print(len(df))


# In[25]:



#Z.plt.plot(x, df['doppler'])
#y = np.log(df['prob_detection'])
y = df['phase']
#y = df['sps_track']
#y = df['phase']
Z.plt.plot(x, y)


# In[5]:


print(len(tmp.tsync.stats))


# In[24]:



Z.plt.plot(dfsync['skip'])
Z.plt.plot(dfsync['double'])


# In[ ]:


Z.plt.plot(t, df['doppler'])


# In[71]:


Z.plt.plot(t, df['phase'])


# In[85]:


Z.plt.plot(x, np.log(df['prob_detection']))


# In[84]:


get_ipython().run_line_magic('matplotlib', 'qt')
x = df.index
y = df['doppler']
y = df['phase_diff']
y = np.unwrap(df['ang_at'])
#Z.plt.plot(x, np.log(df['prob_detection']))
#Z.plt.plot(x, df['ang_at'])
#Z.plt.plot(x, df['b0_phase'])
#Z.plt.plot(x, df['nb0_phase'])
Z.plt.plot(x, np.abs(df['phase_diff']) >= (np.pi/2))
#Z.plt.plot(x, df['phase_diff'])


# In[19]:


idx = df['ts'].searchsorted(low)
df.iloc[idx]


# In[37]:


low = 0
tds = res.ds.y[low:]
p = get_p(target.sat_num, acq_sym=2, l1ca_repeat=1)
sig = tds[:p.acq_proc_size]
doppler_space =  np.arange(-10000, 10000, p.conf.doppler_step)
Z.pprint(analyse1(p, doppler_space, sig[:p.acq_proc_size]))


# In[15]:


sig.shape


# In[ ]:





# In[13]:


res.lst[4].sat_num


# In[12]:


df


# In[15]:


get_ipython().run_line_magic('matplotlib', 'qt')
for x in res.lst:
  if x.sat_num != 29: continue
  dx = list([y.prob_detection for y in x.output])
  doppler = list([y.doppler for y in x.output])
  ts = np.array([y.chip_start for y in x.output]) / x.conf.sample_rate
  #Z.plt.plot(doppler)
  Z.plt.plot(dx)
  break
    
    


# In[27]:



data = Z.DataFile('/home/benoit/programmation/dsp/data/test.dat', typ='complex8')
ds = data.to_ds()[:8000000]

orig_center_freq = 1575e6
orig_sample_rate = 8.7381e6
samples_per_chip = 8



# In[4]:


data = Z.DataFile('/home/benoit/programmation/dsp/data/gps/gps_4msps_complex.12Sep2005_1575.42_c64.dat', typ=np.complex64)


# In[11]:


42e6/4092/20


# In[ ]:


ds = data.to_ds()[:2000000]
ds.y = ds.y[1000:]


# In[12]:


orig_center_freq = 1575.42e6
orig_sample_rate = 4e6
samples_per_chip = 4


# In[13]:



chips_per_prn = opa_gps.kGPS_CHIPS_PER_PRN
prn_freq = opa_gps.kGPS_PRN_FREQ
bpsk_rate = opa_gps.kGPS_CHIP_FREQ
sample_rate = samples_per_chip * chips_per_prn * prn_freq
center_freq = opa_gps.kGPS_CENTER_FREQ


# In[5]:


g = K.GraphHelper(run_in_jupyter=1)
if 0:
  water = Z.DataOp.FreqWaterfall_DS(nds, 2**10)
  resx = water.new_y(Z.dsp_utils.g_fft_helper.map(water.y, quant=0.01))
  g.create_plot(images=[resx.transpose()])
else:
  doppler = 500
  p = get_p(26, 1, l1ca_repeat=1)
  interval = p.conf.pulse_shape
  ix = p.get_doppler_shift(interval, doppler)
  
  dy = np.abs(Z.signal.correlate(r.dbg.noise, ix, mode='valid'))
  g.create_plot(plots=[dy])
g.run()


# In[21]:


from gnuradio.filter import firdes
from gnuradio import gr, blocks, filter


taps = firdes.low_pass(1.0, orig_sample_rate, bpsk_rate * 1.1, bpsk_rate * 0.5, firdes.WIN_HAMMING, 6.76)
fil = filter.freq_xlating_fir_filter_ccc(1, taps, (center_freq - orig_center_freq), orig_sample_rate)

resampler = filter.mmse_resampler_cc(0, orig_sample_rate / sample_rate)
tmp = ds.y.astype(np.complex64)
#tmp = list(map(complex, ds.y))

if 1:
  ny = opa_dsp_base.run_block(opa_dsp_base.make_graph(chain=[tuple(ds.y), fil, resampler]), out_type=np.complex64)
else:
  ny = opa_dsp_base.run_block(opa_dsp_base.make_graph(chain=[tuple(ds.y), resampler]), out_type=np.complex64)
nds = ds.make_new('prepare', y=ny, samp_rate=sample_rate)


# In[29]:


nds.to_file('/tmp/test.dat', typ=np.complex64)


# In[9]:



def get_p(sat_num, acq_sym=1, l1ca_repeat=20):
  symbol_rate=prn_freq // l1ca_repeat
  doppler_max_step  =  symbol_rate / 2
  conf = cmisc.Attributize(
      sample_rate=sample_rate,
      sps=samples_per_chip * chips_per_prn * l1ca_repeat,
      symbol_rate=symbol_rate,
      n_acq_symbol=acq_sym,
      max_doppler_shift=0,
      doppler_space_size=1,
      SNR_THRESH=10,
      skip_ratio=0,
      jitter=0,
      pulse_shape = list(opa_gps.get_l1ca_pulse(sat_num, samples_per_chip)) * l1ca_repeat,
      doppler_step = doppler_max_step / 4,
    track_doppler=None, acq_doppler=None,
  )
  p = opa_gps.OpaXpsk(conf)
  return p


# In[10]:


def analyse1(p, doppler_space, sig):
  tb=[]
  print(len(doppler_space))
  for doppler in doppler_space:
    r = p.analyse_chip(doppler, sig)
    tb.append(r)
  mx = max(tb, key=lambda x: x.snr)
  mx2 = max(tb, key=lambda x: x.prob_detection)
  return cmisc.Attr(by_snr=mx, by_prob=mx2)


# In[15]:


doppler_space =  np.arange(1040, 1070, 1)
doppler_space = (1057,)
#doppler_space = np.arange(-2000, 2000, 300)
p = get_p(26, 10, l1ca_repeat=1)
print(len(nds.y), p.acq_proc_size)
sig = nds.y[p.acq_proc_size:p.acq_proc_size*2]
r = p.analyse_chip(doppler_space[0], sig ,dbg=1, clear_sig_type2=1)


# In[26]:


print(opa_dsp_base.DbHelper.sig_power(r.typ2_sig))
print(opa_dsp_base.DbHelper.sig_power(r.dbg.noise))
print(len(p.conf.pulse_shape))
print(p.conf.sps)
print(r.snr)


# In[21]:



g = K.GraphHelper(run_in_jupyter=1)
g.create_plot(plots=[np.abs(r.typ2_sig)])
g.create_plot(plots=[np.abs(r.dbg.noise)])
#g.create_plot(plots=[np.abs(Z.signal.correlate(r.dbg.noise, , mode='valid'))])

g.run()


# In[21]:


sig = r.typ2_sig
#sig = nds.y
mp = {
  8:np.arange(-10000, 10000, p.conf.doppler_step),
  3:np.linspace(-500, 500, 40),
  1:np.linspace(1000, 1900, 40),
  26:np.linspace(1040, 1070, 10),
}

sid = 8
p = get_p(sid, 2, l1ca_repeat=1)
doppler_space=mp[sid]
print(len(doppler_space))
sig = sig[p.acq_proc_size//3:]
Z.pprint(analyse1(p, doppler_space, sig[:p.acq_proc_size]))


# In[17]:


sid = 4
p = get_p(sid, acq_sym=2, l1ca_repeat=2)
sig = nds.y[:p.acq_proc_size]
doppler_space =  np.arange(-10000, 10000, p.conf.doppler_step)
Z.pprint(analyse1(p, doppler_space, sig[:p.acq_proc_size]))


# In[53]:


doppler_space =  np.arange(-2000, 2000, doppler_step)
sig = RMX.dbg.noise[p.acq_proc_size: p.acq_proc_size*2]
print(p.acq_proc_size, len(mx.dbg.noise))


# In[34]:



g = K.GraphHelper(run_in_jupyter=1)
#g.create_plot(plots=[np.abs(sig)])
g.create_plot(plots=[np.abs(nds.y[p.acq_proc_size:p.acq_proc_size*2])])
g.run()


# In[22]:



sats = (1,4,5,9,29)
sats_mp = {
  1 : (5062,),
  4 : (3874,),
  5 : (750,),
  9 : (1187,),
  29 : (1437,),
}
#1 6.982685490120044e-05 5062.5
#4 7.495027931625486e-10 3875.0
#5 3.018314105318609e-07 750.0
#9 0.00013229216830179524 1187.5
#29 4.773959005888173e-14 1437.5


# In[ ]:


sats = range(opa_gps.Satellites.nsats())


# In[30]:


nsyms = 5
l1ca_repeat=1
p0 = get_p(0, nsyms, l1ca_repeat=l1ca_repeat)
sig = nds.y
sig = sig[:p0.acq_proc_size]
doppler_space =  np.arange(-10000, 10000, p0.conf.doppler_step)
print(len(sig), p0.acq_proc_size)

tb = []
#sats = (1, 3, 8, 26)
for sid, doppler_space in sats_mp.items():
  if sid != 1: continue
  p = get_p(sid, nsyms, l1ca_repeat=l1ca_repeat)
  d0 = doppler_space[0]
  #doppler_space = np.linspace(d0-200, d0+200, 20)
  mx = analyse1(p, doppler_space, sig)
  tb.append(cmisc.Attr(sid=sid, res=mx))
  print(sid, mx.by_snr.snr, mx.by_prob.prob_detection, mx.by_prob.doppler)


# In[21]:


for e in tb:
  px =  1 - e.res.by_prob.prob_detection
  if px < 1e-3:
    print(e.sid, px, e.res.by_prob.doppler)


# In[ ]:


#1 6.982685490120044e-05 5062.5
#4 7.495027931625486e-10 3875.0
#5 3.018314105318609e-07 750.0
#9 0.00013229216830179524 1187.5
#29 4.773959005888173e-14 1437.5


# In[52]:


RMX = mx


# In[70]:


target_doppler = 1000
r = p.analyse_chip(target_doppler, sig, dbg=0)
r


# In[75]:


get_ipython().run_line_magic('matplotlib', 'qt')
Z.plt.figure()
interval = nds.y[:conf.sps]
dy = np.abs(Z.signal.correlate(interval, nds.y[:3*conf.sps], mode='full'))
dx = np.array(range(len(dy)))

Z.plt.plot(dx, dy)
  
8738133


# In[114]:



sat_num  = 26
prn = opa_gps.get_l1ca_pulse(sat_num, samples_per_chip)
prn = list(prn) + list(prn)
t = 2*np.pi * 1j * np.arange(len(prn)) / conf.sample_rate


# In[128]:



shift_seq = np.exp(p.pulse_t_acq * target_doppler) * dx
correl = Z.signal.correlate(shift_seq, conf.pulse_shape, mode='valid')
Z.plt.plot(np.abs(correl))


# In[27]:


Z.FileFormatHelper.Write('/tmp/test.pickle', cmisc.Attr(conf=conf, ds=ds))


# In[3]:



dd = Z.FileFormatHelper.Read('/tmp/test.pickle')
conf = dd.conf
ds = dd.ds


# In[4]:


import scipy.io as sio
#r = sio.loadmat('/home/benoit/programmation/dsp/data/tmp/observables.mat')
r = sio.loadmat()


# In[36]:


get_ipython().run_line_magic('matplotlib', 'qt')
import h5py
mpx = {}

keys = None
pattern = '/home/benoit/programmation/dsp/data/gnss_4msps/epl_tracking_ch_*.mat'
pattern = '/home/benoit/programmation/dsp/data/test_spain/epl_tracking_ch_*.mat'
for fname in cmisc.get_input([pattern]):
  print('go, ',fname)
  with h5py.File(fname, 'r') as f:
    keys = list(f.keys())

    x = np.array(f['carrier_doppler_hz'])
    if len(x.shape) != 2: continue
    prn = f['PRN'][0][0]
    n = x.shape[1]
    mpx[prn-1] = -x[0][0]


    
    #print(f['PRN'][0])
    tx = f['code_error_filt_chips'][:,0]
    print('\n\n',prn)
    print(np.mean(tx))
    Z.plt.plot(f['code_error_filt_chips'][:,0], label=str(prn))
    Z.plt.legend()
    #print(np.array(f['code_error_chips']))
    #print(np.array(f['PRN_start_sample_count']))
    #print(np.array(f['Prompt_I']))
Z.pprint(mpx)
print(keys)


# In[ ]:





# In[ ]:




