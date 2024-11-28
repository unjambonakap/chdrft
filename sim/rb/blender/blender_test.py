#!/usr/bin/env python

import chdrft.utils.misc as cmisc
import numpy as np
from chdrft.utils.opa_types import *

from chdrft.sim.blender import BlenderPhysHelper
from chdrft.display.blender import clear_scene, AnimationSceneHelper
from chdrft.sim.base import compute_cam_parameters
from chdrft.sim.rb.rb_gen import *
from chdrft.sim.rb.rb_player import SceneController, InputControllerParameters
from chdrft.sim.rb.ctrl import *

from chdrft.utils.rx_helpers import ImageIO
from chdrft.config.blender import init_blender
from chdrft.sim.rb.scenes import *
import bpy

init_blender()  # required to make qt work

global flags, cache
flags = None
cache = None


def go():
  clear_scene()
  dt_base = 1e-2
  use_gamepad=0
  use_jit = 0
  mode = 5
  override_ctrl = np.array([-10])

  q0  = None
  match mode:
    case 0:
      idx = simple_box
      mdata = ModelData(func_id=idx)
      end_conds = ControlInputEndConds(q=np.array([1]), qd=np.array([0]))
      cspec = ControlSpec(
          mdata=mdata,
          cparams=ControlParameters(ndt_const=1, dt=dt_base, integ_nsteps=1, nt=5, use_jit=use_jit),
          end_conds=end_conds
      )

      i0 = ControlInputState(q=np.zeros(1), qd=np.zeros(1))
    case 1:
      idx = scene_T2D_idx
      qd0 = np.array([10, 3, -2])
    case 2:
      idx = scene_T_idx
      qd0 = np.array([0, 0, 0, 10, 0, 0.1])
    case 3:
      idx = control_simple_idx1
      qd0 = np.array([0, 0, 0, 0, 0, 0, 1])
    case 4:
      idx = control_simple_idx2
      qd0 = np.array([0, 0, 0, 0, 1, 1, 1, 3])
    case 5:
      idx = control_simple_idx3
      qd0 = np.array([0,0,0,1, -3, -2, 0, 0, 0])

      qd0 = np.array([0, 0, 0, 0, 0, 1, 1])
      idx = control_simple_idx1

      idx = two_mass_idx
      qd0=np.array([0,0,0,0,0,1])

      idx = inv_pendulum_idx
      qd0=np.array([0,0])
      q0=np.array([0,0])


  mdata = ModelData(func_id=idx)
  if q0 is None:
    q0 = mdata.ss.q_desc.default

  i0 = ControlInputState(q=q0, qd=qd0)
  cspec = ControlSpec(
      mdata=mdata,
      cparams=ControlParameters(ndt_const=1, dt=dt_base, integ_nsteps=1, nt=5, use_jit=use_jit),
      end_conds=None
  )

  params = cspec.cparams

  solver = RBSolver(spec=cspec)
  sctx = solver.spec.mdata.sctx
  fspec = ControlFullSpec(consts=ControlInput(state=i0), spec=cspec)


  ctrl = SceneController(
      fspec=fspec,
      dt_base=dt_base,
      use_jit=use_jit,
      use_gamepad=use_gamepad,
      parameters=InputControllerParameters(map=lambda x: x),
  )

  ctrl.setup()
  ctrl.s0.stop = 0
  ctrl.s0.precision = 1
  ctrl.s0.fix_mom = False
  sh = ctrl.sh

  sx = ctrl.sim
  tg = sx.root
  data = np.ones((100, 1))

  #sx.load_state(**params.i_state)

  helper = BlenderPhysHelper()
  helper.load(tg)
  cam_loc = np.array([5, 0, 0])
  aabb = tg.self_link.aabb()
  cam_params = compute_cam_parameters(
      cam_loc, aabb.center, Vec3.Z().vdata, aabb.points[:, :3], blender=True
  )
  helper.cam.data.angle_y = cam_params.angle_box.yn
  helper.cam.mat_world = cam_params.toworld

  #r0l = sctx.compose([wlink])
  #tg = r0l
  #px.plot(by_col=True, fx=0.2)

  #for i in range(10):
  #  print('FUU ', tg.self_link.get_particles(10000000).angular_momentum())
  #return

  #osync = ObjectSync(helper.obj2blender[rbl.child], helper.cam)


  params=ControlParameters(ndt_const=3, dt=1e-1, integ_nsteps=1, nt=40, use_jit=use_jit)

  data = np.array([[-25.8103984,],
       [39.9247077,],
       [36.8656438,],
       [5.3030351,],
       [23.5012944,],
       [36.0444049,],
       [34.7305171,],
       [21.9130388,],
       [30.3634881,],
       [35.9985798,],
       [3.5225636,],
       [-27.0954699,],
       [-39.6237018,],
       [-24.3805839,],
       [-39.9820066,],
       [-39.9847459,],
       [-39.0331285,],
       [-39.4192280,],
       [-39.9214837,],
       [-6.2897321,],
       [8.8205823,],
       [-39.4638568,],
       [-39.7914606,]])





  animation = AnimationSceneHelper(frame_step=1)
  animation.start()
  helper.update(animation)

  sx.load_state(i0)
  print('FUU u', ctrl.ctrl_obs.last)
  freq = 60 * 2
  root = sx.rootl

  if 0:
    ctrl.debug(True, True)
    ctrl.obs_scene.connect_to(ImageIO(f=lambda *args: helper.update()))
    if not use_gamepad: ctrl.override_ctrl = override_ctrl

    @cmisc.logged_failsafe
    def do_update():
      #ctrl.update()
      if ctrl.ctrl_obs.last:
        ctrl._update_scene(ctrl.ctrl_obs.last)
      helper.update()
      mom = root.compute_momentum(Transform.From(), SpatialVector.Vector())
      print(ctrl.sh.t, 'mom >>> ', mom.around(root.agg_com), root.agg_com)
      return 1 / freq

    do_update()
    bpy.app.timers.register(do_update)
    #th = TimerHelper()
    #th.add(ctrl.update, 1e-1)
    #th.add(lambda: print('123'), 1)
    #input()
    return helper

  t = 0
  for i, ex in enumerate(data):
    for ix in range(params.ndt_const):
      print()
      print()
      print()
      print()
      print(t, i, len(data))
      t += params.dt

      for jk in range(params.integ_nsteps):
        sh.runner.advance(sx, params.actual_dt, ex)
        #sx.update(params.actual_dt, ex)

      for x in sctx.obj2name.keys():
        rl = x.self_link
      print(sx.dump_state())

      helper.update(animation)

  animation.finish()
  cam = helper.cam

  return helper


helper = go()
