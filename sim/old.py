


#class RBCompactor:
#
#  def __init__(self, wl: Transform, rotvec_w: Vec3):
#    self.wl = wl
#    self.statics: list[Tuple[Transform, RigidBodyLink]] = []
#    self.rotvec_w = rotvec_w
#    self.res_children: list[RigidBodyLink] = []
#
#  @classmethod
#  def Build(cls, link: RigidBodyLink) -> RigidBodyLink:
#    return link.child.ctx.compose(cls.Compact(link))
#
#  @classmethod
#  def Compact(cls,
#              link: RigidBodyLink,
#              wl: Transform = None,
#              rotvec_w: Vec3 = None) -> list[RigidBodyLink]:
#    if wl is None:
#      wl = link.wl
#      rotvec_w = link.rotvec_w
#    return cls(wl, rotvec_w).go(link)
#
#  def go(self, link: RigidBodyLink) -> list[RigidBodyLink]:
#    self.dfs(link, Transform.From())
#    mass = sum([x.child.spec.mass for wl, x in self.statics], 0)
#    com = Vec3.Zero()
#    tensor = InertialTensor()
#    tsf = self.wl.clone()
#    if mass > 0:
#      com = sum([wl.pos_v * x.child.spec.mass for wl, x in self.statics], Vec3.Zero()) / mass
#      tensor = sum(
#          [
#              x.child.spec.inertial_tensor.shift_inertial_tensor(com - wl.pos_v, x.child.spec.mass
#                                                                ).get_world_tensor(wl)
#              for wl, x in self.statics
#          ], InertialTensor()
#      )
#      tsf.pos_v -= com
#
#    spec = SolidSpec(
#        mass=mass,
#        inertial_tensor=tensor,
#    )
#    base_name = '|'.join(x.child.name for wl, x in self.statics)
#    cur = RigidBodyLink(
#        child=RigidBody(spec=spec, ctx=link.child.ctx, base_name=base_name),
#        wl=tsf,
#        move_desc=MoveDesc.From(self.rotvec_w)
#    )
#    self.res_children.append(cur)
#
#    return self.res_children
#
#  def dfs(self, link: RigidBodyLink, wl: Transform):
#    self.statics.append((wl, link))
#    for clink in link.child.links:
#      nwl = wl @ clink.wl
#      if clink.link_data.static:
#        self.dfs(clink, nwl)
#      else:
#        self.res_children.extend(
#            RBCompactor.Compact(clink, nwl, self.rotvec_w + wl.tsf_rot @ clink.rotvec_w)
#        )
#
