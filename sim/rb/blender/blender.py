#!/usr/bin/env python
# coding: utf-8

# In[1]:


init_jupyter()


# In[2]:


import bpy
import sys
import traceback as tb
import addon_utils
from mathutils import Matrix
from chdrft.display.blender import clear_scene

addon_utils.enable('io_kspblender')
clear_scene()
bpy.context.preferences.addons.get('io_object_mu').preferences.GameData = '/home/benoit/work/links/Kerbal Space Program/GameData'

#bpy.ops.import_object.ksp_mu(filepath="/home/benoit/programmation/projects/ksp/assets/M1DV+.mu")
bpy.ops.import_object.ksp_craft(filepath="/home/benoit/work/links/Kerbal Space Program/saves/aaa/Ships/VAB/SpaceX Falcon 9 Block 5.craft")

#bpy.ops.export_mesh2.stl(filepath='/tmp/res.stl', use_selection=True, use_mesh_modifiers=False)
#return


# In[39]:


b = bpy.data.objects['KK.SPX.Merlin1DV+']
a = bpy.data.collections['KK.SPX.Merlin1D++:partmodel']


# In[41]:


print(a.mumodelprops.config)


# In[38]:


for x in bpy.data.collections:
    print('>>> ', x.name, '<<<', x.mumodelprops.config[:10])


# In[34]:


b.mumodelprops.config


# In[30]:


b.muproperties.i


# In[13]:


u = x.to_mesh()


# In[9]:


for x in a.objects:
    print(x)


# In[9]:


objname = 'KK_SpX_MvacDp'
objname = 'SpaceX Falcon 9 Block 5'
obj = bpy.data.objects[objname]
me = MeshExporter(bl=lambda x: 'launchClamp' in x or 'KK.SpX.FRH' in x or 'Gridfin' in x or 'LandingLeg' in x)
me.walk(obj, Matrix())
len(me.data)


# In[3]:


from chdrft.display.blender import blender_obj_to_meshio
filter_bad = lambda x: 'launchClamp' in x or 'KK.SpX.FRH' in x or 'Gridfin' in x or 'LandingLeg' in x

objname = 'KK_SpX_MvacDp'
objname = 'KK.SPX.F9FT.Interstage.RCS'
objname = 'SpaceX Falcon 9 Block 5'
obj = bpy.data.objects[objname]
res = blender_obj_to_meshio(obj, filter_bad)


# In[6]:


res.write('/home/benoit/programmation/projects/rockets/data/block5.stl')


# In[5]:


import os
os.getcwd()


# In[ ]:





# In[28]:


lst = [x for x in me.data if x.name == 'nozzle']
lst = me.data
for x in lst:
    print(x.chain)
e = merge_meshdata(lst)
import meshio
m = meshio.Mesh(e.vertices, [('triangle', e.faces)])
m.write('/tmp/res.stl')


# In[37]:


obj.delta_rotation_quaternion

