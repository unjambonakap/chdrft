import System
import clr
import sys
import os
sys.path.append(r'C:\Users\JUICER_004\benoit\work\repo\ironpython\ConsoleApp1\bin\Debug')
clr.AddReference('Microsoft.Build.dll')
clr.AddReference('Microsoft.Build.Utilities.Core.dll')
clr.AddReference('Microsoft.Build.Framework.dll')
import Microsoft.Build.Evaluation
from Microsoft.Build.Evaluation import Project
flags = None
#from System.Collection.Generic import *
from System.Collections.Generic import *
kvp = KeyValuePair[str,str]
dictss = Dictionary[str, str]


proj_dirs='''
Tools/SEM_ScannerLib
'''
proj_list = proj_dirs.strip().split()

projfile = './Tools/test/test.csproj'
projfile = './Tools/Benoit.AllLib/Benoit.AllLib.csproj'

def get_files(pdir):
  res = dict(cs=list(), xaml=list(), xaml_cs=list())
  pdir =  os.path.normpath(pdir)
  for root, dirs, files in os.walk(pdir):
    components = root.split(os.path.sep)
    if 'bin' in components or 'obj' in components: continue

    for f in files:
      assert not f.endswith('.g.cs')
      fpath = os.path.join(root, f)
      if f.endswith('.xaml.cs'): res['xaml_cs'].append(fpath)
      elif f.endswith('.cs'):  res['cs'].append(fpath)
      elif f.endswith('.xaml'):  res['xaml'].append(fpath)
  return res

retrieve_typs = ('Compile', 'Page')

def get_files2(slndir, pfile):
  print('Processing ', pfile)
  properties = {'SolutionDir': slndir}
  proj = Project(pfile, dictss(properties), None)
  res = dict()
  for typ in retrieve_typs:
    res[typ] = list(proj.GetItems(typ))
  return res

def extract_proj_info(slndir, proj_dir):
  proj_name = os.path.basename(proj_dir)
  pname = os.path.join(proj_dir, proj_name+'.csproj')
  #files = get_files(proj_dir)
  files = get_files2(slndir, pname)
  res = dict(files=files, paket=os.path.join(proj_dir, 'paket.references'), proj=proj_dir)
  return res

def test_extract(ctx):
  res = extract_proj_info('./', 'Tools/Benoit.Stuff.Lib')
  print(res)
  return
  for proj in proj_list:
    res = extract_proj_info('./', proj)
    print(res)



def create_cs_file(p, rel, x):
  proj.AddItem('Compile', os.path.join(rel, x), [kvp('Link', x)])
def create_xaml_cs_file(p, rel, x):
  proj.AddItem('Compile', os.path.join(rel, x), [kvp('Link', x)])
  proj.AddItem('Compile', os.path.join(rel, x), [kvp('Link', x)])
  proj.AddItem('Compile', os.path.join(rel, x), [kvp('Link', x)])


def create_global_proj(ctx):
  relpath = '../../'
  proj = Project(projfile)
  #proj.RemoveItems(proj.GetItems('Compile'))
  #proj.RemoveItems(proj.GetItems('Folder'))
  #proj.RemoveItems(proj.GetItems('Page'))

  pakets=  set()
  folders =set()
  seen = set()
  for projdir in proj_list:
    res = extract_proj_info('./', projdir)
    pdir = res['proj']
    files = res['files']
    pakets.update(map(str.strip, open(res['paket'], 'r').readlines()))
    #cs, xaml, xaml_cs = files['cs'], files['xaml'], files['xaml_cs']
    #all_files = cs + xaml + xaml_cs

    for typ in retrieve_typs:
      for obj in files[typ]:
        fname = obj.UnevaluatedInclude
        fdir = os.path.dirname(fname)

        fullpath = os.path.normpath(os.path.join(relpath, projdir, fname))
        if fdir.find('$(SolutionDir)')!=-1:
          fdir = fdir.replace('$(SolutionDir)', '.')
        else:
          fdir = os.path.join(projdir, fdir)

        folders.add(fdir)
        if fullpath in seen: continue
        seen.add(fullpath)


        items = proj.AddItem(typ, fullpath)
        assert len(items) == 1
        item = items[0]
        item.SetMetadataValue('Link', os.path.join(projdir, fname))
        for metadata in obj.Metadata:
          item.SetMetadataValue(metadata.Name, metadata.UnevaluatedValue)


  added_folder=  set()
  for folder in folders:
    while folder:
      if folder.endswith('\\'): folder = folder[:-1]
      if folder in added_folder: break
      added_folder.add(folder)
      proj.AddItem('Folder', folder)
      folder = os.path.dirname(folder)


  print('\n'.join(pakets))
  proj.Save()

def add_embedded_resources(ctx):
  relpath = '../../'
  proj = Project(projfile)
  global retrieve_typs
  retrieve_typs = ['EmbeddedResource']

  for typ in retrieve_typs: proj.RemoveItems(proj.GetItems(typ))
  pakets=  set()
  folders =set()
  seen = set()
  for projdir in proj_list:
    res = extract_proj_info('./', projdir)
    pdir = res['proj']
    files = res['files']

    for typ in retrieve_typs:
      for obj in files[typ]:
        fname = obj.UnevaluatedInclude
        fdir = os.path.dirname(fname)

        fullpath = os.path.normpath(os.path.join(relpath, projdir, fname))
        if fdir.find('$(SolutionDir)')!=-1:
          fdir = fdir.replace('$(SolutionDir)', '.')
        else:
          fdir = os.path.join(projdir, fdir)

        folders.add(fdir)
        if fullpath in seen: continue
        seen.add(fullpath)


        items = proj.AddItem(typ, fullpath)
        assert len(items) == 1
        item = items[0]
        item.SetMetadataValue('Link', os.path.join(projdir, fname))
        for metadata in obj.Metadata:
          item.SetMetadataValue(metadata.Name, metadata.UnevaluatedValue)


  added_folder=  set()
  for folder in folders:
    while folder:
      if folder.endswith('\\'): folder = folder[:-1]
      if folder in added_folder: break
      added_folder.add(folder)
      proj.AddItem('Folder', folder)
      folder = os.path.dirname(folder)


  print('\n'.join(pakets))
  proj.Save()

def test1(ctx):
  proj_fullpath = os.path.join(ctx.proj, projfile)
  proj = Project(proj_fullpath)
  proj.RemoveItems(proj.GetItems('Compile'))
  proj.RemoveItems(proj.GetItems('Folder'))



  proj.AddItem('Folder', r'Test')
  proj.AddItem('Compile', r'..\SEM_Scanner\Misc\SEMInteractions.cs')
  proj.AddItem('Page', r'..\Benoit.Stuff.Lib\Assets\TestDictionary.xaml', [kvp('Link', 'Test/TestDictionary.xaml')])
  proj.Save()
#      <Generator>MSBuild:Compile</Generator>
#      <SubType>Designer</SubType>
#    </Page>

def dump(ctx):
  proj = Project(ctx.proj)
  for x in proj.GetItems('Compile'):
    print(x.EvaluatedInclude, x.ItemType, x.UnevaluatedInclude, x.GetMetadataValue('Link'))



def main():
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('--proj', type=str)
  parser.add_argument('--action', type=str)
  args = sys.argv
  idx = args.index('--')
  global flags

  flags = parser.parse_args(sys.argv[idx+1:])
  globals()[flags.action](flags)

if __name__ == '__main__':
  main()
