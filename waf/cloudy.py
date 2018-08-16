#!/usr/bin/env python

from chdrft.utils.misc import Attributize, OpaInit, cwdpath
from chdrft.gen.opa_clang import OpaIndex, OpaModifier, EasyFilter
from chdrft.gen.types import OpaType
from clang.cindex import conf, CursorKind
import re
import argparse


FLAGS=None


def generate_builder(idx):

    print('\n\nBUILDER\n')
    take={}
    for k, v in idx.typs.items():
        if not k: continue
        if not v.comment: continue

        com=v.comment.decode()
        m=re.search('TGEN: ([\w ,]*)', com)
        if not m: continue

        want=[x.rstrip().lstrip() for x in m.group(1).split(',')]
        all='ALL' in want
        tb=[]

        for field in v.fields:
            if not all and not field.name in want: continue
            tb.append(field)
        assert len(tb)
        take[v]=tb


    with open(FLAGS.dest_file, 'w') as f:
        for k, v in take.items():
            loads=[]
            stores=[]

            for pos,field in enumerate(v):
                typ=field.typ.desc
                    loads.append('{}.load(*any.add_any());'.format(field.name))
                    stores.append('{}.store(any.any({});'.format(field.name, pos))
                elif typ.starts_with('std::shared_ptr'):
                elif typ.start_with('std::basic_string'):
                elif typ.start_with('std::basic_string'):
                    loads.append('opa::threading::Baseany.set_{}(')





            body_load='\n'.join(loads)
            body_store='\n'.join(stores)
            s="""
void {name}::load(const AnyMsg &any){
{body_load}
}
void {name}::store(AnyMsg &any){
{body_store}
}
"""
            f.write(s.format(name=k.desc, **locals())







def main():

    parser=argparse.ArgumentParser()
    parser.add_argument('dest_file',type=cwdpath, required=True) 

    global FLAGS
    FLAGS=parser.parse_args()

    content = """
    #include <algorithm>
    namespace KAPPA{
    struct JAMBON{};
    }

    template<class A1, class A2>
    struct B{
    A2 xx;
    A1 bb;
    std::shared_ptr<A1> ptr;
    };

B<double,KAPPA::JAMBON> FUUU;

        //namespace test{
        //int x;
        //class Y{};
        //template <int kInt, typename T, typename X, bool kBool>
        //class foo{
        //T x;
        //X y;
        //std::pair<T,X> pp;
        //};
        //}

        //test::foo<-7, float, bool, true> abc;
        //template<class X>
        //struct Test{
        //test::foo<3,X,bool, false> uuu;

        //};

        //Test<int> kappa;
    """

    args = ['-isystem/usr/lib/llvm-3.6/lib/clang/3.6.2/include', '-I/home/benoit/programmation/opa/common/inc']

    filter = EasyFilter(filter_file=lambda x: x.is_input)
    x = OpaIndex.create_index(filter,
                              cpp_mode=True,

                              filename='/home/benoit/programmation/hack/hacklu/12/given/solve.cpp',
                              #file_content=content,
                              args=args)

    generate_builder(x)


OpaInit(globals(), main)

