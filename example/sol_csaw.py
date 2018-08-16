#!/usr/bin/python


import chdrft.GdbDebugger
import time
import sys
import struct
import itertools as it
import curses.ascii
import binascii
import elftools.elf.elffile as EF
import random
import traceback

from chdrft.DebuggerCommon import Status

def display_as_hex(item):
    if isinstance(item, (int, long)):
        print hex(item)
    else:
        print repr(item)

sys.displayhook = display_as_hex

class Runner:
    def __init__(self):
        self.msg=""
        self.firstRun=True
        self.last=None
        self.wantContext=False
        self.verbose=True

        self.charset=[]
        for i in range(128):
            if curses.ascii.isprint(chr(i)) and chr(i)!="'":
                self.charset.append(chr(i))


    def init(self, args):
        self.x=chdrft.GdbDebugger.GdbDebugger()

        self.x.loadDebugger()
        entryBpt=self.x.addEntryBreakpoint()

        VERIFY_FUNC_CALL=0x080534F5
        verifyBpt=self.x.setOrEnableBreakpoint(VERIFY_FUNC_CALL)


        self.x.startDebugger(args)

        self.x.resume()

        y=self.x.reg
        print hex(y.esi)
        print hex(y.edi)
        self.passAddr=self.x.getU32(y.esi+y.edi-0x34)
        self.resultAddr=y.esp+0x28
        assert self.x.getMemory(self.passAddr, len(args))==args

        self.x.resume()
        self.waitEnd()

        self.x.deleteBreakpoint(entryBpt)
        self.x.deleteBreakpoint(verifyBpt)

    def dispMem(self):
        esp=self.x.getRegister('esp')
        l=0x300

        mem=self.x.getMemory(esp, l)
        tb=''.join([mem[j] for j in range(l)])
        tb=binascii.hexlify(tb)

        if self.last is not None:
            for i in range(len(tb)):
                c=' '
                if tb[i]!=self.last[i]:
                     c=tb[i]
                sys.stdout.write(c)
            print ""
        self.last=tb

    def waitEnd(self):
        while True:

            status=self.x.getStatus()
            if status not in (Status.RUNNING, Status.STOPPED):
                break
            #self.dispMem()

            msg=self.getMessage()
            if self.verbose:
                print "EIP=0x%08x"%self.x.getEip()
                print "RESULT=0x%08x"%self.x.getU32(self.resultAddr)
                if msg is not None:
                    print msg
                print ""

                if self.wantContext:
                    self.x.showContext()

            #self.x.showContext()
            self.x.resume()



    def doRun1(self, args):
        def stopHandler(x):
            msg=None
            if x.isBpActive(self.bPass):
                msg="Pass >> %08x"%x.getU32(self.curPassAddr)
            elif x.isBpActive(self.bResult):
                #msg="Result >> %08x"%x.getU32(self.resultAddr)
                pass

        self.setMessage(msg)


        self.x.startDebugger(args)
        self.x.setStopHandler(self.stopHandler)
        self.msg=None
        self.waitEnd()

        ans=0
        #ans=self.bPass.hit_count
        return ans




    def getMessage(self):
        return self.msg

    def setMessage(self, msg):
        self.msg=msg



    def go(self):

        sol=[0x66, 0x6c, 0x61, 0x67, 0x7b]
        res=[chr(x) for x in sol]
        print(''.join(res))
        args=['a']*0x26



        def trySolve(res):
            for i in range(len(res)):
                args[i]=res[i]


            tmp=''.join(args)
            print tmp
            ans=self.doRun1(tmp)
            if ans!=5:
                print " result >> ",ans
                print len(tmp)
                print ""


        self.init('a'*0x26)
        res.append(' ')
        res.append(' ')
        res.append(' ')
        res.append('f')
        res.append('x')
        res.append('A')
        res.append(' ')
        res.append(';')
        res.append('s')
        res.append('&')
        trySolve(res)
        return


        for i in range(2):
            res.append('a')


        for x in it.product(charset, charset):
            for i in range(len(x)):
                res[len(sol)+i]=x[i]
            trySolve(res)

    def go2(self):
        pw=bytearray('flag{')
        pw+='a'*(0x25-len(pw))
        pw+='}'

        charset=[]
        for i in range(26):
            charset.append(chr(ord('a')+i))


        self.init(pw)

        fx=self.x.getFilePath()
        textSection=None
        with open(fx, 'r') as f:
            elf=EF.ELFFile(f)
            textSection=elf.get_section_by_name('.text')

        s1=textSection['sh_addr']
        self.textSectionRange=(s1, s1+textSection['sh_size'])



        offset=21
        self.bp1=self.x.setHardwareBreakpoint(self.passAddr+offset)

        self.msg=None
        self.cnt=0

        def considerBpt(eip):
            if eip<self.textSectionRange[0] or self.textSectionRange[1]<=eip:
                return False
            if eip in (0x8048aaa, 0x8048ab4, 0x8048abe, 0x8048ac8):
                return False
            return True

        def stopHandler1(x):
            try:
                self.wantContext=False
                if x.isBpActive(self.bp1):
                    eip=self.x.reg.eip
                    if considerBpt(eip):
                        self.cnt+=1
                        self.wantContext=True
            except Exception as e:
                self.setMessage("Exception >> "+str(e))


        self.x.setStopHandler(stopHandler1)

        n=4
        for i in it.product(charset, repeat=n):
            print i
            for j in range(n):
                pw[5+j]=i[j]
            pw[5:9]='1_ea'
            print pw

            self.cnt=0
            self.x.startDebugger(pw)
            self.waitEnd()
            if self.cnt!=0:
                print "FOUND >> >", pw
                print "count >> ", self.cnt
                sys.exit(0)
            print "FAIL"
            break

        print map(hex, self.textSectionRange)



    def eval3(self, pw):
        self.out1=None
        self.out2=None
        self.out3=None
        self.out4=None
        self.out42=None
        self.out5=None
        self.out7=[]
        self.out8=None
        self.out9=None
        self.out10=None

        self.cnt=0
        self.countBpPass=False
        self.ret=None

        def stopHandler(x):
            r=self.x.reg
            try:
                self.wantContext=False

                if x.isBpActive(self.bp1):
                    self.out1=r.cl
                    self.countBpPass=True

                elif x.isBpActive(self.bp2):
                    self.out2=r.ecx

                elif x.isBpActive(self.bp3):
                    n=r.edx
                    e1=r.ecx
                    e2=r.eax
                    s1=self.x.getMemory(e1, n)
                    s2=self.x.getMemory(e2, n)
                    self.out3=(n,s1,s2,e1,e2)

                elif x.isBpActive(self.bp4):
                    self.out4=r.eax
                elif x.isBpActive(self.bp42):
                    self.out42=r.eax

                elif x.isBpActive(self.bp5):
                    self.out5=r.ecx
                    self.countBpPass=False


                elif x.isBpActive(self.bp6):
                    self.countBpPass=True
                    self.cnt=0
                elif x.isBpActive(self.ebp6):
                    pass
                elif self.bpPass is not None and x.isBpActive(self.bpPass):
                    if self.countBpPass:
                        self.cnt=self.cnt+1

                elif x.isBpActive(self.bp7):
                    self.out7.append((r.esi, r.eax))
                elif x.isBpActive(self.bp8):
                    self.out8=self.x.getU32(r.esp+0x104)
                elif x.isBpActive(self.bp9):
                    self.out9=self.x.getU32(r.esp+0x40)

                elif x.isBpActive(self.bp10):
                    self.out10=r.eax

                elif x.isBpActive(self.bpRet):
                    self.ret=r.eax



            except Exception as e:
                self.setMessage("Exception >> "+str(e)+
                                "\nStacktrace: "+traceback.format_exc())
                self.verbose=False

        self.x.setStopHandler(stopHandler)
        self.x.startDebugger(pw, silent=True)
        self.waitEnd()


    def go3(self):
        nx=0x26
        pw=bytearray('flag{')
        pw+='X8eA'
        pw+=' 6lv'

        pw+="a"*(nx-len(pw))
        pw[nx-1]='}'

        self.init(pw)
        self.x.deleteAllBreakpoints()


        self.verbose=False
        self.bp1=self.x.setOrEnableBreakpoint(0x804a6a9)
        self.bp2=self.x.setOrEnableBreakpoint(0x804a6bb)
        self.bp1=self.x.setOrEnableBreakpoint(0x804a6a9)
        self.bp3=self.x.setOrEnableBreakpoint(0x804bee3)
        self.bp4=self.x.setOrEnableBreakpoint(0x804be76)
        self.bp42=self.x.setOrEnableBreakpoint(0x804be8b)
        self.bp5=self.x.setOrEnableBreakpoint(0x804c8e7)

        self.bp6=self.x.setOrEnableBreakpoint(0x804ab6b)
        self.ebp6=self.x.setOrEnableBreakpoint(0x804ab70)

        self.bp7=self.x.setOrEnableBreakpoint(0x804d626)
        self.bp8=self.x.setOrEnableBreakpoint(0x804de8f)
        self.bp9=self.x.setOrEnableBreakpoint(0x804ee03)
        self.bp10=self.x.setOrEnableBreakpoint(0x804eea2)
        self.bpRet=self.x.setOrEnableBreakpoint(0x8053535)

        self.bpPass=None


        #for i in range(nx):
        #    self.bpPass=self.x.setHardwareBreakpoint(self.passAddr+i)
        #    self.eval3(pw)
        #    print i, self.cnt, self.countBpPass
        #    self.bpPass.delete()
        #sys.exit(0)

        #for i in '0123456':
        #    pw[11]=i
        #    self.eval3(pw)
        #    print 'ok ', pw, self.out4, self.out42
        #sys.exit(0)

        #t1='o! fee'
        #pw[13:19]=t1

        #self.eval3(pw)
        #print self.out5


        #pw[13]='b'
        #self.eval3(pw)
        #print pw, self.out1, self.out2, self.out3, self.out2, self.out5
        #print self.out3[3]-self.passAddr

        #for s in range(9, nx-1):
        #    for j in range(5):
        #        pw[s]=random.choice(self.charset)
        #        self.eval3(pw)
        #        print s, pw, self.out1, self.out2, self.out3, self.out2, self.out5
        #sys.exit(0)

        pw[13]='a'
        pw[16]='0'
        pw[17]='F'
        pw[18]='a'
        pw[19]='a'
        pw[20]='d'


        #self.eval3(pw)
        #print pw, self.out1, self.out2, self.out3, self.out2, self.out5
        #sys.exit(0)

        #s=20
        #l=1
        #self.bpPass=self.x.setHardwareBreakpoint(self.passAddr+s)
        #sx=self.charset
        #sx='0123456789abcdef'
        #for j in it.product(sx, repeat=l):
        ##for s in range(14,nx-1):
        #    pw[s:s+l]=j
        #    #pw[s]=random.choice(self.charset)

        #    self.eval3(pw)
        #    print s, pw, self.out1, self.out2, self.out3, self.out2, self.out5
        #    #if self.out4!=None:
        #    #    #print "'%s',"%j
        #    #    print pw, self.out4, self.out42, self.cnt



        #print pw
        #self.eval3(pw)
        #print pw, self.out7

        pw[27]='7'
        pw[28]='1'
        pw[25]='3'
        pw[26]='1'
        pw[23]='5'
        pw[24]='0'
        pw[21]='5'
        pw[22]='7'

        ##for s in range(21, nx-1):
        ##    old=pw[s]
        ##    for j in range(5):
        ##        pw[s]=random.choice(self.charset)
        ##        self.eval3(pw)
        ##        print s, pw, self.out7
        ##    pw[s]=old
        ##sys.exit(0)

        #l=2
        #sset=range(27, nx-1)
        #sset=[21]
        #wh=3
        #for s in sset:
        #    sx=self.charset
        #    sx='0123456789abcdefABCDEF'
        #    for j in it.product(sx, repeat=l):
        #        pw[s:s+l]=j
        #        self.eval3(pw)
        #        print self.out7

        #        if len(self.out7)>wh:
        #            u=self.out7[wh]
        #            if u[0]==u[1]:
        #                print s, pw, self.out7
        #            pw[s]='a'


        #print pw
        #self.eval3(pw)
        #print pw, self.out8
        #for s in range(21, nx-1):
        #    old=pw[s]
        #    for j in range(50):
        #        pw[s]=random.choice(self.charset)
        #        self.eval3(pw)
        #        if self.out8 is not None:
        #            print s, pw, hex(self.out8)
        #    pw[s]=old
        ##sys.exit(0)

        #sset=[21,22]
        #l=1
        #for s in sset:
        #    sx='0123456789abcdefABCDEF'
        #    sx=self.charset
        #    for j in it.product(sx, repeat=l):
        #        pw[s:s+l]=j
        #        self.eval3(pw)

        #        if self.out8==0x169:
        #            print s, pw, self.out8

        pw[21]='^'

        #print pw
        #self.eval3(pw)
        #print pw, self.out8
        #for s in range(29, nx-1):
        #    old=pw[s]
        #    for j in range(50):
        #        pw[s]=random.choice(self.charset)
        #        self.eval3(pw)
        #        if self.out9 is not None:
        #            print s, pw, hex(self.out9)
        #    pw[s]=old
        ##sys.exit(0)

        #sset=[35]
        #l=2
        #for s in sset:
        #    sx=self.charset
        #    sx='0123456789abcdefABCDEF'
        #    for j in it.product(sx, repeat=l):
        #        pw[s:s+l]=j
        #        self.eval3(pw)

        #        if self.out9<2:
        #            print s, pw, self.out9

        #pw[29]='2'
        #pw[30]='b'
        #pw[31]='2'
        #pw[32]='C'
        #pw[33]='2'
        #pw[34]='C'
        pw[35]='2'
        pw[36]='B'
        #flag{X8eA 6lvaaa0Faad^7503171aaAaaa2B}

        print pw
        self.eval3(pw)
        print pw, self.out9, self.ret
        for s in range(31, nx-1):
            old=pw[s]
            for j in range(20):
                pw[s]=random.choice(self.charset)
                self.eval3(pw)
                print s, pw, self.out9, int(self.ret)
            #pw[s]=old
        #sys.exit(0)



def go():
    u=Runner()
    #u.go2()

    u.go3()



