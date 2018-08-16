import idc
import idaapi
import idautils

class Status:
    NONE = 0
    RUNNING = 1
    TERMINATED = 2

class IdaDebugger:
    def __init__(self):
        pass

    def doSyncCall(self, *args, **kwargs):
        def tmpFunc():
            try:
                res=self.activeFunc(self, *args, **kwargs)
                self.tmpres=res
            except Exception as e:
                print e
                raise e
            return 0

        res=idaapi.execute_sync(tmpFunc, idaapi.MFF_FAST)

        if res==-1:
            raise Exception('idaapi execute_sync call failed, was for '+str(tmpFunc))
        return self.tmpres


    def __getattr__(self, name):
        syncName="_sync_"+name
        if syncName in IdaDebugger.__dict__:
            func=IdaDebugger.__dict__[syncName]
            self.activeFunc=func
            return self.doSyncCall
        else:
            raise AttributeError

    def _sync_disableBreakpoints(self):
        n=idc.GetBptQty()
        for i in range(n):
            ea=idc.GetBptEA(i)
            idc.EnableBpt(ea, False)

    def _sync_setOrEnableBreakpoint(self, ea):
        if not idc.AddBpt(ea):
            idc.EnableBpt(ea, True)


    def _sync_stopDebugger(self):
        idc.StopDebugger()
        idc.GetDebuggerEvent(idc.WFNE_SUSP, -1)

    def _sync_addEntryBreakpoint(self):
        idc.SetDebuggerOptions(idc.DOPT_ENTRY_BPT)

    def _sync_loadDebugger(self):
        return idc.LoadDebugger('linux', 0)


    def _sync_startDebugger(self, args):
        idc.StartDebugger('', args, '')

    def _sync_resume(self):
        return idc.GetDebuggerEvent(idc.WFNE_CONT | idc.WFNE_SUSP, -1)

    def _sync_wait(self):
        return idc.GetDebuggerEvent(idc.WFNE_SUSP, -1)

    def _sync_stepInto(self):
        idc.StepInto()
        idc.GetDebuggerEvent(idc.WFNE_SUSP, -1)

    def _sync_getStatus(self):
        status=idc.GetDebuggerEvent(idc.WFNE_SUSP, -1)
        if status==None:
            return Status.NONE
        elif status==idc.NOTASK:
            return Status.TERMINATED
        else:
            return Status.RUNNING


    def getEip(self):
        return idautils.cpu.eip

    def getRegister(self, reg):
        return idautils.cpu.__getattr__(reg)

    def getMemory(self, pos, l):
        return idc.DbgRead(pos, l)

if __name__ == '__main__':
    x=IdaDebugger()
    x.resume()


