#include <vector>
#include <list>
#include <map>
#include <set>
#include <deque>
#include <queue>
#include <stack>
#include <algorithm>
#include <numeric>
#include <utility>
#include <sstream>
#include <iostream>
#include <cstdlib>
#include <string>
#include <cstring>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <cassert>
#include <climits>
//#include <ext/hash_map>
#include <signal.h>
#include <ucontext.h>


using namespace std;
using namespace __gnu_cxx;



#define REP(i,n) for(int i = 0; i < int(n); ++i)
#define REPV(i, n) for (int i = (n) - 1; (int)i >= 0; --i)
#define FOR(i, a, b) for(int i = (int)(a); i < (int)(b); ++i)

#define FE(i,t) for (__typeof((t).begin())i=(t).begin();i!=(t).end();++i)
#define FEV(i,t) for (__typeof((t).rbegin())i=(t).rbegin();i!=(t).rend();++i)

#define two(x) (1LL << (x))
#define ALL(a) (a).begin(), (a).end()


#define pb push_back
#define ST first
#define ND second
#define MP(x,y) make_pair(x, y)

typedef long long ll;
typedef unsigned long long ull;
typedef unsigned int uint;
typedef pair<int,int> pii;
typedef vector<int> vi;
typedef vector<string> vs;
typedef signed char s8;
typedef unsigned char u8;
typedef signed short s16;
typedef unsigned short u16;
typedef signed int s32;
typedef unsigned int u32;
typedef signed long long s64;
typedef unsigned long long u64;

template<class T> void checkmin(T &a, T b){if (b<a)a=b;}
template<class T> void checkmax(T &a, T b){if (b>a)a=b;}
template<class T> void out(T t[], int n){REP(i, n)cout<<t[i]<<" "; cout<<endl;}
template<class T> void out(vector<T> t, int n=-1){for (int i=0; i<(n==-1?t.size():n); ++i) cout<<t[i]<<" "; cout<<endl;}
inline int count_bit(int n){return (n==0)?0:1+count_bit(n&(n-1));}
inline int low_bit(int n){return (n^n-1)&n;}
inline int ctz(int n){return (n==0?-1:ctz(n>>1)+1);}
int toInt(string s){int a; istringstream(s)>>a; return a;}
string toStr(int a){ostringstream os; os<<a; return os.str();}

const int bufsize=1<<26;
char charset[256];
int nx=0;
char buf[bufsize];
#include "data.h"

static ucontext_t mainContext;
int signalFail;

void signalHandler(int signum, siginfo_t* info, void *oldContext) {
    if (signum=SIGFPE){
        signalFail=1;
        setcontext(&mainContext);
    }
}


extern int callMain(const char *pass, char *buf, int buflen) asm("callMain");

int go(const char *pass){
    int sz=0x10000;
    int res;
    char p2[sz];
    strncpy(p2, pass, sz-1);
    signalFail=0;
    assert(getcontext(&mainContext)==0);
    mainContext.uc_mcontext.gregs[REG_EIP]=(int)&&PTR;
    res=callMain(p2, buf, bufsize);
PTR:
    return res;
}

int getMode(){
    int mode=-1;
    FILE *f=fopen("./solver_mode.in", "rb");
    fread(&mode, 4, 1, f);
    fclose(f);
    return mode;
}


void doMode1(){
    char pass[0x27];
    const char *prefix="flag{";
    memset(pass, 'a', 0x26);
    memcpy(pass, prefix, strlen(prefix));

    pass[0x25]='}';
    pass[0x26]=0;


    char charset[256];
    int nx=0;
    for (int i=0; i<128; ++i) if (isprint(i)) charset[nx++]=i;

    int cnt=0;
    int offset=5;
    for (int i=0; i<NumCnd2; ++i){
        memcpy(pass+offset, mode2Cnds[i], strlen(mode2Cnds[i]));
        int res=go(pass);
        if (res!=-4){
            printf("OK %c %c %c %c\n", pass[5], pass[6], pass[7], pass[8]);
            ++cnt;
            //goto eb;
        }
    }
    printf("FOUND >>%d/%d\n", cnt, NumCnd2);
}


void doMode2(){
    char pass[0x27];
    const char *prefix="flag{abcd";
    memset(pass, 'a', 0x26);
    memcpy(pass, prefix, strlen(prefix));

    pass[0x25]='}';
    pass[0x26]=0;



    int cnt=0;
    int offset=5;
    REP(a,nx){
        printf("on %d/%d, have %d\n", a,nx,cnt);
        REP(b,nx){
            REP(c,nx) {
                REP(d,nx){
                    pass[5+0]=charset[a];
                    pass[5+1]=charset[b];
                    pass[5+2]=charset[c];
                    pass[5+3]=charset[d];

                    int res=go(pass);
                    if (res==0xfeed){
                        printf("OK %c %c %c %c\n", pass[5], pass[6], pass[7], pass[8]);
                        ++cnt;
                        //goto eb;
                    }
                }
            }
eb:;
        }
    }
done:;
     printf("FOUND >>%d\n", cnt);
     int res=go(pass)&0xffff;
     printf("RSE>> %x\n", res);
}

void doMode3(){
    char pass[0x27];
    const char *prefix="flag{";
    memset(pass, 'a', 0x26);
    memcpy(pass, prefix, strlen(prefix));

    pass[0x25]='}';
    pass[0x26]=0;

    set<char> seen;

    int pos=9;
    REP(i,NumCnd2) REP(a,nx){
        printf("ON %s\n", pass);
        REP(b,nx) REP(c,nx) REP(d,nx){
            memcpy(pass+5,mode2Cnds[i],strlen(mode2Cnds[i]));
            memcpy(pass+5, "X8eA", 4);
            pass[pos]=charset[a];
            pass[pos+1]=charset[b];
            pass[pos+2]=charset[c];
            pass[pos+3]=charset[d];


            int res=go(pass);
            seen.insert(res);
            if (signalFail) continue;
            if (res==0x6){
                printf("=====OK %s\n", pass);
                //goto eb;
            }
        }
    }
    FE(it,seen) printf(">> %d\n", *it);
done:;
}
void doMode4(){
    char pass[0x27];
    const char *prefix="flag{";
    memset(pass, 'a', 0x26);
    memcpy(pass, prefix, strlen(prefix));
    memcpy(pass+9, " 6lv", 4);

    pass[13]='b';
    pass[0x25]='}';
    pass[0x26]=0;

    set<char> seen;

    int pos=16;
    REP(a,nx) REP(b,nx) REP(c,nx) REP(d,nx){
        pass[pos]=charset[a];
        pass[pos+1]=charset[b];
        pass[pos+2]=charset[c];
        pass[pos+3]=charset[d];

        int res=go(pass);
        if (signalFail) continue;
        if (res==0){
            printf("=====OK %s\n", pass);
            //goto eb;
        }
    }
}




void test(){
}

__attribute__((constructor)) void before(void) {
    int mode=getMode();

    struct sigaction sa;
    sa.sa_sigaction=&signalHandler;
    sa.sa_flags=SA_RESTART | SA_SIGINFO;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGFPE, &sa, NULL);

    for (int i=0; i<128; ++i) if (isprint(i)) charset[nx++]=i;

    printf("GOT MODE >> %d\n", mode);

    if (mode==1)
        doMode1();
    else if (mode==2)
        doMode2();
    else if (mode==3)
        doMode3();
    else if (mode==4)
        doMode4();
    else
        test();



}

