
#special registers for cc1110
cc1110_prefix='''
.globl P0
.globl SP
.globl DPL0
.globl DPH0
.globl DPL1
.globl DPH1
.globl U0CSR
.globl PCON
.globl TCON
.globl P0IFG
.globl P1IFG
.globl P2IFG
.globl PICTL
.globl P1IEN
.globl _SFR8E
.globl P0INP
.globl P1
.globl RFIM
.globl DPS
.globl MPAGE
.globl _XPAGE
.globl _SFR94
.globl _SFR95
.globl _SFR96
.globl _SFR97
.globl S0CON
.globl _SFR99
.globl IEN2
.globl S1CON
.globl T2CT
.globl T2PR
.globl T2CTL
.globl _SFR9F
.globl P2
.globl WORIRQ
.globl WORCTRL
.globl WOREVT0
.globl WOREVT1
.globl WORTIME0
.globl WORTIME1
.globl _SFRA7
.globl IEN0
.globl IP0
.globl _SFRAA
.globl FWT
.globl FADDRL
.globl FADDRH
.globl FCTL
.globl FWDATA
.globl _SFRB0
.globl ENCDI
.globl ENCDO
.globl ENCCS
.globl ADCCON1
.globl ADCCON2
.globl ADCCON3
.globl _SFRB7
.globl IEN1
.globl IP1
.globl ADCL
.globl ADCH
.globl RNDL
.globl RNDH
.globl SLEEP
.globl _SFRBF
.globl IRCON
.globl U0DBUF
.globl U0BAUD
.globl _SFRC3
.globl U0UCR
.globl U0GCR
.globl CLKCON
.globl MEMCTR
.globl _SFRC8
.globl WDCTL
.globl T3CNT
.globl T3CTL
.globl T3CCTL0
.globl T3CC0
.globl T3CCTL1
.globl T3CC1
.globl PSW
.globl DMAIRQ
.globl DMA1CFGL
.globl DMA1CFGH
.globl DMA0CFGL
.globl DMA0CFGH
.globl DMAARM
.globl DMAREQ
.globl TIMIF
.globl RFD
.globl T1CC0L
.globl T1CC0H
.globl T1CC1L
.globl T1CC1H
.globl T1CC2L
.globl T1CC2H
.globl ACC
.globl RFST
.globl T1CNTL
.globl T1CNTH
.globl T1CTL
.globl T1CCTL0
.globl T1CCTL1
.globl T1CCTL2
.globl IRCON2
.globl RFIF
.globl T4CNT
.globl T4CTL
.globl T4CCTL0
.globl T4CC0
.globl T4CCTL1
.globl T4CC1
.globl B
.globl PERCFG
.globl ADCCFG
.globl P0SEL
.globl P1SEL
.globl P2SEL
.globl P1INP
.globl P2INP
.globl U1CSR
.globl U1DBUF
.globl U1BAUD
.globl U1UCR
.globl U1GCR
.globl P0DIR
.globl P1DIR
.globl P2DIR

        .area RSEG    (ABS,DATA)
        .org 0x0000
P0      =       0x0080
SP      =       0x0081
DPL0    =       0x0082
DPH0    =       0x0083
DPL1    =       0x0084
DPH1    =       0x0085
U0CSR   =       0x0086
PCON    =       0x0087
TCON    =       0x0088
P0IFG   =       0x0089
P1IFG   =       0x008a
P2IFG   =       0x008b
PICTL   =       0x008c
P1IEN   =       0x008d
_SFR8E  =       0x008e
P0INP   =       0x008f
P1      =       0x0090
RFIM    =       0x0091
DPS     =       0x0092
MPAGE   =       0x0093
_XPAGE  =       0x0093
_SFR94  =       0x0094
_SFR95  =       0x0095
_SFR96  =       0x0096
_SFR97  =       0x0097
S0CON   =       0x0098
_SFR99  =       0x0099
IEN2    =       0x009a
S1CON   =       0x009b
T2CT    =       0x009c
T2PR    =       0x009d
T2CTL   =       0x009e
_SFR9F  =       0x009f
P2      =       0x00a0
WORIRQ  =       0x00a1
WORCTRL =       0x00a2
WOREVT0 =       0x00a3
WOREVT1 =       0x00a4
WORTIME0        =       0x00a5
WORTIME1        =       0x00a6
_SFRA7  =       0x00a7
IEN0    =       0x00a8
IP0     =       0x00a9
_SFRAA  =       0x00aa
FWT     =       0x00ab
FADDRL  =       0x00ac
FADDRH  =       0x00ad
FCTL    =       0x00ae
FWDATA  =       0x00af
_SFRB0  =       0x00b0
ENCDI   =       0x00b1
ENCDO   =       0x00b2
ENCCS   =       0x00b3
ADCCON1 =       0x00b4
ADCCON2 =       0x00b5
ADCCON3 =       0x00b6
_SFRB7  =       0x00b7
IEN1    =       0x00b8
IP1     =       0x00b9
ADCL    =       0x00ba
ADCH    =       0x00bb
RNDL    =       0x00bc
RNDH    =       0x00bd
SLEEP   =       0x00be
_SFRBF  =       0x00bf
IRCON   =       0x00c0
U0DBUF  =       0x00c1
U0BAUD  =       0x00c2
_SFRC3  =       0x00c3
U0UCR   =       0x00c4
U0GCR   =       0x00c5
CLKCON  =       0x00c6
MEMCTR  =       0x00c7
_SFRC8  =       0x00c8
WDCTL   =       0x00c9
T3CNT   =       0x00ca
T3CTL   =       0x00cb
T3CCTL0 =       0x00cc
T3CC0   =       0x00cd
T3CCTL1 =       0x00ce
T3CC1   =       0x00cf
PSW     =       0x00d0
DMAIRQ  =       0x00d1
DMA1CFGL        =       0x00d2
DMA1CFGH        =       0x00d3
DMA0CFGL        =       0x00d4
DMA0CFGH        =       0x00d5
DMAARM  =       0x00d6
DMAREQ  =       0x00d7
TIMIF   =       0x00d8
RFD     =       0x00d9
T1CC0L  =       0x00da
T1CC0H  =       0x00db
T1CC1L  =       0x00dc
T1CC1H  =       0x00dd
T1CC2L  =       0x00de
T1CC2H  =       0x00df
ACC     =       0x00e0
RFST    =       0x00e1
T1CNTL  =       0x00e2
T1CNTH  =       0x00e3
T1CTL   =       0x00e4
T1CCTL0 =       0x00e5
T1CCTL1 =       0x00e6
T1CCTL2 =       0x00e7
IRCON2  =       0x00e8
RFIF    =       0x00e9
T4CNT   =       0x00ea
T4CTL   =       0x00eb
T4CCTL0 =       0x00ec
T4CC0   =       0x00ed
T4CCTL1 =       0x00ee
T4CC1   =       0x00ef
B       =       0x00f0
PERCFG  =       0x00f1
ADCCFG  =       0x00f2
P0SEL   =       0x00f3
P1SEL   =       0x00f4
P2SEL   =       0x00f5
P1INP   =       0x00f6
P2INP   =       0x00f7
U1CSR   =       0x00f8
U1DBUF  =       0x00f9
U1BAUD  =       0x00fa
U1UCR   =       0x00fb
U1GCR   =       0x00fc
P0DIR   =       0x00fd
P1DIR   =       0x00fe
P2DIR   =       0x00ff

;--------------------------------------------------------
; special function bits
;--------------------------------------------------------
        .area RSEG    (ABS,DATA)
        .org 0x0000
P0_0   =       0x0080
P0_1   =       0x0081
P0_2   =       0x0082
P0_3   =       0x0083
P0_4   =       0x0084
P0_5   =       0x0085
P0_6   =       0x0086
P0_7   =       0x0087
IT0    =       0x0088
RFTXRXIF       =       0x0089
IT1    =       0x008a
URX0IF =       0x008b
ADCIF  =       0x008d
URX1IF =       0x008f
P1_0   =       0x0090
P1_1   =       0x0091
P1_2   =       0x0092
P1_3   =       0x0093
P1_4   =       0x0094
P1_5   =       0x0095
P1_6   =       0x0096
P1_7   =       0x0097
ENCIF_0        =       0x0098
ENCIF_1        =       0x0099
P2_0   =       0x00a0
P2_1   =       0x00a1
P2_2   =       0x00a2
P2_3   =       0x00a3
P2_4   =       0x00a4
P2_5   =       0x00a5
P2_6   =       0x00a6
P2_7   =       0x00a7
RFTXRXIE       =       0x00a8
ADCIE  =       0x00a9
URX0IE =       0x00aa
URX1IE =       0x00ab
ENCIE  =       0x00ac
STIE   =       0x00ad
EA     =       0x00af
DMAIE  =       0x00b8
T1IE   =       0x00b9
T2IE   =       0x00ba
T3IE   =       0x00bb
T4IE   =       0x00bc
P0IE   =       0x00bd
DMAIF  =       0x00c0
T1IF   =       0x00c1
T2IF   =       0x00c2
T3IF   =       0x00c3
T4IF   =       0x00c4
P0IF   =       0x00c5
STIF   =       0x00c7
P      =       0x00d0
F1     =       0x00d1
OV     =       0x00d2
RS0    =       0x00d3
RS1    =       0x00d4
F0     =       0x00d5
AC     =       0x00d6
CY     =       0x00d7
T3OVFIF        =       0x00d8
T3CH0IF        =       0x00d9
T3CH1IF        =       0x00da
T4OVFIF        =       0x00db
T4CH0IF        =       0x00dc
T4CH1IF        =       0x00dd
OVFIM  =       0x00de
ACC_ERASE  =       0x00e0
ACC_WRITE  =       0x00e1
ACC_2  =       0x00e2
ACC_3  =       0x00e3
ACC_CONTRD  =       0x00e4
ACC_5  =       0x00e5
ACC_SWBSY  =       0x00e6
ACC_BUSY  =       0x00e7
P2IF   =       0x00e8
UTX0IF =       0x00e9
UTX1IF =       0x00ea
P1IF   =       0x00eb
WDTIF  =       0x00ec
B_0    =       0x00f0
B_1    =       0x00f1
B_2    =       0x00f2
B_3    =       0x00f3
B_4    =       0x00f4
B_5    =       0x00f5
B_6    =       0x00f6
B_7    =       0x00f7
ACTIVE =       0x00f8
TX_BYTE        =       0x00f9
RX_BYTE        =       0x00fa
ERR    =       0x00fb
FE     =       0x00fc
SLAVE  =       0x00fd
RE     =       0x00fe
MODE   =       0x00ff
'''
