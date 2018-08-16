global callMain
DEFAULT ABS
callMain:
;password, buf, buflen
;ret=int

    push esi
    push edi
    mov esi, [esp+010h]
    mov edi, [esp+014h]
    mov eax, [esp+0ch]
    sub esp, 1000h


    mov [esi+edi-34h], eax
    mov [esp+1C0h], esi
    lea eax, [esi+edi]
    mov [esp+1BCh], eax
    lea eax, [esi+edi-38h]
    mov dword [esp+60h], 0
    mov [esp+40h], eax
    lea eax, [esp+28h]
    mov [esp], eax
    mov eax, 8048660h
    push ret1
    jmp eax
ret1:
    sub esp, 4
    mov eax, [esp+28h]


    add esp, 1000h
    pop edi
    pop esi
    retn

