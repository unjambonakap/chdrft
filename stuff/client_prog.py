def main():
  a = ctypes.c_float()
  b = ctypes.c_float()
  c = ctypes.c_float()
  a16, b16 = ctypes.c_short(), ctypes.c_short()

  if 1:
    sx = ctypes.c_int(0)
    sy = ctypes.c_int(300)
    width, height = 768, 484
    nx = ctypes.c_int(width)
    ny = ctypes.c_int(height)
    buf = (ctypes.c_char * (width * height))()
    x.GetOverlayImagePart(sx, sy, nx, ny, buf)
    sys.stdout.write(buf.raw)
    return

  if 0:

    if 1:
      x.GetPosition(ctypes.byref(a), ctypes.byref(b), ctypes.byref(c))
      print(a, b, c)
      x.SetPositionXY(ctypes.c_float(0.210), ctypes.c_float(-0.290))
      for i in range(10):
        x.IsStageMoving(ctypes.byref(a16))
        print(a16)
        if a16 == 0: break
      return
      s = ctypes.create_unicode_buffer('1'*39)
      s2 = ctypes.create_unicode_buffer('bonjour')
      x.SetDatabarUserText(s2)
      a = ctypes.cast(s, ctypes.c_wchar_p)
      x.GetDatabarUserText(ctypes.byref(a))
      print(a)
    else:
      #s = ctypes.create_unicode_buffer('T2.TIF')
      s = ctypes.create_unicode_buffer('C:\\benoit\\T2.TIF')
      #b = ctypes.c_short(1<<13 | 1<<4)
      b = ctypes.c_short(0<<13 | 1<<4)
      x.WriteTIFFImage(ctypes.addressof(s) ,b)
    return

  if 1:
    x.GetPosition(ctypes.byref(a), ctypes.byref(b), ctypes.byref(c))
    print(a, b, c)
    x.GetHighTension(ctypes.byref(a))
    print(a)
    x.GetContrast(ctypes.byref(a))
    print(a)
    x.GetBrightness(ctypes.byref(a))
    print(a)
    x.GetSASizeX(ctypes.byref(a))
    print('sa size', a)
    x.GetSASizeY(ctypes.byref(a))
    print(a)
    x.GetZ(ctypes.byref(a))
    print(a)
    x.GetMagnification(ctypes.byref(a))
    print(a)
    x.GetScreenMetrics(ctypes.byref(a), ctypes.byref(b))
    print(a, b)
    x.GetNrOfLines(ctypes.byref(a16))
    x.GetLineTime(ctypes.byref(a16))
    print('nr LINES >> ',a16)

    x.ReadDBFloat(ctypes.c_short(109), ctypes.byref(a))
    print(a)
  #a = ctypes.c_int(1)
  #b = ctypes.c_int()
  #x.Test(a, ctypes.byref(b))

  else:
    x.SetFilterMode(ctypes.c_short(2), ctypes.c_short(0))
    for i in range(1000):
      time.sleep(0.01)
      x.GetFilterMode(ctypes.byref(a16), ctypes.byref(b16))
      if a16.value == 3:
        width, height =  768, 484
        buf = (ctypes.c_char * (width * height))()
        x.GetImage(buf)
        print(base64.b64encode(buf.raw))
        return
    print('failed')


main()
