#!/usr/bin/env python
from pyknon.genmidi import Midi
from pyknon.music import NoteSeq, Note, Rest

from chdrft.cmds import CmdsList
from chdrft.main import app
from chdrft.utils.cmdify import ActionHandler
from chdrft.utils.misc import Attributize
import chdrft.utils.misc as cmisc
import glog
import random

global flags, cache
flags = None
cache = None

####### First we'll generate all piano notes

# Pyknon dubs C5 as the value 0, so Note(0) give you C5
# (C in the 5th octave).  So we have to move 51 keys to the
# left to get the far left key on a piano.
first_note = Note("A,,,,,")  # The far-left key on the piano


def key_number(n, octave=0):
  return (n.octave + octave) * 12 + n.value - first_note.value


def note_from_key_number(k, octave=0):
  if isinstance(k, Note): return k
  if isinstance(k, int): return Note(k - 51 + octave * 12)
  lst = []
  for x in k:
    lst.append(note_from_key_number(x, octave=octave))
  return lst


def intervals(notes):
  interval = []
  for n in range(len(notes) - 1):
    interval.append(notes[n + 1] - notes[n])
  return interval


piano_notes = list(map(note_from_key_number, list(range(88))))
#midi = Midi(1, tempo=80)
#midi.seq_notes(piano_notes, track=0)
#midi.write("piano_keys.mid")

####### Next we'll examine a major and minor scale, and look at the intervals between
####### each of their notes

middle_c = Note("C,")  # key_number=39
note_nums = list(map(key_number, piano_notes))
print("All piano notes:", note_nums)
print("Middle C key number:", key_number(middle_c))

# A, means drop the octave, C' means raise the octave.
# Also, in a NoteSeq, Pyknon stays in the same octave unless explicitly
# changed by using either , or '.
C_major = NoteSeq("C D E F G A B C'' ")
A_minor = NoteSeq("A, B C' D E F G A")

# Note, when defining a NoteSeq, all notes are by default in the same
# octave as the starting note.
print("C major (staring with middle C):", list(map(key_number, NoteSeq("C, D E F G A B"))))
print("C major:", list(map(key_number, C_major)))
print("Intervals for C major:", intervals(list(map(key_number, C_major))))
print("A minor:", list(map(key_number, A_minor)))
print("Intervals A minor:", intervals(list(map(key_number, A_minor))))

####### Last we'll generate some chords in a major and minor keys.


def major_chord(root, octave=0):
  root_key_num = key_number(root, octave)
  return [root_key_num, root_key_num + 4, root_key_num + 7]


def minor_chord(root, octave=0):
  root_key_num = key_number(root, octave)
  return [root_key_num, root_key_num + 3, root_key_num + 7]


def dim_chord(root, octave=0):
  root_key_num = key_number(root, octave)
  return [root_key_num, root_key_num + 3, root_key_num + 6]


def aug_chord(root, octave=0):
  root_key_num = key_number(root, octave)
  return [root_key_num, root_key_num + 4, root_key_num + 8]


def w(x):
  return x + 2


def h(x):
  return x + 1


def build_scale(scale_desc, root, octave=0):
  pos = root + octave * 12
  res = [pos]

  for i in scale_desc:
    pos = i(pos)
    res.append(pos)
  return res


maj_scale = [w, w, h, w, w, w, h]
min_scale = [w, h, w, w, h, w, w]

# Chord qualities: M m m M M m d (M)
major_chord_progression = [major_chord, minor_chord, minor_chord, major_chord, \
                           major_chord, minor_chord, dim_chord]

# Chord qualities: m d M m m M M (m)
minor_chord_progression = [minor_chord, dim_chord, major_chord, minor_chord, \
                           minor_chord, major_chord, major_chord]

intervals = [3, 4, 5, 7, 8, 9, 12]

####### Generate C major chords ###
C_maj_chords = []
for i in range(len(major_chord_progression)):
  C_maj_chords.append(major_chord_progression[i](C_major[i]))


def list_intervals(scale, octave_range):
  for octave in octave_range:
    for interval in intervals:
      for pos in scale:
        target = key_number(pos, octave=octave)
        if can_interval(target, interval, scale):
          yield (target, target + interval)


def list_keys(scale, octave_range):
  for octave in octave_range:
    for pos in scale:
      target = key_number(pos, octave=octave)
      yield (target,)


def list_chords(scale, octave_range):

  for octave in octave_range:
    for i in range(len(major_chord_progression)):
      yield major_chord_progression[i](scale[i], octave=octave)


def get_scale_data(scale, octave_range):
  return Attributize(
      chords=list(list_chords(scale, octave_range)),
      intervals=list(list_intervals(scale, octave_range)),
      singlekeys=list(list_keys(scale, octave_range)),
  )


# Throw a "mistake" in there to hear the difference
#C_maj_chords.append([Note("C"), Note("F#"), Note("Bb")])


def can_interval(pos, interval, scale):
  found = 0
  pos = note_from_key_number(pos)
  for e in scale:
    if pos.value == e.value: found |= 1
    if (pos.value + interval) % 12 == e.value: found |= 2
  return found == 3
