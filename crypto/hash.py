import crypto_swig
import struct
import binascii


class HashType:
  SHA1 = 0,
  SHA256 = 1,
  SHA512 = 2,


def hash_factory(hash_type):
  if hash_type == HashType.SHA1:
    return crypto_swig.Sha1()
  elif hash_type == HashType.SHA256:
    return crypto_swig.Sha256()
  elif hash_type == HashType.SHA512:
    return crypto_swig.Sha512()
  else:
    assert False


def length_extension(hash_type, base_hash, base_len, extra):
  h = hash_factory(hash_type)

  suffix = [0x80]
  block_size = h.get_block_size()
  append_len_size = h.get_append_len_size()
  target_mod = block_size - append_len_size

  suffix += [0] * ((target_mod - 1 - base_len) % block_size)

  if append_len_size == 8:
    suffix += struct.pack('>Q', base_len * 8)
  elif append_len_size == 16:
    suffix += struct.pack('>QQ', 0, base_len * 8)
  else:
    assert False

  h.set_context(base_hash, base_len + len(suffix))
  h.update(extra)
  valid_hash = h.get()

  suffix += extra
  suffix = bytes(suffix)
  return suffix, valid_hash
