#/usr/bin/env python

import keyring
import argparse
import getpass


class Keyring:
    store_key = 'opa_keyring'

    def __init__(self):
        self.kr = keyring.backends.Gnome.Keyring()

    def get_or_set(self, key, help=''):
        pwd = self.kr.get_password(self.store_key, key)
        if not pwd:
            pwd = getpass.getpass(help)
            self.kr.set_password(self.store_key, key, pwd)
        return pwd

    def delete(self, key):
        try:
            self.kr.delete_password(self.store_key, key)
        except keyring.errors.PasswordDeleteError:
            print('failed to delete password {}'.format(key))
            pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--key', type=str, required=True)
    args = parser.parse_args()

    x = Keyring()
    x.delete(args.key)
