#!/usr/bin/env python
from chdrft.utils.misc import Attributize

class Action:
  action_name = ''

  def do(self):
    assert 0

  def configure_args(self, parser):
    return


  @staticmethod
  def register_actions(parser, *actions, required=False, **kwargs):
    sp = parser.add_subparsers()
    sp.required = required
    res=Attributize()
    actions=list(actions)

    for k,v in kwargs.items():
      actions.append(Action.create_action(k, v))

    for action in actions:
      if isinstance(action, list) or isinstance(action, tuple):
        action = Action.create_action(*action)

      parser = sp.add_parser(action.action_name)
      parser.set_defaults(func=action.do)
      action.configure_args(parser)
      res[action.action_name] = parser

    return res


  @staticmethod
  def create_action(action_name, func, *args_func):
    new_action = Action()
    new_action.action_name = action_name
    new_action.do = lambda: func()

    def configure_args_func(parser):
      for x in args_func:
        args_func(parser)

    new_action.configure_args = configure_args_func
    return new_action


  @staticmethod
  def proc_action(flags):
    flags.func()
