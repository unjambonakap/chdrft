from parsimonious.grammar import Grammar
from parsimonious import NodeVisitor
from parsimonious import expressions
from chdrft.utils.misc import to_list, Attributize, to_dict

BaseGrammar = '''
digit = ~"[0-9]"
int = "-"? digit+
word = ~"[0-9a-zA-Z_]+"
float = int ("." int)?
whitespaces = ~"[ \t]*"
whitespaces1 = ~"[ \t]+"
to_eol_w = whitespaces "\\n"
to_eol = ~"[^\\n]*\\n"
word_list = word ("," whitespaces word)*
'''


def build_grammar(grammar):
  grammar = grammar + BaseGrammar
  return Grammar(grammar)


class BaseVisitor(NodeVisitor):

  def __init__(self, ignore_nodes=[], atoms=[], guess_mode=True):
    ignore_nodes = list(ignore_nodes)
    ignore_nodes.extend(to_list('whitespaces whitespaces1 to_eol to_eol_w'))
    self.ignore_nodes = ignore_nodes
    self.atoms=atoms

    self.guess_mode = guess_mode

  def visit_int(self, node, vc):
    return int(node.text)

  def visit_float(self, node, vc):
    return float(node.text)

  def visit_word(self, node, vc):
    return str(node.text)

  def generic_visit(self, node, vc):
    name = node.expr_name
    print('on ', name)
    if name in self.atoms:
      return node.text
    if self.guess_mode:
      lst = []
      for n,out in zip(node, vc):
        if out is not None:
          lst.append([n.expr_name, out])
      if len(lst)==0:
         return None

      if isinstance(node.expr, expressions.Compound):
        if isinstance(node.expr, expressions.OneOf):
          if len(lst)==0:
            return None
          assert len(lst)==1
          return Attributize(**to_dict(lst))
        elif isinstance(node.expr, expressions.Sequence):
          if len(lst)==1:
            return lst[0][1]
          return Attributize(**to_dict(lst))
        else:
          return [x[1] for x in lst]
      else:
        return None

    else:
      if name in self.ignore_nodes:
        res = None
        for n in node:
          cur = self.visit(n)
          if cur is not None:
            assert res is None
            res = cur
        return res
      else:
        assert 0
