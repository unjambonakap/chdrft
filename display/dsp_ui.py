from chdrft.config.env import qt_imports
import pyqtgraph as pg
import glog

from chdrft.utils.misc import to_list
from chdrft.utils.fmt import fmt
from chdrft.display.utils import *
import chdrft.dsp.line as dsp_line


def console_txt_to_html(txt):
  if isinstance(txt, list):
    txt = '\n'.join(txt)
  txt = txt.replace('\n', '<br/>')
  return txt


class StatsWidget(pg.GraphicsWidget, pg.GraphicsWidgetAnchor):

  def __init__(self, plot):
    pg.GraphicsWidget.__init__(self)
    pg.GraphicsWidgetAnchor.__init__(self)
    self.setFlag(self.ItemIgnoresTransformations)
    self.layout = qt_imports.QtWidgets.QGraphicsGridLayout()
    self.setLayout(self.layout)
    self.plot = plot

    self.setParentItem(self.plot.graphicsItem())
    self.label = pg.LabelItem(justify='right')
    #self.label.setParentItem(self.plot.graphicsItem())
    self.layout.addItem(self.label, 0, 0)
    self.label.show()
    self.anchor((0, 0), (0.9, 0), offset=(0, 0))
    self.set_text('abcddef')

  def set_text(self, text):
    self.label.setText(console_txt_to_html('\n'.join(to_list(text))))


class DspTools:

  def __init__(self, plot):
    self.plot = plot
    self.regions = self.plot.regions
    self.stats = StatsWidget(plot)
    self.stats.show()

  def setup_menu(self, base_menu):
    menu = base_menu.add_menu('dsp')
    menu.addAction('stats', self.action_stats)
    menu.addAction('i2c', self.action_i2c)

  def get_active_region(self):
    return self.plot.regions.active_region

  def get_filtered_data_in_region(self, region):
    interval = Intervals(region.getRegion())
    data = []
    for plot in self.plot.opa_plots:
      data.append([plot, interval.filter_dataset(plot.data)])
    return interval, data

  def action_stats(self):
    region = self.get_active_region()
    if not region:
      return
    self.stats_label.show()
    interval, data_list = self.get_filtered_data_in_region(region)

    res = []
    for plot, data in data_list:
      stats = dsp_line.compute_stats(filtered)
      res.append([plot, stats])
    self.stats.set_text(self.get_all_stats_text(interval, res))

  def get_all_stats_text(self, interval, stats_list):
    res = [str(interval)]
    for plot, stats in stats_list:
      text = self.get_stats_text(plot, stats)
      res.extend(text)
    return res

  def get_stats_text(self, plot, stats):
    return '%s: %s' % (plot.ds.name, str(stats))

  def action_i2c(self):
    region = self.get_active_region()
    if not region:
      glog.info('no active region for i2c data')
      return
    interval, data_list = self.get_filtered_data_in_region(region)
    if len(data_list) != 2:
      glog.info('Can\'t compute i2c stuff if not only two datasets')
      return

    res = dsp_line.compute_i2c_data([data for _, data in data_list])
    lst = []
    lst.append(fmt(res).bit().v)
    lst.append(fmt(res).bin2byte().byte_str().v)
    print('\n'.join(lst))

  def notify_remove_region(self, region):
    pass
