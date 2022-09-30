import sys
import os
import logging
import bpy
from PyQt5 import QtWidgets, QtCore

logger = logging.getLogger('qtutils')
logger.setLevel('INFO')


class QtWindowEventLoop(bpy.types.Operator):
  """Allows PyQt or PySide to run inside Blender"""

  bl_idname = 'screen.qt_event_loop'
  bl_label = 'Qt Event Loop'

  def __init__(self, widget=None, *args, **kwargs):
    self._widget = widget
    self._args = args
    self._kwargs = kwargs
    self.widget = None

  def modal(self, context, event):
    # bpy.context.window_manager
    wm = context.window_manager

    if self.widget and not self.widget.isVisible():
      # if widget is closed
      logger.debug('finish modal operator')
      wm.event_timer_remove(self._timer)
      return {'FINISHED'}
    else:
      logger.debug('process the events for Qt window')
      self.event_loop.processEvents()
      self.app.sendPostedEvents(None, 0)

    return {'PASS_THROUGH'}

  def execute(self, context):
    logger.debug('execute operator')

    self.app = QtWidgets.QApplication.instance()
    # instance() gives the possibility to have multiple windows
    # and close it one by one

    if not self.app:
      # create the first instance
      self.app = QtWidgets.QApplication(sys.argv)

    if 'stylesheet' in self._kwargs:
      stylesheet = self._kwargs['stylesheet']
      self.set_stylesheet(self.app, stylesheet)

    self.event_loop = QtCore.QEventLoop()
    #self.widget = self._widget(*self._args, **self._kwargs)

    logger.debug(self.app)
    #logger.debug(self.widget)

    # run modal
    wm = context.window_manager
    self._timer = wm.event_timer_add(1 / 120, window=context.window)
    context.window_manager.modal_handler_add(self)

    return {'RUNNING_MODAL'}

  def set_stylesheet(self, app, filepath):
    file_qss = QtCore.QFile(filepath)
    if file_qss.exists():
      file_qss.open(QtCore.QFile.ReadOnly)
      stylesheet = QtCore.QTextStream(file_qss).readAll()
      app.setStyleSheet(stylesheet)
      file_qss.close()

blender_init = False

def init_blender():
  import bpy
  bpy.utils.register_class(QtWindowEventLoop)
  bpy.ops.screen.qt_event_loop()
  blender_init = True
