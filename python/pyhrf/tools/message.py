# -*- coding: iso-8859-1 -*-
# Copyright CEA and IFR 49 (2000-2005)
#
#  This software and supporting documentation were developed by
#      CEA/DSV/SHFJ and IFR 49
#      4 place du General Leclerc
#      91401 Orsay cedex
#      France
#
# This software is governed by the CeCILL license version 2 under
# French law and abiding by the rules of distribution of free software.
# You can  use, modify and/or redistribute the software under the
# terms of the CeCILL license version 2 as circulated by CEA, CNRS
# and INRIA at the following URL "http://www.cecill.info".
#
# As a counterpart to the access to the source code and  rights to copy,
# modify and redistribute granted by the license, users are provided only
# with a limited warranty  and the software's author,  the holder of the
# economic rights,  and the successive licensors  have only  limited
# liability.
#
# In this respect, the user's attention is drawn to the risks associated
# with loading,  using,  modifying and/or developing or reproducing the
# software by the user in light of its specific status of free software,
# that may mean  that it is complicated to manipulate,  and  that  also
# therefore means  that it is reserved for developers  and  experienced
# professionals having in-depth computer knowledge. Users are therefore
# encouraged to load and test the software's suitability as regards their
# requirements in conditions enabling the security of their systems and/or
# data to be ensured and,  more generally, to use and operate it in the
# same conditions as regards security.
#
# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license version 2 and that you accept its terms.

'''
This package provide a mean to print colored message to a standard terminal if
color is available else message are print in black and white mode. If stdout
is redirected in a file or piped to an other program, the output is made
black and white to avoid issus with strange caracters that defined colors
in terminals. Remember that all messages are print to stdout.

This package exists in 3 places, for some very good arguments :
     datamind.tools.message
     soma.wip.message
     pyhrf.tools.message
     
To use these functionalities, play with 'msg' instance. Here, some classical
uses :

    msg.info('something cool happend'):
    msg.error(self, 'too bad, an error'):
    msg.warning(self, 'something strange but not fatal'):
    msg.write_list(('no color', ('color in red', 'red'))):
    msg.write('simple colored write function')
    msg.string('string to colored string')

'''

import sys

# add your shell colors here :
colors = {
		'gray' : '\033[01;30m',
		'bold_red' : '\033[01;31m',
		'bold_green' : '\033[01;32m',
		'bold_yellow' : '\033[01;33m',
		'bold_blue' : '\033[01;34m',
		'bold_purple' : '\033[01;35m',
		'bold_cyan' : '\033[01;36m',
		'bold_white' : '\033[01;37m',
		'black' : '\033[02;30m',
		'dark_red' : '\033[02;31m',
		'dark_green' : '\033[02;32m',
		'dark_yellow' : '\033[02;33m',
		'dark_blue' : '\033[02;34m',
		'dark_purple' : '\033[02;35m',
		'dark_cyan' : '\033[02;36m',
		'dark_gray' : '\033[02;37m',
		'red' : '\033[03;31m',
		'green' : '\033[03;32m',
		'yellow' : '\033[03;33m',
		'blue' : '\033[03;34m',
		'purple' : '\033[03;35m',
		'cyan' : '\033[03;36m',
		'gray' : '\033[03;37m',
		'underline_red' : '\033[04;31m',
		'underline_green' : '\033[04;32m',
		'underline_yellow' : '\033[04;33m',
		'underline_blue' : '\033[04;34m',
		'underline_purple' : '\033[04;35m',
		'underline_cyan' : '\033[04;36m',
		'underline_gray' : '\033[04;37m',
		'underline_white' : '\033[04;38m',
		'thin_red' : '\033[05;31m',
		'thin_green' : '\033[05;32m',
		'thin_yellow' : '\033[05;33m',
		'thin_blue' : '\033[05;34m',
		'thin_purple' : '\033[05;35m',
		'thin_cyan' : '\033[05;36m',
		'thin_gray' : '\033[05;37m',
		'invert_red' : '\033[07;31m',
		'invert_green' : '\033[07;32m',
		'invert_yellow' : '\033[07;33m',
		'invert_blue' : '\033[07;34m',
		'invert_purple' : '\033[07;35m',
		'invert_cyan' : '\033[07;36m',
		'invert_gray' : '\033[07;37m',
		'invert_white' : '\033[07;38m',
		'cross_red' : '\033[09;31m',
		'cross_green' : '\033[09;32m',
		'cross_yellow' : '\033[09;33m',
		'cross_blue' : '\033[09;34m',
		'cross_purple' : '\033[09;35m',
		'cross_cyan' : '\033[09;36m',
		'cross_gray' : '\033[09;37m',
		'cross_white' : '\033[09;38m',
		'back' : '\x1b[00m'}


class MessageColor(object):
	def haveColor(self):
		return True

	def info(self, msg):
		print ' *', colors['bold_green'] + 'Info' + \
				colors['back'], ':', msg

	def error(self, msg):
		sys.stderr.write(' * ' + colors['bold_red'] + 'Error' + \
			colors['back'] + ' : ' + msg + '\n')

	def warning(self, msg):
		print ' *', colors['bold_yellow'] + 'Warning' + \
			colors['back'], ':', msg

	def write_list(self, msg_list):
		for l in msg_list :
			if type(l) in [list, tuple]:
				self.write(*l)
			else:	self.write(l)

	def write(self, msg, color = 'back'):
		sys.stdout.write(self.string(msg, color))

	def string(self, msg, color = 'back'):
		if color[0] == '\033':
			c = color
		else:	c = colors[color]
		return c + str(msg) + colors['back']

	string = classmethod(string)
	write = classmethod(write)
	write_list = classmethod(write_list)
	error = classmethod(error)
	warning = classmethod(warning)
	info = classmethod(info)

class MessageNoColor(MessageColor):
	def haveColor(self):
		return False

	def info(self, msg):
		print ' * Info :', msg

	def error(self, msg):
		sys.stderr.write(' * Error : ' + msg + '\n')

	def warning(self, msg):
		print ' * Warning :', msg

	def string(self, msg, color = 'back'):
		return str(msg)

	string = classmethod(string)
	error = classmethod(error)
	warning = classmethod(warning)
	info = classmethod(info)

class Message(object):
	import sys

	# FIXME : add settings check for colors here
	if sys.stdout.isatty():
		_msg = MessageColor
	else:	_msg = MessageNoColor

	def __new__(cls, *args, **kwargs):
		return Message._msg.__new__(Message._msg, *args, **kwargs)

class NoMessage(object):
	def haveColor(self):
		return False

	def info(self, msg): pass
	def error(self, msg): pass
	def warning(self, msg): pass
	def string(self, msg, color = 'back'): pass
	def write(self, msg, color = 'back'): pass
	def write_list(self, msg_list): pass



msg = Message()
